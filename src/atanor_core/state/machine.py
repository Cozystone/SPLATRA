"""Unified hologram state machine — the heart of the engine.

Cache-miss flow (PRD §4 Phase 4)::

    IDLE -> GENERATING (~5s, morph animation runs concurrently)
         -> [VERIFYING]  (only on DePIN refinement: held-out PSNR gate)
         -> SWAP_READY   (verified cartridge pinned, hot-swap signal)
         -> DISPLAYED    (0ms swap done, field is live)

Honesty (PRD §4 note 3): **"0ms" is a cache-HIT asymptotic target.** A cache
hit goes straight to DISPLAYED with no generation thread. A cache MISS spends
~5s on a background generation thread while a coalescence morph animates; the
hot-swap itself is what is ~0ms. This distinction is enforced by the code
structure, not just documented.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..deformation.fourier import FourierDeformer
from ..domain.sgf import Cartridge, GaussianField


class HoloState(str, Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    MORPHING = "MORPHING"
    VERIFYING = "VERIFYING"
    SWAP_READY = "SWAP_READY"
    DISPLAYED = "DISPLAYED"
    ERROR = "ERROR"


@dataclass
class EngineEvent:
    state: HoloState
    info: Dict[str, Any] = _field(default_factory=dict)


class HologramEngine:
    """Port-only state machine. Swap mock<->real adapters freely.

    Args (all ports):
        mapper:     GraphMapperPort
        generator:  GeneratorPort
        rasterizer: RasterizerPort
        compressor: CompressorPort
        verifier:   VerifierPort
        coalesce_turbulence: turbulence used for the cache-miss waiting morph.
    """

    def __init__(
        self,
        mapper,
        generator,
        rasterizer,
        compressor,
        verifier,
        coalesce_turbulence: float = 0.35,
    ) -> None:
        self.mapper = mapper
        self.generator = generator
        self.rasterizer = rasterizer
        self.compressor = compressor
        self.verifier = verifier
        self.coalesce_turbulence = float(coalesce_turbulence)

        self.state: HoloState = HoloState.IDLE
        self.field: Optional[GaussianField] = None
        self.deformer: Optional[FourierDeformer] = None
        self.cache: Dict[str, Cartridge] = {}

        self._events: List[EngineEvent] = []
        # Background generation bookkeeping.
        self._gen_thread: Optional[threading.Thread] = None
        self._gen_lock = threading.Lock()
        self._pending_field: Optional[GaussianField] = None
        self._gen_name: Optional[str] = None
        self._gen_error: Optional[str] = None
        # Knowledge-graph edge overlay (rendered, not part of the morph buffer).
        self._edges: List[tuple] = []
        self._edge_samples: int = 7
        self.show_edges: bool = True

    # -- events ------------------------------------------------------------ #
    def _emit(self, state: HoloState, **info: Any) -> None:
        self.state = state
        self._events.append(EngineEvent(state=state, info=info))

    def drain_events(self) -> List[EngineEvent]:
        """Return and clear emitted events (observe SWAP_READY here, not in tick)."""
        evs = self._events
        self._events = []
        return evs

    # -- static graph render ---------------------------------------------- #
    def render_knowledge_hologram(self, graph: Dict[str, Any]) -> GaussianField:
        """Map a graph to a static field and display it (direct path)."""
        field = self.mapper.map(graph)
        self.field = field
        self.deformer = FourierDeformer(field.means)
        # Parse edges into node-index pairs (node order == field order).
        ids = [str(n.get("id", i)) for i, n in enumerate(graph.get("nodes", []))]
        idx = {nid: i for i, nid in enumerate(ids)}
        self._edges = []
        for e in graph.get("edges", []):
            si, di = idx.get(str(e.get("src"))), idx.get(str(e.get("dst")))
            if si is not None and di is not None and si != di:
                self._edges.append((si, di))
        self._emit(HoloState.DISPLAYED, num_gaussians=field.num_gaussians, source="graph")
        return field

    # -- multi-agent relayout --------------------------------------------- #
    def relayout(self, new_positions: np.ndarray, turbulence: float = 0.1) -> None:
        """Morph the current field to new positions (debate-driven relayout)."""
        if self.field is None or self.deformer is None:
            raise RuntimeError("nothing to relayout; render a graph first")
        self.deformer.trigger_morph(np.asarray(new_positions, dtype=np.float32), turbulence)
        self._emit(HoloState.MORPHING, turbulence=float(turbulence))

    # -- async generation -------------------------------------------------- #
    def generate_3d_object(
        self,
        name: str,
        mv_images: np.ndarray,
        cam_rays: np.ndarray,
        placeholder: Optional[GaussianField] = None,
    ) -> str:
        """Generate (or fetch from cache) a 3D object.

        Cache HIT  -> straight to DISPLAYED, no generation thread (the "0ms" case).
        Cache MISS -> spawn a background generation thread AND immediately start a
                      coalescence morph; state becomes GENERATING.
        """
        if name in self.cache:
            # Cache hit: 0ms asymptotic path. No GENERATING, no thread.
            cart = self.cache[name]
            self.field = self.compressor.decompress(cart)
            self.deformer = FourierDeformer(self.field.means)
            self._edges = []  # an object, not a graph
            self._emit(
                HoloState.DISPLAYED,
                name=name,
                cache="hit",
                num_gaussians=self.field.num_gaussians,
            )
            return "hit"

        # Cache miss: background generation (~5s in reality).
        if placeholder is not None:
            self.field = placeholder
            self.deformer = FourierDeformer(placeholder.means)

        # Immediately start a coalescence morph on the current field so the
        # viewer sees motion while generation runs in the background.
        if self.field is not None and self.deformer is not None:
            self.deformer.trigger_morph(self.field.means.copy(), self.coalesce_turbulence)

        self._pending_field = None
        self._gen_name = name
        self._gen_error = None

        def _worker(mv, rays):
            try:
                result = self.generator.generate(mv, rays)
                with self._gen_lock:
                    self._pending_field = result
            except Exception as exc:  # pragma: no cover - mock never raises
                with self._gen_lock:
                    self._gen_error = str(exc)

        self._gen_thread = threading.Thread(
            target=_worker, args=(mv_images, cam_rays), daemon=True
        )
        self._gen_thread.start()
        self._emit(HoloState.GENERATING, name=name, cache="miss")
        return "miss"

    # -- the tick: advance morph, detect completion, verify, swap --------- #
    def tick(
        self,
        name: Optional[str] = None,
        reference_views: Optional[Sequence[np.ndarray]] = None,
        cameras: Optional[Sequence[Dict[str, Any]]] = None,
        dt: float = 0.033,
    ) -> HoloState:
        """Advance one frame. May emit SWAP_READY and DISPLAYED in one call.

        Observe emitted events via :meth:`drain_events`, not this return value
        (PRD §4 timing warning).
        """
        # Advance the coalescence / relayout morph animation.
        if self.deformer is not None and not self.deformer.done:
            self.deformer.step(dt=dt)

        with self._gen_lock:
            pending = self._pending_field
            gen_error = self._gen_error

        if gen_error is not None:
            self._emit(HoloState.ERROR, error=gen_error)
            self._gen_error = None
            return self.state

        # Still generating (thread not finished) -> stay in GENERATING.
        if self.state == HoloState.GENERATING and pending is None:
            return self.state

        # Generation just completed: verify -> compress -> pin -> swap.
        if pending is not None:
            gen_name = name or self._gen_name or "object"

            verified = True
            if reference_views and cameras:
                self._emit(HoloState.VERIFYING, name=gen_name)
                verified = bool(self.verifier.verify(pending, reference_views, cameras))
                if not verified:
                    # Untrusted DePIN result failed the gate -> reject, no swap.
                    with self._gen_lock:
                        self._pending_field = None
                    self._emit(HoloState.ERROR, name=gen_name, reason="psnr_gate_reject")
                    return self.state

            cartridge = self.compressor.compress(gen_name, pending)
            cartridge.verified = verified
            self.cache[gen_name] = cartridge  # pin
            self._emit(
                HoloState.SWAP_READY,
                name=gen_name,
                verified=verified,
                estimated_bytes=cartridge.meta.get("estimated_compressed_bytes"),
            )

            # 0ms hot-swap: replace the live buffer in one assignment.
            self.field = pending
            self.deformer = FourierDeformer(self.field.means)
            self._edges = []  # generated object replaces the graph
            with self._gen_lock:
                self._pending_field = None
            self._gen_thread = None
            self._emit(
                HoloState.DISPLAYED,
                name=gen_name,
                cache="miss",
                num_gaussians=self.field.num_gaussians,
            )
            return self.state

        return self.state

    # -- render ------------------------------------------------------------ #
    def _edge_overlay(self, node_field: GaussianField) -> GaussianField:
        """Append thin, dim Gaussians sampled along edges (render-time only).

        Edges are recomputed from current node positions every frame, so they
        track morphs without being part of the fixed-N deformation buffer.
        """
        if not (self.show_edges and self._edges):
            return node_field
        m = node_field.means
        sh0 = node_field.sh[:, 0, :]
        s = np.linspace(0.15, 0.85, self._edge_samples, dtype=np.float32)
        pts, cols = [], []
        for si, di in self._edges:
            seg = m[si][None, :] * (1 - s[:, None]) + m[di][None, :] * s[:, None]
            col = sh0[si][None, :] * (1 - s[:, None]) + sh0[di][None, :] * s[:, None]
            pts.append(seg)
            cols.append(col)
        ep = np.concatenate(pts, 0).astype(np.float32)
        ec = np.concatenate(cols, 0).astype(np.float32)
        ne = ep.shape[0]
        escales = np.log(np.full((ne, 3), 0.012, dtype=np.float32))
        equats = np.zeros((ne, 4), dtype=np.float32)
        equats[:, 0] = 1.0
        eop = np.full((ne,), -0.4, dtype=np.float32)  # sigmoid ~0.4, dim
        k = node_field.sh.shape[1]
        esh = np.zeros((ne, k, 3), dtype=np.float32)
        esh[:, 0, :] = ec * 0.7  # slightly dimmer than nodes

        return GaussianField(
            means=np.concatenate([m, ep], 0),
            scales=np.concatenate([node_field.scales, escales], 0),
            quats=np.concatenate([node_field.quats, equats], 0),
            opacities=np.concatenate([node_field.opacities, eop], 0),
            sh=np.concatenate([node_field.sh, esh], 0),
            sh_degree=node_field.sh_degree,
        )

    def render(
        self, viewmat: np.ndarray, K: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Render the current field (deformed positions if mid-morph, + edges)."""
        if self.field is None:
            raise RuntimeError("nothing to render; call render_knowledge_hologram first")
        live = self.field
        if self.deformer is not None and not self.deformer.done:
            live = self.field.copy()
            live.means = self.deformer.positions
        live = self._edge_overlay(live)
        return self.rasterizer.render(live, viewmat, K, width, height)
