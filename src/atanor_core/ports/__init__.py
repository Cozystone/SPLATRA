"""Abstract ports (Ports & Adapters / Clean Architecture).

These interfaces depend **only** on the pure :mod:`atanor_core.domain` layer.
They know nothing about torch, gsplat, diffusers, or any heavy adapter. The
state machine and plugin API talk only to these ports, so a mock PoC adapter
and a real GPU adapter are interchangeable.

Honesty note (PRD §7): :class:`TabularisPort` and :class:`MiroFishPort` are
**spec-undefined placeholders**. ATANOR's philosophy names them but does not
define them. We expose them as abstract slots ONLY. Implementing them with
guessed behavior would bake a wrong assumption into the architecture. When a
human provides a spec, an adapter gets plugged into the slot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..domain.sgf import Cartridge, DeformationCoeffs, GaussianField


@runtime_checkable
class GraphMapperPort(Protocol):
    """Map a knowledge graph (JSON-ish dict) to a static GaussianField."""

    def map(self, graph: Dict[str, Any]) -> GaussianField: ...


@runtime_checkable
class DeformerPort(Protocol):
    """MLP-free spectral deformation: build coeffs and step them in time."""

    def trigger_morph(
        self, new_positions: np.ndarray, turbulence: float = 0.0
    ) -> None: ...

    def step(self, t: Optional[float] = None, dt: float = 0.033) -> np.ndarray: ...

    @property
    def done(self) -> bool: ...


@runtime_checkable
class GeneratorPort(Protocol):
    """Generate a GaussianField from multi-view images + camera rays."""

    def generate(
        self, mv_images: np.ndarray, cam_rays: np.ndarray
    ) -> GaussianField: ...


@runtime_checkable
class RasterizerPort(Protocol):
    """Render a GaussianField to an [H, W, 3] image in [0, 1]."""

    def render(
        self,
        field: GaussianField,
        viewmat: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray: ...


@runtime_checkable
class CompressorPort(Protocol):
    """Compress / decompress a GaussianField into / out of a Cartridge."""

    def compress(self, name: str, field: GaussianField) -> Cartridge: ...

    def decompress(self, cartridge: Cartridge) -> GaussianField: ...

    def estimate_compressed_bytes(self, field: GaussianField) -> int: ...


@runtime_checkable
class VerifierPort(Protocol):
    """Client-side quality gate over a held-out view (e.g. PSNR)."""

    def verify(
        self,
        field: GaussianField,
        reference_views: Sequence[np.ndarray],
        cameras: Sequence[Dict[str, Any]],
    ) -> bool: ...


@runtime_checkable
class LLMPort(Protocol):
    """Chat with tool-calling. Returns {"content": str, "tool_calls": [...]}.

    Each tool call is ``{"name": str, "arguments": dict}``. Adapters: a local
    Ollama client, or an offline heuristic intent parser (no LLM required).
    """

    def chat(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]: ...


@runtime_checkable
class VectorIndexPort(Protocol):
    """Vector index for query-relevance Gaussian culling (e.g. turbovec_rs)."""

    def build(self, vectors: np.ndarray) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> List[int]: ...


# --------------------------------------------------------------------------- #
# Spec-undefined placeholders — DO NOT IMPLEMENT (PRD §7.1).
# --------------------------------------------------------------------------- #


@runtime_checkable
class TabularisPort(Protocol):
    """PLACEHOLDER. Spec undefined. Reserved abstract slot only.

    ATANOR names "Tabularis" but provides no definition. No method contract is
    asserted here on purpose; a real interface will be carved when a human
    supplies the spec. Until then, no adapter should claim to satisfy this.
    """

    ...


@runtime_checkable
class MiroFishPort(Protocol):
    """PLACEHOLDER. Spec undefined. Reserved abstract slot only.

    ATANOR names "MiroFish" but provides no definition. See
    :class:`TabularisPort`. Intentionally empty; awaiting spec.
    """

    ...


__all__ = [
    "GraphMapperPort",
    "DeformerPort",
    "GeneratorPort",
    "RasterizerPort",
    "CompressorPort",
    "VerifierPort",
    "LLMPort",
    "VectorIndexPort",
    "TabularisPort",
    "MiroFishPort",
]
