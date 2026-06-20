"""End-to-end pipeline tests — CPU/numpy only, no torch, no downloads.

Target: ``pytest -q`` -> 7 passed. Each test maps to a Definition-of-Done
gate (G1..G6) from the PRD.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from atanor_core import build_default_engine
from atanor_core.deformation.fourier import FourierDeformer
from atanor_core.domain.sgf import GaussianField, rgb_to_sh_dc, sh_dc_to_rgb
from atanor_core.mapping.graph_mapper import GraphMapper
from atanor_core.state.machine import HoloState
from atanor_core.state.rasterizer import CPURasterizer, default_intrinsics, look_at
from atanor_core.verification.psnr_gate import PSNRGate


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_graph(n: int = 24, dim: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    nodes = []
    for i in range(n):
        nodes.append(
            {
                "id": f"n{i}",
                "embedding": rng.normal(size=dim).astype(np.float32).tolist(),
                "centrality": float(rng.uniform(0.0, 10.0)),
                "importance": float(rng.uniform(0.1, 0.9)),
                "category": int(i % 12),
            }
        )
    edges = [{"src": f"n{i}", "dst": f"n{(i + 1) % n}"} for i in range(n)]
    return {"nodes": nodes, "edges": edges}


def make_mv_images(color=(0.2, 0.7, 0.4), v: int = 4, h: int = 8, w: int = 8) -> np.ndarray:
    img = np.zeros((1, v, 3, h, w), dtype=np.float32)
    for c in range(3):
        img[:, :, c, :, :] = color[c]
    return img


def make_camera(width: int, height: int):
    viewmat = look_at(eye=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0))
    K = default_intrinsics(width, height, fov_deg=60.0)
    return viewmat, K


def make_dense_field(seed: int, n: int = 300) -> GaussianField:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32)
    scales = np.log(np.full((n, 3), 0.15, dtype=np.float32))
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.full((n,), 2.5, dtype=np.float32)
    sh = np.zeros((n, 4, 3), dtype=np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32))
    return GaussianField(means, scales, quats, opacities, sh, sh_degree=1)


def run_until_displayed(engine, name: str, max_ticks: int = 400, **tick_kw):
    """Tick until the named object is cached, collecting emitted states."""
    seen = []
    for _ in range(max_ticks):
        engine.tick(name, **tick_kw)
        for ev in engine.drain_events():
            seen.append(ev.state)
        if name in engine.cache and engine.state == HoloState.DISPLAYED:
            break
        time.sleep(0.005)
    return seen


# --------------------------------------------------------------------------- #
# G1 — graph ingest shapes
# --------------------------------------------------------------------------- #
def test_graph_ingest_shapes():
    engine = build_default_engine()
    graph = make_graph(n=24)
    field = engine.render_knowledge_hologram(graph)

    assert field.means.shape == (24, 3)
    assert field.sh.shape == (24, 4, 3)  # sh_degree=1 -> K=4
    assert field.quats.shape == (24, 4)
    assert engine.state == HoloState.DISPLAYED
    # positions normalized into the [-1, 1] cube
    assert field.means.min() >= -1.0001 and field.means.max() <= 1.0001


# --------------------------------------------------------------------------- #
# G1 — color roundtrip
# --------------------------------------------------------------------------- #
def test_color_roundtrip():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 1.0, size=(50, 3)).astype(np.float32)
    back = sh_dc_to_rgb(rgb_to_sh_dc(rgb))
    assert np.allclose(back, rgb, atol=1e-5)


# --------------------------------------------------------------------------- #
# G6 — render produces an image
# --------------------------------------------------------------------------- #
def test_render_produces_image():
    engine = build_default_engine()
    engine.render_knowledge_hologram(make_graph(n=24))
    viewmat, K = make_camera(64, 64)
    img = engine.render(viewmat, K, 64, 64)

    assert img.shape == (64, 64, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.sum() > 0.0  # something was actually drawn


# --------------------------------------------------------------------------- #
# G2 — Fourier morph converges
# --------------------------------------------------------------------------- #
def test_fourier_morph_converges():
    rng = np.random.default_rng(2)
    origin = rng.uniform(-1.0, 1.0, size=(32, 3)).astype(np.float32)
    target = rng.uniform(-1.0, 1.0, size=(32, 3)).astype(np.float32)

    deformer = FourierDeformer(origin, period=1.0, seed=7)
    deformer.trigger_morph(target, turbulence=0.3)
    pos = origin
    for _ in range(40):
        pos = deformer.step(dt=0.033)

    assert deformer.done
    assert np.allclose(pos, target, atol=1e-4)  # turbulence annealed to 0


# --------------------------------------------------------------------------- #
# G3 — cache-miss state machine completes
# --------------------------------------------------------------------------- #
def test_cache_miss_state_machine():
    engine = build_default_engine()
    engine.render_knowledge_hologram(make_graph(n=24))
    engine.drain_events()  # clear the initial DISPLAYED

    result = engine.generate_3d_object("chair", make_mv_images(), cam_rays=None)
    assert result == "miss"

    gen_states = [e.state for e in engine.drain_events()]
    assert HoloState.GENERATING in gen_states

    seen = run_until_displayed(engine, "chair")

    assert HoloState.SWAP_READY in seen
    assert engine.state == HoloState.DISPLAYED
    assert "chair" in engine.cache
    assert engine.cache["chair"].verified


# --------------------------------------------------------------------------- #
# G4 — cache hit is instant (no GENERATING)
# --------------------------------------------------------------------------- #
def test_cache_hit_is_instant():
    engine = build_default_engine()
    engine.render_knowledge_hologram(make_graph(n=24))

    # First call: miss -> generate -> cache it.
    engine.generate_3d_object("lamp", make_mv_images(), cam_rays=None)
    run_until_displayed(engine, "lamp")
    engine.drain_events()  # clear everything

    # Second call: must be an instant cache hit, no GENERATING emitted.
    result = engine.generate_3d_object("lamp", make_mv_images(), cam_rays=None)
    hit_states = [e.state for e in engine.drain_events()]

    assert result == "hit"
    assert HoloState.GENERATING not in hit_states
    assert HoloState.DISPLAYED in hit_states
    assert engine.state == HoloState.DISPLAYED


# --------------------------------------------------------------------------- #
# G5 — PSNR gate rejects garbage
# --------------------------------------------------------------------------- #
def test_psnr_gate_rejects_garbage():
    rasterizer = CPURasterizer()
    gate = PSNRGate(rasterizer, abs_floor_db=18.0, rel_drop_db=3.0)

    width = height = 64
    viewmat, K = make_camera(width, height)
    camera = {"viewmat": viewmat, "K": K}

    good = make_dense_field(seed=0)
    reference = rasterizer.render(good, viewmat, K, width, height)

    # A faithful field reproduces the reference -> high PSNR -> accept.
    assert gate.verify(good, [reference], [camera]) is True
    assert gate.last_psnr is None or gate.last_psnr > 18.0

    # A different (garbage) field does not -> low PSNR -> reject.
    garbage = make_dense_field(seed=999)
    assert gate.verify(garbage, [reference], [camera]) is False
    assert gate.last_psnr is not None and gate.last_psnr < 18.0
