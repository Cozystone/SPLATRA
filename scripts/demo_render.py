"""Headless end-to-end demo (PRD §4 Phase 6 / Gate 6).

Runs the whole pipeline on CPU/numpy and dumps three PPM frames to ``out/``:

    out/01_graph.ppm      static knowledge-graph hologram
    out/02_morphed.ppm    after a multi-agent relayout morph
    out/03_generated.ppm  after a cache-miss 3D object generation + hot-swap

It prints the cache-miss state transitions GENERATING -> SWAP_READY -> DISPLAYED.

Note (PRD §6): use ``np.eye(4, dtype=np.float32)`` with the dtype as a KEYWORD
— ``np.eye``'s second positional arg is ``M`` (columns), not the dtype.
"""

from __future__ import annotations

import os
import time

import numpy as np

from atanor_core import build_default_engine
from atanor_core.state.machine import HoloState
from atanor_core.state.rasterizer import default_intrinsics, look_at

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def write_ppm(path: str, img: np.ndarray) -> None:
    """Write an [H, W, 3] float image in [0, 1] as a binary PPM (P6)."""
    h, w, _ = img.shape
    data = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(data.tobytes())


def make_graph(n: int = 36, dim: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    nodes = [
        {
            "id": f"n{i}",
            "embedding": rng.normal(size=dim).astype(np.float32).tolist(),
            "centrality": float(rng.uniform(0.0, 10.0)),
            "importance": float(rng.uniform(0.2, 0.95)),
            "category": int(i % 12),
        }
        for i in range(n)
    ]
    edges = [{"src": f"n{i}", "dst": f"n{(i + 1) % n}"} for i in range(n)]
    return {"nodes": nodes, "edges": edges}


def make_mv_images(color=(0.9, 0.4, 0.2)) -> np.ndarray:
    img = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    for c in range(3):
        img[:, :, c, :, :] = color[c]
    return img


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    W = H = 256
    # dtype as a KEYWORD (np.eye's 2nd positional is M, not dtype).
    _identity = np.eye(4, dtype=np.float32)
    viewmat = look_at(eye=(0.0, 0.0, -3.2), target=(0.0, 0.0, 0.0))
    K = default_intrinsics(W, H, fov_deg=55.0)

    engine = build_default_engine()

    # 1) Static graph hologram.
    graph = make_graph(n=36)
    engine.render_knowledge_hologram(graph)
    for ev in engine.drain_events():
        print(f"[graph]   -> {ev.state.value}")
    write_ppm(os.path.join(OUT, "01_graph.ppm"), engine.render(viewmat, K, W, H))
    print("wrote out/01_graph.ppm")

    # 2) Multi-agent relayout morph.
    rng = np.random.default_rng(123)
    new_positions = engine.field.means + rng.normal(0, 0.25, size=engine.field.means.shape)
    new_positions = np.clip(new_positions, -1.0, 1.0).astype(np.float32)
    engine.relayout(new_positions, turbulence=0.15)
    for _ in range(20):
        engine.tick()
    for ev in engine.drain_events():
        print(f"[morph]   -> {ev.state.value}")
    write_ppm(os.path.join(OUT, "02_morphed.ppm"), engine.render(viewmat, K, W, H))
    print("wrote out/02_morphed.ppm")

    # 3) Cache-miss 3D object generation + hot-swap.
    engine.generate_3d_object("demo-orb", make_mv_images(), cam_rays=None)
    for ev in engine.drain_events():
        print(f"[gen]     -> {ev.state.value}")
    for _ in range(400):
        engine.tick("demo-orb")
        for ev in engine.drain_events():
            print(f"[gen]     -> {ev.state.value}")
        if "demo-orb" in engine.cache and engine.state == HoloState.DISPLAYED:
            break
        time.sleep(0.005)
    write_ppm(os.path.join(OUT, "03_generated.ppm"), engine.render(viewmat, K, W, H))
    print("wrote out/03_generated.ppm")

    assert engine.state == HoloState.DISPLAYED, "demo did not reach DISPLAYED"
    print("\nDONE. Cache-miss path observed: GENERATING -> SWAP_READY -> DISPLAYED")


if __name__ == "__main__":
    main()
