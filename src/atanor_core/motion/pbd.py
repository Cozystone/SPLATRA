# -*- coding: utf-8 -*-
"""Position-based dynamics — soft-body life for particle fields.

The rig (live_rig / autorig) computes WHERE particles should go; PBD makes the
journey physical: inertia, lag, jiggle, settle. Verlet integration pulls each
particle toward its skinned target while distance constraints (sampled from the
home shape) keep local structure rigid — so a driven tail whips and wobbles
instead of teleporting, and flesh trails a beat behind the bone.

Pure numpy, fully vectorized, bounded memory: constraints come from a voxel-hash
pairing (O(N log N) build, ~2 edges/particle), no N x N anything. The same
integrator maps 1:1 onto a GPU/shader implementation later (like motion/flow.py).
"""
from __future__ import annotations

import numpy as np


def _voxel_edges(home: np.ndarray, voxel: float, passes: int = 2, seed: int = 0):
    """Pair points that share a voxel: sort by voxel key, link consecutive rows.
    A second shuffled pass adds cross-links so constraints aren't one chain."""
    P = np.asarray(home, np.float32)
    keys = np.floor(P / voxel).astype(np.int64)
    edges = []
    rng = np.random.default_rng(seed)
    order0 = np.arange(len(P))
    for p in range(passes):
        order = order0 if p == 0 else rng.permutation(len(P))
        k = keys[order]
        srt = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
        o = order[srt]
        same = (keys[o[:-1]] == keys[o[1:]]).all(1)
        edges.append(np.stack([o[:-1][same], o[1:][same]], axis=1))
    E = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
    return E[:, 0], E[:, 1]


class SoftBody:
    """Verlet + distance constraints, driven by per-frame target positions."""

    def __init__(self, home: np.ndarray, voxel: float | None = None,
                 follow: float = 0.30, damping: float = 0.88,
                 stiffness: float = 0.5, iters: int = 2, seed: int = 0):
        self.home = np.asarray(home, np.float32).copy()
        span = float(np.abs(self.home - self.home.mean(0)).max()) + 1e-6
        self.voxel = float(voxel) if voxel else span * 0.06
        self.i, self.j = _voxel_edges(self.home, self.voxel, seed=seed)
        d = self.home[self.i] - self.home[self.j]
        self.rest = np.linalg.norm(d, axis=1) + 1e-8
        self.x = self.home.copy()
        self.xprev = self.home.copy()
        self.follow = follow
        self.damping = damping
        self.stiffness = stiffness
        self.iters = iters
        self.max_step = self.voxel * 2.5          # hard bound: no explosions

    @property
    def n_constraints(self) -> int:
        return len(self.rest)

    def step(self, targets: np.ndarray) -> np.ndarray:
        """One frame: inertia + pull toward targets, then constraint projection."""
        T = np.asarray(targets, np.float32)
        start = self.x
        inertia = (self.x - self.xprev) * self.damping
        pull = (T - self.x) * self.follow
        self.xprev = self.x
        self.x = self.x + inertia + pull
        for _ in range(self.iters):
            d = self.x[self.i] - self.x[self.j]
            L = np.linalg.norm(d, axis=1) + 1e-8
            corr = d * (((L - self.rest) / L) * 0.5 * self.stiffness)[:, None]
            np.subtract.at(self.x, self.i, corr)
            np.add.at(self.x, self.j, corr)
        # hard bound on the WHOLE frame (incl. constraint pushes): no explosions
        dx = self.x - start
        n = np.linalg.norm(dx, axis=1, keepdims=True)
        dx *= np.minimum(1.0, self.max_step / np.maximum(n, 1e-9))
        self.x = start + dx
        return self.x

    def settle(self, targets: np.ndarray, frames: int = 8) -> np.ndarray:
        """Run several frames toward a fixed target — returns the settled state."""
        for _ in range(frames):
            self.step(targets)
        return self.x
