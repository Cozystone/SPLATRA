# -*- coding: utf-8 -*-
"""Position-based dynamics — soft-body, fluid, and granular life for particles.

The rig (live_rig / autorig) computes WHERE particles should go; PBD makes the
journey physical: inertia, lag, jiggle, settle. Verlet integration pulls each
particle toward its skinned target while distance constraints (sampled from the
home shape) keep local structure rigid — so a driven tail whips and wobbles
instead of teleporting, and flesh trails a beat behind the bone.

MATERIALS — the same integrator, different constraint semantics:
  * flesh  — constraints resist stretch AND compression; follows rig targets.
  * water  — constraints resist COMPRESSION only (incompressible) plus a weak
             cohesion band; pairs are periodically REBUILT from current
             positions (fluid topology changes as it flows); gravity + floor.
  * soil   — constraints BREAK permanently past a strain limit (fracture into
             grains) and never re-form; strong ground friction. Piles hold a
             slope where water flattens — an angle of repose emerges.

Pure numpy, fully vectorized, bounded memory: constraints come from a voxel-hash
pairing (O(N log N) build, ~2 edges/particle), no N x N anything. The same
integrator maps 1:1 onto a GPU/shader implementation later (like motion/flow.py).
"""
from __future__ import annotations

import numpy as np

MATERIALS = {
    # follow damping stiffness gravity  friction cohesion break rebuild substeps
    "flesh": dict(follow=0.30, damping=0.88, stiffness=0.5, gravity=0.0,
                  friction=0.0, compress_only=False, cohesion=0.0,
                  break_strain=None, rebuild_every=0, substeps=1, graph_passes=2),
    "water": dict(follow=0.0, damping=0.97, stiffness=0.6, gravity=0.012,
                  friction=0.02, compress_only=True, cohesion=0.04,
                  break_strain=None, rebuild_every=8, substeps=3, graph_passes=2),
    "soil": dict(follow=0.0, damping=0.80, stiffness=0.9, gravity=0.012,
                 friction=0.55, compress_only=False, cohesion=0.0,
                 break_strain=2.2, rebuild_every=0, substeps=3, graph_passes=5),
}


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


_HALF_OFFSETS = [(dx, dy, dz)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                 if (dx, dy, dz) > (0, 0, 0)]           # 13 half-space neighbours


def _pack(K: np.ndarray) -> np.ndarray:
    return (((K[:, 0] + (1 << 19)) << 42) | ((K[:, 1] + (1 << 19)) << 21)
            | (K[:, 2] + (1 << 19)))


def _grid_pairs(P: np.ndarray, h: float, cap: int = 4):
    """DENSE fixed-radius candidate pairs via a uniform grid — every particle
    against its cell run (3 shifts) and up to `cap` particles in each of the 13
    half-space neighbour cells. Vectorized, no python-per-cell loops. This is
    what CONTACT needs: the stochastic voxel pairing (~2 pairs/particle) lets
    grains interpenetrate freely and piles melt into the floor."""
    K = np.floor(P / h).astype(np.int64)
    key = _pack(K)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pi_all, pj_all = [], []
    for shift in (1, 2, 3):                             # same-cell pairs
        cand = np.arange(len(P) - shift)
        ok = ks[cand] == ks[cand + shift]
        pi_all.append(order[cand[ok]]); pj_all.append(order[cand[ok] + shift])
    for dx, dy, dz in _HALF_OFFSETS:                    # neighbour-cell pairs
        nk = _pack(K + np.array([dx, dy, dz]))
        s = np.searchsorted(ks, nk, side="left")
        e = np.searchsorted(ks, nk, side="right")
        cnt = np.minimum(e - s, cap)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        cum = np.cumsum(cnt) - cnt
        within = np.arange(tot) - np.repeat(cum, cnt)
        pi_all.append(np.repeat(np.arange(len(P)), cnt))
        pj_all.append(order[np.repeat(s, cnt) + within])
    if not pi_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(pi_all), np.concatenate(pj_all)


class SoftBody:
    """Verlet + distance constraints; material decides the constraint semantics.
    Drive with per-frame targets (flesh) or let gravity run it (water/soil)."""

    def __init__(self, home: np.ndarray, voxel: float | None = None,
                 material: str = "flesh", floor: float | None = None,
                 iters: int = 2, seed: int = 0, **overrides):
        cfg = dict(MATERIALS[material]); cfg.update(overrides)
        self.material = material
        self.home = np.asarray(home, np.float32).copy()
        span = float(np.abs(self.home - self.home.mean(0)).max()) + 1e-6
        self.span = span
        if voxel:
            self.voxel = float(voxel)
        else:
            # adaptive: a fixed span fraction starves solid volumes (voxel below
            # the inter-particle spacing -> no shared voxels -> no constraints),
            # so derive it from the measured nearest-neighbour spacing
            rng = np.random.default_rng(seed)
            probe = self.home[rng.choice(len(self.home),
                                         min(400, len(self.home)), replace=False)]
            d = np.linalg.norm(probe[:, None, :] - self.home[None, :, :], axis=2)
            d[d < 1e-9] = np.inf
            nn = float(np.median(d.min(1)))
            self.voxel = float(np.clip(2.5 * nn, span * 0.02, span * 0.25))
        self._seed = seed
        self._graph_passes = int(cfg.get("graph_passes", 2))
        self._rebuild_edges(self.home)
        self.r0 = float(np.median(self.rest))          # fluid rest spacing
        if cfg["compress_only"]:
            self.rest[:] = self.r0                      # water keeps a fixed
                                                        # density target, not
                                                        # whatever it was frozen in
        self.x = self.home.copy()
        self.xprev = self.home.copy()
        self.follow = cfg["follow"]
        self.damping = cfg["damping"]
        self.stiffness = cfg["stiffness"]
        self.gravity = cfg["gravity"] * span          # scale-invariant fall rate
        self.friction = cfg["friction"]
        self.compress_only = cfg["compress_only"]
        self.cohesion = cfg["cohesion"]
        self.break_strain = cfg["break_strain"]
        self.rebuild_every = cfg["rebuild_every"]
        self.substeps = int(cfg.get("substeps", 1))
        self.floor = float(floor) if floor is not None else float(self.home[:, 1].min())
        self.iters = iters
        # hard per-substep displacement bound. For gravity materials this is the
        # anti-tunnelling speed limit: a particle may never cross a whole
        # neighbour layer in one substep, or layers fall through each other and
        # the body collapses into an unrecoverable one-particle-thick pancake.
        if cfg["gravity"]:
            self.max_step = self.r0 * 0.35
            self.iters = max(iters, 3)
            # ground roughness: a per-particle floor offset. A perfectly flat
            # clamp puts every grounded particle at EXACTLY the same height,
            # so contact directions lose their vertical component to the last
            # bit and nothing can ever push upward again (measured degeneracy)
            rr = np.random.default_rng(seed + 1)
            self._floor_eps = rr.uniform(0.0, 0.25 * self.r0,
                                         len(self.home)).astype(np.float32)
        else:
            self.max_step = self.voxel * 2.5
            self._floor_eps = None
        self._frame = 0

    def _rebuild_edges(self, positions: np.ndarray) -> None:
        self.i, self.j = _voxel_edges(positions, self.voxel,
                                      passes=self._graph_passes, seed=self._seed)
        d = positions[self.i] - positions[self.j]
        self.rest = np.linalg.norm(d, axis=1) + 1e-8
        if getattr(self, "compress_only", False):
            self.rest[:] = self.r0                      # fluid: constant spacing
        self.alive = np.ones(len(self.rest), bool)

    @property
    def n_constraints(self) -> int:
        return int(self.alive.sum())

    def _project(self) -> None:
        for _ in range(self.iters):
            ii, jj = self.i[self.alive], self.j[self.alive]
            rest = self.rest[self.alive]
            d = self.x[ii] - self.x[jj]
            L = np.linalg.norm(d, axis=1) + 1e-8
            strain = (L - rest) / L
            if self.compress_only:
                # incompressible: push apart when squeezed; only weak cohesion
                # pulls in a band just beyond rest (a droplet, not a solid)
                k = np.where(strain < 0, self.stiffness,
                             np.where(L < rest * 2.0, self.cohesion, 0.0))
            else:
                k = np.full(len(L), self.stiffness, np.float32)
            corr = d * (strain * 0.5 * k)[:, None]
            np.subtract.at(self.x, ii, corr)
            np.add.at(self.x, jj, corr)
            if self.break_strain is not None:
                self.alive[np.flatnonzero(self.alive)[L > self.rest[self.alive]
                                                      * self.break_strain]] = False

    def _substep(self, targets: np.ndarray | None, dt: float) -> None:
        start = self.x
        inertia = (self.x - self.xprev) * self.damping
        if self.gravity:
            # inelastic impact: near the ground, falling momentum is absorbed
            # instead of crushing the layers below through the floor — distance
            # constraints are translation-invariant, so without this the whole
            # body's momentum steamrolls the contact pushes and it pancakes
            near = self.x[:, 1] < self.floor + 2.5 * self.r0
            inertia[near] *= 0.35
        force = np.zeros_like(self.x)
        if targets is not None and self.follow > 0:
            force += (np.asarray(targets, np.float32) - self.x) * self.follow * dt
        if self.gravity:
            force[:, 1] -= self.gravity * dt
        self.xprev = self.x
        self.x = self.x + inertia + force
        self._project()

        if self.gravity:
            # CONTACT (non-penetration): the sparse material graph does not
            # encode volume — a 3D body can flatten into a plane with every
            # pair still at rest length (measured: pancake, all pairs
            # horizontal, strain ~0). Volume comes from freshly-found near
            # pairs each substep being pushed to at least d0 apart, which
            # forces stacking: N particles simply cannot fit in one layer.
            d0 = self.r0 * 0.8
            ci, cj = _grid_pairs(self.x, d0)
            for _ in range(2):
                d = self.x[ci] - self.x[cj]
                L = np.linalg.norm(d, axis=1) + 1e-8
                pen = L < d0
                if not pen.any():
                    break
                corr = d[pen] * (((L[pen] - d0) / L[pen]) * 0.5)[:, None]
                np.subtract.at(self.x, ci[pen], corr)
                np.add.at(self.x, cj[pen], corr)

        # hard bound on the WHOLE substep (incl. constraint pushes) — for
        # gravity materials this is the anti-tunnelling speed limit
        dx = self.x - start
        n = np.linalg.norm(dx, axis=1, keepdims=True)
        dx *= np.minimum(1.0, self.max_step / np.maximum(n, 1e-9))
        self.x = start + dx

        # floor contact: clamp (to the rough ground), kill downward velocity,
        # apply ground friction
        if self.gravity or self.friction:
            lvl = self.floor if self._floor_eps is None else self.floor + self._floor_eps
            below = self.x[:, 1] < lvl
            if below.any():
                self.x[below, 1] = lvl[below] if self._floor_eps is not None else self.floor
                self.xprev[below, 1] = np.minimum(self.xprev[below, 1], self.x[below, 1])
                if self.friction:
                    keep = 1.0 - self.friction
                    self.xprev[below, 0] = self.x[below, 0] - \
                        (self.x[below, 0] - self.xprev[below, 0]) * keep
                    self.xprev[below, 2] = self.x[below, 2] - \
                        (self.x[below, 2] - self.xprev[below, 2]) * keep

    def step(self, targets: np.ndarray | None = None) -> np.ndarray:
        """One frame = `substeps` micro-steps of integrate+project+contact.
        Targets are optional — water/soil run free under gravity."""
        dt = 1.0 / self.substeps
        for _ in range(self.substeps):
            self._substep(targets, dt)
        self._frame += 1
        if self.rebuild_every and self._frame % self.rebuild_every == 0:
            self._rebuild_edges(self.x)               # fluid: neighbours flow
        return self.x

    def settle(self, targets: np.ndarray | None = None, frames: int = 8) -> np.ndarray:
        """Run several frames toward a fixed target — returns the settled state."""
        for _ in range(frames):
            self.step(targets)
        return self.x
