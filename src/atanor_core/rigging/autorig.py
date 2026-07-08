# -*- coding: utf-8 -*-
"""Part-aware auto-rig — find the limbs, articulate them, invent the secondary motion.

The honest problem the owner named: a generator (TripoSR) hands you a FROZEN
surface shell. It contains no skeleton, no joints, and none of the deformation
detail — wrinkles, folds, jiggle — that only APPEARS when a thing moves. To bring
it alive you must *compute* structure that was never captured:

  1. extremities(): farthest-point sampling from the body core finds the limb TIPS
     (ears, arms, legs, tail, nose) — real parts, not one PCA axis.
  2. bind_parts(): each particle is assigned to the limb whose direction it best
     matches, with a weight that is high near a tip and fades to 0 at the core —
     so limbs swing while the torso stays put (articulation, not a rigid bend).
  3. pose(): each limb rotates about the core independently; a distance-lagged
     radial term adds SECONDARY motion (skin jiggle / soft wobble) that no static
     capture holds — the "invented" detail.

This is the geometric v1 of what needs, eventually, a learned rig predictor. It is
pure numpy, deterministic, and cheap enough that ATANOR (the reasoning layer) can
drive it by choosing per-limb intents ("wave the right arm", "twitch the ears")
while this engine executes the articulation. Same means[N,3] interface as a real
TripoSR field — no sphere required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PartRig:
    core: np.ndarray       # [3] body centroid (root)
    tips: np.ndarray       # [L,3] limb tips
    tipdir: np.ndarray     # [L,3] unit limb directions from core
    limb: np.ndarray       # [N] which limb each particle belongs to
    weight: np.ndarray     # [N] articulation weight (0 at core -> 1 at a tip)
    radial: np.ndarray     # [N] distance from core (for secondary jiggle)

    @property
    def n_limbs(self) -> int:
        return int(self.tips.shape[0])


def extremities(means: np.ndarray, n_tips: int = 6) -> np.ndarray:
    """Farthest-point sampling from the centroid → the protruding limb tips."""
    m = np.asarray(means, dtype=np.float64)
    c = m.mean(axis=0)
    picked = [int(np.argmax(np.linalg.norm(m - c, axis=1)))]
    mind = np.linalg.norm(m - m[picked[0]], axis=1)
    for _ in range(max(1, n_tips) - 1):
        i = int(np.argmax(mind))
        picked.append(i)
        mind = np.minimum(mind, np.linalg.norm(m - m[i], axis=1))
    return m[picked].astype(np.float32)


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bind_parts(means: np.ndarray, n_limbs: int = 6) -> PartRig:
    m = np.asarray(means, dtype=np.float64)
    c = m.mean(axis=0)
    tips = extremities(m, n_limbs).astype(np.float64)
    tipdir = tips - c
    tipdir /= (np.linalg.norm(tipdir, axis=1, keepdims=True) + 1e-9)
    v = m - c
    vlen = np.linalg.norm(v, axis=1)
    vdir = v / (vlen[:, None] + 1e-9)
    cos = vdir @ tipdir.T                       # [N,L] alignment to each limb
    limb = np.argmax(cos, axis=1).astype(np.int32)
    align = cos[np.arange(len(m)), limb]
    # articulate only well-aligned particles that are far from the core
    w = _smooth(align, 0.55, 0.9) * _smooth(vlen / (vlen.max() + 1e-9), 0.25, 0.8)
    return PartRig(core=c.astype(np.float32), tips=tips.astype(np.float32),
                   tipdir=tipdir.astype(np.float32), limb=limb,
                   weight=w.astype(np.float32), radial=vlen.astype(np.float32))


def _perp(u: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w = np.cross(u, ref)
    return w / (np.linalg.norm(w) + 1e-9)


def pose(means: np.ndarray, rig: PartRig, t: float, amp: float = 1.0,
         intents: dict[int, float] | None = None) -> np.ndarray:
    """Articulate each limb about the core by a per-particle weighted angle, plus
    a distance-lagged radial jiggle (the invented secondary motion). `intents`
    lets a controller (ATANOR) set a per-limb drive; default = a gentle idle."""
    m = np.asarray(means, dtype=np.float64)
    c = rig.core.astype(np.float64)
    out = m.copy()
    for l in range(rig.n_limbs):
        sel = rig.limb == l
        if not np.any(sel):
            continue
        drive = (intents or {}).get(l, np.sin(t + l * 1.7))   # per-limb angle driver
        ang = (amp * 0.6 * drive) * rig.weight[sel]           # per-particle angle
        a = _perp(rig.tipdir[l].astype(np.float64))           # rotation axis ⟂ limb
        v = m[sel] - c
        ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
        cross = np.cross(np.broadcast_to(a, v.shape), v)
        dot = (v @ a)[:, None]
        out[sel] = c + v * ca + cross * sa + a[None, :] * dot * (1 - ca)
    # secondary: distance-lagged radial jiggle — soft skin wobble, tips wobble most
    lag = np.sin(t * 1.6 - rig.radial * 3.2) * (0.02 * amp)
    vdir = (m - c) / (rig.radial[:, None] + 1e-9)
    out += vdir * (lag * _smooth(rig.radial, 0.15, 0.8))[:, None]
    return out.astype(np.float32)
