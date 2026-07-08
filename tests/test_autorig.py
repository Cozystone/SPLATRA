# -*- coding: utf-8 -*-
"""Part-aware auto-rig: finds limbs, articulates them, body stays, ATANOR can drive."""

from __future__ import annotations

import numpy as np

from atanor_core.rigging.autorig import bind_parts, extremities, pose


def _creature(seed=0):
    rng = np.random.default_rng(seed)
    body = rng.normal(scale=0.35, size=(3000, 3))
    c = body.mean(0)
    parts = [body]
    for d in [(1, .3, 0), (-1, .3, 0), (0, 1, 0), (.3, -.9, .4), (.3, -.9, -.4)]:
        d = np.array(d, float); d /= np.linalg.norm(d)
        tv = np.linspace(0.4, 1.4, 1000)[:, None]
        parts.append(c + d * tv + rng.normal(scale=0.06, size=(1000, 3)))
    return np.vstack(parts).astype(np.float32)


def test_finds_multiple_extremities_not_one_axis():
    m = _creature()
    tips = extremities(m, 6)
    assert tips.shape == (6, 3)
    # the tips are spread out (real limbs), not clustered on a single line
    spread = np.linalg.norm(tips - tips.mean(0), axis=1).mean()
    assert spread > 0.6


def test_limbs_articulate_while_core_stays():
    m = _creature()
    rig = bind_parts(m, 6)
    p0, p1 = pose(m, rig, 0.0), pose(m, rig, 1.6)
    moved = np.linalg.norm(p1 - p0, axis=1)
    core = rig.radial < 0.3
    tips = rig.radial > rig.radial.max() * 0.8
    assert moved[core].mean() < moved[tips].mean()      # body stable, limbs swing
    assert moved[tips].mean() > 0.05


def test_atanor_can_drive_a_single_limb():
    m = _creature()
    rig = bind_parts(m, 6)
    out = pose(m, rig, 0.0, intents={2: 2.0})           # drive ONLY limb 2
    other = rig.limb != 2
    sel2 = rig.limb == 2
    assert np.linalg.norm(out[sel2] - m[sel2], axis=1).mean() > \
           np.linalg.norm(out[other] - m[other], axis=1).mean()


def test_deterministic_and_shape_bounded():
    m = _creature()
    rig = bind_parts(m, 6)
    a, b = pose(m, rig, 0.9), pose(m, rig, 0.9)
    assert np.allclose(a, b)
    # not exploded
    assert np.ptp(pose(m, rig, 1.0)[:, 0]) < np.ptp(m[:, 0]) * 2.0
