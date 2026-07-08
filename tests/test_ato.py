# -*- coding: utf-8 -*-
"""아토 (Ato): the machine's own character plugs into every living channel."""

from __future__ import annotations

import numpy as np

from atanor_core.avatar.ato import build_ato
from atanor_core.rigging.parts import find_eyes
from atanor_core.rigging.live_rig import bind_joints, decompose_chains


def test_ato_builds_clean():
    pos, col, scale, quat, opa = build_ato(20000)
    assert len(pos) > 10000
    assert np.isfinite(pos).all() and np.isfinite(col).all()
    assert col.min() >= 0.0 and col.max() <= 1.0
    assert (opa > 0).all()
    # deterministic per seed
    pos2, *_ = build_ato(20000)
    assert np.allclose(pos, pos2)


def test_ato_eyes_are_found_as_a_pair():
    pos, col, *_ = build_ato(40000)
    eyes = find_eyes(pos, col)
    assert len(eyes) == 2
    xs = sorted(e["center"][0] for e in eyes)
    assert xs[0] < -0.1 < 0.1 < xs[1]          # mirrored left/right
    assert all(e["center"][1] > 0.3 for e in eyes)   # on the face, not the boots


def test_ato_limbs_form_chains():
    pos, *_ = build_ato(30000)
    # geometric sanity without the trained net: hand/boot tips exist and a rig
    # binds — chains partition whatever joints a predictor supplies
    joints = np.array([[1.28, 0.10, 0.05], [1.46, -0.28, 0.10],
                       [-1.28, 0.10, 0.05], [-1.46, -0.28, 0.10],
                       [0.33, -0.95, 0.03], [-0.33, -0.95, 0.03]], np.float32)
    rig = bind_joints(pos, joints)
    chains = decompose_chains(rig)
    flat = sorted(j for ch in chains for j in ch)
    assert flat == list(range(len(joints)))
    assert (rig.weight > 0.3).sum() > 500      # gloves/boots articulate
