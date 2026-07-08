# -*- coding: utf-8 -*-
"""Auto-rigging: skeleton extraction + LBS make a surface shell move as a body."""

from __future__ import annotations

import numpy as np

from atanor_core.rigging.skeleton import extract_skeleton, bind, pose, animate_positions


def _rod(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    r = 0.25 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([x, r * np.cos(a), r * np.sin(a)], 1).astype(np.float32)


def test_skeleton_spans_the_body():
    m = _rod()
    J = extract_skeleton(m, n_joints=10)
    assert J.shape == (10, 3)
    # the joint chain spans the object's long axis
    assert np.ptp(J[:, 0]) > 1.5


def test_binding_covers_all_particles():
    m = _rod()
    rig = bind(m, extract_skeleton(m, 10))
    assert rig.bone_of.shape == (len(m),)
    assert rig.n_bones == 9
    assert set(np.unique(rig.bone_of)).issubset(set(range(9)))


def test_rig_animates_with_a_displacement_gradient():
    m = _rod()
    rig = bind(m, extract_skeleton(m, 10))
    d0 = animate_positions(m, rig, t=0.0, amplitude=0.7)
    d1 = animate_positions(m, rig, t=1.6, amplitude=0.7)
    moved = np.linalg.norm(d1 - d0, axis=1)
    # SOMETHING moves (it's a live body, not frozen)
    assert moved.max() > 0.05
    # anchored end (near joint 0) moves less than the far end — articulated motion
    root = rig.joints[0]
    dist_to_root = np.linalg.norm(m - root, axis=1)
    near = dist_to_root < np.percentile(dist_to_root, 20)
    far = dist_to_root > np.percentile(dist_to_root, 80)
    assert moved[near].mean() < moved[far].mean()


def test_shape_is_preserved_not_exploded():
    m = _rod()
    rig = bind(m, extract_skeleton(m, 10))
    d = animate_positions(m, rig, t=1.0, amplitude=0.6)
    # bounding box stays comparable (rigid bones, not a blow-up)
    assert 0.6 < np.ptp(d[:, 0]) / np.ptp(m[:, 0]) < 1.6


def test_deterministic():
    m = _rod()
    rig = bind(m, extract_skeleton(m, 10))
    a = animate_positions(m, rig, t=0.9, amplitude=0.5)
    b = animate_positions(m, rig, t=0.9, amplitude=0.5)
    assert np.allclose(a, b)
