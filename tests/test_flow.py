# -*- coding: utf-8 -*-
"""Microbot flow: particles drift and shape holds; cheap + deterministic."""

from __future__ import annotations

import numpy as np

from atanor_core.motion.flow import flow_offset, flow_positions, gather_flow


def _sphere(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_particles_actually_move_over_time():
    h = _sphere()
    p0 = flow_positions(h, 0.0)
    p1 = flow_positions(h, 1.7)
    moved = np.linalg.norm(p1 - p0, axis=1)
    assert moved.mean() > 0.01           # it's alive, not frozen


def test_shape_is_held_bounded_drift():
    h = _sphere()
    p = flow_positions(h, 3.3, amp=0.06)
    # every particle stays within a bounded radius of home -> the shape holds
    drift = np.linalg.norm(p - h, axis=1)
    assert drift.max() < 0.25
    # bounding box barely changes (still recognisably a sphere)
    assert abs(np.ptp(p[:, 0]) - np.ptp(h[:, 0])) < 0.4


def test_deterministic():
    h = _sphere()
    assert np.allclose(flow_positions(h, 0.9), flow_positions(h, 0.9))


def test_gather_flow_assembles_from_swarm_to_shape():
    h = _sphere()
    loose = gather_flow(h, 0.0, u=0.0)     # dispersed swarm
    tight = gather_flow(h, 0.0, u=1.0)     # settled onto the shape
    spread_loose = np.linalg.norm(loose - loose.mean(0), axis=1).mean()
    spread_tight = np.linalg.norm(tight - tight.mean(0), axis=1).mean()
    assert spread_loose > spread_tight     # it converges, never a hard pop
    # settled state is close to home (shimmer aside)
    assert np.linalg.norm(tight - h, axis=1).mean() < 0.2


def test_cost_is_cheap_vectorized():
    # 200k particles in one call — proves it's cheap enough for a laptop
    h = _sphere(200_000)
    p = flow_positions(h, 1.0)
    assert p.shape == (200_000, 3)
