# -*- coding: utf-8 -*-
"""Live rig on real fields: SPL2 loading, joint binding, FK chain posing, PBD."""

from __future__ import annotations

import os
import struct

import numpy as np

from atanor_core.rigging.live_rig import (load_spl2, bind_joints, pose_joints,
                                          pose_chain)
from atanor_core.motion.pbd import SoftBody

PIKA = os.path.join(os.path.dirname(__file__), "..", "viewer", "samples", "pikachu.bin")


def _tadpole(n=3000, seed=0):
    """Round body + a straight tail along -x, with two known tail joints."""
    rng = np.random.default_rng(seed)
    body = rng.normal(scale=0.3, size=(n // 2, 3))
    t = rng.uniform(0.4, 2.0, n // 2)[:, None]
    tail = np.array([-1.0, 0.0, 0.0]) * t + rng.normal(scale=0.05, size=(n // 2, 3))
    P = np.vstack([body, tail]).astype(np.float32)
    joints = np.array([[-0.7, 0, 0], [-1.4, 0, 0]], np.float32)
    return P, joints


def test_load_spl2_real_pikachu():
    if not os.path.exists(PIKA):
        return  # sample not present on this checkout
    pos, col, scale, quat, opa = load_spl2(PIKA)
    n = len(pos)
    assert n > 50_000 and col.shape == (n, 3) and quat.shape == (n, 4)
    assert np.isfinite(pos).all()
    assert 0.0 <= opa.min() and opa.max() <= 1.0


def test_bind_weights_are_distal():
    P, joints = _tadpole()
    rig = bind_joints(P, joints)
    tail_tip = P[:, 0] < -1.6
    core = np.linalg.norm(P, axis=1) < 0.3
    assert rig.weight[tail_tip].mean() > 0.8      # far past a joint: full drive
    assert rig.weight[core].mean() < 0.05         # body: stays


def test_pose_chain_compounds_and_stays_coherent():
    P, joints = _tadpole()
    rig = bind_joints(P, joints)
    out = pose_chain(P, rig, [0, 1], drive=0.7)
    tip = P[:, 0] < -1.6
    mid = (P[:, 0] < -1.0) & (P[:, 0] > -1.3)
    body = np.linalg.norm(P, axis=1) < 0.3
    mv = np.linalg.norm(out - P, axis=1)
    # FK: the tip accumulates BOTH joint rotations -> moves more than mid-tail
    assert mv[tip].mean() > mv[mid].mean() > 0.01
    assert mv[body].mean() < 0.01
    # coherence: neighbouring tail particles stay neighbours (no shearing spray)
    tp = out[tip]
    spread = np.linalg.norm(tp - tp.mean(0), axis=1).mean()
    spread0 = np.linalg.norm(P[tip] - P[tip].mean(0), axis=1).mean()
    assert spread < spread0 * 1.5


def test_pose_joints_idle_and_intents():
    P, joints = _tadpole()
    rig = bind_joints(P, joints)
    assert np.allclose(pose_joints(P, rig, amp=0.0), P)           # amp 0 = still
    out = pose_joints(P, rig, amp=0.0, intents={1: 0.8})          # drive ONE joint
    mv = np.linalg.norm(out - P, axis=1)
    assert mv[rig.assign == 1].max() > 0.05
    assert mv[rig.assign == 0].max() < 1e-5


def test_softbody_rest_and_follow_through():
    P, joints = _tadpole(n=2000)
    soft = SoftBody(P)
    assert soft.n_constraints > len(P) * 0.5
    x = soft.settle(P, frames=5)
    assert np.linalg.norm(x - P, axis=1).max() < 1e-4             # rest is stable
    # jump the target: one step lags behind, settling closes the gap
    T = P + np.array([0.4, 0.0, 0.0], np.float32)
    x1 = soft.step(T).copy()
    lag1 = np.linalg.norm(x1 - T, axis=1).mean()
    xs = soft.settle(T, frames=12)
    lags = np.linalg.norm(xs - T, axis=1).mean()
    assert lag1 > lags                                            # follow-through
    assert lags < 0.1
    # local structure survived the ride
    d = np.linalg.norm(xs[soft.i] - xs[soft.j], axis=1)
    assert np.abs(d - soft.rest).mean() < soft.voxel * 0.5


def test_softbody_bounded_never_explodes():
    P, _ = _tadpole(n=1500)
    soft = SoftBody(P)
    wild = P * 50.0                                               # absurd target
    x = soft.settle(wild, frames=3)
    assert np.isfinite(x).all()
    step = np.linalg.norm(x - P, axis=1).max()
    assert step <= soft.max_step * 3 * 1.01 + 1e-5                # clamped per frame
