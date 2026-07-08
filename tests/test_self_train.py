# -*- coding: utf-8 -*-
"""Self-training on real fields: consensus pseudo-labels, no forgetting."""

from __future__ import annotations

import numpy as np
import pytest

from atanor_core.rigging.synth_rigs import make_creature
from atanor_core.rigging.rig_predictor import RigPredictor, _TORCH
from atanor_core.rigging.self_train import consensus_joints, pseudo_shape, self_train
from atanor_core.rigging.live_rig import predict_rig_joints


def test_pseudo_shape_matches_training_format():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(5000, 3)).astype(np.float32)
    joints = np.array([[0.5, 0, 0], [-0.5, 0, 0]], np.float32)
    s = pseudo_shape(cloud, joints, n_sub=800)
    assert s["points"].shape == (800, 3)
    assert s["jointness"].shape == (800,)
    assert 0.0 <= s["jointness"].min() and s["jointness"].max() <= 1.0
    # canonical frame like make_creature output
    assert np.abs(s["points"]).max() <= 1.0 + 1e-5


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_self_train_cycle_runs_and_keeps_predictor_usable():
    rng = np.random.default_rng(1)
    synth = [make_creature(rng, 600) for _ in range(24)]
    rp = RigPredictor()
    rp.train(synth, epochs=60, device="cpu")
    # a "real" cloud the teacher never saw: one of its own creatures, denser
    cloud = make_creature(rng, 2500)["points"]
    stats = self_train(rp, [cloud], synth, copies=2, rounds=4,
                       epochs=40, device="cpu")
    assert stats["pseudo_shapes"] in (0, 2)          # consensus may abstain
    j = predict_rig_joints(cloud, rp, ensemble=3)
    assert j.shape[1] == 3 if len(j) else True       # still produces sane output
    s = make_creature(rng, 600)
    assert np.isfinite(rp.jointness(s["points"])).all()


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_consensus_is_deterministic():
    rng = np.random.default_rng(2)
    synth = [make_creature(rng, 600) for _ in range(24)]
    rp = RigPredictor()
    rp.train(synth, epochs=60, device="cpu")
    cloud = make_creature(rng, 2000)["points"]
    a = consensus_joints(cloud, rp, rounds=3)
    b = consensus_joints(cloud, rp, rounds=3)
    assert len(a) == len(b)
    if len(a):
        assert np.allclose(a, b, atol=1e-5)
