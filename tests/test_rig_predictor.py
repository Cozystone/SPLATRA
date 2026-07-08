# -*- coding: utf-8 -*-
"""Learned rig predictor recovers INTERNAL joints (knees/spine) that farthest-point
sampling cannot — trained on procedural creatures whose skeleton is known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from atanor_core.rigging.synth_rigs import make_creature
from atanor_core.rigging.rig_predictor import features, RigPredictor, _TORCH
from atanor_core.rigging.autorig import extremities


def test_make_creature_has_internal_joints():
    s = make_creature(np.random.default_rng(1), 900)
    # a quadruped: spine-mid + 4 knees are internal (>=2 bones), tips are not
    assert len(s["internal_joints"]) >= 5
    assert len(s["internal_joints"]) < len(s["joints"])
    assert s["points"].shape[1] == 3
    # jointness peaks near a real joint and is low far away
    dj = np.linalg.norm(s["points"][:, None, :] - s["joints"][None, :, :], axis=2).min(1)
    assert np.corrcoef(dj, s["jointness"])[0, 1] < -0.5


def test_features_shape_and_finite():
    s = make_creature(np.random.default_rng(2), 600)
    f = features(s["points"])
    assert f.shape == (600, 10)
    assert np.isfinite(f).all()


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_jointness_discriminates_after_training():
    rng = np.random.default_rng(3)
    train = [make_creature(rng, 700) for _ in range(40)]
    rp = RigPredictor()
    rp.train(train, epochs=120, device="cpu")
    s = make_creature(rng, 700)
    j = rp.jointness(s["points"])
    # learned score tracks ground truth and has a real peak (not a flat mean)
    assert np.corrcoef(j, s["jointness"])[0, 1] > 0.5
    assert j.max() > 0.4


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_learned_beats_geometric_on_internal_joints():
    rng = np.random.default_rng(4)
    train = [make_creature(rng, 700) for _ in range(50)]
    rp = RigPredictor()
    rp.train(train, epochs=150, device="cpu")

    def internal_recall(pred, gt, tol=0.18):
        if len(pred) == 0:
            return 0.0
        d = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2).min(1)
        return float((d < tol).mean())

    learned, geo = [], []
    for _ in range(6):
        s = make_creature(rng, 700)
        gt = s["joints"][s["internal_joints"]]
        learned.append(internal_recall(rp.predict_joints(s["points"]), gt))
        tips = extremities(s["points"], 8)
        tips = (tips - s["points"].mean(0)) / (np.abs(s["points"] - s["points"].mean(0)).max() + 1e-6)
        geo.append(internal_recall(tips, gt))

    assert np.mean(learned) > 0.5          # learned finds most internal joints
    assert np.mean(learned) > np.mean(geo) + 0.3   # decisively beats geometry
