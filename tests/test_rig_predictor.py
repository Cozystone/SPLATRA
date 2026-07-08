# -*- coding: utf-8 -*-
"""Learned rig predictor recovers INTERNAL joints (knees/spine) that farthest-
point sampling cannot — trained on procedural shapes whose skeleton is known
exactly, across creature AND general-mesh families."""

from __future__ import annotations

import numpy as np
import pytest

from atanor_core.rigging.synth_rigs import make_creature, KINDS
from atanor_core.rigging.rig_predictor import features, RigPredictor, _TORCH
from atanor_core.rigging.autorig import extremities


def test_every_family_builds_with_internal_joints():
    rng = np.random.default_rng(1)
    for kind in KINDS:
        s = make_creature(rng, 900, kind=kind)
        assert len(s["internal_joints"]) >= 3, kind
        assert len(s["internal_joints"]) < len(s["joints"]), kind
        assert np.isfinite(s["points"]).all(), kind
        # jointness anticorrelates with distance to the nearest joint (the
        # gaussian label is nonlinear, so the linear proxy is loose — slab-heavy
        # kinds like furniture sit around -0.3)
        dj = np.linalg.norm(s["points"][:, None, :] - s["joints"][None, :, :],
                            axis=2).min(1)
        assert np.corrcoef(dj, s["jointness"])[0, 1] < -0.25, kind


def test_features_shape_and_finite():
    s = make_creature(np.random.default_rng(2), 600)
    f = features(s["points"])
    assert f.shape == (len(s["points"]), 10)
    assert np.isfinite(f).all()


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_jointness_discriminates_after_training():
    rng = np.random.default_rng(3)
    train = [make_creature(rng, 700) for _ in range(80)]
    rp = RigPredictor()
    rp.train(train, epochs=250, device="cpu")
    corr, peak = [], []
    for _ in range(4):
        s = make_creature(rng, 700)
        j = rp.jointness(s["points"])
        corr.append(np.corrcoef(j, s["jointness"])[0, 1])
        peak.append(j.max())
    # learned score tracks ground truth and has a real peak (not a flat mean)
    assert np.mean(corr) > 0.45
    assert np.mean(peak) > 0.5


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_learned_beats_geometric_on_internal_joints():
    rng = np.random.default_rng(4)
    train = [make_creature(rng, 700) for _ in range(80)]
    rp = RigPredictor()
    rp.train(train, epochs=250, device="cpu")

    def internal_recall(pred, gt, tol=0.18):
        if len(pred) == 0:
            return 0.0
        d = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2).min(1)
        return float((d < tol).mean())

    learned, geo = [], []
    for _ in range(8):
        s = make_creature(rng, 700)
        gt = s["joints"][s["internal_joints"]]
        learned.append(internal_recall(rp.predict_joints(s["points"]), gt))
        tips = extremities(s["points"], 8)
        tips = (tips - s["points"].mean(0)) / (np.abs(s["points"] - s["points"].mean(0)).max() + 1e-6)
        geo.append(internal_recall(tips, gt))

    # CPU-scale training on the hard general distribution: still clearly ahead
    assert np.mean(learned) > 0.25
    assert np.mean(learned) > np.mean(geo) + 0.15


@pytest.mark.skipif(not _TORCH, reason="torch required")
def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    train = [make_creature(rng, 500) for _ in range(10)]
    rp = RigPredictor()
    rp.train(train, epochs=20, device="cpu")
    p = str(tmp_path / "w.pt")
    rp.save(p)
    rp2 = RigPredictor.load(p)
    s = make_creature(rng, 500)
    assert np.allclose(rp.jointness(s["points"]), rp2.jointness(s["points"]),
                       atol=1e-5)
