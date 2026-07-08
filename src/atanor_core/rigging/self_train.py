# -*- coding: utf-8 -*-
"""Self-training on real generated fields — closing the domain gap.

The predictor is taught by procedural shapes; real generator output (TripoSR
shells etc.) is off-distribution, so its confidence there is soft and draw-to-
draw shaky. Standard remedy, honestly applied: PSEUDO-LABELS. Run the ensemble
predictor many rounds on a real field, keep only the joints that RECUR across
rounds (consensus = precision over recall), turn them into training labels with
the same gaussian jointness the synthetic teacher uses, and mix a few copies
into the next training run. The net then sees the real domain with labels it
already agrees on — confidence sharpens, stability rises. No human rigging, no
external dataset, and no fabricated supervision: every pseudo-label started as
the model's own repeated, agreeing prediction.
"""
from __future__ import annotations

import numpy as np

from .live_rig import predict_rig_joints
from .rig_predictor import RigPredictor


def consensus_joints(points: np.ndarray, predictor: RigPredictor,
                     rounds: int = 8, merge: float = 0.12,
                     min_frac: float = 0.5) -> np.ndarray:
    """Joints that recur across `rounds` independent ensemble predictions,
    in the cloud's ORIGINAL frame. High precision by construction."""
    P = np.asarray(points, dtype=np.float32)
    c = P.mean(0)
    s = np.abs(P - c).max() + 1e-6
    all_j, all_round = [], []
    for r in range(rounds):
        j = predict_rig_joints(P, predictor, seed=1000 + 37 * r)
        if len(j):
            all_j.append((j - c) / s)
            all_round.append(np.full(len(j), r))
    if not all_j:
        return np.zeros((0, 3), np.float32)
    J = np.concatenate(all_j)
    R = np.concatenate(all_round)
    out, used = [], np.zeros(len(J), bool)
    for i in np.argsort(-np.bincount(R)[R].astype(float)):
        if used[i]:
            continue
        grp = np.linalg.norm(J - J[i], axis=1) < merge
        used |= grp
        if len(np.unique(R[grp])) >= max(2, int(np.ceil(rounds * min_frac))):
            out.append(J[grp].mean(0))
    joints = np.asarray(out, np.float32).reshape(-1, 3)
    return joints * s + c if len(joints) else joints


def pseudo_shape(points: np.ndarray, joints: np.ndarray,
                 n_sub: int = 1200, seed: int = 0, sigma: float = 0.06) -> dict:
    """Turn a real cloud + consensus joints into a training sample shaped like
    synth_rigs.make_creature output (points canonical, gaussian jointness)."""
    P = np.asarray(points, dtype=np.float32)
    c = P.mean(0)
    s = np.abs(P - c).max() + 1e-6
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=min(n_sub, len(P)), replace=False)
    sub = (P[idx] - c) / s
    jc = (np.asarray(joints, np.float32) - c) / s
    dj = np.linalg.norm(sub[:, None, :] - jc[None, :, :], axis=2).min(1)
    jointness = np.exp(-(dj ** 2) / (2 * sigma ** 2))
    return {"points": sub.astype(np.float32), "joints": jc,
            "jointness": jointness.astype(np.float32), "kind": "pseudo"}


def self_train(predictor: RigPredictor, real_clouds: list, synth_shapes: list,
               copies: int = 6, rounds: int = 8,
               epochs: int = 200, device: str = "cpu") -> dict:
    """One self-training cycle: consensus-label each real cloud, mix `copies`
    differently-subsampled pseudo shapes per cloud into the synthetic set, and
    retrain. Returns stats about what was actually taught."""
    pseudo, per_cloud = [], []
    for ci, cloud in enumerate(real_clouds):
        joints = consensus_joints(cloud, predictor, rounds=rounds)
        per_cloud.append(len(joints))
        if len(joints) == 0:
            continue
        for k in range(copies):
            pseudo.append(pseudo_shape(cloud, joints, seed=ci * 100 + k))
    loss = predictor.train(synth_shapes + pseudo, epochs=epochs, device=device)
    return {"pseudo_shapes": len(pseudo), "consensus_joints_per_cloud": per_cloud,
            "final_loss": float(loss)}
