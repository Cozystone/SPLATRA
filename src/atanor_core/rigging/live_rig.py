# -*- coding: utf-8 -*-
"""Live rig — run the learned rig predictor on a REAL generated field and drive it.

Chain: surface shell (TripoSR/SPL2 cartridge) -> predicted joints (rig_predictor,
internal joints included) -> joint binding -> articulated pose. ATANOR supplies
per-joint intents; this module is the musculature that executes them.

The predictor's features need an N x N distance matrix, so joints are predicted on
a bounded subsample (joints are global structure; a few thousand points carry it),
then EVERY particle is bound to the predicted joints for posing.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .rig_predictor import RigPredictor


def load_spl2(path: str):
    """Read an SPL2 cartridge -> (pos[N,3], col[N,3], scale[N,3] linear,
    quat[N,4], opa[N] linear). The exact format _pack_cartridge writes."""
    with open(path, "rb") as fh:
        blob = fh.read()
    assert blob[:4] == b"SPL2", "not an SPL2 cartridge"
    n = struct.unpack("<I", blob[4:8])[0]
    f = np.frombuffer(blob, np.float32, offset=8)
    o = 0
    pos = f[o:o + n * 3].reshape(n, 3); o += n * 3
    col = f[o:o + n * 3].reshape(n, 3); o += n * 3
    scale = f[o:o + n * 3].reshape(n, 3); o += n * 3
    quat = f[o:o + n * 4].reshape(n, 4); o += n * 4
    opa = f[o:o + n]
    return pos.copy(), col.copy(), scale.copy(), quat.copy(), opa.copy()


def predict_rig_joints(points: np.ndarray, predictor: RigPredictor,
                       n_sub: int = 1200, seed: int = 0, ensemble: int = 5,
                       merge: float = 0.12, min_votes: int = 2) -> np.ndarray:
    """Predict joint positions for a full particle cloud, returned in the CLOUD's
    original coordinate frame (the predictor works canonically).

    A single random subsample is UNSTABLE (the confidence peak — and with it the
    adaptive cut — swings with the draw). And a raw score cut cannot separate a
    weak-but-real joint region from body noise (their SCORES overlap; what
    differs is that a real joint is a CLUSTER that recurs in every subsample).
    So: pool hot points from several subsamples with a low cut, mean-shift the
    union, then rank clusters by vote count (how many subsamples contributed)
    and mass — recurring clusters survive, one-draw noise dies. The RNG is
    seeded from the cloud itself, so the same field always yields the same rig."""
    P = np.asarray(points, dtype=np.float32)
    c = P.mean(0)
    s = np.abs(P - c).max() + 1e-6
    base = int(np.abs(P[:64]).sum() * 1e3) % (2 ** 31)     # deterministic per field
    rng = np.random.default_rng(base + seed)
    ensemble = max(1, ensemble)
    hot, wj, vote = [], [], []
    for e in range(ensemble):
        idx = rng.choice(len(P), size=min(n_sub, len(P)), replace=False)
        sub = (P[idx] - c) / s
        j = predictor.jointness(sub)
        cut = max(0.12, 0.35 * float(j.max()))
        sel = j > cut
        hot.append(sub[sel]); wj.append(j[sel]); vote.append(np.full(sel.sum(), e))
    hot = np.concatenate(hot); wj = np.concatenate(wj); vote = np.concatenate(vote)
    if len(hot) == 0:
        return np.zeros((0, 3), np.float32)
    clusters = []
    used = np.zeros(len(hot), bool)
    for idx in np.argsort(-wj):
        if used[idx]:
            continue
        grp = np.linalg.norm(hot - hot[idx], axis=1) < merge
        used |= grp
        clusters.append((np.average(hot[grp], axis=0, weights=wj[grp]),
                         float(wj[grp].sum()), len(np.unique(vote[grp]))))
    if not clusters:
        return np.zeros((0, 3), np.float32)
    top_mass = max(m for _, m, _ in clusters)
    need_votes = max(min_votes, int(np.ceil(ensemble * 0.6)))
    joints = [cen for cen, mass, votes in clusters
              if votes >= min(need_votes, ensemble) and mass >= 0.15 * top_mass]
    joints = np.asarray(joints[:16], np.float32).reshape(-1, 3)
    return joints * s + c if len(joints) else joints


@dataclass
class JointRig:
    joints: np.ndarray      # [J,3] predicted joint positions (original frame)
    assign: np.ndarray      # [N]   nearest joint per particle
    weight: np.ndarray      # [N]   0..1, how fully the particle follows its joint
    outward: np.ndarray     # [J,3] unit body-centre -> joint (the limb direction)
    centroid: np.ndarray    # [3]


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bind_joints(points: np.ndarray, joints: np.ndarray) -> JointRig:
    """Bind every particle to a predicted joint it sits DISTAL to (past it, away
    from the body centre) — that is exactly what bends at a knee. Among the
    joints a particle is distal to, the nearest wins. Weight ramps with distal
    reach normalised by the local JOINT SPACING (with many predicted joints the
    global scale is far too coarse — every weight would round to zero)."""
    P = np.asarray(points, dtype=np.float32)
    c = P.mean(0)
    J = len(joints)
    d = np.linalg.norm(P[:, None, :] - joints[None, :, :], axis=2)   # [N,J]
    outward = joints - c
    outward /= (np.linalg.norm(outward, axis=1, keepdims=True) + 1e-6)
    proj = np.einsum("njd,jd->nj", P[:, None, :] - joints[None, :, :], outward)

    # local reach scale = median nearest-neighbour distance between joints
    if J > 1:
        jd = np.linalg.norm(joints[:, None, :] - joints[None, :, :], axis=2)
        jd[np.arange(J), np.arange(J)] = np.inf
        reach = float(np.median(jd.min(1)))
    else:
        reach = float(np.abs(P - c).max()) * 0.35
    reach = max(reach, 1e-4)

    distal = proj > 0.0
    d_masked = np.where(distal, d, np.inf)
    assign = d_masked.argmin(1)
    has = distal.any(1)
    assign[~has] = d.argmin(1)[~has]                     # core points: nearest, weight 0
    p_sel = proj[np.arange(len(P)), assign]
    weight = np.where(has, _smooth(p_sel / reach, 0.0, 0.5), 0.0)
    return JointRig(joints=joints.astype(np.float32), assign=assign,
                    weight=weight.astype(np.float32),
                    outward=outward.astype(np.float32), centroid=c.astype(np.float32))


def _rodrigues(v: np.ndarray, axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return (v * ca + np.cross(axis, v) * sa
            + axis * np.einsum("nd,nd->n", axis, v)[:, None] * (1 - ca))


def pose_chain(points: np.ndarray, rig: JointRig, chain: list[int],
               drive: float, band: float | None = None) -> np.ndarray:
    """True forward kinematics along a joint CHAIN (e.g. a tail or a limb).

    Joints are visited root->tip; each rotates everything past it — including
    the remaining chain joints — so rotations COMPOUND into a coherent curl.
    Per-particle angle saturates to the full drive just past each joint (the
    smooth band is only a skinning blend), so segments stay rigid instead of
    shearing apart the way independent per-joint rotation does."""
    P = np.asarray(points, dtype=np.float32).copy()
    member = np.isin(rig.assign, np.asarray(chain))
    J = rig.joints.copy()
    order = sorted(chain, key=lambda j: np.linalg.norm(J[j] - rig.centroid))
    if band is None:
        seg = [np.linalg.norm(J[order[k + 1]] - J[order[k]])
               for k in range(len(order) - 1)]
        band = (float(np.median(seg)) if seg else 0.3) * 0.5
    up = np.array([0.0, 1.0, 0.0], np.float32)
    prev = rig.centroid
    for k in order:
        dir_k = J[k] - prev
        n = np.linalg.norm(dir_k)
        dir_k = dir_k / n if n > 1e-5 else rig.outward[k]
        axis = np.cross(dir_k, up)
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-4 else np.array([1.0, 0.0, 0.0], np.float32)
        proj = (P - J[k]) @ dir_k
        w = _smooth(proj / band, 0.0, 1.0) * member          # saturates -> rigid
        sel = w > 1e-3
        if sel.any():
            ang = drive * w[sel]
            P[sel] = J[k] + _rodrigues(P[sel] - J[k], np.repeat(axis[None], sel.sum(), 0), ang)
        # the rest of the chain rides along (true FK)
        rest = [j for j in order if np.linalg.norm(J[j] - rig.centroid) >
                np.linalg.norm(J[k] - rig.centroid) + 1e-9]
        if rest:
            r = np.asarray(rest)
            J[r] = J[k] + _rodrigues(J[r] - J[k], np.repeat(axis[None], len(r), 0),
                                     np.full(len(r), drive, np.float32))
        prev = J[k]
    return P


def pose_joints(points: np.ndarray, rig: JointRig, t: float = 0.0,
                amp: float = 0.8, intents: dict | None = None) -> np.ndarray:
    """Articulate the cloud about the predicted joints.

    intents: {joint_index: drive} — ATANOR's per-joint commands. Without intents
    every joint swings gently (idle life). The distal side of each joint rotates
    about it; the body side stays — a real bend, not a global wobble."""
    P = np.asarray(points, dtype=np.float32).copy()
    J = len(rig.joints)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    for j in range(J):
        drive = (intents or {}).get(j)
        ang_j = float(drive) if drive is not None else amp * np.sin(t + j * 1.7) * 0.35
        if abs(ang_j) < 1e-4:
            continue
        sel = rig.assign == j
        if not sel.any():
            continue
        axis1 = np.cross(rig.outward[j], up)
        n = np.linalg.norm(axis1)
        axis1 = axis1 / n if n > 1e-4 else np.array([1.0, 0.0, 0.0], np.float32)
        rel = P[sel] - rig.joints[j]
        ang = ang_j * rig.weight[sel]
        axis = np.repeat(axis1[None, :], sel.sum(), 0)
        P[sel] = rig.joints[j] + _rodrigues(rel, axis, ang)
    return P
