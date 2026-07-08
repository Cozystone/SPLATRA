# -*- coding: utf-8 -*-
"""Synthetic rigged creatures — supervised training data for the rig predictor.

A learned rig predictor generalises to ANY generated model, but a real RigNet
needs thousands of hand-rigged meshes we do not have. So we MANUFACTURE the
supervision: procedural creatures whose skeleton — INCLUDING internal joints
(knees, elbows, spine) that farthest-point sampling can never find — is known
exactly, with a point cloud skinned around it.

The DISTRIBUTION is what the net learns, so it must be broad enough to cover
real generated fields (TripoSR-style):
  * body plans: quadruped / biped / chunky blob-with-stubs (pikachu-like)
  * head appendages (ears) with their own bend joints
  * wide proportion ranges (stubby..lanky), random limb splay, random yaw
  * SURFACE sampling (hollow shells) mixed with volume — real 3DGS fields are
    surface shells, so half the creatures are skinned as shells.
Points are canonicalised (centre + unit scale) — the same normalisation applied
to a real field at predict time.
"""
from __future__ import annotations

import numpy as np

# a skeleton is a list of joints [J,3] and bones (pairs of joint indices).
# "internal" joints are the ones with >=2 bones — the ones that actually bend.


def _limb(rng, joints, bones, root_idx, direction, length, splay=0.35):
    """Attach a 2-bone limb (root -> mid -> tip): every limb owns an internal
    bend joint (knee/elbow/ear-base). Direction is jittered by `splay`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d + rng.normal(scale=splay, size=3)
    d /= np.linalg.norm(d) + 1e-9
    base = joints[root_idx]
    bend = rng.uniform(0.35, 0.65)                      # where the mid joint sits
    kink = rng.normal(scale=0.25, size=3)               # mid joint bends off-axis
    mid = base + d * length * bend + kink * length * 0.25
    tip = mid + d * length * (1.0 - bend) + rng.normal(scale=0.08, size=3) * length
    mi = len(joints); joints.append(mid); bones.append((root_idx, mi))
    ti = len(joints); joints.append(tip); bones.append((mi, ti))
    return mi, ti


def make_creature(rng: np.random.Generator, n_points: int = 1600, kind: str | None = None):
    if kind is None:
        kind = ["quad", "biped", "blob"][int(rng.integers(0, 3))]

    torso_len = rng.uniform(0.5, 1.2)
    body_y = rng.uniform(0.4, 0.8)
    joints = [np.array([-torso_len / 2, body_y, 0.0])]               # 0 hip
    joints.append(np.array([0.0, body_y + rng.uniform(-0.08, 0.08), 0.0]))  # 1 spine-mid
    joints.append(np.array([torso_len / 2, body_y, 0.0]))            # 2 shoulder
    bones = [(0, 1), (1, 2)]
    thick_body = rng.uniform(0.14, 0.42)                             # thin..chunky
    limb_len = rng.uniform(0.3, 0.9)
    limb_thick = thick_body * rng.uniform(0.25, 0.55)

    # head on a neck (neck joint = internal once ears attach)
    head = np.array([torso_len / 2 + rng.uniform(0.15, 0.4),
                     body_y + rng.uniform(0.15, 0.45), 0.0])
    hi = len(joints); joints.append(head); bones.append((2, hi))

    if kind == "quad":
        for (rj, z) in [(0, 1), (0, -1), (2, 1), (2, -1)]:
            _limb(rng, joints, bones, rj, (0.05, -1.0, 0.35 * z), limb_len)
        if rng.random() < 0.7:                                       # tail
            _limb(rng, joints, bones, 0, (-1.0, 0.4, 0.0), limb_len * 0.8)
    elif kind == "biped":
        for z in (1, -1):                                            # legs w/ knees
            _limb(rng, joints, bones, 0, (0.0, -1.0, 0.3 * z), limb_len)
        for z in (1, -1):                                            # arms w/ elbows
            _limb(rng, joints, bones, 2, (0.3, -0.5, 0.8 * z), limb_len * 0.9)
    else:                                                            # blob + stubs
        n_stub = int(rng.integers(4, 7))
        for s in range(n_stub):
            az = 2 * np.pi * s / n_stub + rng.uniform(-0.3, 0.3)
            d = (0.4 * np.cos(az), -0.9, 0.7 * np.sin(az))
            _limb(rng, joints, bones, int(rng.integers(0, 3)), d, limb_len * 0.7)
    # ears / head appendages with their own bend joints (0-2)
    for z in list(range(int(rng.integers(0, 3)))):
        _limb(rng, joints, bones, hi, (0.1, 1.0, 0.5 * (z * 2 - 1)), limb_len * 0.6)

    joints = np.array(joints, dtype=np.float64)

    # skin: points along each bone as a tube — SURFACE shell for half the
    # creatures (real 3DGS fields are hollow shells), gaussian volume otherwise
    surface = rng.random() < 0.5
    head_r = thick_body * rng.uniform(0.7, 1.4)
    pts, pbone = [], []
    per = max(1, n_points // (len(bones) + 2))
    for bi, (a, b) in enumerate(bones):
        A, B = joints[a], joints[b]
        seglen = np.linalg.norm(B - A) + 1e-6
        thick = thick_body if bi < 2 else (head_r * 0.5 if bi == 2 else limb_thick)
        t = rng.uniform(0, 1, per)
        centers = A[None] + t[:, None] * (B - A)[None]
        off = rng.normal(size=(per, 3))
        d = (B - A) / seglen
        off -= (off @ d)[:, None] * d[None]              # wrap around the bone
        if surface:
            off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
            off *= thick * rng.uniform(0.9, 1.1, (per, 1))
        else:
            off *= thick
        pts.append(centers + off)
        pbone.append(np.full(per, bi))
    # head blob + (for blob kind) torso blob — chunky mass, not just tubes
    blobs = [(joints[3], head_r)]
    if kind == "blob":
        blobs.append((joints[1], thick_body * rng.uniform(1.6, 2.4)))
    for centre, radius in blobs:
        off = rng.normal(size=(per * 2, 3))
        if surface:
            off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
        off *= radius * rng.uniform(0.85, 1.1, (per * 2, 1))
        pts.append(centre[None] + off)
        pbone.append(np.full(per * 2, 2))                # head bone bucket
    P = np.vstack(pts)
    pbone = np.concatenate(pbone).astype(np.int64)

    # random yaw so nothing is tied to the x-axis spine convention
    yaw = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    P = P @ R.T
    joints = joints @ R.T

    # canonicalise: centre + unit scale (same as we'll do to a real field)
    c = P.mean(0); P = P - c; joints = joints - c
    s = np.abs(P).max() + 1e-6; P = P / s; joints = joints / s

    # per-point jointness = closeness to the nearest joint
    dj = np.linalg.norm(P[:, None, :] - joints[None, :, :], axis=2).min(1)
    jointness = np.exp(-(dj ** 2) / (2 * 0.06 ** 2))

    internal = np.array([j for j in range(len(joints))
                         if sum((a == j) + (b == j) for a, b in bones) >= 2])
    return {
        "points": P.astype(np.float32), "joints": joints.astype(np.float32),
        "bones": bones, "point_bone": pbone, "jointness": jointness.astype(np.float32),
        "internal_joints": internal, "kind": kind,
    }
