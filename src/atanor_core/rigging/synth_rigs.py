# -*- coding: utf-8 -*-
"""Synthetic rigged creatures — supervised training data for the rig predictor.

A learned rig predictor generalises to ANY generated model, but a real RigNet
needs thousands of hand-rigged meshes we do not have. So we MANUFACTURE the
supervision: procedural creatures whose skeleton — INCLUDING internal joints
(knees, spine) that farthest-point sampling can never find — is known exactly,
with a point cloud skinned around it. Vary proportions/angles/limb counts so the
predictor learns the distribution, not one shape. Points are canonicalised
(centre + unit scale) — the same normalisation applied to a real field at predict.
"""
from __future__ import annotations

import numpy as np

# a skeleton is a list of joints [J,3] and bones (pairs of joint indices).
# joint 0 is the root (hip). "internal" joints are the non-tip ones (knees, spine).


def make_creature(rng: np.random.Random, n_points: int = 1600):
    hipx = -0.5 + rng.uniform(-0.1, 0.1)
    shx = 0.5 + rng.uniform(-0.1, 0.1)
    bodyY = rng.uniform(0.45, 0.75)
    legLen = rng.uniform(0.5, 0.85)
    joints = [np.array([hipx, bodyY, 0.0])]           # 0 hip (root)
    joints.append(np.array([(hipx + shx) / 2, bodyY + rng.uniform(-0.05, 0.05), 0.0]))  # 1 spine-mid (internal)
    joints.append(np.array([shx, bodyY, 0.0]))        # 2 shoulder
    bones = [(0, 1), (1, 2)]
    # head (from shoulder up-forward)
    joints.append(np.array([shx + 0.25, bodyY + 0.35, 0.0])); bones.append((2, len(joints) - 1))
    # tail (from hip back-up)
    joints.append(np.array([hipx - 0.3, bodyY + 0.15, 0.0])); bones.append((0, len(joints) - 1))
    # 4 legs, each with a KNEE (internal joint): root -> knee -> foot
    for (rootJ, z) in [(0, 0.18), (0, -0.18), (2, 0.18), (2, -0.18)]:
        base = joints[rootJ]
        kneeY = bodyY - legLen * 0.5 + rng.uniform(-0.05, 0.05)
        kneeX = base[0] + rng.uniform(-0.08, 0.08)
        knee = np.array([kneeX, kneeY, z])
        foot = np.array([kneeX + rng.uniform(-0.05, 0.15), bodyY - legLen, z])
        ki = len(joints); joints.append(knee); bones.append((rootJ, ki))       # thigh
        fi = len(joints); joints.append(foot); bones.append((ki, fi))          # shin (knee internal)
    joints = np.array(joints, dtype=np.float64)

    # skin: sample points along each bone as a tapered tube
    radii = {}
    pts, pbone = [], []
    per = max(1, n_points // len(bones))
    for bi, (a, b) in enumerate(bones):
        A, B = joints[a], joints[b]
        seglen = np.linalg.norm(B - A) + 1e-6
        # body/spine bones are thick, legs/head/tail thin
        thick = 0.22 if bi < 2 else (0.12 if bi in (2, 3) else 0.07)
        t = rng.uniform(0, 1, per)
        centers = A[None] + t[:, None] * (B - A)[None]
        # random tube offset perpendicular-ish
        off = rng.normal(size=(per, 3)) * thick
        # remove component along the bone so it wraps the segment
        d = (B - A) / seglen
        off -= (off @ d)[:, None] * d[None]
        pts.append(centers + off)
        pbone.append(np.full(per, bi))
    P = np.vstack(pts)
    pbone = np.concatenate(pbone).astype(np.int64)

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
        "internal_joints": internal,
    }
