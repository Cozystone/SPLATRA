# -*- coding: utf-8 -*-
"""Synthetic rigged shapes — supervised training data for the rig predictor.

A learned rig predictor generalises to ANY generated model, but a real RigNet
needs thousands of hand-rigged meshes we do not have. So we MANUFACTURE the
supervision: procedural shapes whose skeleton — INCLUDING internal joints
(knees, elbows, spine) that farthest-point sampling can never find — is known
exactly, with a point cloud skinned around it.

The DISTRIBUTION is what the net learns, so it must cover what generators emit:
  * creature plans: quadruped / biped / chunky blob-with-stubs / humanoid / snake
  * GENERAL meshes:  tree (trunk+branches), furniture (slab + legs)
  * head appendages incl. WIDE-BASE TAPERED cones (pikachu-style ears) with a
    bend joint — wide bases were invisible to the constant-radius tube teacher
  * wide proportion ranges, random limb splay, random yaw
  * SURFACE sampling (hollow shells) mixed with volume — real 3DGS fields are
    surface shells
  * generation-artifact augmentation: anisotropic scale, noise, patch dropout
Points are canonicalised (centre + unit scale) — the same normalisation applied
to a real field at predict time.
"""
from __future__ import annotations

import numpy as np

KINDS = ("quad", "biped", "blob", "humanoid", "snake", "tree", "furniture")

# a skeleton is a list of joints [J,3], bones (pairs of joint indices), and a
# per-bone (thick_at_a, thick_at_b) taper. "internal" joints have >=2 bones.


def _limb(rng, joints, bones, taper, root_idx, direction, length,
          thick, splay=0.35, tip_thick=None):
    """Attach a 2-bone limb (root -> mid -> tip): every limb owns an internal
    bend joint (knee/elbow/branch fork). Tapers from `thick` to `tip_thick`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d + rng.normal(scale=splay, size=3)
    d /= np.linalg.norm(d) + 1e-9
    base = joints[root_idx]
    bend = rng.uniform(0.35, 0.65)
    kink = rng.normal(scale=0.25, size=3)
    mid = base + d * length * bend + kink * length * 0.25
    tip = mid + d * length * (1.0 - bend) + rng.normal(scale=0.08, size=3) * length
    tt = thick * 0.5 if tip_thick is None else tip_thick
    mi = len(joints); joints.append(mid); bones.append((root_idx, mi))
    taper.append((thick, (thick + tt) / 2))
    ti = len(joints); joints.append(tip); bones.append((mi, ti))
    taper.append(((thick + tt) / 2, tt))
    return mi, ti


def _skeleton(rng, kind):
    """Build (joints list, bones, taper, blobs[(centre, radius)]) for a kind."""
    joints, bones, taper, blobs = [], [], [], []
    thick_body = rng.uniform(0.14, 0.42)
    limb_len = rng.uniform(0.3, 0.9)
    limb_thick = thick_body * rng.uniform(0.25, 0.55)

    if kind == "snake":
        # pure chain: every joint but the two ends is internal
        n_seg = int(rng.integers(4, 8))
        p = np.array([0.0, rng.uniform(0.3, 0.6), 0.0])
        heading = np.array([1.0, 0.0, 0.0])
        joints.append(p.copy())
        for _ in range(n_seg):
            heading = heading + rng.normal(scale=0.45, size=3)
            heading /= np.linalg.norm(heading) + 1e-9
            seg = rng.uniform(0.25, 0.5)
            p = p + heading * seg
            joints.append(p.copy())
            bones.append((len(joints) - 2, len(joints) - 1))
            taper.append((thick_body * 0.6, thick_body * 0.6))
        return joints, bones, taper, blobs

    if kind == "tree":
        # trunk of 2-3 segments up, then 3-6 tapering branches — general mesh
        n_trunk = int(rng.integers(2, 4))
        p = np.array([0.0, 0.0, 0.0])
        joints.append(p.copy())
        for _ in range(n_trunk):
            p = p + np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.7),
                              rng.uniform(-0.1, 0.1)])
            joints.append(p.copy())
            bones.append((len(joints) - 2, len(joints) - 1))
            taper.append((thick_body, thick_body * 0.75))
        for _ in range(int(rng.integers(3, 7))):
            root = int(rng.integers(1, len(joints)))
            az = rng.uniform(0, 2 * np.pi)
            d = (np.cos(az), rng.uniform(0.2, 0.9), np.sin(az))
            _limb(rng, joints, bones, taper, root, d, limb_len,
                  limb_thick, tip_thick=limb_thick * 0.3)
        return joints, bones, taper, blobs

    if kind == "furniture":
        # slab top + 3-4 straight legs with a mid joint — tables/chairs/stools
        h = rng.uniform(0.5, 0.9)
        half = rng.uniform(0.35, 0.7)
        c0 = np.array([-half, h, 0.0]); c1 = np.array([half, h, 0.0])
        joints.extend([c0, (c0 + c1) / 2, c1])
        bones.extend([(0, 1), (1, 2)])
        taper.extend([(thick_body * 0.6, thick_body * 0.6)] * 2)
        blobs.append(((c0 + c1) / 2, half * 1.1, "slab"))
        for (rj, z) in [(0, 1), (0, -1), (2, 1), (2, -1)][: int(rng.integers(3, 5))]:
            _limb(rng, joints, bones, taper, rj, (0.0, -1.0, 0.3 * z), h * 0.95,
                  limb_thick, splay=0.1, tip_thick=limb_thick * 0.8)
        return joints, bones, taper, blobs

    # creature bodies share a horizontal torso ------------------------------
    torso_len = rng.uniform(0.5, 1.2)
    body_y = rng.uniform(0.4, 0.8)
    vertical = kind == "humanoid"
    if vertical:            # humanoid torso is upright: hip below, shoulder up
        joints.append(np.array([0.0, body_y - torso_len / 2, 0.0]))       # 0 hip
        joints.append(np.array([0.0, body_y + rng.uniform(-0.05, 0.05), 0.0]))
        joints.append(np.array([0.0, body_y + torso_len / 2, 0.0]))       # 2 shoulder
    else:
        joints.append(np.array([-torso_len / 2, body_y, 0.0]))            # 0 hip
        joints.append(np.array([0.0, body_y + rng.uniform(-0.08, 0.08), 0.0]))
        joints.append(np.array([torso_len / 2, body_y, 0.0]))             # 2 shoulder
    bones.extend([(0, 1), (1, 2)])
    taper.extend([(thick_body, thick_body)] * 2)

    head_r = thick_body * rng.uniform(0.7, 1.4)
    if vertical:
        head = joints[2] + np.array([0.0, rng.uniform(0.15, 0.3), 0.0])
    else:
        head = joints[2] + np.array([rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.45), 0.0])
    hi = len(joints); joints.append(head)
    bones.append((2, hi)); taper.append((head_r * 0.5, head_r * 0.5))
    blobs.append((head, head_r, "ball"))

    if kind == "quad":
        for (rj, z) in [(0, 1), (0, -1), (2, 1), (2, -1)]:
            _limb(rng, joints, bones, taper, rj, (0.05, -1.0, 0.35 * z),
                  limb_len, limb_thick)
        if rng.random() < 0.7:
            _limb(rng, joints, bones, taper, 0, (-1.0, 0.4, 0.0),
                  limb_len * 0.8, limb_thick)
    elif kind in ("biped", "humanoid"):
        for z in (1, -1):
            _limb(rng, joints, bones, taper, 0, (0.0, -1.0, 0.3 * z),
                  limb_len, limb_thick)
        for z in (1, -1):
            _limb(rng, joints, bones, taper, 2, (0.3, -0.5, 0.8 * z),
                  limb_len * 0.9, limb_thick)
    else:                                                        # blob + stubs
        n_stub = int(rng.integers(4, 7))
        for si in range(n_stub):
            az = 2 * np.pi * si / n_stub + rng.uniform(-0.3, 0.3)
            d = (0.4 * np.cos(az), -0.9, 0.7 * np.sin(az))
            _limb(rng, joints, bones, taper, int(rng.integers(0, 3)), d,
                  limb_len * 0.7, limb_thick)
        blobs.append((joints[1], thick_body * rng.uniform(1.6, 2.4), "ball"))

    # ears: WIDE-BASE tapered cones half the time (pikachu-style), thin otherwise
    for z in list(range(int(rng.integers(0, 3)))):
        wide = rng.random() < 0.5
        base_th = head_r * rng.uniform(0.45, 0.75) if wide else limb_thick
        _limb(rng, joints, bones, taper, hi, (0.1, 1.0, 0.5 * (z * 2 - 1)),
              limb_len * rng.uniform(0.5, 0.9), base_th,
              tip_thick=base_th * (0.12 if wide else 0.5))
    return joints, bones, taper, blobs


def make_creature(rng: np.random.Generator, n_points: int = 1600,
                  kind: str | None = None, augment: bool = True):
    if kind is None:
        kind = KINDS[int(rng.integers(0, len(KINDS)))]
    joints, bones, taper, blobs = _skeleton(rng, kind)
    joints = [np.asarray(j, dtype=np.float64) for j in joints]

    # skin: points along each bone as a LINEARLY TAPERED tube — SURFACE shell
    # for half the shapes (real 3DGS fields are hollow), gaussian volume else
    surface = rng.random() < 0.5
    pts, pbone = [], []
    per = max(1, n_points // (len(bones) + 2 * max(1, len(blobs))))
    for bi, (a, b) in enumerate(bones):
        A, B = joints[a], joints[b]
        seglen = np.linalg.norm(B - A) + 1e-6
        t = rng.uniform(0, 1, per)
        thick = taper[bi][0] * (1 - t) + taper[bi][1] * t          # taper along bone
        centers = A[None] + t[:, None] * (B - A)[None]
        off = rng.normal(size=(per, 3))
        d = (B - A) / seglen
        off -= (off @ d)[:, None] * d[None]
        if surface:
            off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
            off *= (thick * rng.uniform(0.9, 1.1, per))[:, None]
        else:
            off *= thick[:, None]
        pts.append(centers + off)
        pbone.append(np.full(per, bi))
    for centre, radius, style in blobs:
        n_blob = per * 2
        off = rng.normal(size=(n_blob, 3))
        if surface:
            off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
        off *= radius * rng.uniform(0.85, 1.1, (n_blob, 1))
        if style == "slab":                                        # flat top
            off[:, 1] *= 0.12
        pts.append(np.asarray(centre)[None] + off)
        pbone.append(np.full(n_blob, min(2, len(bones) - 1)))
    P = np.vstack(pts)
    pbone = np.concatenate(pbone).astype(np.int64)
    joints = np.array(joints, dtype=np.float64)

    # random yaw so nothing is tied to the x-axis convention
    yaw = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    P = P @ R.T
    joints = joints @ R.T

    if augment:
        # generation artifacts: anisotropic squash, sensor noise, missing patches
        ani = rng.uniform(0.75, 1.3, 3)
        P *= ani; joints *= ani
        span = float(np.abs(P - P.mean(0)).max()) + 1e-6
        P += rng.normal(scale=span * rng.uniform(0.002, 0.012), size=P.shape)
        for _ in range(int(rng.integers(0, 3))):                   # dropout holes
            hole = P[int(rng.integers(0, len(P)))]
            keep = np.linalg.norm(P - hole, axis=1) > span * rng.uniform(0.05, 0.12)
            if keep.sum() > len(P) * 0.7:
                P, pbone = P[keep], pbone[keep]

    # canonicalise: centre + unit scale (same as a real field at predict time)
    c = P.mean(0); P = P - c; joints = joints - c
    s = np.abs(P).max() + 1e-6; P = P / s; joints = joints / s

    dj = np.linalg.norm(P[:, None, :] - joints[None, :, :], axis=2).min(1)
    jointness = np.exp(-(dj ** 2) / (2 * 0.06 ** 2))

    internal = np.array([j for j in range(len(joints))
                         if sum((a == j) + (b == j) for a, b in bones) >= 2])
    return {
        "points": P.astype(np.float32), "joints": joints.astype(np.float32),
        "bones": bones, "point_bone": pbone, "jointness": jointness.astype(np.float32),
        "internal_joints": internal, "kind": kind,
    }
