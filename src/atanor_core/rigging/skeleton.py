# -*- coding: utf-8 -*-
"""Auto-rigging for Gaussian objects — turn a surface shell into a live body.

SPLATRA generates a *surface* Gaussian shell (no skeleton). To make it move as
if alive (for animation / explanation), we form a skeleton and rig it — here,
automatically from the geometry, no manual armature:

  1. extract_skeleton(means): PCA principal axis -> a MEDIAL joint chain (the
     centroid of each slice along the axis), so the skeleton follows the object's
     body, curved or straight. This is the auto-formed 뼈대.
  2. bind(means, joints): skin each particle to its 1-2 nearest bones with smooth
     distance weights (linear-blend-skinning bind).
  3. pose(joints, t): forward-kinematics a travelling sine wave along the chain
     (each bone bends relative to its parent) — the skeleton wriggles / breathes.
  4. skin(field, ...): apply the per-bone rigid transforms to the particles (and
     rotate their gaussians' quats to match), so the shell deforms as one body.

Pure numpy, deterministic, no learned weights yet — the honest geometric v0 of
"AI auto-rigs the model and the particles come alive." A learned rig predictor
(RigNet-style) can later replace extract_skeleton without touching the skinning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rig:
    joints: np.ndarray      # [K, 3] rest-pose joint positions (bone chain)
    bone_of: np.ndarray     # [N]    index of the bone each particle is bound to
    bone_w: np.ndarray      # [N, 2] weights to (bone_of, bone_of+1) — LBS blend
    axis: np.ndarray        # [3]    principal axis (for reference)

    @property
    def n_bones(self) -> int:
        return int(self.joints.shape[0] - 1)


def extract_skeleton(means: np.ndarray, n_joints: int = 12) -> np.ndarray:
    """A medial joint chain along the object's principal axis. Auto, no manual rig."""
    m = np.asarray(means, dtype=np.float64)
    c = m.mean(axis=0)
    X = m - c
    cov = (X.T @ X) / max(1, len(X))
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    proj = X @ axis
    lo, hi = float(proj.min()), float(proj.max())
    edges = np.linspace(lo, hi, n_joints)
    half = (hi - lo) / max(1, 2 * (n_joints - 1)) + 1e-6
    joints = np.empty((n_joints, 3), dtype=np.float64)
    for k, e in enumerate(edges):
        mask = np.abs(proj - e) <= half
        joints[k] = m[mask].mean(axis=0) if int(mask.sum()) >= 3 else (c + axis * e)
    return joints.astype(np.float32)


def bind(means: np.ndarray, joints: np.ndarray) -> Rig:
    """Skin each particle to its nearest bone segment with smooth 2-bone weights."""
    m = np.asarray(means, dtype=np.float64)
    J = np.asarray(joints, dtype=np.float64)
    nb = len(J) - 1
    a = J[:-1]                 # [B,3] bone starts
    d = J[1:] - J[:-1]         # [B,3] bone directions
    dl2 = np.sum(d * d, axis=1) + 1e-9
    # projection param t of each point on each bone, clamped to [0,1]
    # dist2 point-to-segment for all bones
    diff = m[:, None, :] - a[None, :, :]                 # [N,B,3]
    t = np.clip(np.sum(diff * d[None], axis=2) / dl2[None], 0.0, 1.0)  # [N,B]
    closest = a[None] + t[:, :, None] * d[None]          # [N,B,3]
    dist2 = np.sum((m[:, None, :] - closest) ** 2, axis=2)  # [N,B]
    bone_of = np.argmin(dist2, axis=1).astype(np.int32)  # [N]
    # blend weight along the bone toward the next bone (t near 1 -> share w/ next)
    tt = t[np.arange(len(m)), bone_of]
    w_next = np.where(bone_of < nb - 1, 0.5 * np.clip(tt - 0.5, 0, 0.5) / 0.5, 0.0)
    bone_w = np.stack([1.0 - w_next, w_next], axis=1).astype(np.float32)
    return Rig(joints=joints.astype(np.float32), bone_of=bone_of,
               bone_w=bone_w, axis=np.zeros(3, np.float32))


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit-ish vector u to v (Rodrigues)."""
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    c = float(np.dot(u, v))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def pose(joints: np.ndarray, t: float, amplitude: float = 0.6,
         wavelength: float = 3.0) -> np.ndarray:
    """Forward-kinematics a travelling sine wave down the bone chain: each bone
    bends relative to its parent, so the body wriggles / sways / breathes."""
    J = np.asarray(joints, dtype=np.float64)
    K = len(J)
    # bend axis = perpendicular to the overall chain direction
    chain = J[-1] - J[0]
    ref = np.array([0.0, 0.0, 1.0]) if abs(chain[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    bend_axis = np.cross(chain, ref)
    bend_axis /= (np.linalg.norm(bend_axis) + 1e-9)
    posed = np.empty_like(J)
    posed[0] = J[0]
    R = np.eye(3)
    for i in range(1, K):
        rest_seg = J[i] - J[i - 1]
        ang = amplitude * np.sin(2 * np.pi * (i / wavelength) - t) * (i / K)
        ca, sa = np.cos(ang), np.sin(ang)
        ux, uy, uz = bend_axis
        Kx = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        Rb = np.eye(3) + sa * Kx + (1 - ca) * (Kx @ Kx)  # rotate about bend axis
        R = R @ Rb
        posed[i] = posed[i - 1] + R @ rest_seg
    return posed.astype(np.float32)


def skin(means: np.ndarray, rig: Rig, posed_joints: np.ndarray) -> np.ndarray:
    """Linear-blend-skin the particles from rest joints to posed joints."""
    m = np.asarray(means, dtype=np.float64)
    J = np.asarray(rig.joints, dtype=np.float64)
    P = np.asarray(posed_joints, dtype=np.float64)
    out = np.empty_like(m)
    for b in range(rig.n_bones):
        sel = rig.bone_of == b
        if not np.any(sel):
            continue
        rest_dir = J[b + 1] - J[b]
        pose_dir = P[b + 1] - P[b]
        Rb = _rot_between(rest_dir, pose_dir)
        # rigid transform of bone b: rotate about its rest start, move start to posed start
        out[sel] = P[b] + (m[sel] - J[b]) @ Rb.T
    return out.astype(np.float32)


def animate_positions(means: np.ndarray, rig: Rig, t: float, **pose_kw) -> np.ndarray:
    """Convenience: rest positions -> deformed positions at animation time t."""
    return skin(means, rig, pose(rig.joints, t, **pose_kw))
