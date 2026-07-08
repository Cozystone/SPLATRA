# -*- coding: utf-8 -*-
"""Semantic parts: eye detection is shape-agnostic and honestly abstains."""

from __future__ import annotations

import os

import numpy as np

from atanor_core.rigging.parts import find_eyes
from atanor_core.rigging.live_rig import load_spl2, bind_joints, decompose_chains

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "viewer", "samples")


def _synthetic_face(n=4000, seed=0):
    """Yellow ball with two dark eye spots — no species, just the pattern."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    c = np.tile(np.array([0.9, 0.8, 0.2], np.float32), (n, 1))
    for ez in (0.35, -0.35):
        eye = np.array([0.9, 0.25, ez], np.float32)
        eye /= np.linalg.norm(eye)
        m = np.linalg.norm(p - eye, axis=1) < 0.18
        c[m] = [0.05, 0.05, 0.08]
    return p, c


def test_finds_the_two_dark_spots_on_any_shape():
    p, c = _synthetic_face()
    eyes = find_eyes(p, c)
    assert len(eyes) == 2
    zs = sorted(e["center"][2] for e in eyes)
    assert zs[0] < 0 < zs[1]                  # a left/right pair


def test_abstains_without_contrast():
    p, _ = _synthetic_face()
    plain = np.tile(np.array([0.5, 0.5, 0.5], np.float32), (len(p), 1))
    assert find_eyes(p, plain) == []


def test_real_pikachu_and_torus():
    pika = os.path.join(SAMPLES, "pikachu.bin")
    torus = os.path.join(SAMPLES, "torus.bin")
    if not (os.path.exists(pika) and os.path.exists(torus)):
        return
    pos, col, *_ = load_spl2(pika)
    assert len(find_eyes(pos, col)) == 2      # both eyes, merged, paired
    pos, col, *_ = load_spl2(torus)
    assert find_eyes(pos, col) == []          # a torus has no eyes


def test_decompose_chains_partitions_all_joints():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(3000, 3)).astype(np.float32)
    joints = np.array([[1.2, 0, 0], [1.8, 0, 0], [-1.2, 0.2, 0],
                       [0, 1.5, 0], [0, 2.2, 0]], np.float32)
    rig = bind_joints(pts, joints)
    chains = decompose_chains(rig)
    flat = [j for ch in chains for j in ch]
    assert sorted(flat) == list(range(len(joints)))         # exactly once each
    for ch in chains:                                       # root->tip ordering
        d = [np.linalg.norm(joints[j] - rig.centroid) for j in ch]
        assert d == sorted(d)
