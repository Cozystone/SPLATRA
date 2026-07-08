# -*- coding: utf-8 -*-
"""Semantic micro-parts — find the pieces of ANY generated model that deserve
their own real-time channel (eyes first; the same machinery later fits mouths,
nostrils, buttons, wheels, leaves).

An eye, on virtually any generated creature or character, is a SMALL, COMPACT
cluster of particles whose color contrasts hard with its local surroundings,
sitting in the upper half of the body. That definition is shape-agnostic — no
species, no template, no name. Detection is geometric + photometric only:

  1. score each particle by local color contrast (its color vs the mean color
     within a neighbourhood radius),
  2. cluster the high-contrast particles (greedy mean-shift, like the rig),
  3. keep clusters that are small (radius << body), compact, elevated, and
     appear in a left/right-ish PAIR when two qualify.

Returns spheres (center, radius) in the cloud's own frame — exactly what a
blink/gaze shader channel consumes. Abstains (empty list) on shapes with no
such structure (a torus has no eyes, and should have none)."""
from __future__ import annotations

import numpy as np


def _subsample(P, C, n, seed=0):
    if len(P) <= n:
        return P, C
    idx = np.random.default_rng(seed).choice(len(P), n, replace=False)
    return P[idx], C[idx]


def find_eyes(points: np.ndarray, colors: np.ndarray, n_sub: int = 6000,
              contrast_cut: float = 0.35, max_eyes: int = 4) -> list[dict]:
    """Detect eye-like spots on any particle model. colors in [0,1] RGB."""
    P0 = np.asarray(points, np.float32)
    C0 = np.asarray(colors, np.float32)
    c = P0.mean(0)
    s = float(np.abs(P0 - c).max()) + 1e-6
    P, C = _subsample((P0 - c) / s, C0, n_sub)

    # 1) local color contrast within a neighbourhood radius
    # (squared-distance via matmul: no [n,n,3] intermediate)
    r_nb = 0.10
    sq = (P * P).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    nb = D2 < r_nb * r_nb
    denom = nb.sum(1, keepdims=True).astype(np.float32)
    local_mean = (nb.astype(np.float32) @ C) / np.maximum(denom, 1.0)
    contrast = np.linalg.norm(C - local_mean, axis=1)

    hot = contrast > contrast_cut
    if not hot.any():
        return []
    HP, HC, HW = P[hot], C[hot], contrast[hot]

    # 2) greedy weighted clustering of the hot spots
    clusters = []
    used = np.zeros(len(HP), bool)
    for i in np.argsort(-HW):
        if used[i]:
            continue
        grp = np.linalg.norm(HP - HP[i], axis=1) < 0.09
        used |= grp
        if grp.sum() < 4:
            continue
        centre = np.average(HP[grp], axis=0, weights=HW[grp])
        radius = float(np.linalg.norm(HP[grp] - centre, axis=1).mean() * 1.6)
        clusters.append({"centre": centre, "radius": radius,
                         "mass": float(HW[grp].sum()),
                         "color": HC[grp].mean(0)})

    # merge overlapping clusters — one big eye otherwise shows up several times
    clusters.sort(key=lambda e: -e["mass"])
    merged = []
    for cl in clusters:
        home = None
        for m in merged:
            if np.linalg.norm(cl["centre"] - m["centre"]) < (cl["radius"] + m["radius"]) * 0.8:
                home = m
                break
        if home is None:
            merged.append(cl)
        else:
            w1, w2 = home["mass"], cl["mass"]
            home["centre"] = (home["centre"] * w1 + cl["centre"] * w2) / (w1 + w2)
            # grow gently, not to full cover: chained merges of one eye's pieces
            # (white + pupil + lash) otherwise inflate past the size cap and the
            # eye rejects itself. The blink shader smoothsteps past the radius
            # anyway, so a slight under-cover is harmless.
            d = float(np.linalg.norm(home["centre"] - cl["centre"]))
            home["radius"] = max(home["radius"], 0.6 * d + 0.75 * cl["radius"])
            home["mass"] = w1 + w2

    # 3) eye plausibility: small, elevated, not the global centre
    eyes = []
    for cl in merged:
        if cl["radius"] > 0.22 or cl["radius"] < 0.015:
            continue
        if cl["centre"][1] < np.quantile(P[:, 1], 0.35):     # lower body: no
            continue
        eyes.append(cl)
    eyes.sort(key=lambda e: -e["mass"])

    # prefer a left/right PAIR (mirrored across the x or z axis) — searched over
    # ALL plausible clusters BEFORE truncation: real eyes are often out-massed
    # by decorative contrast (rims, stripes), but decorations don't come in
    # mirrored pairs of similar size; eyes do.
    best = None
    y_upper = float(np.quantile(P[:, 1], 0.5))     # eyes live in the upper body
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            a, b = eyes[i]["centre"], eyes[j]["centre"]
            if (a[1] + b[1]) * 0.5 < y_upper:      # a low pair is decoration
                continue
            similar = (min(eyes[i]["radius"], eyes[j]["radius"])
                       / max(eyes[i]["radius"], eyes[j]["radius"])) > 0.5
            for ax in (0, 2):
                mirror = (abs(a[ax] + b[ax]) < 0.12 and abs(a[1] - b[1]) < 0.10
                          and abs(a[ax]) > 0.05 and similar)
                if mirror:
                    score = eyes[i]["mass"] + eyes[j]["mass"]
                    if best is None or score > best[0]:
                        best = (score, i, j)
    if best:
        eyes = [eyes[best[1]], eyes[best[2]]]
    else:
        eyes = eyes[:max_eyes]

    return [{"center": (np.asarray(e["centre"]) * s + c).tolist(),
             "radius": float(e["radius"] * s),
             "color": [round(float(x), 3) for x in e["color"]]} for e in eyes]
