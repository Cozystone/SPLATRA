# -*- coding: utf-8 -*-
"""아토 (Ato) — ATANOR's machine character. A retro cartoon clock, reinterpreted.

Inspired by the wit of a certain animated clock, rebuilt from scratch as an
ORIGINAL: an ATANOR-orange clock face with big blinking eyes, thin gloved
arms, stubby boots, and a spark antenna. Procedural on purpose — the geometry
is exactly what the living pipeline consumes:

  * big white eyes + dark pupils  -> parts.find_eyes -> GPU blink channel
  * two-segment arms and legs     -> rig predictor chains -> FK gestures
  * one body, known proportions   -> /v1/embody avatar; the LIVE self's
                                     hormones (dopamine, cortisol...) drive
                                     amp/tempo/droop of every gesture

Everything is surface-sampled Gaussians (pos, col, scale, quat, opa) in the
same [-1,1]-ish frame the viewer expects. Deterministic per seed."""
from __future__ import annotations

import numpy as np

ORANGE = (0.83, 0.32, 0.12)          # ATANOR accent, #d2521f
DARK = (0.10, 0.07, 0.05)
CREAM = (0.96, 0.93, 0.86)
BOOT = (0.85, 0.38, 0.15)


def _disk(rng, centre, radius, depth, n, color, axis="z"):
    """Surface-sample a rounded disk (a squashed sphere)."""
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    p[:, 0] *= radius
    p[:, 1] *= radius
    p[:, 2] *= depth
    if axis == "x":
        p = p[:, [2, 1, 0]]
    return centre + p, np.tile(color, (n, 1))


def _tube(rng, a, b, r, n, color, taper=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = rng.uniform(0, 1, n)
    c = a[None] + t[:, None] * (b - a)[None]
    d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    off = rng.normal(size=(n, 3))
    off -= (off @ d)[:, None] * d[None]
    off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
    rr = r * (1 - (1 - taper) * t)
    return c + off * rr[:, None], np.tile(color, (n, 1))


def _ball(rng, centre, r, n, color, squash=(1, 1, 1)):
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    p *= r
    p *= np.asarray(squash, float)
    return centre + p, np.tile(color, (n, 1))


def build_ato(n_points: int = 70000, seed: int = 7):
    """Build the Ato field -> (pos, col, scale, quat, opa) float32 arrays."""
    rng = np.random.default_rng(seed)
    P, C, S = [], [], []          # points, colors, per-part gaussian size

    def add(p, c, size):
        P.append(p); C.append(c); S.append(np.full(len(p), size))

    u = n_points // 100           # budget unit (percent)

    # ── clock body: fat orange disk facing +z. TWO layers — surface shell +
    # inner fill — so the body reads SOLID, not gauzy ─────────────────────────
    body_r = 1.0
    add(*_disk(rng, [0, 0.35, 0], body_r, 0.22, 30 * u, ORANGE), 0.036)
    inner = rng.normal(size=(12 * u, 3))
    inner /= np.linalg.norm(inner, axis=1, keepdims=True) + 1e-9
    inner *= rng.uniform(0, 1, (12 * u, 1)) ** 0.5     # bias toward the surface
    inner[:, 0] *= body_r * 0.94
    inner[:, 1] *= body_r * 0.94
    inner[:, 2] *= 0.18
    add(inner + [0, 0.35, 0], np.tile(ORANGE, (12 * u, 1)), 0.040)
    # face rim — WARM brown, deliberately low-contrast against the orange so
    # the rim arc never out-scores the pupils in the eye detector
    ang = rng.uniform(0, 2 * np.pi, 6 * u)
    rr = body_r * rng.uniform(0.965, 1.0, 6 * u)
    rim = np.stack([rr * np.cos(ang), 0.35 + rr * np.sin(ang) * 0.98,
                    rng.uniform(0.05, 0.20, 6 * u)], axis=1)
    add(rim, np.tile((0.55, 0.24, 0.10), (6 * u, 1)), 0.018)

    # ── 12 tick marks on the face ────────────────────────────────────────────
    for k in range(12):
        a = k * np.pi / 6
        r0, r1 = (0.80, 0.94) if k % 3 == 0 else (0.86, 0.94)
        t = rng.uniform(0, 1, u)
        pts = np.stack([np.sin(a) * (r0 + (r1 - r0) * t),
                        0.35 + np.cos(a) * (r0 + (r1 - r0) * t),
                        np.full(u, 0.225)], axis=1)
        pts += rng.normal(scale=0.012, size=pts.shape)
        add(pts, np.tile(DARK, (u, 1)), 0.016)

    # ── clock hands: small, low on the face (10:10-ish V, kept clear of the
    # eyes so the face reads friendly, not furrowed) ─────────────────────────
    pin = np.array([0, 0.18, 0.23])
    for a, ln in ((-2.2, 0.30), (2.35, 0.24)):       # pointing down-left/right
        d = np.array([np.sin(a), np.cos(a), 0.0])
        add(*_tube(rng, pin, pin + d * ln, 0.030, u, DARK, taper=0.5), 0.015)
    add(*_ball(rng, pin + [0, 0, 0.01], 0.05, u, DARK), 0.015)   # centre pin

    # ── eyes: big cream ovals + round pupils + CATCHLIGHT (the life dot) ─────
    for sx in (-1, 1):
        ec = np.array([0.34 * sx, 0.62, 0.24])
        add(*_ball(rng, ec, 0.21, 5 * u, CREAM, squash=(0.78, 1.0, 0.16)), 0.024)
        pc = ec + [0.015 * sx, -0.02, 0.06]
        add(*_ball(rng, pc, 0.115, 3 * u, DARK, squash=(0.85, 1.0, 0.28)), 0.020)
        add(*_ball(rng, pc + [-0.035 * sx, 0.045, 0.05], 0.032, u, (1.0, 1.0, 1.0)),
            0.020)                                               # catchlight
        for lk in (-1, 0, 1):                                     # lashes
            base = ec + [0.10 * sx * (0.6 + 0.3 * abs(lk)), 0.19, 0.0]
            tip = base + [0.07 * sx * lk * 0.5 + 0.04 * sx, 0.10, 0.0]
            add(*_tube(rng, base, tip, 0.014, u // 3 or 1, DARK), 0.012)

    # ── smile: a ∪ arc — LOWEST in the middle (the sign the frown version
    # got wrong), sitting under the hands ────────────────────────────────────
    a = rng.uniform(-0.62, 0.62, 3 * u)
    smile = np.stack([a * 0.55, -0.28 + 0.30 * (a ** 2),
                      np.full(3 * u, 0.235)], axis=1)
    smile += rng.normal(scale=0.012, size=smile.shape)
    add(smile, np.tile(DARK, (3 * u, 1)), 0.019)

    # ── arms: shoulder -> elbow -> glove (two segments = an FK chain) ───────
    for sx in (-1, 1):
        sh = np.array([0.92 * sx, 0.45, 0.0])
        el = np.array([1.28 * sx, 0.10, 0.05])
        ha = np.array([1.46 * sx, -0.28, 0.10])
        add(*_tube(rng, sh, el, 0.055, 3 * u, ORANGE, taper=0.85), 0.020)
        add(*_tube(rng, el, ha, 0.048, 3 * u, ORANGE, taper=0.85), 0.020)
        add(*_ball(rng, ha, 0.16, 3 * u, CREAM, squash=(1, 0.8, 0.55)), 0.022)
        for f in (-1, 0, 1):                                      # mitten fingers
            tip = ha + [0.13 * sx, -0.10 + 0.07 * f, 0.02 * f]
            add(*_tube(rng, ha, tip, 0.045, u, CREAM), 0.018)

    # ── legs + boots ─────────────────────────────────────────────────────────
    for sx in (-1, 1):
        hip = np.array([0.30 * sx, -0.55, 0.0])
        kn = np.array([0.33 * sx, -0.95, 0.03])
        ft = np.array([0.34 * sx, -1.28, 0.05])
        add(*_tube(rng, hip, kn, 0.040, 2 * u, DARK), 0.016)
        add(*_tube(rng, kn, ft, 0.036, 2 * u, DARK), 0.016)
        add(*_ball(rng, ft + [0.06 * sx, -0.05, 0.10], 0.17, 3 * u, BOOT,
                   squash=(0.9, 0.55, 1.25)), 0.022)

    # ── spark antenna (the ATANOR-ish reinterpretation of the hair swoosh).
    # Kept LOW-CONTRAST against the body on purpose: the eye detector keys on
    # local color contrast, and a bright tip up top would out-score the pupils.
    t = rng.uniform(0, 1, 2 * u)
    curl = np.stack([0.10 * np.sin(t * 4.5), 1.38 + 0.30 * t,
                     0.05 * np.cos(t * 4.5)], axis=1)
    curl += rng.normal(scale=0.015, size=curl.shape)
    add(curl, np.tile(ORANGE, (2 * u, 1)), 0.018)
    add(*_ball(rng, [0.10 * np.sin(4.5), 1.71, 0.05 * np.cos(4.5)], 0.07,
               u, (0.90, 0.42, 0.18)), 0.022)

    pos = np.vstack(P).astype(np.float32)
    col = np.vstack(C).astype(np.float32)
    sizes = np.concatenate(S).astype(np.float32)
    n = len(pos)
    scale = np.repeat(sizes[:, None], 3, axis=1)
    quat = np.zeros((n, 4), np.float32); quat[:, 0] = 1.0
    opa = np.full(n, 0.985, np.float32)              # solid, not gauzy
    return pos, col, scale.astype(np.float32), quat, opa
