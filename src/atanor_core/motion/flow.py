# -*- coding: utf-8 -*-
"""Microbot flow — particles drift on a living flow field, the shape holds.

The Big-Hero-6 microbot look: instead of a static shell (or particles popping
fresh on every generation), every particle CONTINUOUSLY flows on a smooth
curl-like field, wandering a bounded distance from its home so the object stays
recognisable while shimmering as if alive.

Design constraints the owner set:
  * runs on a LOW-END laptop GPU — the field is STATELESS (an offset from home,
    no integration, no per-frame history) and costs ~9 sin/cos per particle, so
    the identical formula drops straight into a vertex shader (uAnimMode==10) and
    animates 200k particles on integrated graphics.
  * no LLM, no training — hand-authored motion that mimics how an imagined thing
    quietly breathes and swirls.

Pure numpy, deterministic per (home, t). The viewer shader mirrors this exactly.
"""
from __future__ import annotations

import numpy as np

# phase drift per axis — coprime-ish so the flow never obviously loops
_DRIFT = np.array([0.60, 0.50, 0.70], dtype=np.float32)


def flow_offset(home: np.ndarray, t: float, amp: float = 0.06,
                freq: float = 2.3, swirl: float = 0.5) -> np.ndarray:
    """A bounded, divergence-low, shape-holding displacement field at time t.
    Two scales: fine curl (local shimmer) + a slow large swirl (the whole cloud
    breathes/orbits gently)."""
    h = np.asarray(home, dtype=np.float32)
    q = h * freq + (t * _DRIFT)[None, :]
    # fine curl-like flow (each component driven by the OTHER two axes -> swirly)
    flow = np.stack([
        np.sin(q[:, 1]) * np.cos(q[:, 2]),
        np.sin(q[:, 2]) * np.cos(q[:, 0]),
        np.sin(q[:, 0]) * np.cos(q[:, 1]),
    ], axis=1)
    # slow large-scale swirl so the cloud feels alive as a whole, not just jitter
    flow[:, 0] += swirl * np.sin(t * 0.30 + h[:, 1] * 1.5)
    flow[:, 1] += swirl * np.cos(t * 0.27 + h[:, 2] * 1.5)
    flow[:, 2] += swirl * np.sin(t * 0.33 + h[:, 0] * 1.5)
    return (flow * amp).astype(np.float32)


def flow_positions(home: np.ndarray, t: float, **kw) -> np.ndarray:
    """Home positions -> live flowing positions at time t (shape preserved)."""
    return (np.asarray(home, dtype=np.float32) + flow_offset(home, t, **kw)).astype(np.float32)


def gather_flow(home: np.ndarray, t: float, u: float, scatter_radius: float = 1.4,
                seed: int = 0, **flow_kw) -> np.ndarray:
    """A NEW object 'flows in' instead of popping: at u=0 particles are dispersed
    (a loose swarm), at u=1 they have settled onto `home` — and the microbot flow
    shimmer plays throughout, so assembly looks like a swarm converging, never a
    hard cut. u in [0,1]."""
    h = np.asarray(home, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # each particle's dispersed start: pushed out along its own direction + jitter
    dirs = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-6)
    disp = h + dirs * (0.5 + scatter_radius) + (rng.random(h.shape) - 0.5) * scatter_radius
    e = float(np.clip(u, 0.0, 1.0))
    e = e * e * (3 - 2 * e)  # smoothstep
    base = disp * (1 - e) + h * e
    # shimmer strength eases in as it settles (loose swarm shimmers more)
    return (base + flow_offset(base, t, **flow_kw) * (0.5 + 0.5 * (1 - e))).astype(np.float32)
