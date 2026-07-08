# -*- coding: utf-8 -*-
"""Mood drive — ATANOR's inner state becomes body motion.

The self's digital hormones (cortisol / dopamine / noradrenaline) and vitals
(energy / valence / curiosity) already modulate its inner life; this module is
the bridge that lets the SAME state drive the rigged particle body, so emotion
becomes movement:

  dopamine, valence, energy  -> larger, livelier swings (amp, tempo up)
  noradrenaline              -> sharp, quick motion (tempo up)
  cortisol                   -> smaller guarded motion + fine tremor (jitter)
                                and a drooped posture bias
  low energy                 -> everything slows

Deterministic pure functions: state -> motion params -> per-joint drives at
time t. No randomness, no I/O — the caller supplies the live state snapshot
(e.g. from /api/selfhood/live) and a clock."""
from __future__ import annotations

import math

def _clip(x, lo, hi):
    return max(lo, min(hi, float(x)))


def motion_params(state: dict) -> dict:
    """Map a selfhood state snapshot to bounded motion parameters."""
    h = (state.get("homeostasis") or {}).get("hormones") or state.get("hormones") or {}
    v = state.get("vitals") or {}
    cort = float(h.get("cortisol", 0.0))
    dopa = float(h.get("dopamine", 0.0))
    nora = float(h.get("noradrenaline", 0.0))
    energy = float(v.get("energy", 0.6))
    valence = float(v.get("valence", 0.4))
    curiosity = float(v.get("curiosity", 0.5))

    amp = _clip(0.15 + 0.55 * dopa + 0.30 * valence + 0.20 * energy - 0.40 * cort,
                0.04, 0.9)
    tempo = _clip(0.6 + 1.2 * nora + 0.6 * dopa + 0.4 * curiosity
                  - 0.4 * (1.0 - energy), 0.2, 2.5)
    jitter = _clip(0.03 + 0.45 * cort, 0.0, 0.4)
    droop = _clip(0.45 * cort + 0.35 * max(0.0, 0.45 - valence), 0.0, 0.5)
    return {"amp": amp, "tempo": tempo, "jitter": jitter, "droop": droop}


def chain_drives(params: dict, joints: list[int], t: float) -> dict[int, float]:
    """Per-joint drive angles at time t: a phase-staggered sway scaled by mood,
    plus tension tremor, minus postural droop. Bounded by construction."""
    amp, tempo = params["amp"], params["tempo"]
    jitter, droop = params["jitter"], params["droop"]
    out = {}
    for k, j in enumerate(joints):
        sway = amp * math.sin(tempo * t + k * 1.7)
        tremor = jitter * 0.3 * math.sin(7.3 * t + k * 2.9)
        out[j] = sway + tremor - droop * 0.35
    return out
