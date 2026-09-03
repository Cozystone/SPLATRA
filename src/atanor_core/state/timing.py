"""Learned timing — the progress bar gets more accurate the more it is used.

A step-counting bar is honest about *what* is happening but useless about *when*
it will end: generating a car body and generating a wheel are both "one part",
yet one takes twenty times longer. And the costs here are not guessable up front —
they depend on this machine, on whether the models are warm, on how much VRAM
Docker happens to be holding.

So measure instead of guess. Every timed operation is recorded under a key that
describes the work ("part:full:d3"), and the estimate for that key is an
exponential moving average of what it has actually cost here. The averages persist
to disk, so the estimate keeps improving across restarts rather than starting from
nothing every session.

The EMA is deliberately quick to adapt (alpha 0.35): machine state changes — a
model warms up, Docker frees the GPU — and a bar that keeps quoting last week's
cold-start time is worse than one that tracks the last few runs.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

_ALPHA = 0.28
_LOCK = threading.Lock()

_PATH = os.environ.get(
    "SPLATRA_TIMING_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "out", "timing.json"))

_store: Optional[Dict[str, Dict[str, float]]] = None


def _load() -> Dict[str, Dict[str, float]]:
    global _store
    if _store is None:
        try:
            with open(_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _store = data if isinstance(data, dict) else {}
        except Exception:
            _store = {}
    return _store


def _save() -> None:
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(_store or {}, f, indent=1)
    except Exception:
        pass


def record(key: str, seconds: float) -> None:
    """Fold one observed duration into the running estimate for ``key``."""
    if seconds <= 0 or seconds > 3600:
        return
    with _LOCK:
        st = _load()
        cur = st.get(key)
        if not cur:
            st[key] = {"ema": float(seconds), "n": 1.0,
                       "min": float(seconds), "max": float(seconds)}
        else:
            ema = float(cur.get("ema", seconds))
            # One slow run (a cold model, a moment of GPU contention) must not
            # define the estimate: fold in a clamped observation so a spike nudges
            # the average instead of jumping it, while a genuine shift still gets
            # there over a few runs.
            obs = max(0.4 * ema, min(2.5 * ema, float(seconds)))
            cur["ema"] = (1 - _ALPHA) * ema + _ALPHA * obs
            cur["n"] = float(cur.get("n", 0)) + 1.0
            cur["min"] = min(float(cur.get("min", seconds)), float(seconds))
            cur["max"] = max(float(cur.get("max", seconds)), float(seconds))
        _save()


def estimate(key: str, default: float) -> float:
    """Best guess for ``key`` in seconds — the learned average, or ``default``."""
    st = _load()
    cur = st.get(key)
    if not cur:
        return float(default)
    return float(cur.get("ema", default))


def samples(key: str) -> int:
    st = _load()
    return int((st.get(key) or {}).get("n", 0))


def confidence(key: str) -> float:
    """0..1 — how much the estimate has been backed by real observations."""
    n = samples(key)
    return min(1.0, n / 5.0)


def snapshot() -> Dict[str, Any]:
    return {k: dict(v) for k, v in _load().items()}
