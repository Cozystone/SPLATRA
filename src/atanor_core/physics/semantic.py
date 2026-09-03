"""Semantic material understanding — the "what is this?" layer.

Skeletons only describe hinged bodies. A dress, a flag, a candle flame, a pile of
sand and a jellyfish have no joints, yet they all move. So instead of predicting
bones we ask what the object is MADE OF and what DRIVES it, and hand those
material fields to a constraint solver (client side) that produces the motion.

This mirrors the current literature — GaussianProperty / PUGS / NeRF2Physics feed
a vision-language model's material estimate into a physical simulation rather than
hand-authoring animation. We use the generation prompt (which we already have, and
which is exactly the semantic label those papers work to recover) plus the local
LLM, with a keyword fallback so it never hard-fails.

Output (normalized model space, y up, roughly [-1, 1])::

    {"kind": "creature", "has_eyes": true, "anchor": "feet",
     "regions": [{"name": "skirt", "y": [-1.0, -0.1], "radial": [0.3, 1.0],
                  "material": "cloth", "stiffness": 0.08, "damping": 0.06,
                  "drive": "gravity+wind", "gain": 1.0}, ...]}

``material`` decides which constraints the solver builds, ``drive`` decides which
forces act. Nothing here encodes a walk cycle — the motion is emergent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

MATERIALS = ("rigid", "soft", "cloth", "strand", "granular", "fluid")
DRIVES = ("none", "gravity", "wind", "breath", "jiggle", "flow", "gravity+wind")

_SYS = (
    "You are a physics-grounding model. Given an object description, describe what "
    "it is MADE OF so a simulator can move it. Output ONLY JSON: "
    '{"kind":"<creature|plant|cloth|vehicle|food|furniture|liquid|granular|rigid|toy>",'
    '"has_eyes":<true|false>,"anchor":"<feet|base|top|none>","regions":[...]}. '
    'Each region: {"name":"<part>","y":[<lo>,<hi>],"radial":[<lo>,<hi>],'
    '"material":"<rigid|soft|cloth|strand|granular|fluid>","stiffness":<0.01..1.0>,'
    '"damping":<0.01..0.3>,"drive":"<none|gravity|wind|breath|jiggle|flow|gravity+wind>",'
    '"gain":<0..2>}. '
    "y is the vertical band in the object's own normalized box (-1 bottom, +1 top); "
    "radial is distance from the vertical axis (0 core, 1 outer shell). Use 2-5 "
    "regions that genuinely differ. Examples: a dress is a stiff torso plus a cloth "
    "skirt driven by gravity+wind; a tree is a rigid trunk with cloth-like foliage on "
    "wind; a candle is rigid wax with a fluid flame; a person is a soft body with "
    "breath and strand hair; a rock is one rigid region with drive none. Set has_eyes "
    "only for things that really have eyes. anchor is what stays planted (feet for a "
    "standing creature, base for furniture, none for something floating). Object: "
)

_KIND_RULES = (
    ("creature", ("person", "man", "woman", "girl", "boy", "cat", "dog", "bird",
                  "dragon", "robot", "pikachu", "monster", "animal", "horse",
                  "fish", "character", "human", "duck", "rabbit", "bear")),
    ("plant", ("tree", "flower", "plant", "grass", "leaf", "bush", "branch")),
    ("cloth", ("flag", "dress", "curtain", "cape", "scarf", "shirt", "cloth",
               "banner", "towel", "skirt")),
    ("liquid", ("water", "wave", "liquid", "juice", "milk", "fountain", "splash")),
    ("granular", ("sand", "dust", "powder", "gravel", "snow")),
    ("vehicle", ("car", "truck", "plane", "airplane", "ship", "boat", "rocket", "bike")),
    ("food", ("apple", "banana", "cake", "bread", "fruit", "pizza", "burger",
              "donut", "egg", "strawberry", "lemon", "peach")),
    ("furniture", ("chair", "table", "desk", "lamp", "sofa", "shelf", "bed", "teapot")),
)


def _heuristic(prompt: str) -> Dict[str, Any]:
    """No-LLM fallback: map the prompt to a coarse but honest material layout."""
    p = (prompt or "").lower()
    kind = "rigid"
    for name, words in _KIND_RULES:
        if any(w in p for w in words):
            kind = name
            break

    if kind == "creature":
        regions = [
            {"name": "body", "y": [-1, 1], "radial": [0, 1], "material": "soft",
             "stiffness": 0.55, "damping": 0.12, "drive": "breath", "gain": 0.35},
            {"name": "limbs", "y": [-1, 0.35], "radial": [0.45, 1], "material": "soft",
             "stiffness": 0.28, "damping": 0.09, "drive": "none", "gain": 0.0},
        ]
        spec = {"kind": kind, "has_eyes": True, "anchor": "feet", "regions": regions}
    elif kind == "plant":
        spec = {"kind": kind, "has_eyes": False, "anchor": "base", "regions": [
            {"name": "trunk", "y": [-1, -0.2], "radial": [0, 0.45], "material": "rigid",
             "stiffness": 0.95, "damping": 0.2, "drive": "none", "gain": 0.2},
            {"name": "foliage", "y": [-0.2, 1], "radial": [0, 1], "material": "cloth",
             "stiffness": 0.12, "damping": 0.07, "drive": "wind", "gain": 1.2}]}
    elif kind == "cloth":
        spec = {"kind": kind, "has_eyes": False, "anchor": "top", "regions": [
            {"name": "fabric", "y": [-1, 1], "radial": [0, 1], "material": "cloth",
             "stiffness": 0.06, "damping": 0.05, "drive": "gravity+wind", "gain": 1.4}]}
    elif kind == "liquid":
        spec = {"kind": kind, "has_eyes": False, "anchor": "none", "regions": [
            {"name": "body", "y": [-1, 1], "radial": [0, 1], "material": "fluid",
             "stiffness": 0.03, "damping": 0.08, "drive": "flow", "gain": 1.2}]}
    elif kind == "granular":
        spec = {"kind": kind, "has_eyes": False, "anchor": "base", "regions": [
            {"name": "grains", "y": [-1, 1], "radial": [0, 1], "material": "granular",
             "stiffness": 0.04, "damping": 0.2, "drive": "gravity", "gain": 1.0}]}
    else:
        soft = kind in ("food", "toy")
        spec = {"kind": kind, "has_eyes": False,
                "anchor": "base" if kind in ("furniture", "vehicle") else "none",
                "regions": [
                    {"name": "body", "y": [-1, 1], "radial": [0, 1],
                     "material": "soft" if soft else "rigid",
                     "stiffness": 0.45 if soft else 0.9,
                     "damping": 0.12,
                     "drive": "none",          # a resting solid does not squirm
                     "gain": 0.0}]}
    spec["engine"] = "heuristic"
    return spec


def _sanitize(spec: Any) -> Optional[Dict[str, Any]]:
    """Clamp an LLM answer into the contract; return None if unusable."""
    if not isinstance(spec, dict):
        return None

    def pair(v, lo, hi, dflt):
        try:
            a, b = float(v[0]), float(v[1])
        except Exception:
            return dflt
        a, b = max(lo, min(hi, a)), max(lo, min(hi, b))
        return [min(a, b), max(a, b)]

    def num(v, lo, hi, dflt):
        try:
            return max(lo, min(hi, float(v)))
        except Exception:
            return dflt

    regions: List[Dict[str, Any]] = []
    for r in (spec.get("regions") or [])[:6]:
        if not isinstance(r, dict):
            continue
        mat = str(r.get("material", "soft")).lower()
        drv = str(r.get("drive", "none")).lower()
        regions.append({
            "name": str(r.get("name", "part"))[:24],
            "y": pair(r.get("y"), -1.0, 1.0, [-1.0, 1.0]),
            "radial": pair(r.get("radial"), 0.0, 1.0, [0.0, 1.0]),
            "material": mat if mat in MATERIALS else "soft",
            "stiffness": num(r.get("stiffness"), 0.01, 1.0, 0.5),
            "damping": num(r.get("damping"), 0.01, 0.3, 0.1),
            "drive": drv if drv in DRIVES else "none",
            "gain": num(r.get("gain"), 0.0, 2.0, 1.0),
        })
    if not regions:
        return None
    anchor = str(spec.get("anchor", "none")).lower()
    return {"kind": str(spec.get("kind", "rigid"))[:24],
            "has_eyes": bool(spec.get("has_eyes", False)),
            "anchor": anchor if anchor in ("feet", "base", "top", "none") else "none",
            "regions": regions}


def material_spec(prompt: str, ollama_model: Optional[str] = None,
                  host: str = "http://localhost:11434") -> Dict[str, Any]:
    """prompt -> physical material layout. LLM when reachable, else keyword rules."""
    if ollama_model:
        import urllib.request

        payload = {"model": ollama_model, "format": "json", "stream": False,
                   "messages": [{"role": "system", "content": _SYS + (prompt or "")},
                                {"role": "user", "content": prompt or "object"}]}
        try:
            req = urllib.request.Request(
                host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (data.get("message", {}) or {}).get("content", "") or ""
            try:
                parsed = json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, re.S)
                parsed = json.loads(m.group(0)) if m else None
            clean = _sanitize(parsed)
            if clean:
                clean["engine"] = "ollama:" + ollama_model
                return clean
        except Exception:
            pass
    return _heuristic(prompt)
