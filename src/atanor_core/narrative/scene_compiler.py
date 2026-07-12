# -*- coding: utf-8 -*-
"""Imagination compiler — a thought (concepts + relations) → a SPLATRA scene spec.

The owner's vision (2026-07-11): "SPLATRA should render everything the AI thinks and wants to
explain — its inner imagination — as a real-time animated Gaussian splatting." This is the first
slice: the bridge that turns the *structure of a thought* into a scene SPLATRA can render.

It is deterministic and No-LLM. It reads only the GRAPH the engine already activated when it
reasoned — the concepts in play and the relations among them — and lays them out as particle
objects with a timed motion script. Nothing here is knowledge; it is PRESENTATION logic (how to
*show* a relation), the visual counterpart of the Korean surface tables. The knowledge stays in
the graph; this only decides where a concept floats and how a relation moves.

Motion is chosen by the SEMANTIC KIND of the relation, not a per-predicate table:
  * structural  (is_a / part_of / 포함)          → nest / orbit   (one belongs to another)
  * transform   (되다 / 변하다 / 끓다 / 녹다)      → morph / rise   (a thing becomes another)
  * causal      (때문 / 유발 / 이끌다)             → flow           (particles stream cause→effect)
  * link        (everything else)                 → tether         (a soft connective pulse)
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

from .cloud import shape_spec as _shape_spec

# Relation-kind cues (surface hints, not knowledge). Extend freely — a miss just falls to "link".
_STRUCTURAL = ("is_a", "part_of", "instance_of", "속하", "포함", "구성", "종류")
_TRANSFORM = ("되다", "변하", "끓", "녹", "얼", "타오르", "자라", "피", "전환", "become", "turn_into")
_CAUSAL = ("때문", "유발", "이끌", "일으키", "cause", "leads_to", "원인", "결과")

_MOTION_BY_KIND = {"structural": "nest", "transform": "morph", "causal": "flow", "link": "tether"}


def _kind_of(predicate: str) -> str:
    p = (predicate or "").lower()
    for cue in _STRUCTURAL:
        if cue in p:
            return "structural"
    for cue in _TRANSFORM:
        if cue in p:
            return "transform"
    for cue in _CAUSAL:
        if cue in p:
            return "causal"
    return "link"


# PHYSICS MOTION (owner 2026-07-12, the "Jarvis directive"): a relation that names a real motion
# drives a real motion — an apple that FALLS accelerates down, a moon that ORBITS circles its
# planet. The physics is chosen by the relation's SEMANTICS (presentation logic, like the shape
# archetypes) — never a per-concept narrative table. When the graph names no motion, the scene
# honestly falls back to the structural/tether kinds; the cinema emerges only where it is grounded.
_FALL = ("떨어", "낙하", "추락", "떨군", "낙과", "fall", "drop", "descend", "gravity_pull")
_ORBIT = ("궤도", "공전", "회전", "돈다", "돌다", "도는", "선회", "orbit", "revolve", "circle", "spin")
_ATTRACT = ("끌", "당기", "중력", "인력", "이끌", "attract", "pull", "gravitate")
_EMIT = ("방출", "발산", "방사", "비추", "퍼지", "퍼진", "번지", "emit", "radiate", "shine", "spread")
_RISE = ("오르", "상승", "증발", "떠오르", "치솟", "rise", "ascend", "evaporate")
_MOTION_CUES = ((_FALL, "fall"), (_ORBIT, "orbit"), (_ATTRACT, "attract"),
                (_EMIT, "emit"), (_RISE, "rise"))


def _physics_of(predicate: str) -> str | None:
    """A physics motion name if the relation predicate names one, else None (fall back to a kind)."""
    p = (predicate or "").lower()
    for cues, name in _MOTION_CUES:
        if any(c in p for c in cues):
            return name
    return None


def _motion_entry(predicate: str, from_id: str, to_id: str, t: float) -> tuple[dict[str, Any], str]:
    """Build one timeline motion entry + its link kind. Physics motions carry the params a renderer
    needs to DRIVE them (accel for a fall, a center for an orbit, a target for attraction)."""
    phys = _physics_of(predicate)
    if phys == "fall":
        return {"t": t, "action": "fall", "target": to_id, "accel": 1.0}, "motion"
    if phys == "orbit":
        return {"t": t, "action": "orbit", "target": to_id, "around": from_id,
                "radius": 0.55, "period": 3.0}, "motion"
    if phys == "attract":
        return {"t": t, "action": "attract", "target": to_id, "toward": from_id, "strength": 1.0}, "motion"
    if phys == "emit":
        return {"t": t, "action": "emit", "source": from_id, "target": to_id}, "motion"
    if phys == "rise":
        return {"t": t, "action": "rise", "target": to_id, "accel": -0.6}, "motion"
    kind = _kind_of(predicate)
    return {"t": t, "from": from_id, "to": to_id, "action": _MOTION_BY_KIND[kind]}, kind


def _hue(seed: str) -> float:
    """A stable hue in [0,360) from a concept id — same concept always the same color."""
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6], 16)
    return (h % 360)


def _archetype(concept: dict[str, Any]) -> str:
    """Coarse particle shape from the concept's own type + description signals (graph-derived, not
    a name table). A liquid/gas reads as a blob, a process as a swirl, a place as a field, else a
    sphere. Reads `desc` too so a curated concept with no explicit type still gets a fitting shape."""
    t = " ".join(str(concept.get(k, "")) for k in ("type", "kind", "category", "desc")).lower()
    label = str(concept.get("label", ""))
    if any(c in t for c in ("liquid", "fluid", "gas", "water", "액체", "기체", "물")) or label in ("물", "불", "연기", "구름", "비", "바다", "강"):
        return "blob"
    if any(c in t for c in ("process", "event", "action", "행동", "과정", "현상", "운동")):
        return "swirl"
    if any(c in t for c in ("place", "location", "지역", "장소", "도시", "나라", "지방")):
        return "field"
    return "sphere"


def compile_scene(
    concepts: Iterable[dict[str, Any]],
    relations: Iterable[dict[str, Any]] | None = None,
    *,
    subject_id: str | None = None,
    duration: float = 6.0,
) -> dict[str, Any]:
    """Compile a thought into a SPLATRA scene spec.

    concepts:  [{"id", "label", optional "type"/"kind", optional "weight"}]
    relations: [{"from", "to", "predicate"}]
    subject_id: the concept to center (defaults to the first / heaviest).
    Returns {objects, links, motion, duration, meta} — coordinates normalized to a unit sphere.
    """
    cs = [dict(c) for c in concepts if c.get("id")]
    rels = [dict(r) for r in (relations or []) if r.get("from") and r.get("to")]
    if not cs:
        return {"objects": [], "links": [], "motion": [], "duration": 0.0,
                "meta": {"empty": True}}

    # center the subject (explicit, else the heaviest, else the first)
    if subject_id is None:
        subject_id = max(cs, key=lambda c: float(c.get("weight", 0.0) or 0.0)).get("id") or cs[0]["id"]
    others = [c for c in cs if c["id"] != subject_id]

    objects: list[dict[str, Any]] = []
    positions: dict[str, list[float]] = {}
    # subject at the origin, everything else on a ring around it (stable, legible layout)
    positions[subject_id] = [0.0, 0.0, 0.0]
    n = max(1, len(others))
    for i, c in enumerate(others):
        ang = 2.0 * math.pi * i / n
        positions[c["id"]] = [round(math.cos(ang), 4), round(0.35 * math.sin(2 * ang), 4),
                              round(math.sin(ang), 4)]
    for c in cs:
        cid = c["id"]
        is_subj = cid == subject_id
        objects.append({
            "id": cid, "label": c.get("label", cid), "archetype": _archetype(c),
            "pos": positions[cid], "scale": round(1.0 if is_subj else 0.6, 3),
            "hue": _hue(cid), "role": "subject" if is_subj else "satellite",
            "shape": _shape_spec(c),          # structural silhouette (form + seed) for detailed particles
        })

    links: list[dict[str, Any]] = []
    motion: list[dict[str, Any]] = []
    # objects fade in first (subject leads), staggered
    t = 0.0
    step = min(0.5, (duration * 0.4) / max(1, len(objects)))
    for o in sorted(objects, key=lambda o: 0 if o["role"] == "subject" else 1):
        motion.append({"t": round(t, 3), "target": o["id"], "action": "appear"})
        t += step
    # then relations animate, each carrying its kind's motion
    rel_start = round(t + 0.2, 3)
    span = max(0.4, duration - rel_start)
    for j, r in enumerate(rels):
        at = round(rel_start + span * (j / max(1, len(rels))), 3)
        entry, kind = _motion_entry(r.get("predicate", ""), r["from"], r["to"], at)
        links.append({"from": r["from"], "to": r["to"], "predicate": r.get("predicate", ""),
                      "kind": kind})
        motion.append(entry)

    return {
        "objects": objects, "links": links, "motion": motion,
        "duration": round(duration, 3),
        "meta": {"subject": subject_id, "n_objects": len(objects), "n_relations": len(rels),
                 "empty": False},
    }
