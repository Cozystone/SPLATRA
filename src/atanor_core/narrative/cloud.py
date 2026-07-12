# -*- coding: utf-8 -*-
"""SPLATRA point-cloud detail — abstract disks → recognizable object silhouettes (owner's 2026-07-12
"진짜 사물 디테일" order). A concept becomes a STRUCTURAL FORM (a tree branches, a bottle is a vessel,
a person is humanoid) instead of a uniform blob, so "나무" materializes tree-shaped behind the glass.

Two truths kept honest:
  * this is the PIPELINE + a structural bootstrap — the forms are procedural silhouettes keyed by the
    graph's TYPE (a presentation vocabulary, like the blob/swirl archetypes, not a per-concept table);
  * a real learned 3DGS asset drops into the SAME pipeline: `decimate_importance` reduces its Gaussians
    to the output budget, and the renderer draws that instead of the procedural form. Not faked as trained.

RESOURCE SAFETY (owner's design question) lives at the engine's mouth — never stream raw Gaussians to a
2D canvas. `lod_budget` bounds points-per-object by depth × salience under a hard per-frame ceiling; the
renderer's frame-time governor scales it down further under load. The wire carries a tiny SHAPE SPEC
(form + seed + params), not points — the bounded cloud is generated client-side where the frame budget
is measured.
"""
from __future__ import annotations

import hashlib
from typing import Any

# hard ceilings — a 2D canvas draws one arc() per point per frame, so total points/frame is the budget
CEILING = 2000              # never exceed this many drawn points across the whole scene
SUBJECT_BASE = 420          # the focus object at full LoD
NEIGHBOR_BASE = 190         # a neighbour at full LoD (before depth falloff)

# graph TYPE / description tokens → a structural form. Token-keyed so it GENERALIZES (any 식물 → 가지형),
# never a per-concept entry. Order matters: earlier wins.
_FORMS: list[tuple[str, tuple[str, ...]]] = [
    ("branching", ("나무", "tree", "식물", "plant", "숲", "forest", "가지", "잎", "꽃", "flower")),
    ("humanoid", ("사람", "person", "인간", "human", "여자", "남자", "아이", "인물")),
    ("creature", ("동물", "animal", "개", "dog", "고양이", "cat", "새", "bird", "물고기", "fish", "말", "곰")),
    ("vessel", ("병", "bottle", "컵", "cup", "잔", "glass", "꽃병", "vase", "그릇", "bowl", "주전자", "container")),
    ("orb", ("별", "star", "태양", "sun", "행성", "planet", "달", "moon", "공", "ball", "사과", "apple",
             "오렌지", "orange", "지구", "earth", "과일", "fruit")),
    ("columnar", ("건물", "building", "탑", "tower", "기둥", "빌딩", "집", "house", "산", "mountain")),
    ("radial", ("바퀴", "wheel", "원", "circle", "톱니", "gear", "해바라기")),
    ("blob", ("물", "water", "구름", "cloud", "액체", "liquid", "기체", "gas", "연기", "smoke")),
]


def form_for(concept: dict[str, Any]) -> str:
    """The concept's structural form from its graph type/description. A presentation choice, not a fact;
    unknown types fall back to 'orb' (a clean solid), never a lie about the shape."""
    hay = " ".join(str(concept.get(k, "")) for k in ("type", "label", "canonical_name",
                                                      "desc", "short_description")).lower()
    for form, toks in _FORMS:
        if any(t in hay for t in toks):
            return form
    return "orb"


def _seed(text: str) -> int:
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8], 16)


def shape_spec(concept: dict[str, Any]) -> dict[str, Any]:
    """The compact descriptor sent over the wire (NOT points): the client expands it to a bounded cloud.
    A stable seed per concept keeps the silhouette identical frame to frame (a tree stays that tree)."""
    return {"form": form_for(concept),
            "seed": _seed(concept.get("concept_id") or concept.get("canonical_name") or concept.get("label") or "")}


def lod_budget(*, salience: float, depth: float, is_subject: bool = False) -> int:
    """Points to draw for one object: base LoD by role, decayed by distance behind the glass. Bounded so
    a whole scene of these stays under the per-frame CEILING (the renderer's governor may shrink further)."""
    base = SUBJECT_BASE if is_subject else NEIGHBOR_BASE
    depth_falloff = 1.0 / (1.0 + max(0.0, float(depth)) * 1.2)      # farther → fewer points
    n = base * max(0.15, min(1.0, float(salience))) * depth_falloff
    return max(24, min(SUBJECT_BASE, int(n)))                       # never fewer than a legible core


def decimate_importance(points: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Reduce a REAL 3DGS cloud to the output budget by keeping the most important Gaussians (opacity ×
    scale define the silhouette; interior redundancy is dropped). The pipeline a learned asset flows
    through — invariant is the OUTPUT budget, so the compression ratio adapts to the source size."""
    if budget <= 0 or not points:
        return []
    if len(points) <= budget:
        return list(points)
    ranked = sorted(points, key=lambda p: float(p.get("w", p.get("opacity", 1.0))) *
                    (1.0 + float(p.get("scale", 0.0))), reverse=True)
    return ranked[:budget]


def scene_point_estimate(objects: list[dict[str, Any]]) -> int:
    """Contract check: the total points a scene WOULD draw at full LoD — used to prove the ceiling holds."""
    total = 0
    for o in objects:
        total += lod_budget(salience=float(o.get("weight", 0.5)),
                            depth=abs(float((o.get("pos") or [0, 0, 0])[2])),
                            is_subject=(o.get("role") == "subject"))
    return total
