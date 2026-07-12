"""Offline heuristic "LLM" — keyword/regex intent parser, no model needed.

This lets the chat UI work with zero dependencies and zero Ollama. It is NOT a
language model; it is an honest rule-based intent router that emits the same
tool-call structure a real LLM would.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

_SHAPE_WORDS = {
    "sphere": "sphere", "ball": "sphere", "orb": "sphere", "구": "sphere", "공": "sphere",
    "cube": "cube", "box": "cube", "큐브": "cube", "상자": "cube", "정육면체": "cube",
    "torus": "torus", "donut": "torus", "doughnut": "torus", "ring": "torus",
    "도넛": "torus", "토러스": "torus", "고리": "torus",
    "spiral": "spiral", "helix": "spiral", "나선": "spiral", "스프링": "spiral",
}
_GRAPH_WORDS = [
    "graph", "knowledge", "network", "node", "nodes", "그래프", "지식", "네트워크",
    "노드", "관계", "맵", "map",
]
_GEN_WORDS = [
    "generate", "make", "create", "build", "render an object", "object",
    "생성", "만들", "그려", "띄워", "보여",
]


def match_shape(text: str):
    """Return the procedural shape named in the text, or None if none is named."""
    t = text.lower()
    for word, shape in _SHAPE_WORDS.items():
        if word in t:
            return shape
    return None


# --- multi-object scene detection (지구와 달 -> two objects, not one blob) ---------
# ALWAYS-split separators. Korean noun conjunctions attach to the LEFT noun and are
# followed by a space before the next noun ("지구와 달", "책과 연필"). Requiring the
# trailing space is the boundary-safe rule that avoids the 사과(apple) trap — there 과 is
# word-interior (followed by 를, not a space) so it must NOT split. Commas/그리고/및/slash
# are unambiguous list separators.
_ALWAYS_SPLIT = re.compile(
    r"(?:와|과|랑|이랑|하고)\s+"          # Korean conjunction + space (boundary-safe)
    r"|\s*(?:그리고|및|,|、|/|;|·)\s*"     # explicit list separators
)
# English "and"/"vs" is split CONDITIONALLY (see split_objects) — bare "and" joins
# adjectives ("black and white cube" is ONE object), so it only splits a real list.
_EN_CONJ = re.compile(r"\s+(?:and|vs\.?|versus)\s+", re.IGNORECASE)
_STRIP_EN = re.compile(
    r"\b(?:generate|make|create|build|show|render|draw|explain|compare|please|"
    r"the|a|an)\b",
    re.IGNORECASE,
)
# Only UNAMBIGUOUS object josa (을/를) and multi-char generation verbs are peeled. The
# 1-char subject josa 이/가/은/는 are NOT peeled: they are indistinguishable from a noun's
# final syllable without morphology (고양'이'=cat, 종'이'=paper), and peeling them
# corrupted '고양이' -> '고양'.
_STRIP_KO_VERB = re.compile(
    r"\s*(?:보여줘|보여|만들어줘|만들어|그려줘|그려|생성해줘|생성|설명해줘|설명|"
    r"비교해줘|비교|해줘)$"
)
_STRIP_KO_OBJJOSA = re.compile(r"(?:을|를)$")
_RELATION_HINT = re.compile(
    r"관계|연결|이어|vs\.?|versus|비교|compare|link|connect|사이", re.IGNORECASE
)


def _clean_object(fragment: str) -> str:
    s = _STRIP_EN.sub(" ", fragment)
    s = s.strip(" \t\r\n'\"()[]{}。.!?~-·")
    for _ in range(3):
        s2 = _STRIP_KO_VERB.sub("", s).strip()
        s2 = _STRIP_KO_OBJJOSA.sub("", s2).strip()
        if s2 == s:
            break
        s = s2
    return s.strip(" \t'\"-")


def split_objects(text: str) -> List[str]:
    """Return >=2 distinct object prompts iff the text names multiple things; else []."""
    has_comma = bool(re.search(r"[,、;/·]", text))
    refined: List[str] = []
    for part in _ALWAYS_SPLIT.split(text):
        subs = _EN_CONJ.split(part)
        # A bare English "and" only makes a list when the text is comma-punctuated
        # ("a, b and c") or every side names a shape ("a red SPHERE and a blue CUBE");
        # otherwise keep it whole so "black and white cube" stays one object.
        if len(subs) >= 2 and (has_comma or all(match_shape(s) for s in subs)):
            refined.extend(subs)
        else:
            refined.append(part)
    out: List[str] = []
    seen = set()
    for frag in refined:
        p = _clean_object(frag)
        if len(p) < 1 or p.isdigit():
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out if len(out) >= 2 else []


def detect_shape(text: str) -> str:
    return match_shape(text) or "sphere"


def sample_graph(n: int = 18, seed: int = 0) -> Dict[str, Any]:
    """Deterministic sample knowledge graph (ring + random cross links)."""
    rng = np.random.default_rng(seed)
    nodes = [
        {
            "id": f"n{i}",
            "embedding": rng.normal(size=12).astype(np.float32).tolist(),
            "centrality": float(rng.uniform(0.5, 10.0)),
            "importance": float(rng.uniform(0.25, 0.95)),
            "category": int(i % 12),
        }
        for i in range(n)
    ]
    edges = [{"src": f"n{i}", "dst": f"n{(i + 1) % n}"} for i in range(n)]
    for _ in range(n // 2):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            edges.append({"src": f"n{a}", "dst": f"n{b}"})
    return {"nodes": nodes, "edges": edges}


class HeuristicLLM:
    """Rule-based intent router (offline fallback, not a real LLM)."""

    name = "heuristic"

    def chat(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = str(m.get("content", ""))
                break
        t = user.lower()

        wants_graph = any(w in t for w in _GRAPH_WORDS)
        wants_gen = any(w in t for w in _GEN_WORDS) or any(
            w in t for w in _SHAPE_WORDS
        )

        # Multi-object request ("지구와 달", "sun, earth and moon") -> ONE explain_scene
        # call so several objects coexist in a shared space, instead of the whole phrase
        # collapsing into a single blob. Only when the caller offers explain_scene and the
        # text is not a knowledge-graph request.
        has_scene_tool = any(
            (tool.get("function") or {}).get("name") == "explain_scene"
            for tool in (tools or [])
        )
        if has_scene_tool and not wants_graph:
            objs = split_objects(user)
            if len(objs) >= 2:
                objects = [{"prompt": name, "id": name} for name in objs[:8]]
                links = (
                    [[objs[i], objs[i + 1]] for i in range(len(objs[:8]) - 1)]
                    if _RELATION_HINT.search(user)
                    else []
                )
                joined = ", ".join(objs[:8])
                return {
                    "content": f"Building a scene with {joined}.",
                    "tool_calls": [
                        {"name": "explain_scene",
                         "arguments": {"objects": objects, "links": links}}
                    ],
                }

        if wants_graph and not wants_gen:
            m = re.search(r"(\d{1,3})", t)
            n = int(m.group(1)) if m else 18
            n = max(3, min(n, 120))
            return {
                "content": f"Visualizing a knowledge-graph hologram with {n} nodes.",
                "tool_calls": [
                    {"name": "render_knowledge_hologram", "arguments": {"_sample_n": n}}
                ],
            }

        # Only set an explicit shape when one is actually named; otherwise leave
        # it unset so the backend can synthesize the real object (text->3D).
        shape = match_shape(t)
        prompt = user.strip() or (shape or "object")
        args = {"prompt": prompt}
        if shape:
            args["shape"] = shape
        if wants_gen:
            what = f"a 3D '{shape}'" if shape else f"'{prompt}'"
            return {
                "content": f"Generating {what} object hologram.",
                "tool_calls": [{"name": "generate_3d_object", "arguments": args}],
            }

        # Default: treat free text as an object prompt.
        return {
            "content": (
                "I'll render that as a 3D object. Try: 'generate a torus', "
                "'show a knowledge graph with 24 nodes', or 'make a blue cube'."
            ),
            "tool_calls": [{"name": "generate_3d_object", "arguments": args}],
        }
