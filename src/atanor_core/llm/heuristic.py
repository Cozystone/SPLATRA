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
