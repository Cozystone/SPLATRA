"""Structural decomposition — "a car" is a body, wheels, seats and an engine.

A single-view lift can only ever produce a shell: whatever the camera saw, inflated.
Ask it for a car and you get a car-shaped crust with nothing inside, because nothing
in the pipeline ever knew that a car *contains* seats.

So we split the problem. The LLM already knows what a car is made of — that
knowledge is exactly the part image reconstruction cannot recover — while the
generator is good at single concrete objects. Have the LLM name the parts and say
where each belongs, generate the parts independently, assemble them into one scene.
The result has an interior: hide the exterior layer and the seats are really there.

This is the cheap cousin of PartCrafter / PartGen (compositional part-level 3D
diffusion). Those denoise all parts jointly so the parts genuinely fit one another;
here the fit is only as good as the placement. Honest trade: no new weights, works
today, and the structure is real rather than implied.

Placement uses a discrete anchor vocabulary rather than raw coordinates. Small
local models reliably answer "bottom-corners" but hand back [0,0,0] for every part
when asked for numbers, so the words are turned into geometry here instead.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

LAYERS = ("exterior", "interior", "structure")

_ANCHORS: Dict[str, List[Tuple[float, float, float]]] = {
    "center": [(0.0, 0.0, 0.0)],
    "front": [(0.0, 0.0, 0.55)],
    "back": [(0.0, 0.0, -0.55)],
    "left": [(-0.55, 0.0, 0.0)],
    "right": [(0.55, 0.0, 0.0)],
    "top": [(0.0, 0.55, 0.0)],
    "bottom": [(0.0, -0.55, 0.0)],
    "front-top": [(0.0, 0.40, 0.45)],
    "front-bottom": [(0.0, -0.35, 0.45)],
    "back-top": [(0.0, 0.40, -0.45)],
    "back-bottom": [(0.0, -0.35, -0.45)],
    "sides": [(-0.40, 0.0, 0.0), (0.40, 0.0, 0.0)],
    "bottom-corners": [(-0.45, -0.45, 0.45), (0.45, -0.45, 0.45),
                       (-0.45, -0.45, -0.45), (0.45, -0.45, -0.45)],
    "top-corners": [(-0.45, 0.45, 0.45), (0.45, 0.45, 0.45),
                    (-0.45, 0.45, -0.45), (0.45, 0.45, -0.45)],
}

_ANCHOR_WORDS = ", ".join(_ANCHORS)

_SYS = (
    "You decompose an object into the parts it is actually MADE of, including the "
    "parts hidden inside. Output ONLY JSON: "
    '{"object":"<name>","has_interior":<true|false>,"parts":[{"name":"<part>",'
    '"prompt":"<one concrete object, '
    'in English, that can be generated on its own>","where":"<anchor>",'
    '"count":<1|2|4>,"scale":<0.05..1.0>,'
    '"layer":"<exterior|interior|structure>"}]}. '
    "where must be one of: " + _ANCHOR_WORDS + ". Use count 2 with where=sides for "
    "a symmetric pair, and count 4 with where=bottom-corners for wheels or legs. "
    "scale is the part's size relative to the whole. Give 3-8 parts. The outer "
    "shell comes first as layer exterior with where=center and scale 1.0; anything "
    "you would only see by opening the object is interior; load-bearing frames are "
    "structure. Example for a car: body shell (exterior, center, 1.0), wheels "
    "(exterior, bottom-corners, count 4, 0.22), seats (interior, sides, count 2, "
    "0.2), steering wheel (interior, front, 0.12), engine block (interior, "
    "front-bottom, 0.3). "
    "has_interior is true only when the object has a designed inside you "
    "cannot see from outside — a car (cabin, engine), a computer (boards), "
    "a building (rooms), a camera (sensor, mirror). It is false for solid "
    "natural things and simple shapes: fruit, a rock, a ball, a log, a "
    "flower. A stem or a seed does not count as an interior. Object: "
)

# Trusted layouts for the things people reach for first; everything else falls back
# to a single part, which is the honest answer to "we do not know its insides".
_KNOWN: Dict[str, List[Tuple[str, str, str, int, float, str]]] = {
    "car": [
        ("body", "a car body shell", "center", 1, 1.0, "exterior"),
        ("wheel", "a car wheel", "bottom-corners", 4, 0.22, "exterior"),
        ("seat", "a car seat", "sides", 2, 0.22, "interior"),
        ("engine", "a car engine block", "front-bottom", 1, 0.30, "interior"),
    ],
    "house": [
        ("shell", "a small house", "center", 1, 1.0, "exterior"),
        ("roof", "a house roof", "top", 1, 0.90, "exterior"),
        ("table", "a wooden table", "center", 1, 0.25, "interior"),
        ("chair", "a wooden chair", "sides", 2, 0.20, "interior"),
    ],
    "computer": [
        ("case", "a desktop computer case", "center", 1, 1.0, "exterior"),
        ("board", "a computer motherboard", "back", 1, 0.60, "interior"),
        ("fan", "a computer cooling fan", "front-top", 1, 0.25, "interior"),
    ],
}


# Small models label everything "exterior" about half the time, which makes the
# reveal useless. The words themselves are unambiguous, so correct the label from
# the part name rather than trusting the field.
_INSIDE_WORDS = ("seat", "engine", "motor", "board", "chip", "battery", "cabin",
                 "interior", "inside", "gear", "piston", "wire", "cable", "frame",
                 "skeleton", "bone", "organ", "heart", "fuel", "tank", "dashboard",
                 "steering", "pedal", "filling", "core", "stuffing", "spring")


def _infer_layer(name: str, prompt: str, given: str) -> str:
    text = (name + " " + prompt).lower()
    if any(w in text for w in _INSIDE_WORDS):
        return "interior"
    return given



# The model answers "does this have a designed inside?" well (10/12 on a mixed set)
# but still calls a football hollow because of its bladder. These are shapes whose
# inside is never worth generating, so they are settled here rather than asked.
_SOLID_WORDS = ("ball", "sphere", "orb", "marble", "rock", "stone", "pebble",
                "boulder", "brick", "log", "egg", "apple", "banana", "orange",
                "lemon", "pear", "peach", "grape", "tomato", "potato", "cube",
                "pyramid", "cone", "donut", "cookie", "candy", "coin")


def _has_interior(prompt: str, claimed: bool) -> bool:
    p = (prompt or "").lower()
    if any(w in p for w in _SOLID_WORDS):
        return False
    return claimed


def _places(where: str, count: int, layer: str) -> List[List[float]]:
    """Anchor word -> up to ``count`` positions; interiors are pulled inward so the
    seats land inside the cabin instead of on the paintwork."""
    spots = _ANCHORS.get(where, _ANCHORS["center"])
    if count > len(spots):
        spots = (spots * ((count // len(spots)) + 1))[:count]
    spots = spots[:max(1, count)]
    k = 0.55 if layer in ("interior", "structure") else 1.0
    return [[round(x * k, 3), round(y * k, 3), round(z * k, 3)] for x, y, z in spots]


def _expand(name: str, prompt: str, where: str, count: int, scale: float,
            layer: str) -> List[Dict[str, Any]]:
    out = []
    for i, at in enumerate(_places(where, count, layer)):
        out.append({"name": name if count == 1 else "%s%d" % (name, i + 1),
                    "prompt": prompt, "at": at, "scale": scale, "layer": layer})
    return out


def _fallback(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    for key, parts in _KNOWN.items():
        if key in p:
            out: List[Dict[str, Any]] = []
            for name, q, where, cnt, sc, layer in parts:
                out.extend(_expand(name, q, where, cnt, sc, layer))
            return {"object": key, "engine": "builtin",
                    "has_interior": True, "parts": out}
    return {"object": (prompt or "object").strip()[:40], "engine": "single",
            "has_interior": False,
            "parts": [{"name": "whole", "prompt": prompt or "object",
                       "at": [0.0, 0.0, 0.0], "scale": 1.0, "layer": "exterior"}]}


def _tighten(prompt: str, name: str) -> str:
    """Keep part prompts to something an image generator can actually draw.

    Asked for "one concrete object" the planner sometimes answers with a
    definition instead — "round components at the bottom corners for a car to
    move on". Diffusion models render that as a muddle of corners and cars, and
    the muddle is what gets reconstructed and bolted onto the object. The part's
    own name is the concrete noun that was wanted, so fall back to it whenever the
    prompt has drifted into prose.
    """
    q = " ".join((prompt or "").split())
    if len(q.split()) <= 4:
        return q[:80]
    nm = " ".join((name or "").split()).strip()
    nm = re.sub(r"[\s_-]*\d+$", "", nm)          # "wheels1" -> "wheels"
    return (nm or q)[:80]


def _sanitize(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    parts: List[Dict[str, Any]] = []
    for r in (data.get("parts") or [])[:8]:
        if not isinstance(r, dict):
            continue
        q = str(r.get("prompt") or r.get("name") or "").strip()
        if not q:
            continue
        try:
            scale = max(0.05, min(1.0, float(r.get("scale", 0.3))))
        except Exception:
            scale = 0.3
        layer = str(r.get("layer", "exterior")).lower()
        if layer not in LAYERS:
            layer = "exterior"
        where = str(r.get("where", "center")).lower()
        try:
            cnt = max(1, min(4, int(r.get("count", 1))))
        except Exception:
            cnt = 1
        nm = str(r.get("name", "part"))[:24]
        parts.extend(_expand(nm, _tighten(q, nm), where, cnt, scale,
                             _infer_layer(nm, q, layer)))
    if not parts:
        return None
    return {"object": str(data.get("object", "object"))[:40],
            "has_interior": bool(data.get("has_interior", False)),
            "parts": parts[:12]}


def _name_the_shell(parts: List[Dict[str, Any]], prompt: str) -> None:
    """Ask for the whole object when generating the outermost part.

    The planner names the biggest part after its role — "car body", "chassis",
    "outer casing" — which is the right word for a parts list and the wrong words
    for an image generator. "car body" renders a bare stamped panel about as often
    as it renders a car, and a flat panel makes a flat reconstruction, which is
    where the occasional slab-shaped car came from. The exterior shell of a car is
    just a car, so ask for that: the shell then comes out in the proportions and
    the upright orientation the rest of the parts were placed against.
    """
    want = (prompt or "").strip()
    if not want:
        return
    outer = [q for q in parts if q.get("layer") == "exterior"]
    if not outer:
        return
    shell = max(outer, key=lambda q: float(q.get("scale") or 0.0))
    if float(shell.get("scale") or 0.0) >= 0.9:
        shell["prompt"] = want


def decompose(prompt: str, ollama_model: Optional[str] = None,
              host: str = "http://localhost:11434",
              use_research: bool = True) -> Dict[str, Any]:
    """prompt -> the parts it is made of, placed in the object's own frame.

    With use_research the object is looked up first and the retrieved article
    text is handed to the model as evidence, so an unfamiliar thing is read about
    rather than guessed at. Retrieval failing is not fatal — we simply fall back to
    what the model already knows.
    """
    found = ""
    if use_research:
        try:
            from .research import evidence
            found = evidence(prompt)
        except Exception:
            found = ""
    if ollama_model:
        import urllib.request

        system = _SYS + (prompt or "")
        if found:
            system += ("\n\nUse this retrieved reference as the ground truth for "
                       "what the object is made of; prefer its terminology over "
                       "your own assumptions.\n" + found)
        payload = {"model": ollama_model, "format": "json", "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt or "object"}]}
        try:
            req = urllib.request.Request(
                host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (data.get("message", {}) or {}).get("content", "") or ""
            try:
                parsed = json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, re.S)
                parsed = json.loads(m.group(0)) if m else None
            clean = _sanitize(parsed)
            if clean and len(clean["parts"]) > 1:
                _name_the_shell(clean["parts"], prompt)
                clean["has_interior"] = _has_interior(prompt,
                                                      clean.get("has_interior", False))
                clean["engine"] = "ollama:" + ollama_model
                clean["researched"] = bool(found)
                return clean
        except Exception:
            pass
    return _fallback(prompt)
