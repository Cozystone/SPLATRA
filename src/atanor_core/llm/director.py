"""Narration director — the LLM authors a *time-synced* script that speaks while
manipulating the 3D model (the "JARVIS" mode).

The model is generated first; then the LLM writes a short script: a list of steps,
each a phrase to **say** plus an **action** to fire as that phrase is spoken. The
browser speaks the phrase (TTS) and runs the action at the same moment, so speech
and the particle model move together::

    "피카츄는"          -> reveal / face camera
    "꼬리에"            -> focus + rotate to the tail
    "전기를 모아"        -> charge FX gathering at the tail
    "방출합니다"         -> discharge burst

Action vocabulary (client interprets these):
    focus    {yaw,pitch,dist,ms}     camera move
    spin     {ms}                    slow turntable
    charge   {center:[x,y,z],ms}     energy gathers at a point on the model
    discharge{center:[x,y,z],ms}     burst of particles from that point
    pulse    {ms}                    whole-model glow
    none

Coordinates are in the model's normalized [-1,1] cube (the LLM guesses anchors
like "tail" ≈ a point). Falls back to a generic reveal→spin→pulse script if no
LLM is available.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_SYS = (
    "You are a 3D explainer that BUILDS A SCENE out of particle models and narrates "
    "it. Output ONLY JSON: a list of 5-12 steps, each "
    "{{\"say\":\"<short phrase in {lang}>\",\"action\":{{...}}}}. You compose the "
    "whole scene freely to teach '{topic}'. Start by SPAWNING the object(s) you "
    "need, placed apart in the world (the world spans about [-2,2]); then narrate, "
    "move things, point and label. Actions:\n"
    "- spawn {{\"type\":\"spawn\",\"prompt\":\"<a single concrete object, in English>\",\"id\":\"<short name>\",\"at\":[x,y,z],\"scale\":<0.3..1.0>}} create a 3D object at a spot\n"
    "- move {{\"type\":\"move\",\"id\":\"<name>\",\"to\":[x,y,z],\"ms\":1200}} move/animate an object (e.g. something falling)\n"
    "- focus {{\"type\":\"focus\",\"yaw\":<rad>,\"pitch\":<rad>,\"dist\":<3..6>,\"ms\":1200}} rotate/zoom the camera\n"
    "- arrow {{\"type\":\"arrow\",\"from\":[x,y,z],\"to\":[x,y,z],\"ms\":3000}} point, or show a force/flow/direction in the air\n"
    "- label {{\"type\":\"label\",\"at\":[x,y,z],\"text\":\"<short>\",\"ms\":3500}} name something\n"
    "- scale {{\"type\":\"scale\",\"to\":<0.3..1.8>,\"ms\":900}}, spin {{\"type\":\"spin\",\"ms\":1600}}, "
    "charge/discharge {{\"type\":\"charge\",\"center\":[x,y,z],\"ms\":1500}} energy FX, pulse, none.\n"
    "- animate {{\"type\":\"animate\",\"style\":\"breathe|sway|bounce|float|jelly|wave|spin|walk|handwave|stop\",\"ms\":3000}} "
    "make the model move (breathe=living thing, sway=plant/tree, bounce/jelly=soft/playful, "
    "float=light/floating, wave=water/flag, walk=a person/animal walking with swinging "
    "limbs, handwave=a character waving hello) — bring an object to life as you describe it.\n"
    "Example — gravity: spawn a tree at [-0.8,0,0]; spawn an apple in its branches "
    "at [-0.6,0.6,0]; spawn a person under it at [-0.6,-0.8,0]; then move the apple "
    "down onto the person's head and add a downward arrow for the force. For a "
    "single-object topic, just spawn it once and inspect it with focus/arrow/label. "
    "Make each short phrase line up with its visual. Explain: {topic}"
)


def _extract_json_list(text: str) -> Optional[list]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.S)        # first [...] block
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _normalize(steps: Any) -> List[Dict[str, Any]]:
    out = []
    if isinstance(steps, dict):
        steps = steps.get("steps") or steps.get("script") or []
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        say = str(s.get("say") or s.get("text") or "").strip()
        act = s.get("action") if isinstance(s.get("action"), dict) else {"type": "none"}
        if "type" not in act:
            act = {"type": "none"}
        if say or act.get("type") != "none":
            out.append({"say": say, "action": act})
    return out


_TAIL = [0.0, -0.3, -0.45]


def _heuristic_script(topic: str, lang: str) -> List[Dict[str, Any]]:
    """No-LLM fallback: chunk the text into ~2-word phrases and pick an action by
    keyword (전기→charge, 방출→discharge, 꼬리→focus tail, …) so the visual still
    tracks the words. The LLM director produces richer scripts."""
    words = re.sub(r"[.,!?·]", " ", topic).split()
    phrases = [" ".join(words[i:i + 2]) for i in range(0, len(words), 2)] or [topic]
    spin = [{"type": "spin", "ms": 1400}, {"type": "pulse", "ms": 700},
            {"type": "focus", "yaw": 0.6, "pitch": 0.3, "dist": 3.2, "ms": 1200}]
    # first: spawn the subject (the LLM does richer scenes; this is the fallback)
    out = [{"say": phrases[0], "action": {"type": "spawn", "prompt": topic,
            "id": "obj", "at": [0, 0, 0], "scale": 1.0}}]
    for i, p in enumerate(phrases[1:]):
        if any(k in p for k in ("전기", "스파크", "번개", "spark", "electric", "lightning", "energy", "에너지")):
            a = {"type": "charge", "center": _TAIL, "ms": 1500}
        elif any(k in p for k in ("방출", "발사", "폭발", "쏘", "discharge", "release", "emit", "blast", "fire")):
            a = {"type": "discharge", "center": _TAIL, "ms": 700}
        elif any(k in p for k in ("꼬리", "tail", "뒤", "back")):
            a = {"type": "focus", "yaw": 2.4, "pitch": 0.2, "dist": 2.6, "ms": 1200}
        elif any(k in p for k in ("머리", "얼굴", "head", "face", "앞", "front")):
            a = {"type": "focus", "yaw": 0.0, "pitch": 0.1, "dist": 2.6, "ms": 1200}
        else:
            a = spin[i % 3]
        out.append({"say": p, "action": a})
    return out[:8]


def make_script(topic: str, lang: str = "ko",
                ollama_model: Optional[str] = None,
                host: str = "http://localhost:11434") -> Dict[str, Any]:
    """Return {engine, script:[{say,action}]}. Uses Ollama (format=json) if a model
    is given and reachable; otherwise the heuristic fallback."""
    if ollama_model:
        import urllib.request

        sys_prompt = _SYS.format(lang=lang, topic=topic)
        payload = {
            "model": ollama_model,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": topic}],
            "format": "json", "stream": False,
        }
        try:
            req = urllib.request.Request(
                f"{host.rstrip('/')}/api/chat", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = (data.get("message", {}) or {}).get("content", "") or ""
            steps = _normalize(_extract_json_list(content))
            if steps:
                return {"engine": f"ollama:{ollama_model}", "script": steps}
        except Exception:
            pass
    return {"engine": "heuristic", "script": _heuristic_script(topic, lang)}
