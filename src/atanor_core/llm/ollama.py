"""Local Ollama adapter (stdlib urllib, no extra deps).

Two routes, tried in order, so *any* local model can drive the engine:

1. **Native tool-calling** (``/api/chat`` with ``tools``) — for models that
   support it (llama3.1, qwen2.5, mistral-nemo, …).
2. **Prompted JSON mode** — if the model has no native tool support (Ollama
   returns HTTP 400 "does not support tools", e.g. dolphin3), we re-ask with
   ``format:"json"`` and a schema-describing system prompt, then parse the JSON
   into the same tool-call structure.

If both fail (Ollama down, or unparseable), it degrades to
:class:`HeuristicLLM`. The returned dict carries an ``engine`` field naming the
route that actually produced the result.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .heuristic import HeuristicLLM

DEFAULT_HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You drive a 3D Gaussian hologram engine. Use the provided tools. "
    "Call render_knowledge_hologram to visualize a knowledge graph, or "
    "generate_3d_object to create a 3D object from a text prompt. "
    "Prefer calling a tool over answering in prose."
)

# Used when the model lacks native tool support; forced via format="json".
JSON_PROMPT = (
    "You control a 3D Gaussian hologram engine. For the user's request, choose "
    "exactly ONE tool and reply with ONLY a JSON object (no prose, no markdown):\n"
    '1) {"tool":"generate_3d_object","arguments":{"prompt":"<the object in '
    'English, e.g. a red apple, a pikachu, a teapot>"}}  — to create a 3D model '
    "of ANY object from a description. Translate the object to English. Do NOT "
    'add a "shape" field unless the user literally asks for a geometric primitive '
    '(then add "shape":"sphere|cube|torus|spiral").\n'
    '2) {"tool":"render_knowledge_hologram","arguments":{"_sample_n":<int 3-120>}}'
    "  — to visualize a knowledge graph / network of N nodes.\n"
    "Output one JSON object of that exact form."
)


class OllamaError(RuntimeError):
    pass


def list_models(host: str = DEFAULT_HOST, timeout: float = 3.0) -> List[str]:
    """Return installed Ollama model names, or [] if unreachable."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _normalize_call(obj: Any) -> Optional[Dict[str, Any]]:
    """Coerce a loose model-emitted object into {"name", "arguments"}."""
    if not isinstance(obj, dict):
        return None
    if obj.get("tool_calls"):
        first = obj["tool_calls"][0]
        if isinstance(first, dict):
            fn = first.get("function", first)
            return _normalize_call(
                {"tool": fn.get("name") or first.get("tool"),
                 "arguments": fn.get("arguments") or first.get("arguments") or {}}
            )
    name = obj.get("tool") or obj.get("name") or obj.get("function")
    args = obj.get("arguments") or obj.get("args") or {}
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        args = name.get("arguments", args)
        name = name.get("name")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"prompt": args}
    if not name:  # infer from shape of the object
        if any(k in obj for k in ("_sample_n", "nodes", "n_nodes", "num_nodes")):
            n = obj.get("_sample_n") or obj.get("n_nodes") or obj.get("num_nodes") or 18
            return {"name": "render_knowledge_hologram", "arguments": {"_sample_n": int(n)}}
        if "shape" in obj or "prompt" in obj:
            return {
                "name": "generate_3d_object",
                "arguments": {k: obj[k] for k in ("prompt", "shape") if k in obj},
            }
        return None
    if not isinstance(args, dict):
        args = {}
    return {"name": str(name), "arguments": args}


class OllamaClient:
    """Tool-calling chat via a local Ollama server (native + JSON fallback)."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        fallback: Optional[HeuristicLLM] = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = float(timeout)
        self.name = f"ollama:{model}"
        self.fallback = fallback or HeuristicLLM()

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:200]
            except Exception:
                pass
            raise OllamaError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Ollama unreachable at {self.host} ({exc}). Is `ollama serve` running?"
            ) from exc

    def _chat_native(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        data = self._post(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "tools": tools,
                "stream": False,
            }
        )
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            norm = _normalize_call(tc)
            if norm:
                calls.append(norm)
        if not calls:
            return None
        return {"engine": f"{self.name} (native)", "content": content, "tool_calls": calls}

    def _chat_prompted(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        data = self._post(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": JSON_PROMPT}] + messages,
                "format": "json",
                "stream": False,
            }
        )
        content = (data.get("message", {}) or {}).get("content", "") or ""
        try:
            obj = json.loads(content)
        except Exception:
            return None
        norm = _normalize_call(obj)
        if not norm:
            return None
        return {"engine": f"{self.name} (json)", "content": "", "tool_calls": [norm]}

    def chat(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # 1) native tool-calling
        try:
            out = self._chat_native(messages, tools)
            if out:
                return out
        except OllamaError:
            pass  # likely "does not support tools" -> try prompted JSON

        # 2) prompted JSON mode (works for models without tool support)
        try:
            out = self._chat_prompted(messages)
            if out:
                return out
        except OllamaError:
            pass

        # 3) offline heuristic fallback
        fb = self.fallback.chat(messages, tools)
        fb["engine"] = f"{self.name}→heuristic"
        fb["content"] = (fb.get("content", "") + " (Ollama gave no tool call; used heuristic)").strip()
        return fb
