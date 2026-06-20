"""FastAPI plugin API + OpenAI tools schema + local-LLM chat UI.

Design rule (PRD §1.2 / §5): **never ship raw 3D buffers to the LLM.** Tool
responses carry only an SGF summary (DC-level: counts, sizes, bbox), a small
cartridge handle, and a hot-swap signal.

This server also hosts a local chat UI (``/``) that wires a local LLM (Ollama)
to the engine. The browser shows the *actual* image the CPU EWA rasterizer
renders (via ``/v1/frame``), orbitable with the mouse — not a fake scatter.

Run::

    pip install -e ".[api]"
    uvicorn apps.plugin_api:app --reload      # then open http://localhost:8000
"""

from __future__ import annotations

import os
import re
import struct
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - api extra not installed
    raise RuntimeError(
        "FastAPI/pydantic missing. Install API extras: pip install -e '.[api]'"
    ) from exc

from atanor_core import build_default_engine
from atanor_core.domain.sgf import GaussianField
from atanor_core.llm.heuristic import HeuristicLLM, detect_shape, sample_graph
from atanor_core.llm.ollama import OllamaClient, list_models
from atanor_core.state.machine import HoloState
from atanor_core.state.rasterizer import default_intrinsics, orbit_camera

app = FastAPI(title="atanor-hologram-core", version="0.1.0")

_VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer")

# Single-process PoC engine + in-memory job table.
_engine = build_default_engine()
_jobs: Dict[str, Dict[str, Any]] = {}
_viewer_sockets: List["WebSocket"] = []
_heuristic = HeuristicLLM()

# Always have something on screen for the first frame.
_engine.render_knowledge_hologram(sample_graph(n=18, seed=1))
_engine.drain_events()


# --------------------------------------------------------------------------- #
# OpenAI tool schema (function calling) — shared with Ollama
# --------------------------------------------------------------------------- #
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "render_knowledge_hologram",
            "description": (
                "Visualize a knowledge graph as a 3D Gaussian particle hologram. "
                "Returns an SGF summary and a cartridge handle, NOT raw buffers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "embedding": {"type": "array", "items": {"type": "number"}},
                                "centrality": {"type": "number"},
                                "importance": {"type": "number"},
                                "category": {"type": "integer"},
                            },
                            "required": ["id"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string"},
                                "dst": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_3d_object",
            "description": (
                "Generate a 3D object hologram from a text prompt. quality is "
                "fast|refined; shape is sphere|cube|torus|spiral."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "shape": {
                        "type": "string",
                        "enum": ["sphere", "cube", "torus", "spiral"],
                    },
                    "quality": {"type": "string", "enum": ["fast", "refined"]},
                },
                "required": ["prompt"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class GraphNode(BaseModel):
    id: str
    embedding: Optional[List[float]] = None
    centrality: float = 0.0
    importance: float = 0.5
    category: int = 0


class GraphEdge(BaseModel):
    src: Optional[str] = None
    dst: Optional[str] = None


class RenderGraphRequest(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    prompt: str
    shape: Optional[str] = None
    quality: str = "fast"


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None      # e.g. "llama3.1"; None -> heuristic
    use_ollama: bool = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sgf_summary(field: GaussianField) -> Dict[str, Any]:
    """DC-level summary only — NOT the raw buffer (PRD §5)."""
    mn = field.means.min(axis=0).tolist()
    mx = field.means.max(axis=0).tolist()
    return {
        "num_gaussians": field.num_gaussians,
        "sh_degree": field.sh_degree,
        "raw_bytes": field.nbytes(),
        "bbox_min": [round(float(v), 4) for v in mn],
        "bbox_max": [round(float(v), 4) for v in mx],
    }


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48] or "object"


# Vivid named colors (honor "blue cube", "red torus", incl. common Korean).
_COLORS = {
    "red": (0.92, 0.16, 0.16), "빨강": (0.92, 0.16, 0.16), "빨간": (0.92, 0.16, 0.16),
    "orange": (0.96, 0.52, 0.12), "주황": (0.96, 0.52, 0.12),
    "yellow": (0.96, 0.86, 0.22), "노랑": (0.96, 0.86, 0.22), "노란": (0.96, 0.86, 0.22),
    "gold": (0.86, 0.7, 0.26), "금색": (0.86, 0.7, 0.26),
    "green": (0.22, 0.82, 0.34), "초록": (0.22, 0.82, 0.34), "녹색": (0.22, 0.82, 0.34),
    "teal": (0.18, 0.8, 0.74), "cyan": (0.2, 0.85, 0.9), "청록": (0.18, 0.8, 0.74),
    "blue": (0.26, 0.48, 0.96), "파랑": (0.26, 0.48, 0.96), "파란": (0.26, 0.48, 0.96),
    "purple": (0.62, 0.32, 0.92), "violet": (0.62, 0.32, 0.92), "보라": (0.62, 0.32, 0.92),
    "pink": (0.96, 0.42, 0.72), "magenta": (0.95, 0.3, 0.75), "분홍": (0.96, 0.42, 0.72),
    "핑크": (0.96, 0.42, 0.72),
    "white": (0.9, 0.92, 0.96), "흰": (0.9, 0.92, 0.96), "하양": (0.9, 0.92, 0.96),
}


def _detect_color(prompt: str) -> Optional[tuple]:
    t = prompt.lower()
    for word, rgb in _COLORS.items():
        if word in t:
            return rgb
    return None


def _prompt_to_mv(prompt: str) -> np.ndarray:
    """Synthetic multi-view color (mock input): honor color words, else a vivid
    deterministic hue from the prompt hash."""
    rgb = _detect_color(prompt)
    if rgb is None:
        h = abs(hash(prompt))
        c = np.array([(h >> 0) & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF], dtype=np.float32) / 255.0
        m = float(c.max())
        c = (c / m * 0.9) if m > 1e-3 else np.array([0.4, 0.55, 0.95], dtype=np.float32)
        rgb = tuple(np.clip(c, 0.14, 1.0).tolist())
    img = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    for ch in range(3):
        img[:, :, ch, :, :] = rgb[ch]
    return img


def _png_bytes(img01: np.ndarray) -> bytes:
    """Encode an [H,W,3] float image in [0,1] as PNG (stdlib zlib, no Pillow)."""
    arr = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    h, w, _ = arr.shape
    raw = bytearray()
    stride = w * 3
    flat = arr.reshape(h, stride)
    for y in range(h):
        raw.append(0)  # filter type 0 (None)
        raw.extend(flat[y].tobytes())
    comp = zlib.compress(bytes(raw), 6)

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


def _drive_generation(name: str, max_ticks: int = 600) -> bool:
    """Tick the engine until the named object is generated + displayed."""
    for _ in range(max_ticks):
        _engine.tick(name)
        if name in _engine.cache and _engine.state == HoloState.DISPLAYED:
            return True
        time.sleep(0.003)
    return name in _engine.cache


def _execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """Run one LLM tool call against the engine; return a small action record."""
    name = call.get("name", "")
    args = call.get("arguments", {}) or {}

    if name == "render_knowledge_hologram":
        nodes = args.get("nodes")
        if nodes:
            graph = {"nodes": nodes, "edges": args.get("edges", []) or []}
        else:
            graph = sample_graph(n=int(args.get("_sample_n", 18)), seed=1)
        field = _engine.render_knowledge_hologram(graph)
        return {
            "tool": name,
            "ok": True,
            "cartridge_id": f"graph-{uuid.uuid4().hex[:8]}",
            "sgf": _sgf_summary(field),
            "edges": len(_engine._edges),
        }

    if name == "generate_3d_object":
        prompt = str(args.get("prompt", "object"))
        shape = args.get("shape") or detect_shape(prompt)
        obj = _slug(prompt)
        _engine.generate_3d_object(obj, _prompt_to_mv(prompt), cam_rays={"shape": shape})
        done = _drive_generation(obj)
        rec: Dict[str, Any] = {"tool": name, "ok": done, "name": obj, "shape": shape}
        if done and _engine.field is not None:
            rec["sgf"] = _sgf_summary(_engine.field)
            rec["verified"] = bool(_engine.cache[obj].verified)
            rec["hot_swap"] = True
        return rec

    return {"tool": name, "ok": False, "error": "unknown tool"}


# --------------------------------------------------------------------------- #
# UI + introspection
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = os.path.join(_VIEWER_DIR, "studio.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/tools")
def get_tools() -> Dict[str, Any]:
    return {"tools": OPENAI_TOOLS}


@app.get("/v1/models")
def get_models() -> Dict[str, Any]:
    models = list_models()
    return {"ollama_available": bool(models), "models": models}


@app.get("/v1/state")
def get_state() -> Dict[str, Any]:
    resp: Dict[str, Any] = {"state": _engine.state.value, "edges": len(_engine._edges)}
    if _engine.field is not None:
        resp["sgf"] = _sgf_summary(_engine.field)
    return resp


# --------------------------------------------------------------------------- #
# Live frame (the actual EWA-rendered image)
# --------------------------------------------------------------------------- #
@app.get("/v1/frame")
def frame(
    yaw: float = 0.6,
    pitch: float = 0.35,
    dist: float = 3.2,
    w: int = 480,
    h: int = 480,
    fov: float = 55.0,
) -> Response:
    if _engine.field is None:
        raise HTTPException(status_code=409, detail="nothing rendered yet")
    viewmat = orbit_camera(yaw, pitch, dist)
    K = default_intrinsics(w, h, fov_deg=fov)
    img = _engine.render(viewmat, K, w, h)
    png = _png_bytes(img)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------- #
# Direct tool endpoints (LLM plugin contract)
# --------------------------------------------------------------------------- #
@app.post("/v1/render_knowledge_hologram")
def render_knowledge_hologram(req: RenderGraphRequest) -> Dict[str, Any]:
    graph = {
        "nodes": [n.model_dump() for n in req.nodes],
        "edges": [e.model_dump() for e in req.edges],
    }
    field = _engine.render_knowledge_hologram(graph)
    cartridge_id = f"graph-{uuid.uuid4().hex[:8]}"
    return {
        "status": "displayed",
        "cartridge_id": cartridge_id,
        "sgf": _sgf_summary(field),  # DC summary only, no raw buffer
        "hot_swap": True,
        "viewer_pull_url": f"/viewer/pull/{cartridge_id}",
    }


@app.post("/v1/generate_3d_object")
def generate_3d_object(req: GenerateRequest) -> Dict[str, Any]:
    name = _slug(req.prompt)
    shape = req.shape or detect_shape(req.prompt)
    result = _engine.generate_3d_object(name, _prompt_to_mv(req.prompt), cam_rays={"shape": shape})
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"name": name, "done": result == "hit"}
    eta = 0 if result == "hit" else 5  # hit ~instant; miss ~5s (honest ETA)
    return {
        "status": "displayed" if result == "hit" else "generating",
        "job_id": job_id,
        "name": name,
        "shape": shape,
        "cache": result,
        "eta_seconds": eta,
        "poll": f"/v1/job/{job_id}",
    }


@app.get("/v1/job/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    name = job["name"]
    _engine.tick(name)
    events = [{"state": e.state.value, "info": e.info} for e in _engine.drain_events()]
    done = name in _engine.cache and _engine.state == HoloState.DISPLAYED
    job["done"] = done

    resp: Dict[str, Any] = {
        "job_id": job_id,
        "name": name,
        "state": _engine.state.value,
        "events": events,
        "done": done,
    }
    if done and _engine.field is not None:
        resp["sgf"] = _sgf_summary(_engine.field)
        resp["verified"] = bool(_engine.cache[name].verified)
        resp["hot_swap"] = True
    return resp


# --------------------------------------------------------------------------- #
# Chat (local LLM tool-calling loop)
# --------------------------------------------------------------------------- #
@app.post("/v1/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    messages = [{"role": "user", "content": req.message}]

    note = ""
    if req.use_ollama and req.model:
        client = OllamaClient(model=req.model, fallback=_heuristic)
        try:
            out = client.chat(messages, OPENAI_TOOLS)
        except Exception as exc:  # total failure -> heuristic fallback
            out = _heuristic.chat(messages, OPENAI_TOOLS)
            out["engine"] = "heuristic"
            note = f"(Ollama failed: {exc}; used heuristic)"
    else:
        out = _heuristic.chat(messages, OPENAI_TOOLS)
        out.setdefault("engine", "heuristic")
    used = out.get("engine", "heuristic")

    actions = [_execute_tool_call(tc) for tc in out.get("tool_calls", [])]
    return {
        "engine": used,
        "assistant": (out.get("content", "") + (" " + note if note else "")).strip(),
        "actions": actions,
        "state": _engine.state.value,
        "sgf": _sgf_summary(_engine.field) if _engine.field is not None else None,
        "edges": len(_engine._edges),
    }


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket) -> None:
    """Push hot-swap signals + SGF deltas to connected viewers."""
    await ws.accept()
    _viewer_sockets.append(ws)
    try:
        if _engine.field is not None:
            await ws.send_json({"type": "sgf", "sgf": _sgf_summary(_engine.field)})
        while True:
            msg = await ws.receive_text()
            await ws.send_json({"type": "ack", "echo": msg, "state": _engine.state.value})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _viewer_sockets:
            _viewer_sockets.remove(ws)
