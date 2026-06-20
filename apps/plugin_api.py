"""FastAPI plugin API + OpenAI function-calling tool schema.

Design rule (PRD §1.2 / §5): **never ship raw 3D buffers to the LLM.** Tool
responses carry only an SGF summary (DC-level: counts, sizes, bbox), a small
cartridge handle, and a hot-swap signal. The local browser viewer pulls the
cartridge on a side channel (``viewer_pull_url``) and renders it.

Run::

    pip install -e ".[api]"
    uvicorn apps.plugin_api:app --reload
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - api extra not installed
    raise RuntimeError(
        "FastAPI/pydantic missing. Install API extras: pip install -e '.[api]'"
    ) from exc

from atanor_core import build_default_engine
from atanor_core.domain.sgf import GaussianField
from atanor_core.state.machine import HoloState

app = FastAPI(title="atanor-hologram-core", version="0.1.0")

# Single-process PoC engine + in-memory job table.
_engine = build_default_engine()
_jobs: Dict[str, Dict[str, Any]] = {}
_viewer_sockets: List["WebSocket"] = []


# --------------------------------------------------------------------------- #
# OpenAI tool schema (function calling)
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
                                "embedding": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
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
                "required": ["nodes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_3d_object",
            "description": (
                "Generate a 3D object hologram from a text prompt. Returns "
                "immediately with a job id; poll /v1/job/{id} for the hot-swap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "quality": {
                        "type": "string",
                        "enum": ["fast", "refined"],
                        "default": "fast",
                    },
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
    quality: str = "fast"


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
    return s or "object"


def _prompt_to_mv(prompt: str) -> np.ndarray:
    """Deterministic synthetic multi-view color from a prompt (mock input)."""
    h = abs(hash(prompt))
    color = (
        ((h >> 0) & 0xFF) / 255.0,
        ((h >> 8) & 0xFF) / 255.0,
        ((h >> 16) & 0xFF) / 255.0,
    )
    img = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    for c in range(3):
        img[:, :, c, :, :] = color[c]
    return img


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/tools")
def get_tools() -> Dict[str, Any]:
    return {"tools": OPENAI_TOOLS}


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
    mv = _prompt_to_mv(req.prompt)
    result = _engine.generate_3d_object(name, mv, cam_rays=None)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"name": name, "done": result == "hit"}
    # Cache hit is effectively instant; a miss takes ~5s (honest ETA).
    eta = 0 if result == "hit" else 5
    return {
        "status": "displayed" if result == "hit" else "generating",
        "job_id": job_id,
        "name": name,
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


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket) -> None:
    """Push hot-swap signals + SGF deltas to connected viewers."""
    await ws.accept()
    _viewer_sockets.append(ws)
    try:
        # On connect, hand the viewer the current SGF summary if any.
        if _engine.field is not None:
            await ws.send_json({"type": "sgf", "sgf": _sgf_summary(_engine.field)})
        while True:
            # Echo client pings; real impl would push deltas on state changes.
            msg = await ws.receive_text()
            await ws.send_json({"type": "ack", "echo": msg, "state": _engine.state.value})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _viewer_sockets:
            _viewer_sockets.remove(ws)
