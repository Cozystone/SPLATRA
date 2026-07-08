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

import io
import json
import os
import re
import struct
import threading
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from fastapi import (
        FastAPI,
        File,
        HTTPException,
        Response,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - api extra not installed
    raise RuntimeError(
        "FastAPI/pydantic missing. Install API extras: pip install -e '.[api]'"
    ) from exc

from atanor_core import build_default_engine
from atanor_core.deformation.fourier import FourierDeformer
from atanor_core.domain.sgf import GaussianField, sh_dc_to_rgb
from atanor_core.llm.heuristic import _SHAPE_WORDS, HeuristicLLM, detect_shape, sample_graph
from atanor_core.llm.ollama import OllamaClient, list_models
from atanor_core.state.machine import HoloState
from atanor_core.state.rasterizer import default_intrinsics, orbit_camera

# Real LGM image->3D is opt-in (needs CUDA + the `gen` extra + weights).
_USE_LGM = os.environ.get("SPLATRA_LGM", "0") == "1"
_lgm_gen = None  # lazy singleton
# Tiny-SD text->image->3D is opt-in (needs diffusers + a ~1.7GB download).
_USE_SD = os.environ.get("SPLATRA_SD", "0") == "1"
_sd_gen = None  # lazy singleton
# Real multi-view 3D (Zero123++ + visual hull) — opt-in, GPU.
_USE_MV = os.environ.get("SPLATRA_MV", "0") == "1"
_mv_gen = None  # lazy singleton
# Learned single-image 3D (TripoSR) — opt-in, GPU. Highest quality.
_USE_TRIPOSR = os.environ.get("SPLATRA_TRIPOSR", "0") == "1"
_triposr_gen = None  # lazy singleton

if _USE_SD or _USE_MV or _USE_TRIPOSR:
    # diffusers must be imported on the MAIN thread: FastAPI runs sync tool
    # calls in a worker thread, where the lazy diffusers import partially
    # initializes and then fails (measured under uvicorn only:
    # "cannot import name 'AutoPipelineForText2Image'" while the same import
    # succeeds in a plain main-thread process). Warming it here also removes
    # the first-request pipeline-load latency.
    try:
        from diffusers import AutoPipelineForText2Image as _warm_t2i  # noqa: F401
    except Exception:
        pass


def _triposr():
    global _triposr_gen
    if _triposr_gen is None:
        from atanor_core.generation.triposr import TripoSRGenerator

        _triposr_gen = TripoSRGenerator()
        # Share the one text->image pipeline (SDXL ~7GB) instead of letting TripoSR
        # spin up its own second copy — two SDXL pipelines won't fit in 16GB.
        _triposr_gen._t2i = _sd()
    return _triposr_gen


def _sd():
    global _sd_gen
    if _sd_gen is None:
        from atanor_core.generation.text_to_3d import TextTo3DGenerator

        _sd_gen = TextTo3DGenerator()
    return _sd_gen


def _mv():
    global _mv_gen
    if _mv_gen is None:
        from atanor_core.generation.multiview import MultiViewGenerator

        _mv_gen = MultiViewGenerator()
    return _mv_gen

app = FastAPI(title="atanor-hologram-core", version="0.1.0")

# Allow the static Vercel viewer (or any browser) to call this local GPU server.
try:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
except Exception:  # pragma: no cover
    pass


@app.on_event("startup")
def _prewarm_models() -> None:
    """Load the GPU generators in the background at startup so the FIRST user
    request is fast (no cold model load mid-request)."""
    if not (_USE_TRIPOSR or _USE_MV or _USE_SD):
        return

    def warm():
        try:
            if _USE_SD or _USE_TRIPOSR or _USE_MV:
                _sd()._ensure()          # SD-Turbo (conditioning image for all)
            if _USE_TRIPOSR:
                _triposr()._ensure()     # TripoSR triplane model
            elif _USE_MV:
                _mv()._ensure()          # Zero123++
            # one throwaway FULL-pipeline gen so cuDNN autotuning for BOTH SDXL and
            # TripoSR is paid here, not on the user's first request (that first-run
            # cost is the ~25s "first generation is slow" spike).
            try:
                _gen_object("a gray sphere")
            except Exception:
                try:
                    _sd().warmup()
                except Exception:
                    pass
        except Exception:
            pass

    # Resolve diffusers' lazy module + load models entirely in the warm thread so
    # the HTTP server binds immediately (doing it synchronously here delayed the
    # bind by ~30-60s). The concurrent-import race is handled by the load lock in
    # TextTo3DGenerator._ensure, so a request arriving early is safe.
    def warm_all():
        try:
            import diffusers
            diffusers.AutoPipelineForText2Image  # noqa: B018 — force lazy resolution
        except Exception:
            pass
        warm()

    import threading
    threading.Thread(target=warm_all, daemon=True).start()

_VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer")

# Single-process PoC engine + in-memory job table.
# Large particle budget: the browser studio renders on the GPU (WebGL), so we
# can afford dense objects; the CPU rasterizer (/v1/frame) is only a fallback.
_engine = build_default_engine(gen_points=40000)
_jobs: Dict[str, Dict[str, Any]] = {}
_viewer_sockets: List["WebSocket"] = []
_REAL_JOB_MAX_SECONDS = float(os.environ.get("SPLATRA_REAL_JOB_MAX_SECONDS", "180"))


def _run_real_generation_job(job_id: str, name: str, prompt: str, quality: str) -> None:
    try:
        _jobs[job_id].update({"phase": "generating", "worker_started_at": time.time()})
        field, tag = _gen_object(prompt)
        if _jobs[job_id].get("cancelled"):
            return
        _jobs[job_id].update({"phase": "displaying"})
        _display_field(name, field)
        _jobs[job_id].update({
            "done": True,
            "cache": "real_generator",
            "shape": f"{tag}:{quality}",
            "sgf": _sgf_summary(field),
            "verified": True,
            "hot_swap": True,
            "error": None,
            "phase": "complete",
            "finished_at": time.time(),
        })
    except Exception as exc:
        if _jobs[job_id].get("cancelled"):
            return
        _jobs[job_id].update({
            "done": True,
            "cache": "real_generator_failed",
            "shape": f"real_generator_failed:{type(exc).__name__}",
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            "verified": False,
            "hot_swap": False,
            "phase": "failed",
            "finished_at": time.time(),
        })
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
                "Generate a 3D model of ANY object from a text prompt (e.g. 'a red "
                "apple', 'a pikachu', 'a teapot'). Only set 'shape' when the user "
                "literally asks for a geometric primitive; omit it for real objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "the object, in English"},
                    "shape": {
                        "type": "string",
                        "enum": ["sphere", "cube", "torus", "spiral"],
                        "description": "only for literal geometric primitives",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    # ── scene-authoring tools: build a multi-object 3D explanation as you talk ──
    {
        "type": "function",
        "function": {
            "name": "spawn_object",
            "description": (
                "Add a 3D object to the shared explanation scene (many objects "
                "coexist). Give it a short id so you can place/link it later. "
                "Position is optional — omit it to auto-place on a ring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "the object, in English"},
                    "id": {"type": "string", "description": "short handle, e.g. 'sun'"},
                    "position": {"type": "array", "items": {"type": "number"},
                                 "description": "[x,y,z] world position (optional)"},
                    "label": {"type": "string", "description": "floating caption (optional)"},
                    "scale": {"type": "number"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_object",
            "description": "Move an existing scene object to a world position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "position": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["id", "position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_objects",
            "description": "Draw a particle-strand link (relation/arrow) between two scene objects.",
            "parameters": {
                "type": "object",
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                "required": ["src", "dst"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "label_object",
            "description": "Set an object's floating caption.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_scene",
            "description": "Remove all objects and start a fresh explanation scene.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_scene",
            "description": (
                "Build a WHOLE multi-object 3D explanation in one call: several "
                "objects placed in a shared space, linked by relations. Use this "
                "whenever the user asks to explain/compare/show more than one thing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "object in English"},
                                "id": {"type": "string", "description": "short handle"},
                                "label": {"type": "string"},
                            },
                            "required": ["prompt"],
                        },
                    },
                    "links": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "pairs of ids to connect, e.g. [['sun','earth']]",
                    },
                },
                "required": ["objects"],
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
    if s:
        return s[:48]
    # Non-Latin prompts (피카츄/오리/우주선) all collapsed to the SAME "object"
    # slug — cache entries and rig/parts lookups collided across different
    # generations. A deterministic hash keeps each prompt its own identity.
    import hashlib
    return "obj-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


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
        obj = _slug(prompt)
        explicit_shape = args.get("shape") or (
            next((s for w, s in _SHAPE_WORDS.items() if w in prompt.lower()), None)
        )

        # Unknown object (no sphere/cube/torus/spiral word) -> generate the ACTUAL
        # object: real multi-view 3D (Zero123++ + hull) if enabled, else single-view
        # SD lift, else a procedural placeholder.
        if explicit_shape is None and (_USE_TRIPOSR or _USE_MV or _USE_SD):
            try:
                auto = None
                if _USE_TRIPOSR:
                    field = _triposr().generate(prompt); tag = "triposr-3d"
                elif _USE_MV:
                    field = _mv().generate(prompt); tag = "multiview-3d"
                    auto = getattr(_mv(), "last_score", None)
                else:
                    sd = _sd()
                    model_name = str(getattr(sd, "model", "sd")).rstrip("/\\").split("/")[-1].split("\\")[-1] or "sd"
                    field = sd.generate(prompt); tag = f"{model_name}-3d"
                _display_field(obj, field)
                rec = {"tool": name, "ok": True, "name": obj, "shape": tag,
                       "sgf": _sgf_summary(field), "verified": True, "hot_swap": True}
                if auto is not None:
                    rec["auto_score"] = auto      # silhouette-IoU quality 0-100
                return rec
            except Exception as exc:
                try:  # full chained traceback — the lazy diffusers loader masks root causes
                    import traceback as _tb
                    with open("gen_fail.log", "w", encoding="utf-8") as _f:
                        _f.write(_tb.format_exc())
                except Exception:
                    pass
                # QUALITY LADDER: TripoSR sometimes yields an empty volume for a
                # prompt — fall to the SD silhouette lift (still the real object)
                # before ever surrendering to a procedural sphere.
                if _USE_SD and _USE_TRIPOSR:
                    try:
                        field = _sd().generate(prompt)
                        _display_field(obj, field)
                        return {"tool": name, "ok": True, "name": obj, "shape": "sd-fallback-3d",
                                "sgf": _sgf_summary(field), "verified": True, "hot_swap": True}
                    except Exception:
                        pass
                return {"tool": name, "ok": True, "name": obj,
                        "shape": f"sphere (gen failed: {type(exc).__name__}: {str(exc)[:160]})",
                        **_gen_procedural(obj, prompt, "sphere")}

        shape = explicit_shape or "sphere"
        return {"tool": name, "name": obj, "shape": shape, **_gen_procedural(obj, prompt, shape)}

    if name in ("spawn_object", "place_object", "link_objects", "label_object",
                "clear_scene", "explain_scene"):
        return _scene_tool(name, args)

    return {"tool": name, "ok": False, "error": "unknown tool"}


def _gen_procedural(obj: str, prompt: str, shape: str) -> Dict[str, Any]:
    _engine.generate_3d_object(obj, _prompt_to_mv(prompt), cam_rays={"shape": shape})
    done = _drive_generation(obj)
    rec: Dict[str, Any] = {"ok": done}
    if done and _engine.field is not None:
        rec["sgf"] = _sgf_summary(_engine.field)
        rec["verified"] = bool(_engine.cache[obj].verified)
        rec["hot_swap"] = True
    return rec


# --------------------------------------------------------------------------- #
# UI + introspection
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
@app.get("/studio.html", response_class=HTMLResponse)
@app.get("/studio", response_class=HTMLResponse)
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


class ScoreRequest(BaseModel):
    name: str
    prompt: Optional[str] = None
    score: int                       # human 0-100
    note: Optional[str] = None


_FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "out", "feedback.json")


@app.post("/v1/score")
def submit_score(req: ScoreRequest) -> Dict[str, Any]:
    """Human feedback (0-100) for the last generation. Appended to out/feedback.json.

    These ratings are the human-in-the-loop signal; combined with the automatic
    silhouette-IoU score they let the engine prefer better generations (best-of-N
    is driven by the auto score; human scores accumulate for offline tuning)."""
    import json

    os.makedirs(os.path.dirname(_FEEDBACK_PATH), exist_ok=True)
    rows = []
    if os.path.exists(_FEEDBACK_PATH):
        try:
            rows = json.load(open(_FEEDBACK_PATH, encoding="utf-8"))
        except Exception:
            rows = []
    rows.append({"name": req.name, "prompt": req.prompt,
                 "score": int(max(0, min(100, req.score))),
                 "note": req.note, "ts": time.time()})
    json.dump(rows, open(_FEEDBACK_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    scores = [r["score"] for r in rows]
    return {"ok": True, "count": len(rows), "avg": round(sum(scores) / len(scores), 1)}


@app.get("/v1/state")
def get_state() -> Dict[str, Any]:
    resp: Dict[str, Any] = {"state": _engine.state.value, "edges": len(_engine._edges)}
    if _engine.field is not None:
        resp["sgf"] = _sgf_summary(_engine.field)
    return resp


# --------------------------------------------------------------------------- #
# Live frame (the actual EWA-rendered image)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cartridge side channel: the raw Gaussian buffer for the WebGL viewer.
# This is the PRD §1.2 design — heavy buffers go to the LOCAL viewer on a side
# channel (never to the LLM). The browser renders real 3D from this.
# --------------------------------------------------------------------------- #
def _iso(scale: float, n: int) -> np.ndarray:
    return np.tile(np.array([scale, scale, scale], np.float32), (n, 1))


def _ident_quat(n: int) -> np.ndarray:
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1.0
    return q


def _cartridge_arrays():
    """Build (pos[N,3], col[N,3], scale[N,3] linear, quat[N,4], opa[N]) float32.

    Carries full anisotropy (per-splat ellipsoid scale + rotation) so the WebGL
    viewer can do real anisotropic 3DGS. Objects ship their dense oriented-surfel
    field as-is. Graphs are densified for the viewer only (render-time): each node
    gets an isotropic particle halo and each edge an isotropic strand.
    """
    f = _engine.field
    pos = f.means.astype(np.float32)
    col = np.clip(sh_dc_to_rgb(f.sh[:, 0, :]), 0.0, 1.0).astype(np.float32)
    scale = np.exp(f.scales).astype(np.float32)        # log -> linear [N,3]
    quat = f.quats.astype(np.float32)                   # [N,4] (w,x,y,z)
    opa = (1.0 / (1.0 + np.exp(-f.opacities))).astype(np.float32)

    if not _engine._edges:
        return pos, col, scale, quat, opa               # generated object

    rng = np.random.default_rng(7)
    iso = np.exp(f.scales).mean(axis=1).astype(np.float32)  # node radius proxy
    P = [pos]; C = [col]; S = [scale * 1.5]; Q = [quat]; O = [np.clip(opa, 0.6, 1.0)]
    K = 130  # halo particles per node
    for i in range(pos.shape[0]):
        sig = max(float(iso[i]) * 1.6, 0.03)
        P.append(pos[i] + rng.normal(0, sig, size=(K, 3)).astype(np.float32))
        C.append(np.repeat(col[i][None, :], K, 0))
        S.append(_iso(float(iso[i]) * 0.5, K))
        Q.append(_ident_quat(K))
        O.append(np.full(K, 0.22, np.float32))
    M = 26  # samples per edge strand
    s = np.linspace(0.06, 0.94, M, dtype=np.float32)[:, None]
    for a, b in _engine._edges:
        P.append((pos[a][None] * (1 - s) + pos[b][None] * s).astype(np.float32))
        C.append((col[a][None] * (1 - s) + col[b][None] * s).astype(np.float32))
        S.append(_iso(0.011, M))
        Q.append(_ident_quat(M))
        O.append(np.full(M, 0.5, np.float32))
    return (np.concatenate(P, 0), np.concatenate(C, 0), np.concatenate(S, 0),
            np.concatenate(Q, 0), np.concatenate(O, 0))


def _apply_cartridge_budget(pos, col, scale, quat, opa, budget: Optional[int] = None):
    """Return an importance-sampled cartridge view without mutating the source arrays."""
    n = int(pos.shape[0])
    if budget is None or budget <= 0 or budget >= n:
        return pos, col, scale, quat, opa
    k = max(1, int(budget))
    importance = np.asarray(opa, np.float32) * np.max(np.asarray(scale, np.float32), axis=1)
    chosen = np.argpartition(importance, -k)[-k:]
    chosen.sort()
    return pos[chosen], col[chosen], scale[chosen], quat[chosen], opa[chosen]


def _pack_cartridge(pos, col, scale, quat, opa) -> bytes:
    # magic "SPL2" + uint32 N, then pos[N*3] col[N*3] scale[N*3] quat[N*4] opa[N]
    n = int(pos.shape[0])
    return (
        b"SPL2"
        + struct.pack("<I", n)
        + np.ascontiguousarray(pos, np.float32).tobytes()
        + np.ascontiguousarray(col, np.float32).tobytes()
        + np.ascontiguousarray(scale, np.float32).tobytes()
        + np.ascontiguousarray(quat, np.float32).tobytes()
        + np.ascontiguousarray(opa, np.float32).tobytes()
    )


def _pack_cartridge_spl3(pos, col, scale, quat, opa) -> bytes:
    """Pack a quantized SPL3 cartridge.

    Layout:
      magic "SPL3", uint32 N, bbox_min[3] fp32, bbox_max[3] fp32,
      position int16x3 bbox-normalized, color uint8x3, scale fp16x3,
      quaternion int8x4 snorm, opacity uint8.
    """
    n = int(pos.shape[0])
    pos = np.ascontiguousarray(pos, np.float32)
    col = np.ascontiguousarray(np.clip(col, 0.0, 1.0), np.float32)
    scale = np.ascontiguousarray(np.maximum(scale, 0.0), np.float32)
    quat = np.ascontiguousarray(quat, np.float32)
    opa = np.ascontiguousarray(np.clip(opa, 0.0, 1.0), np.float32)

    bbox_min = pos.min(axis=0).astype(np.float32) if n else np.zeros(3, np.float32)
    bbox_max = pos.max(axis=0).astype(np.float32) if n else np.ones(3, np.float32)
    span = np.maximum(bbox_max - bbox_min, np.float32(1e-8))
    pos_q = np.rint(((pos - bbox_min) / span) * 65535.0 - 32768.0)
    pos_q = np.clip(pos_q, -32768, 32767).astype("<i2", copy=False)

    col_q = np.rint(col * 255.0).astype(np.uint8, copy=False)
    scale_q = scale.astype("<f2", copy=False)
    q_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    q_norm = np.where(q_norm > 1e-8, q_norm, 1.0).astype(np.float32)
    quat_q = np.rint(np.clip(quat / q_norm, -1.0, 1.0) * 127.0).astype(np.int8, copy=False)
    opa_q = np.rint(opa * 255.0).astype(np.uint8, copy=False)

    return (
        b"SPL3"
        + struct.pack("<I", n)
        + bbox_min.astype("<f4", copy=False).tobytes()
        + bbox_max.astype("<f4", copy=False).tobytes()
        + np.ascontiguousarray(pos_q).tobytes()
        + np.ascontiguousarray(col_q).tobytes()
        + np.ascontiguousarray(scale_q).tobytes()
        + np.ascontiguousarray(quat_q).tobytes()
        + np.ascontiguousarray(opa_q).tobytes()
    )


@app.get("/v1/cartridge")
def cartridge(format: str = "spl2", budget: Optional[int] = None) -> Response:
    if _engine.field is None:
        raise HTTPException(status_code=409, detail="nothing rendered yet")
    arrays = _apply_cartridge_budget(*_cartridge_arrays(), budget=budget)
    fmt = format.lower().strip()
    if fmt == "spl3":
        blob = _pack_cartridge_spl3(*arrays)
        out_format = "SPL3"
    elif fmt == "spl2":
        blob = _pack_cartridge(*arrays)
        out_format = "SPL2"
    else:
        raise HTTPException(status_code=400, detail="format must be spl2 or spl3")
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store", "X-SPLATRA-Format": out_format})


# --------------------------------------------------------------------------- #
# Image -> 3D (real LGM path, with an honest procedural fallback).
# --------------------------------------------------------------------------- #
def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    """Decode to [256,256,4] RGBA in [0,1] — alpha is kept (it's the best
    foreground mask for transparent character sprites)."""
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGBA").resize((256, 256))
        return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def _dominant_color(img: np.ndarray) -> Tuple[float, float, float]:
    c = img[..., :3].reshape(-1, 3).mean(axis=0)
    m = float(c.max())
    c = (c / m * 0.9) if m > 1e-3 else np.array([0.5, 0.6, 0.9], dtype=np.float32)
    return tuple(np.clip(c, 0.14, 1.0).tolist())


def _color_mv(color) -> np.ndarray:
    img = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    for ch in range(3):
        img[:, :, ch, :, :] = color[ch]
    return img


def _display_field(name: str, field: GaussianField, verified: bool = True) -> None:
    """Hot-swap a ready field into the engine + pin it as a cartridge."""
    _engine.field = field
    _engine.deformer = FourierDeformer(field.means)
    _engine._edges = []
    cart = _engine.compressor.compress(name, field)
    cart.verified = verified
    _engine.cache[name] = cart
    _engine.state = HoloState.DISPLAYED


# ── multi-object explainer scene (REALTIME_EXPLAINER pillar 1) ────────────────
# Many objects coexist in ONE world — placed apart, linked, labelled — and the
# whole scene composites (Scene.flatten) into the single field the renderer +
# viewer already draw. This is the foundation the LLM authoring loop drives.
_scene = None  # type: ignore[var-annotated]


def _field_for_prompt(prompt: str) -> GaussianField:
    """A GaussianField for one prompt — realistic (TripoSR/MV/SD) when enabled and
    the word isn't a known primitive, else a procedural shape. Returns the field
    WITHOUT displaying it, so many can be composited into a scene."""
    shape = next((s for w, s in _SHAPE_WORDS.items() if w in prompt.lower()), None)
    if shape is None and (_USE_TRIPOSR or _USE_MV or _USE_SD):
        try:
            if _USE_TRIPOSR:
                return _triposr().generate(prompt)
            if _USE_MV:
                return _mv().generate(prompt)
            return _sd().generate(prompt)
        except Exception:
            pass
    _gen_procedural(_slug(prompt), prompt, shape or "sphere")
    return _engine.field.copy()


def _rerender_scene() -> int:
    """Composite the current scene and hot-swap it into the renderer."""
    global _scene
    if _scene is not None and _scene.objects:
        _display_field("scene", _scene.flatten())
        return _scene.version
    return 0


def _scene_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """LLM scene-authoring: spawn/place/link/label/clear mutate the shared scene
    and re-render, so the explanation GROWS as the model talks."""
    from atanor_core.domain.scene import Scene, SceneObject

    global _scene
    if _scene is None:
        _scene = Scene()

    if name == "clear_scene":
        _scene = Scene()
        _engine.field = None
        return {"tool": name, "ok": True, "version": 0}

    if name == "explain_scene":
        # whole multi-object explanation in one call (the LLM's preferred path)
        _scene = Scene()
        objs = args.get("objects") or []
        for i, o in enumerate(objs):
            prompt = str(o.get("prompt", "object"))
            oid = _slug(str(o.get("id") or prompt)) or f"obj{i}"
            f = _field_for_prompt(prompt).copy()
            f.means = (f.means - f.means.mean(axis=0)).astype(np.float32)
            so = SceneObject(id=oid, field=f, scale=float(o.get("scale", 1.0) or 1.0),
                             label=o.get("label"))
            pos = o.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                so.position = np.asarray(pos, np.float32)
            elif len(objs) > 1:
                a = 2 * np.pi * i / max(1, len(objs))
                so.position = np.array([2.3 * np.cos(a), 0.0, 2.3 * np.sin(a)], np.float32)
            _scene.add(so)
        for link in (args.get("links") or []):
            if isinstance(link, (list, tuple)) and len(link) >= 2:
                s, d = _slug(str(link[0])), _slug(str(link[1]))
                if s in _scene.objects and d in _scene.objects:
                    _scene.link(s, d)
        return {"tool": name, "ok": True, "objects": list(_scene.objects),
                "links": len(_scene.links), "version": _rerender_scene()}

    if name == "spawn_object":
        prompt = str(args.get("prompt", "object"))
        oid = _slug(str(args.get("id") or prompt)) or f"obj{len(_scene.objects)}"
        f = _field_for_prompt(prompt).copy()
        f.means = (f.means - f.means.mean(axis=0)).astype(np.float32)
        so = SceneObject(id=oid, field=f, scale=float(args.get("scale", 1.0) or 1.0),
                         label=args.get("label"))
        pos = args.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            so.position = np.asarray(pos, np.float32)
        else:
            i = len(_scene.objects)
            a = 2 * np.pi * i / max(1, i + 1)
            so.position = np.array([2.3 * np.cos(a), 0.0, 2.3 * np.sin(a)], np.float32)
        _scene.add(so)
        return {"tool": name, "ok": True, "id": oid,
                "position": [round(float(x), 2) for x in so.position],
                "version": _rerender_scene(), "objects": list(_scene.objects)}

    if name == "place_object":
        oid = _slug(str(args.get("id", "")))
        pos = args.get("position")
        if oid in _scene.objects and isinstance(pos, (list, tuple)) and len(pos) == 3:
            _scene.move(oid, pos)
            return {"tool": name, "ok": True, "id": oid, "version": _rerender_scene()}
        return {"tool": name, "ok": False, "error": "unknown id or bad position"}

    if name == "link_objects":
        s, d = _slug(str(args.get("src", ""))), _slug(str(args.get("dst", "")))
        if s in _scene.objects and d in _scene.objects:
            _scene.link(s, d)
            return {"tool": name, "ok": True, "src": s, "dst": d, "version": _rerender_scene()}
        return {"tool": name, "ok": False, "error": "unknown src/dst"}

    if name == "label_object":
        oid = _slug(str(args.get("id", "")))
        if oid in _scene.objects:
            _scene.objects[oid].label = str(args.get("text", ""))
            _scene.version += 1
            return {"tool": name, "ok": True, "id": oid}
        return {"tool": name, "ok": False, "error": "unknown id"}

    return {"tool": name, "ok": False, "error": "unknown scene tool"}


class ExplainObject(BaseModel):
    id: Optional[str] = None
    prompt: str
    position: Optional[List[float]] = None
    scale: float = 1.0
    label: Optional[str] = None


class ExplainRequest(BaseModel):
    objects: List[ExplainObject]
    links: List[List[Any]] = []          # [src, dst] or [src, dst, {style}]
    clear: bool = True


@app.post("/v1/explain")
def explain(req: ExplainRequest) -> Dict[str, Any]:
    """Build a multi-object 3D explanation: generate each object, place them in one
    world (explicit position or auto-ring), link them, and render the composite."""
    from atanor_core.domain.scene import Scene, SceneObject

    global _scene
    if req.clear or _scene is None:
        _scene = Scene()
    n = max(1, len(req.objects))
    used_ids: List[str] = []
    for i, o in enumerate(req.objects):
        oid = (o.id or _slug(o.prompt) or f"obj{i}")
        f = _field_for_prompt(o.prompt).copy()
        f.means = (f.means - f.means.mean(axis=0)).astype(np.float32)  # object-local center
        so = SceneObject(id=oid, field=f, scale=float(o.scale), label=o.label)
        if o.position is not None and len(o.position) == 3:
            so.position = np.asarray(o.position, np.float32)
        elif n > 1:
            a = 2 * np.pi * i / n
            so.position = np.array([2.3 * np.cos(a), 0.0, 2.3 * np.sin(a)], np.float32)
        _scene.add(so)
        used_ids.append(oid)
    for link in req.links:
        if len(link) >= 2:
            style = link[2] if len(link) > 2 and isinstance(link[2], dict) else {}
            _scene.link(str(link[0]), str(link[1]), **style)
    field = _scene.flatten()
    _display_field("explain", field)
    return {
        "ok": True,
        "objects": [{"id": oid, "label": _scene.objects[oid].label,
                     "position": [round(float(x), 2) for x in _scene.objects[oid].position]}
                    for oid in used_ids],
        "links": len(_scene.links),
        "n_particles": int(field.num_gaussians),
        "version": _scene.version,
    }


@app.post("/v1/explain/clear")
def explain_clear() -> Dict[str, Any]:
    global _scene
    from atanor_core.domain.scene import Scene

    _scene = Scene()
    return {"ok": True, "version": 0}


# ── learned live rig: predicted joints -> FK pose -> PBD soft-body ───────────
# The generated shell gets a LEARNED skeleton (rig_predictor finds the internal
# joints), pose_chain articulates it, PBD adds flesh. ATANOR drives per-joint
# intents through these endpoints.
_RIG_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "data", "rig_predictor_v1.pt")
_live_rig: Dict[str, Any] = {"key": None}


def _rig_for_current_field() -> Dict[str, Any]:
    from atanor_core.rigging.live_rig import bind_joints, predict_rig_joints
    from atanor_core.rigging.rig_predictor import RigPredictor
    from atanor_core.motion.pbd import SoftBody

    if _engine.field is None:
        raise HTTPException(status_code=409, detail="nothing rendered yet")
    f = _engine.field
    key = (int(f.num_gaussians), float(f.means[0, 0]), float(f.means[-1, 2]))
    if _live_rig.get("key") == key:
        return _live_rig
    if not os.path.exists(_RIG_WEIGHTS):
        raise HTTPException(status_code=503, detail="rig predictor weights missing "
                                                    "(train via rigging/rig_predictor)")
    predictor = RigPredictor.load(_RIG_WEIGHTS)
    home = f.means.copy()
    joints = predict_rig_joints(home, predictor)
    if len(joints) == 0:
        raise HTTPException(status_code=422, detail="no joints found on this shape")
    _live_rig.update({
        "key": key, "home": home, "joints": joints,
        "rig": bind_joints(home, joints), "soft": SoftBody(home),
    })
    return _live_rig


def _auto_chain(rig) -> List[int]:
    """Default drive target: the chain of joints extending farthest out in one
    direction from the body (on a creature that's the tail / dominant limb)."""
    d = np.linalg.norm(rig.joints - rig.centroid, axis=1)
    seed = int(d.argmax())
    seed_dir = rig.outward[seed]
    return [j for j in range(len(rig.joints))
            if float(rig.outward[j] @ seed_dir) > 0.6 and d[j] > 0.25 * d[seed]]


class RigAnimateRequest(BaseModel):
    drive: float = 0.6
    chain: Optional[List[int]] = None    # joint indices; None = auto (outermost chain)
    frames: int = 5                      # PBD frames toward the pose (lag/jiggle)
    reset: bool = False


@app.get("/v1/rig")
def rig_info() -> Dict[str, Any]:
    lr = _rig_for_current_field()
    rig = lr["rig"]
    up = np.array([0.0, 1.0, 0.0], np.float32)
    axes = []
    for j in range(len(rig.joints)):
        a = np.cross(rig.outward[j], up)
        n = np.linalg.norm(a)
        axes.append((a / n if n > 1e-4 else np.array([1.0, 0.0, 0.0])).tolist())
    jd = np.linalg.norm(rig.joints[:, None, :] - rig.joints[None, :, :], axis=2)
    jd[np.arange(len(jd)), np.arange(len(jd))] = np.inf
    reach = float(np.median(jd.min(1))) if len(rig.joints) > 1 else 0.3
    from atanor_core.rigging.live_rig import decompose_chains

    return {
        "ok": True,
        "n_joints": len(lr["joints"]),
        "joints": [[round(float(x), 4) for x in j] for j in lr["joints"]],
        "axes": [[round(float(x), 4) for x in a] for a in axes],
        "outward": [[round(float(x), 4) for x in o] for o in rig.outward],
        "reach": round(reach, 4),
        "centroid": [round(float(x), 4) for x in rig.centroid],
        "chains": decompose_chains(rig),
        "auto_chain": _auto_chain(rig),
        "avatar": _embodiment["avatar"],
        "articulating_particles": int((rig.weight > 0.3).sum()),
        "softbody_constraints": int(lr["soft"].n_constraints),
    }


# The machine's emotion may show ONLY in its own character (the avatar). Fields
# generated to explain things are plain objects: rigged, animatable, but never
# mood-driven. Default: object.
_embodiment = {"avatar": False}


@app.post("/v1/embody")
def embody(body: Dict[str, Any]) -> Dict[str, Any]:
    _embodiment["avatar"] = bool(body.get("avatar", True))
    return {"ok": True, "avatar": _embodiment["avatar"]}


_ato_cache: Dict[str, Any] = {}


def _ato_field() -> GaussianField:
    """Ato is deterministic — build ONCE, then every summon is a memcopy.
    This is the instant-summon contract: no rebuild, no re-generation."""
    if "arrays" not in _ato_cache:
        from atanor_core.avatar.ato import build_ato

        pos, col, scale, quat, opa = build_ato()
        C0 = 0.28209479177387814
        op = np.clip(opa, 1e-4, 1 - 1e-4)
        _ato_cache["arrays"] = dict(
            means=pos,
            scales=np.log(np.maximum(scale, 1e-5)).astype(np.float32),
            quats=quat,
            opacities=np.log(op / (1 - op)).astype(np.float32),
            sh=((np.clip(col, 0, 1) - 0.5) / C0)[:, None, :].astype(np.float32))
    a = _ato_cache["arrays"]
    return GaussianField(means=a["means"].copy(), scales=a["scales"],
                         quats=a["quats"], opacities=a["opacities"],
                         sh=a["sh"], sh_degree=0)


@app.post("/v1/avatar")
def summon_avatar(body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Summon 아토 (Ato) — the machine's own character. Cached-instant after
    first build; marks the field as the AVATAR so the live self's hormones may
    drive its gestures (and ONLY its gestures — Ato never melts)."""
    t0 = time.time()
    field = _ato_field()
    _display_field("ato", field)
    _embodiment["avatar"] = True
    return {"ok": True, "name": "ato", "n": int(field.num_gaussians),
            "avatar": True, "summon_ms": round((time.time() - t0) * 1000, 1)}


_parts_cache: Dict[str, Any] = {"key": None}


@app.get("/v1/parts")
def parts_info() -> Dict[str, Any]:
    """Semantic micro-parts of the CURRENT field (eyes for now) — shape-agnostic
    color-contrast detection; abstains on shapes without such structure."""
    from atanor_core.rigging.parts import find_eyes

    if _engine.field is None:
        raise HTTPException(status_code=409, detail="nothing rendered yet")
    if _engine._edges:
        return {"ok": True, "eyes": []}      # a graph has nodes, not a face
    f = _engine.field
    key = (int(f.num_gaussians), float(f.means[0, 0]))
    if _parts_cache.get("key") != key:
        cols = np.clip(sh_dc_to_rgb(f.sh[:, 0, :]), 0.0, 1.0)
        _parts_cache.update({"key": key, "eyes": find_eyes(f.means, cols)})
    return {"ok": True, "eyes": _parts_cache["eyes"]}


@app.post("/v1/dev/load_sample")
def dev_load_sample(body: Dict[str, Any]) -> Dict[str, Any]:
    """Dev helper: load a viewer sample cartridge into the ENGINE so server-side
    endpoints (/v1/rig, /v1/cartridge, /v1/rig_mood) operate on it."""
    from atanor_core.rigging.live_rig import load_spl2

    name = re.sub(r"[^a-z0-9_]", "", str(body.get("name", "pikachu")).lower())
    path = os.path.join(os.path.dirname(__file__), "..", "viewer", "samples", f"{name}.bin")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"no sample {name}")
    pos, col, scale, quat, opa = load_spl2(path)
    C0 = 0.28209479177387814
    op = np.clip(opa, 1e-4, 1 - 1e-4)
    field = GaussianField(
        means=pos.astype(np.float32),
        scales=np.log(np.maximum(scale, 1e-5)).astype(np.float32),
        quats=quat.astype(np.float32),
        opacities=np.log(op / (1 - op)).astype(np.float32),
        sh=((np.clip(col, 0, 1) - 0.5) / C0)[:, None, :].astype(np.float32),
        sh_degree=0)
    _display_field(name, field)
    return {"ok": True, "name": name, "n": int(field.num_gaussians)}


_ATANOR_SELF_URL = os.environ.get("ATANOR_SELF_URL",
                                  "http://127.0.0.1:8502/api/selfhood/live")
_self_state_cache: Dict[str, Any] = {"state": None, "at": 0.0}


class RigMoodRequest(BaseModel):
    state: Optional[Dict[str, Any]] = None   # selfhood snapshot; None = fetch live
    t: Optional[float] = None
    frames: int = 5
    dry: bool = False                        # True: return params only, no pose


@app.post("/v1/rig_mood")
def rig_mood(req: RigMoodRequest) -> Dict[str, Any]:
    """ATANOR's inner state drives the body: hormones/vitals -> motion params ->
    per-joint intents on the predicted rig. The bridge from feeling to gesture."""
    from atanor_core.rigging.live_rig import pose_joints
    from atanor_core.rigging.mood_drive import chain_drives, motion_params

    state = req.state
    if state is None:
        now_t = time.time()
        cached = _self_state_cache.get("state")
        if cached is not None and now_t - _self_state_cache["at"] < 2.0:
            state = cached                    # poll bursts must not stack
        else:
            try:
                import urllib.request

                with urllib.request.urlopen(_ATANOR_SELF_URL, timeout=3) as r:
                    state = json.loads(r.read().decode("utf-8"))
                _self_state_cache.update({"state": state, "at": now_t})
            except Exception as e:
                raise HTTPException(status_code=503,
                                    detail=f"no state given and ATANOR live state "
                                           f"unreachable: {e}")
    params = motion_params(state)
    hormones = ((state.get("homeostasis") or {}).get("hormones")
                or state.get("hormones") or {})
    if req.dry:
        return {"ok": True, "params": params, "hormones": hormones}
    lr = _rig_for_current_field()
    rig, soft, home = lr["rig"], lr["soft"], lr["home"]
    t = float(req.t) if req.t is not None else time.time() % 10000.0
    intents = chain_drives(params, list(range(len(lr["joints"]))), t)
    target = pose_joints(home, rig, amp=0.0, intents=intents)
    x = soft.settle(target, frames=max(1, min(int(req.frames), 60)))
    f = _engine.field
    f.means = x.astype(np.float32)
    _display_field("rigged", f)
    _live_rig["key"] = (int(f.num_gaussians), float(f.means[0, 0]), float(f.means[-1, 2]))
    mv = np.linalg.norm(x - home, axis=1)
    return {"ok": True, "params": params, "hormones": hormones,
            "moved_mean": round(float(mv.mean()), 4),
            "moved_max": round(float(mv.max()), 3)}


@app.post("/v1/rig_animate")
def rig_animate(req: RigAnimateRequest) -> Dict[str, Any]:
    from atanor_core.rigging.live_rig import pose_chain

    lr = _rig_for_current_field()
    rig, soft, home = lr["rig"], lr["soft"], lr["home"]
    chain = req.chain if req.chain else _auto_chain(rig)
    target = home if req.reset else pose_chain(home, rig, chain, float(req.drive))
    x = soft.settle(target, frames=max(1, min(int(req.frames), 60)))
    f = _engine.field
    f.means = x.astype(np.float32)
    _display_field("rigged", f)
    # the field moved but it is still the SAME body: keep the cached rig bound to
    # the original home/joints instead of re-predicting on the posed shape
    _live_rig["key"] = (int(f.num_gaussians), float(f.means[0, 0]), float(f.means[-1, 2]))
    mv = np.linalg.norm(x - home, axis=1)
    return {
        "ok": True, "chain": chain, "drive": float(req.drive),
        "moved_mean": round(float(mv.mean()), 4), "moved_max": round(float(mv.max()), 3),
        "pbd_lag": round(float(np.linalg.norm(x - target, axis=1).mean()), 4),
    }


def _image_to_field(name: str, img: Optional[np.ndarray]) -> Tuple[str, str, GaussianField]:
    """Return (engine_label, note, field). Real LGM if enabled+available, else
    an honest procedural placeholder tinted by the image's dominant color."""
    global _lgm_gen
    # 0a) Learned single-image 3D (TripoSR) — GPU, opt-in. Highest quality.
    if _USE_TRIPOSR and img is not None:
        try:
            field = _triposr().from_image(img)
            return ("triposr→3d", "Learned 3D reconstruction (TripoSR): one image → "
                    "triplane density+color field → 3D point cloud (fills unseen "
                    "geometry with a trained prior). RTX-class GPU.", field)
        except Exception as exc:
            lgm_note = f"TripoSR failed ({type(exc).__name__}); fell back. "
    # 0b) Real multi-view 3D (Zero123++ + visual hull) — GPU, opt-in.
    if _USE_MV and img is not None:
        try:
            field = _mv().from_cond(img)
            return ("multiview→3d", "Real multi-view 3D: Zero123++ generated novel "
                    "views from your image → visual-hull carve → 3D point cloud "
                    "(asymmetric, all sides). RTX-class GPU.", field)
        except Exception as exc:
            lgm_note = f"Multi-view failed ({type(exc).__name__}); fell back. "
    # 1) Full novel-view LGM — GPU only, opt-in.
    if _USE_LGM and img is not None:
        try:
            if _lgm_gen is None:
                from atanor_core.generation.lgm import LGMGenerator

                _lgm_gen = LGMGenerator()
            field = _lgm_gen.from_image(img)
            return ("lgm", "Reconstructed with LGM (image → 4-view diffusion → "
                    "LGM U-Net → 3DGS).", field)
        except Exception as exc:  # NotImplemented (no GPU/weights) or runtime
            lgm_note = (f"LGM unavailable ({type(exc).__name__}); fell back to the "
                        "CPU 2.5D lift. ")
    else:
        lgm_note = ""

    # 2) Real CPU 2.5D RGBD lift — runs anywhere, no weights.
    if img is not None:
        try:
            from atanor_core.generation.bg import cutout
            from atanor_core.generation.image_lift import Image25DGenerator

            cut = cutout(img)                      # rembg U²-Net cutout if present
            used_cut = cut is not None
            field = Image25DGenerator().from_image(cut if used_cut else img)
            note = (lgm_note
                    + ("Background removed (rembg U²-Net) → " if used_cut else "")
                    + "real CPU image→3D: silhouette inflation → closed oriented-surfel "
                    "3DGS volume. Honest: a single-view lift (visible relief inflated), "
                    "not novel-view synthesis (that is the GPU LGM path, SPLATRA_LGM=1).")
            return ("rgbd-lift(2.5D)", note, field)
        except Exception as exc:
            lgm_note += f"2.5D lift failed ({type(exc).__name__}: {str(exc)[:100]}). "

    # 3) Last resort: procedural placeholder tinted by the image.
    color = _dominant_color(img) if img is not None else (0.6, 0.6, 0.7)
    field = _engine.generator.generate(_color_mv(color), cam_rays={"shape": "sphere"})
    return ("mock(procedural)", lgm_note + "Procedural placeholder.", field)


@app.post("/v1/generate_from_image")
async def generate_from_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await image.read()
    img = _decode_image(raw)
    name = _slug(image.filename or "image") or "image"
    engine_label, note, field = _image_to_field(name, img)
    _display_field(name, field)
    resp = {
        "status": "displayed",
        "engine": engine_label,
        "note": note,
        "name": name,
        "state": _engine.state.value,
        "sgf": _sgf_summary(field),
    }
    if engine_label.startswith("multiview") and _mv_gen is not None:
        resp["auto_score"] = getattr(_mv_gen, "last_score", None)
    return resp


# --------------------------------------------------------------------------- #
# Object generation helper (picks the best enabled generator) + JARVIS narrate.
# --------------------------------------------------------------------------- #
def _gen_object(prompt: str):
    """(field, tag) for a text prompt using the best available generator."""
    try:
        from atanor_core.generation.materials import glass_orb_field, looks_like_glass_orb

        if looks_like_glass_orb(prompt):
            return glass_orb_field(), "real_generator_material_glass_orb"
    except Exception:
        pass
    if _USE_TRIPOSR:
        return _triposr().generate(prompt), "triposr-3d"
    if _USE_MV:
        return _mv().generate(prompt), "multiview-3d"
    if _USE_SD:
        sd = _sd()
        model_name = str(getattr(sd, "model", "sd")).rstrip("/\\").split("/")[-1].split("\\")[-1] or "sd"
        return sd.generate(prompt), f"{model_name}-3d"
    shape = detect_shape(prompt)
    return _engine.generator.generate(_prompt_to_mv(prompt), cam_rays={"shape": shape}), shape


class NarrateRequest(BaseModel):
    prompt: str                      # the subject to build, e.g. "pikachu"
    topic: Optional[str] = None      # the explanation; defaults to prompt
    model: Optional[str] = None      # ollama model for the director (None -> heuristic)
    lang: str = "ko"


@app.post("/v1/narrate")
def narrate(req: NarrateRequest) -> Dict[str, Any]:
    """Return a time-synced narration SCRIPT. The script's spawn/move actions
    build & animate a multi-object scene; the browser plays it (TTS + actions +
    /v1/scene/* calls). The scene is cleared so the script starts fresh."""
    from atanor_core.llm.director import make_script

    _scene.objects.clear(); _scene.links.clear(); _scene.version += 1   # fresh scene
    d = make_script(req.topic or req.prompt, lang=req.lang, ollama_model=req.model)
    return {"director": d["engine"], "script": d["script"]}


# --------------------------------------------------------------------------- #
# Multi-object scene (Phase 1) — objects placed in shared space, flattened to
# one field the existing viewer renders. See docs/REALTIME_EXPLAINER.md.
# --------------------------------------------------------------------------- #
from atanor_core.domain.scene import Scene, SceneObject  # noqa: E402

_scene = Scene()


def _scene_display():
    """Flatten the scene into the engine field so /v1/cartridge serves it."""
    _engine.field = _scene.flatten()
    _engine.deformer = FourierDeformer(_engine.field.means)
    _engine._edges = []
    _engine.state = HoloState.DISPLAYED


class SpawnRequest(BaseModel):
    prompt: str
    id: Optional[str] = None
    position: Optional[List[float]] = None
    scale: float = 1.0
    label: Optional[str] = None


class MoveRequest(BaseModel):
    id: str
    position: List[float]


@app.post("/v1/scene/spawn")
def scene_spawn(req: SpawnRequest) -> Dict[str, Any]:
    field, tag = _gen_object(req.prompt)
    oid = req.id or _slug(req.prompt) or f"obj{len(_scene.objects)}"
    pos = np.array(req.position, np.float32) if req.position else None
    _scene.add(SceneObject(id=oid, field=field,
                           position=pos if pos is not None else np.zeros(3, np.float32),
                           scale=float(req.scale), label=req.label))
    if req.position is None:
        _scene.auto_layout()
    _scene_display()
    return {"ok": True, "id": oid, "engine": tag, "objects": list(_scene.objects),
            "sgf": _sgf_summary(_engine.field)}


@app.post("/v1/scene/move")
def scene_move(req: MoveRequest) -> Dict[str, Any]:
    if req.id not in _scene.objects:
        raise HTTPException(404, "no such object")
    _scene.move(req.id, req.position); _scene_display()
    return {"ok": True, "sgf": _sgf_summary(_engine.field)}


@app.post("/v1/scene/clear")
def scene_clear() -> Dict[str, Any]:
    _scene.objects.clear(); _scene.links.clear(); _scene.version += 1
    return {"ok": True}


@app.get("/v1/scene")
def scene_list() -> Dict[str, Any]:
    return {"version": _scene.version,
            "objects": [{"id": o.id, "position": o.position.tolist(),
                         "scale": o.scale, "label": o.label,
                         "num_gaussians": o.field.num_gaussians}
                        for o in _scene.objects.values()],
            "links": [{"src": s, "dst": d} for s, d, _ in _scene.links]}


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
    material_glass_orb = False
    try:
        from atanor_core.generation.materials import looks_like_glass_orb

        material_glass_orb = looks_like_glass_orb(req.prompt)
    except Exception:
        material_glass_orb = False
    quality = (req.quality or "fast").lower().strip()
    quality_wants_real = quality in {"gpu", "realistic", "high", "learned"}
    # A prompt word like "orb" or "ball" is not enough to force the procedural
    # primitive path. For realistic/gpu generation, only the explicit API
    # `shape` parameter means "use a primitive"; otherwise the text-to-3D path
    # gets the first chance. This keeps ATANOR direct-generation requests from
    # silently degrading into low-density mock shapes.
    shape = None if material_glass_orb else req.shape
    real_generator_available = _USE_TRIPOSR or _USE_MV or _USE_SD
    wants_real_generator = material_glass_orb or (shape is None and quality_wants_real)
    if wants_real_generator and (_USE_TRIPOSR or _USE_MV or _USE_SD):
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {
            "name": name,
            "done": False,
            "cache": "real_generator_pending",
            "shape": f"real_generator:{quality}",
            "error": None,
            "phase": "queued",
            "created_at": time.time(),
            "timeout_seconds": _REAL_JOB_MAX_SECONDS,
        }
        threading.Thread(
            target=_run_real_generation_job,
            args=(job_id, name, req.prompt, quality),
            daemon=True,
        ).start()
        return {
            "status": "generating",
            "job_id": job_id,
            "name": name,
            "shape": f"real_generator:{quality}",
            "cache": "real_generator_pending",
            "eta_seconds": 30,
            "poll": f"/v1/job/{job_id}",
        }
    if shape is None and (not quality_wants_real or not real_generator_available):
        shape = next((s for w, s in _SHAPE_WORDS.items() if w in req.prompt.lower()), None)
    result = _engine.generate_3d_object(name, _prompt_to_mv(req.prompt), cam_rays={"shape": shape})
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"name": name, "done": result == "hit", "cache": result, "shape": shape}
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
    cache_state = str(job.get("cache") or "")
    events: List[Dict[str, Any]] = []
    if cache_state.startswith("real_generator"):
        done = bool(job.get("done"))
        created = float(job.get("created_at") or time.time())
        elapsed = max(0.0, time.time() - created)
        timeout = float(job.get("timeout_seconds") or _REAL_JOB_MAX_SECONDS)
        if not done and timeout > 0 and elapsed > timeout:
            job.update({
                "done": True,
                "cache": "real_generator_timeout",
                "shape": f"real_generator_timeout:{cache_state}",
                "phase": "timeout",
                "error": f"real generator exceeded {timeout:.0f}s without completing",
                "verified": False,
                "hot_swap": False,
                "cancelled": True,
                "finished_at": time.time(),
            })
            done = True
    else:
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
        "cache": job.get("cache"),
        "shape": job.get("shape"),
        "phase": job.get("phase"),
    }
    if job.get("created_at"):
        resp["elapsed_seconds"] = round(max(0.0, time.time() - float(job["created_at"])), 3)
    if job.get("timeout_seconds"):
        resp["timeout_seconds"] = job.get("timeout_seconds")
    if job.get("error"):
        resp["error"] = job.get("error")
    if done and cache_state == "real_generator" and isinstance(job.get("sgf"), dict):
        resp["sgf"] = job.get("sgf")
        resp["verified"] = bool(job.get("verified"))
        resp["hot_swap"] = bool(job.get("hot_swap"))
    elif done and _engine.field is not None and name in _engine.cache:
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
