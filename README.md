# atanor-hologram-core

> Standalone 3D Gaussian **hologram + generation** engine, LLM-plugin ready.
> Visualize a knowledge graph as a 3D Gaussian-particle hologram, generate 3D
> objects on demand, and hot-swap them into a local browser viewer — without
> ever shipping heavy 3D buffers to the LLM.

`v0.1.0` · scratch build · **CPU/numpy PoC** (no torch / gsplat / CUDA / model
downloads required to run the tests or the demo).

---

## 1. Intellectual-honesty disclaimer (read this first)

This repo is built to be **honest about what works and what is a stub.** Nothing
mock is dressed up as real.

- **"0ms" is a cache-HIT-only asymptotic target.** A cache hit goes straight to
  `DISPLAYED` with no generation thread. A cache **miss** runs a background
  generation thread (~5s in a real LGM setup) while a coalescence morph
  animates; the *hot-swap itself* is what is ~0ms. The code structure enforces
  this split (`HologramEngine.generate_3d_object` / `tick`).
- **`CPURasterizer` is real EWA splatting, not a mock.** It runs the actual
  3DGS math on the CPU in numpy: anisotropic 3D covariance → projected 2D conic
  → front-to-back alpha compositing with transmittance, plus supersampled AA.
  It is the *same algorithm* `gsplat` runs on the GPU — the only trade-off vs
  `GsplatRasterizer` is **speed, not technique** (~50–85 ms/frame at 480px).
- **`LGMGenerator` is real image→3D wiring** (`generation/lgm.py`), not a stub:
  single image → multi-view diffusion (ImageDream) → LGM U-Net → 14-channel
  3DGS → `GaussianField`, with **sequential VRAM offload** (the two big models
  never co-reside) and **INT8/INT4 quantization** (`generation/quant.py`,
  bitsandbytes) to hold ≤4GB. **Caveat (honest):** it needs a CUDA GPU, the
  `gen` extra, and a one-time weight download — none of which exist in the
  CPU/numpy PoC, so this path is **not exercised by `pytest` and has not been
  run here**. It is enabled with `SPLATRA_LGM=1`; otherwise the API falls back to
  the procedural mock with an explicit note (it never silently pretends). Model
  IDs / activation conventions must be re-verified at wiring time.
- **`Image25DGenerator` is a real CPU image→3D** (`generation/image_lift.py`):
  a 2.5D RGBD lift (numpy + Pillow, no weights, ~20ms) — foreground keying,
  relief-depth estimate, normals from the depth gradient, unprojection to
  oriented-surfel Gaussians + a dim back-shell for volume. It is the default
  image→3D path (runs anywhere). **Honest:** it reconstructs the *visible relief*
  (silhouette-accurate, rounded), not unseen geometry — that's the GPU LGM path.
- **`MockGenerator` is the CPU default** for text prompts — a *procedural* oriented-surfel shape
  (sphere/cube/torus/spiral, Lambert-lit, color-word tinted). Explicitly **not**
  image-to-3D reconstruction; it gives the state machine a plausibly-3D field.
- **`MockCodec`** — identity compression with an **honest size estimate** (not a
  real encode). Real path: Self-Organizing Gaussians + LightGaussian.
- **The local LLM really drives the engine.** `OllamaClient` calls a local
  Ollama server; if the model lacks native tool support it falls back to
  *prompted JSON mode* (`format:"json"`), and if Ollama is down it falls back to
  an offline `HeuristicLLM` (a rule-based intent router, **not** a language
  model). The chat response names which route actually ran.
- **`Tabularis` and `MiroFish` are spec-undefined placeholders.** ATANOR names
  them but does not define them. They exist only as empty abstract ports in
  `ports/`. They are **not** implemented — guessing their behavior would bake a
  wrong assumption into the architecture. An adapter gets plugged in when a spec
  is provided.
- **The PSNR gate is real** (it actually re-renders held-out views and computes
  PSNR). DePIN nodes are untrusted, so any distributed-refinement result must
  pass this client-side gate before display.
- **The SGF unified framing is an unpublished frame.** Its constituent parts
  (3DGS rasterization, Gaussian-Flow-style deformation, LGM, SOG, SDS, PSNR
  verification) are all published, validated techniques. If the unifying frame
  is wrong, the system still works as the sum of its parts.

---

## 2. Quick start

```bash
pip install -e ".[dev]"
pytest -q                      # target: 7 passed (CPU/numpy only, no torch)
python scripts/demo_render.py  # headless pipeline -> out/*.ppm (3 frames)

pip install -e ".[api]"
./scripts/run_api.sh           # then open http://localhost:8000  (chat studio)
```

### Hologram Studio (chat UI)

Open `http://localhost:8000` after starting the API. The left panel is a **real
WebGL2 3D Gaussian point cloud** (tens of thousands of particles) that you
**orbit (drag), zoom (scroll), and pan (right-/shift-drag)** — an actual 3D
model rendered on your GPU, not server-side PNG frames. The browser pulls the
Gaussian buffer once per hot-swap on a side channel (`/v1/cartridge`) and
renders it locally. Type what you want and a local LLM turns it into tool-calls:

- “show a knowledge graph with 30 nodes” → graph hologram (node halos + edge strands)
- “generate a blue torus”, “make a red cube”, “a gold spiral” → dense 3D object
- **drop an image** → real image→3DGS: a CPU **2.5D RGBD lift** runs everywhere
  (foreground key → relief depth → normals → oriented surfels + back-shell, ~20ms,
  no weights); the full novel-view GPU **LGM** path is opt-in (`SPLATRA_LGM=1`)

The viewer is a **real anisotropic 3D Gaussian Splatting rasterizer**: each
splat's 3D covariance (from its scale + quaternion) is projected to a 2D conic
in the vertex shader (EWA), eigen-decomposed to an oriented screen-space
ellipse, and composited **back-to-front** using a per-frame 16-bit counting
**depth sort** — so there are no blending-order artifacts. Objects are emitted
as oriented surfels (disks tangent to the surface). Particles **assemble** (fly
in + reassemble) on load; **⟳ reassemble** replays it, **∞ auto-cycle** loops
disassemble↔reassemble, **size**/**▶ spin** are in the top-right.

> Honesty note: this is genuine anisotropic EWA splatting (the antimatter15/
> gsplat math), the same technique the CPU `CPURasterizer` and the GPU
> `GsplatRasterizer` compute — running in WebGL2 on your GPU. The depth sort is
> exact (back-to-front, every frame the camera moves). The assemble/disassemble
> motion is a client-side scatter↔home interpolation in the vertex shader, in
> the spirit of the SGF morph.

Pick an **Ollama** model from the dropdown to drive it with your local LLM
(works even with models that lack native tool-calling, via prompted JSON mode).
With no Ollama running it transparently uses the offline heuristic router, so
the UI always works. Color words (“blue”, “red”, … incl. Korean) are honored.

### Local LLM (Ollama)

```bash
ollama serve            # start the local server (default :11434)
ollama pull llama3.1    # any model; tool-capable models use the native API,
                        # others fall back to prompted JSON mode automatically
```

## 3. How it works (Spectral Gaussian Field)

Every visual state — static graph, dynamic morph, generated object — is a single
set of Gaussian primitives whose evolution is a spectral series:

```
a(t) = a_DC                                    # static (DC term)
     + Σ_n poly_n · t^n                         # polynomial drift (relayout)
     + anneal(t) · Σ_k [α_k sin + β_k cos]      # annealed Fourier turbulence
```

The DC term is the static field; AC terms are the morph. The same Gaussian
buffer and the same rasterizer render all three cases. `Δa(t)` is a matrix
multiply — sub-millisecond, **zero neural-network inference**.

State machine (cache miss):

```
IDLE → GENERATING (~5s, morph animates)
     → [VERIFYING]  (DePIN only: held-out PSNR gate)
     → SWAP_READY   (verified cartridge pinned, hot-swap signal)
     → DISPLAYED    (0ms swap done, field live)
```

## 4. Architecture (Ports & Adapters)

`domain/` (pure SGF, numpy only) and `ports/` (abstract interfaces) know nothing
about torch/gsplat — so the AI-OS kernel can import them without GPU deps. The
state machine talks only to ports; mock PoC adapters and real GPU adapters are
interchangeable.

| Port | PoC adapter | Real adapter (future) |
|---|---|---|
| `RasterizerPort` | `CPURasterizer` (real EWA, CPU) | `GsplatRasterizer` (gsplat GPU) |
| `GeneratorPort` | `MockGenerator` (procedural) | `LGMGenerator` (Zero123++ → LGM) |
| `CompressorPort` | `MockCodec` | SOG + LightGaussian |
| `VerifierPort` | `PSNRGate` (real) | same + DePIN round-trip |
| `LLMPort` | `HeuristicLLM` / `OllamaClient` | any tool-capable LLM |
| `VectorIndexPort` | — | `turbovec_rs` (Rust, pyo3) |
| `TabularisPort` | **placeholder** | spec undefined |
| `MiroFishPort` | **placeholder** | spec undefined |

## 5. LLM plugin contract

Tool responses **never** contain raw 3D buffers — only an SGF summary
(`num_gaussians`, `sh_degree`, `raw_bytes`, `bbox`), a cartridge handle, and a
hot-swap signal. The local viewer pulls the cartridge on a side channel.

- `GET /` — the Hologram Studio chat UI.
- `GET /tools` — OpenAI function-calling schema.
- `GET /v1/models` — installed Ollama models (`[]` if Ollama is offline).
- `POST /v1/chat` — local-LLM tool-calling loop; runs the chosen tool, returns
  the assistant text, action records, engine state, and SGF summary.
- `GET /v1/cartridge` — the raw Gaussian buffer (binary `SPL2`: positions,
  colors, **scale[3] + quaternion[4]** for full anisotropy, opacities) for the
  WebGL viewer. Graphs are densified (node halos + edge strands) for the viewer
  only. **This is the side channel; never the LLM.**
- `POST /v1/generate_from_image` — upload an image → `LGMGenerator` (if
  `SPLATRA_LGM=1` + GPU) or the honest procedural fallback; hot-swaps the field.
- `GET /v1/frame?yaw&pitch&dist&w&h` — a server-side EWA-rendered PNG (CPU
  fallback / the OpenAI-plugin contract; the studio uses WebGL instead).
- `POST /v1/render_knowledge_hologram` — graph → SGF summary + cartridge handle.
- `POST /v1/generate_3d_object` — returns immediately with a `job_id`.
- `GET /v1/job/{id}` — advances one tick; reports events + done + SGF summary.
- `WS /ws/viewer` — hot-swap signals + SGF deltas.

## 6. Layout

```
src/atanor_core/   domain (SGF) · ports · mapping · deformation
                   · generation (MockGenerator, LGMGenerator, quant) · compression
                   · verification · state (rasterizer, machine) · llm (heuristic, ollama)
apps/plugin_api.py FastAPI + tools schema + chat loop + image→3D + cartridge
viewer/studio.html Hologram Studio: anisotropic WebGL2 3DGS viewer + chat, at /
viewer/index.html  minimal standalone 2D-splat demo page
scripts/           demo_render.py (headless) · run_api.sh
rust/turbovec_rs/  Rust vector-indexer stub (pyo3+maturin planned)
tests/             7 end-to-end tests (CPU only)
```

## 7. License / status

Scratch build for ATANOR. Not production-ready. See §1 before drawing any
performance or capability conclusions.
