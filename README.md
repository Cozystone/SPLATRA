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
- **The PoC adapters are mocks**, clearly labeled:
  - `CPURasterizer` — pure-numpy painter splat. For *coverage / end-to-end
    tests*, **not** render quality. Real quality needs `GsplatRasterizer`
    (torch + gsplat).
  - `MockGenerator` — average-color spherical blob. **Not** real image-to-3D.
    Real path is `LGMGenerator` (Zero123++ → LGM, lazy/heavy, ≤4GB sequential
    offload) — left as a `NotImplementedError`-marked slot.
  - `MockCodec` — identity compression with an **honest size estimate** (not a
    real encode). Real path: Self-Organizing Gaussians + LightGaussian.
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
./scripts/run_api.sh           # plugin API on :8000; open viewer/index.html
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
| `RasterizerPort` | `CPURasterizer` | `GsplatRasterizer` (gsplat) |
| `GeneratorPort` | `MockGenerator` | `LGMGenerator` (Zero123++ → LGM) |
| `CompressorPort` | `MockCodec` | SOG + LightGaussian |
| `VerifierPort` | `PSNRGate` (real) | same + DePIN round-trip |
| `VectorIndexPort` | — | `turbovec_rs` (Rust, pyo3) |
| `TabularisPort` | **placeholder** | spec undefined |
| `MiroFishPort` | **placeholder** | spec undefined |

## 5. LLM plugin contract

Tool responses **never** contain raw 3D buffers — only an SGF summary
(`num_gaussians`, `sh_degree`, `raw_bytes`, `bbox`), a cartridge handle, and a
hot-swap signal. The local viewer pulls the cartridge on a side channel.

- `GET /tools` — OpenAI function-calling schema.
- `POST /v1/render_knowledge_hologram` — graph → SGF summary + cartridge handle.
- `POST /v1/generate_3d_object` — returns immediately with a `job_id`.
- `GET /v1/job/{id}` — advances one tick; reports events + done + SGF summary.
- `WS /ws/viewer` — hot-swap signals + SGF deltas.

## 6. Layout

```
src/atanor_core/   domain (SGF) · ports · mapping · deformation · generation
                   · compression · verification · state (rasterizer, machine)
apps/plugin_api.py FastAPI + OpenAI tools schema
viewer/index.html  lightweight 2D-splat viewer (replace with wgpu/WebGL 3DGS)
scripts/           demo_render.py (headless) · run_api.sh
rust/turbovec_rs/  Rust vector-indexer stub (pyo3+maturin planned)
tests/             7 end-to-end tests (CPU only)
```

## 7. License / status

Scratch build for ATANOR. Not production-ready. See §1 before drawing any
performance or capability conclusions.
