# SPLATRA — LLM-driven real-time 3D particle explainer (design)

> Goal: an LLM **thinks and, at the same time, drives our particle models** to
> build and manipulate a live 3D explanation. Objects live in a **shared 3D
> space** (placed apart, linked, moved, morphed) and the whole scene is authored
> *as the model talks* — like a presenter conjuring a 3D infographic in mid-air.

This is the north star above the current single-object viewer. It has three
pillars: **(1) a multi-object spatial scene**, **(2) an LLM that streams scene
edits while reasoning**, and **(3) generation fast enough to keep up**.

---

## 1. Multi-object spatial scene (free use of space)

Today the viewer shows one Gaussian field. The explainer needs **many** objects
coexisting in one world, each with its own transform — so "사과" and "나무" can
sit apart, an arrow can link them, a label can float above.

**Scene model (`domain/scene.py`):**

```python
@dataclass
class SceneObject:
    id: str
    field: GaussianField          # the object's own particles (object-local coords)
    position: np.ndarray          # [3] world translation
    rotation: np.ndarray          # [4] quaternion
    scale: float                  # uniform world scale
    opacity: float = 1.0          # whole-object fade (spawn/despawn)
    label: Optional[str] = None   # floating caption
    meta: dict = field(default_factory=dict)

@dataclass
class Scene:
    objects: dict[str, SceneObject]     # id -> object
    links: list[tuple[str, str, dict]]  # (src_id, dst_id, style) — relations/arrows
    camera: dict                        # target, yaw, pitch, dist (LLM can steer)
    version: int                        # bumped on every edit (for delta sync)
```

The **cartridge** generalizes from one object to a **scene cartridge**: a header
+ a table of objects (each: transform + its SPL2 blob) + links. The viewer
renders every object's particles transformed by its world matrix into one
depth-sorted draw (the renderer already does anisotropic + sort; we just prepend
a per-object model matrix in the vertex shader, fed via the data texture).

**Auto-layout:** when the LLM spawns N objects without positions, place them on a
grid / circle / force-directed layout (reuse the knowledge-graph mapper's idea).
The LLM can override any position.

---

## 2. The LLM streams scene edits while it reasons

The LLM is given **scene-authoring tools** and told to narrate *and* call them as
it goes. We run a streaming loop: tokens of reasoning come back interleaved with
tool-calls; each tool-call is applied to the `Scene` immediately and pushed to
the viewer over a WebSocket, so the scene **grows as the model speaks**.

**Tool schema (the LLM's "hands"):**

| tool | effect |
|---|---|
| `spawn_object(prompt\|image, id, position?)` | generate a 3DGS object (TripoSR/SD) and add it to the scene |
| `place(id, position)` / `move(id, to, ms)` | set / animate an object's world position |
| `morph(id, prompt)` | regenerate the object and **disassemble→reassemble** in place (we already have this animation) |
| `transform(id, scale?, rotation?)` | resize / rotate |
| `link(src, dst, style)` | draw a particle arrow/relation between two objects |
| `label(id, text)` / `annotate(position, text)` | floating captions |
| `highlight(id)` / `dim(others)` | focus attention (opacity + glow) |
| `focus_camera(id\|position, ms)` | move the camera to frame something |
| `despawn(id)` | dissolve an object into particles and remove it |

**Streaming transport:** `WS /ws/scene` pushes `{op, args, version}` deltas;
`GET /v1/scene` returns the full scene for late joiners. The viewer keeps a local
`Scene` and applies deltas with the existing morph/scatter animations, so every
edit is *animated*, never a hard cut.

**Two clocks (the key idea).** The LLM's reasoning is the *fast* clock (tokens,
sub-second). Object **generation** is the *slow* clock (TripoSR ~20s). So
`spawn_object`/`morph` return immediately with a **placeholder** (a labelled
particle cloud at the target spot, swirling), and swap to the real object when
generation finishes — the LLM never blocks, it keeps narrating and arranging
while objects "develop" in the background. This is exactly the engine's existing
cache-miss state machine (`GENERATING → SWAP_READY → DISPLAYED`), lifted to
per-object in the scene.

---

## 3. Generation performance (keep up with thought)

To make objects appear fast enough that the explanation feels live:

- **Cache + content-addressing.** Key cartridges by `(prompt, quality)`; a repeat
  is a 0ms cache hit (already the design). Persist the cache to disk so common
  objects ("apple", "arrow", "cube") are instant across sessions.
- **Pre-warm & keep-hot.** Load TripoSR/SD-Turbo once at server start (done) and
  keep them resident; the first call already pays the load.
- **Progressive (coarse→fine).** Emit a **coarse** object immediately — a fast
  SD-Turbo single-view *volumetric* cloud (~2s) as the placeholder — then upgrade
  it in place to the **TripoSR** learned mesh (~20s) via the morph animation. The
  viewer shows something instantly and sharpens it.
- **Async background workers.** A small job queue on the GPU: the LLM fires
  `spawn`/`morph` jobs; a worker drains them; results hot-swap into the scene.
  The RTX 5080 (16GB) can hold SD-Turbo + TripoSR resident together (~6GB) and
  still batch.
- **Best-of-N only when it matters.** Use the auto silhouette-IoU score to pick
  the best candidate (already built) — but run N in parallel on the GPU only for
  "hero" objects the LLM marks important.
- **Lighter cartridges.** Down-sample distant/secondary objects (the
  `VectorIndexPort`/`turbovec_rs` slot: cull particles by query relevance), so a
  scene of 20 objects stays under the viewer's particle budget.

---

## 4. Phased plan

1. **Scene model + multi-object renderer.** `domain/scene.py`, scene cartridge
   format, per-object model matrix in the WebGL vertex shader, `GET /v1/scene`.
   (Single object becomes a scene of one — backward compatible.)
2. **Scene edit API + WS deltas.** `spawn/place/move/morph/link/label/focus` →
   `Scene` mutations → `WS /ws/scene`. Viewer applies animated deltas.
3. **LLM authoring loop.** Give the LLM the tool schema; stream reasoning +
   tool-calls; placeholder-then-upgrade per object. (Ollama tool-calling / the
   prompted-JSON path already works.)
4. **Progressive generation + job queue.** Coarse SD-Turbo placeholder →
   TripoSR upgrade; disk cache; relevance-based particle culling.
5. **Polish.** Auto-layout, arrows/labels as particle primitives, camera
   choreography synced to narration, export the scene as a shareable cartridge.

The current repo already provides every primitive this needs: the Gaussian
field + SPL2 cartridge, the anisotropic depth-sorted WebGL renderer, the
disassemble→reassemble morph, the cache-miss state machine, TripoSR/multi-view/
SD generation, the auto-score, and the LLM tool-calling. The explainer is those
parts **composed into a scene graph driven by a streaming LLM**.
