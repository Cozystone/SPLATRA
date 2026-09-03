"""Text-prompted segmentation (CLIPSeg) — the "where is the glass?" layer.

The generator returns one undifferentiated crust. Nothing in it knows that the
upper middle of a car is a windscreen rather than paint, so the windows come out
as opaque as the doors and the cabin we so carefully assembled is sealed inside a
shell nobody can see through.

CLIPSeg answers that question directly: give it the rendered image and the phrase
"car window" and it returns a heat map of where that is. It is a genuine
vision-language model (CLIP backbone, ~150MB, ~0.7s on this GPU), so it generalises
to whatever the object happens to be — "glass", "screen", "lens", "windscreen" —
instead of relying on a hand-written list of where windows usually sit.

The mask is 2D. We lift it by projecting each 3D point back onto the (frontal)
image the reconstruction came from and sampling there, which is approximate but
correct enough to mark a window band through the whole body.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

_MODEL = os.environ.get("SPLATRA_CLIPSEG_MODEL", "CIDAS/clipseg-rd64-refined")

# Phrases that reliably localise see-through material across many objects.
# Measured on generated car renders: "windshield"/"windscreen"/"the windows of the
# car" score ~0.42-0.49 where "glass window" scores 0.04, so phrasing matters more
# than adding synonyms.
# Generic phrases like "transparent glass" fire on any shiny surface — they marked
# 33% of an apple as glass — so only object-specific window phrasing is used.
GLASS_PROMPTS = ("windshield", "car windscreen", "the windows of the car",
                 "window glass of a building", "a window pane in a frame")

_state: Dict[str, object] = {"proc": None, "model": None, "dev": None}


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _ensure():
    if _state["model"] is not None:
        return
    import torch
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPSegProcessor.from_pretrained(_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(_MODEL).to(dev).eval()
    _state.update({"proc": proc, "model": model, "dev": dev, "torch": torch})


def segment(image_rgb: np.ndarray, prompts: List[str]) -> np.ndarray:
    """[H,W,3] in [0,1] + phrases -> [len(prompts), H, W] masks in [0,1]."""
    _ensure()
    import torch
    from PIL import Image

    h, w = image_rgb.shape[:2]
    img = Image.fromarray((np.clip(image_rgb[..., :3], 0, 1) * 255).astype(np.uint8))
    proc, model, dev = _state["proc"], _state["model"], _state["dev"]
    with torch.no_grad():
        inp = proc(text=list(prompts), images=[img] * len(prompts),
                   padding=True, return_tensors="pt").to(dev)
        logits = model(**inp).logits
        if logits.ndim == 2:                      # single prompt comes back squeezed
            logits = logits[None]
        m = torch.sigmoid(logits)
        m = torch.nn.functional.interpolate(
            m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return m.detach().cpu().numpy().astype(np.float32)


def glass_mask(image_rgb: np.ndarray, extra: Optional[List[str]] = None,
               threshold: float = 0.28) -> Optional[np.ndarray]:
    """[H,W] confidence that each pixel is see-through material, or None.

    Returns None when nothing convincing is found, so callers can skip the whole
    transparency path rather than tinting an object that has no glass in it.
    """
    if not available():
        return None
    try:
        prompts = list(GLASS_PROMPTS) + list(extra or [])
        masks = segment(image_rgb, prompts)
    except Exception:
        return None
    best = masks.max(axis=0)
    if float(best.max()) < threshold:
        return None
    # keep only the confident core; CLIPSeg's tails bleed over the whole silhouette
    out = np.clip((best - threshold) / max(1e-3, 1.0 - threshold), 0.0, 1.0)
    hit = float((out > 0.1).mean())
    # A window is a region, not the whole object. Anything covering a quarter of the
    # frame is the model agreeing that a shiny surface looks glassy, not a window.
    if hit < 0.0015 or hit > 0.25:
        return None
    return out.astype(np.float32)


# ── full material palette ───────────────────────────────────────────────────
# Glass was only the first case. The same question — "which pixels are this
# material?" — answers metal, wood, fabric, foliage, water and the rest, so the
# renderer and the physics solver can both be told what the object is actually
# made of instead of being handed one uniform crust. Each entry carries the
# phrasings that localise it (phrasing matters far more than synonym count) and
# the optical response: (opacity multiplier, tint, tint strength).
MATERIAL_PROMPTS: Dict[str, tuple] = {
    "glass":   ("windshield", "car windscreen", "the windows of the car",
                "window glass of a building"),
    "metal":   ("polished chrome metal", "shiny steel surface"),
    "wood":    ("wooden surface with wood grain",),
    "fabric":  ("woven cloth fabric", "textile clothing"),
    "leather": ("leather upholstery",),
    "plastic": ("moulded plastic surface",),
    "stone":   ("rough stone or concrete",),
    "skin":    ("human skin",),
    "fur":     ("animal fur or hair",),
    "foliage": ("green leaves and foliage",),
    "water":   ("the surface of water",),
    "rubber":  ("black rubber tyre",),
}

# optics: how each material should be drawn (alpha scale, rgb tint, tint weight)
MATERIAL_OPTICS: Dict[str, tuple] = {
    "glass":   (0.12, (0.62, 0.74, 0.82), 0.55),
    "metal":   (1.00, (0.80, 0.82, 0.86), 0.18),
    "water":   (0.35, (0.45, 0.62, 0.78), 0.40),
    "rubber":  (1.00, (0.12, 0.12, 0.13), 0.25),
    "fur":     (0.85, (0.00, 0.00, 0.00), 0.00),
    "foliage": (0.92, (0.00, 0.00, 0.00), 0.00),
}

# which simulator material each observed surface behaves as
MATERIAL_PHYSICS: Dict[str, str] = {
    "glass": "rigid", "metal": "rigid", "wood": "rigid", "stone": "rigid",
    "plastic": "rigid", "rubber": "soft", "leather": "soft", "skin": "soft",
    "fabric": "cloth", "fur": "strand", "foliage": "cloth", "water": "fluid",
}


def material_map(image_rgb: np.ndarray, threshold: float = 0.30
                 ) -> Optional[Dict[str, np.ndarray]]:
    """[H,W,3] -> {material: [H,W] confidence} for whatever is confidently present.

    Only materials that are both confident AND localised are returned: a phrase
    that lights up the entire silhouette means the model is free-associating (a
    shiny apple reads as "glass"), not that the object is made of it.
    """
    if not available():
        return None
    names, prompts, spans = [], [], []
    for name, phrases in MATERIAL_PROMPTS.items():
        spans.append((len(prompts), len(prompts) + len(phrases)))
        prompts.extend(phrases)
        names.append(name)
    try:
        masks = segment(image_rgb, prompts)
    except Exception:
        return None
    out: Dict[str, np.ndarray] = {}
    for name, (a, b) in zip(names, spans):
        best = masks[a:b].max(axis=0)
        if float(best.max()) < threshold:
            continue
        m = np.clip((best - threshold) / max(1e-3, 1.0 - threshold), 0.0, 1.0)
        hit = float((m > 0.1).mean())
        if hit < 0.0015 or hit > 0.45:        # noise, or the whole object at once
            continue
        out[name] = m.astype(np.float32)
    return out or None


def dominant_materials(image_rgb: np.ndarray, top: int = 4) -> List[Dict[str, object]]:
    """The materials actually visible, biggest first — a compact summary the rest
    of the pipeline (and the LLM) can reason about."""
    mm = material_map(image_rgb)
    if not mm:
        return []
    rank = sorted(mm.items(), key=lambda kv: -float((kv[1] > 0.1).mean()))
    return [{"material": n, "coverage": round(float((m > 0.1).mean()), 4),
             "physics": MATERIAL_PHYSICS.get(n, "soft")} for n, m in rank[:top]]
