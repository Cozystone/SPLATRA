"""Foreground cutout (background removal) for the image→3D lift.

Uses ``rembg`` (U²-Net saliency segmentation, CPU, ~176MB model on first run)
to produce a clean RGBA cutout so the 2.5D lift reconstructs only the subject —
not the table/background behind it. Falls back to returning the input unchanged
when rembg is unavailable, so it is a strict quality upgrade, never a hard
dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_session = None  # cached rembg session


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


def cutout(img: np.ndarray) -> Optional[np.ndarray]:
    """[H,W,3|4] float [0,1] -> [H,W,4] RGBA float with background removed.

    Returns None if rembg is unavailable (caller then keeps its own keying).
    """
    try:
        from PIL import Image
        from rembg import remove

        rgb = (np.clip(img[..., :3], 0, 1) * 255).astype(np.uint8)
        out = remove(Image.fromarray(rgb), session=_get_session())  # RGBA PIL
        return np.asarray(out.convert("RGBA"), dtype=np.float32) / 255.0
    except Exception:
        return None


def reframe_foreground(rgba: np.ndarray, ratio: float = 0.85,
                       size: int = 512) -> np.ndarray:
    """Recenter+rescale the cutout so the subject occupies ``ratio`` of a square
    canvas with even margin — TripoSR's expected framing. Reconstruction quality
    is very sensitive to this: a frame-filling or off-center subject (common with
    SD product-shot crops) yields a deformed volume. Returns [size,size,4] RGBA.

    The subject's own aspect ratio is preserved; the longer side is scaled to
    ``ratio*size`` and the result is centered, so a frame-filling apple becomes a
    smaller, fully-visible, centered apple.
    """
    a = rgba[..., 3]
    ys, xs = np.where(a > 0.1)
    if ys.size < 16:                       # nothing segmented — leave as-is
        return rgba
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgba[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    from PIL import Image
    scale = (ratio * size) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8), "RGBA")
    pil = pil.resize((nw, nh), Image.LANCZOS)
    small = np.asarray(pil, np.float32) / 255.0
    canvas = np.zeros((size, size, 4), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas
