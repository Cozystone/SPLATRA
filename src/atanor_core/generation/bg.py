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
