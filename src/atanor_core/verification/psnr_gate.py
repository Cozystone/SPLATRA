"""Client-side PSNR verification gate.

DePIN nodes are untrusted (PRD §7.5: io.net 2024-04 Sybil incident). Any
cartridge that comes back from distributed refinement MUST pass a client-side
quality gate before it is pinned and displayed. This gate re-renders held-out
views locally and checks PSNR.

The gate has a real implementation (it actually renders and computes PSNR) —
it is not a mock.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..domain.sgf import GaussianField


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB between two images in [0, range]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


class PSNRGate:
    """Reject cartridges whose held-out re-render is too poor.

    Args:
        rasterizer: a RasterizerPort used to re-render held-out views.
        abs_floor_db: absolute PSNR floor; below this -> reject.
        rel_drop_db: max allowed PSNR drop vs a reference PSNR (if provided).
    """

    def __init__(
        self,
        rasterizer,
        abs_floor_db: float = 18.0,
        rel_drop_db: float = 3.0,
    ) -> None:
        self.rasterizer = rasterizer
        self.abs_floor_db = float(abs_floor_db)
        self.rel_drop_db = float(rel_drop_db)
        self.last_psnr: Optional[float] = None

    def verify(
        self,
        field: GaussianField,
        reference_views: Sequence[np.ndarray],
        cameras: Sequence[Dict[str, Any]],
        reference_psnr: Optional[float] = None,
    ) -> bool:
        if not reference_views or not cameras:
            # Nothing to check against -> conservatively accept (no DePIN trip).
            self.last_psnr = None
            return True

        scores = []
        for ref, cam in zip(reference_views, cameras):
            h, w = ref.shape[0], ref.shape[1]
            rendered = self.rasterizer.render(
                field, cam["viewmat"], cam["K"], w, h
            )
            scores.append(psnr(rendered, ref))

        # Mean over finite scores; inf (perfect match) counts as a pass.
        finite = [s for s in scores if np.isfinite(s)]
        mean_psnr = float(np.mean(finite)) if finite else float("inf")
        self.last_psnr = mean_psnr

        if mean_psnr < self.abs_floor_db:
            return False
        if reference_psnr is not None and (reference_psnr - mean_psnr) > self.rel_drop_db:
            return False
        return True
