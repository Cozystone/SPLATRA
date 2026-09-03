"""Surface hygiene shared by every reconstruction path.

Both learned reconstruction (TripoSR's density field) and geometric
reconstruction (the multi-view visual hull) hand back a *solid*: every voxel the
object occupies. But only the crust of that solid ever has trustworthy colour —
TripoSR's colour field is supervised only where camera rays stop, and the hull
colours only what some view saw frontally, zero-initialising the rest to black.
Buried points outnumber the crust several times over, and a splat one voxel wide
does not fully occlude what sits behind it, so the render looks *through* the
skin into the unsupervised interior: dark, washed out, noisy. The car in the
all-round path came out 94.5% pure black for exactly this reason.

So every path runs the same three steps: carve the solid down to its shell,
rescue any shell point that still has no colour by borrowing from its nearest
coloured neighbour, and (where the point spacing changed) size splats to the
spacing actually measured rather than to the sampling grid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def surface_shell(pts: np.ndarray, cols: np.ndarray, step: float,
                  depth: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the crust of a solid point cloud and drop what is buried inside.

    ``step`` must be the lattice the points were sampled on — indexing by the
    bounding box instead spreads neighbouring samples across voxels, nothing
    reads as 6-connected, and the carve silently keeps everything. The
    occupancy is eroded ``depth`` times (6-neighbourhood) and only what the
    erosion removes survives: a shell with thickness, no interior. Thin sheets
    are all surface and come back whole; if the shell would be implausibly
    sparse the solid is returned rather than an object full of holes.
    """
    if pts.shape[0] < 1000 or step <= 0:
        return pts, cols
    lo = pts.min(0)
    vox = np.rint((pts - lo) / step).astype(np.int64)
    K = int(vox.max()) + 1
    vox = np.clip(vox, 0, K - 1) + 1
    occ = np.zeros((K + 2, K + 2, K + 2), bool)
    occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    inner = occ
    for _ in range(max(1, int(depth))):
        e = inner.copy()
        for ax in (0, 1, 2):
            for sh in (-1, 1):
                e &= np.roll(inner, sh, ax)
        inner = e
    keep = ~inner[vox[:, 0], vox[:, 1], vox[:, 2]]
    if keep.sum() < max(2000, pts.shape[0] * 0.02):
        return pts, cols
    return pts[keep], cols[keep]


def splat_sigma(means: np.ndarray, grid: int) -> float:
    """Splat radius measured from how far apart the points actually ended up.

    Below ~0.7x the median neighbour distance the object fills with holes and
    the background reads through as noise; far above it the surface turns to
    mush. The grid only bounds the answer.
    """
    n = means.shape[0]
    if n < 32:
        return 2.2 / grid
    try:
        from scipy.spatial import cKDTree
        probe = means[np.random.default_rng(0).choice(n, min(4096, n), replace=False)]
        d = cKDTree(means).query(probe, k=2)[0][:, 1]
        nn = float(np.median(d[np.isfinite(d) & (d > 0)]))
    except Exception:
        nn = 2.2 / grid
    return float(np.clip(nn * 0.75, 1.5 / grid, 0.06))


def fill_dark_colors(pts: np.ndarray, cols: np.ndarray,
                     threshold: float = 0.04) -> np.ndarray:
    """Give colourless points the colour of their nearest coloured neighbour.

    The hull zero-initialises colour and only paints voxels some view saw
    frontally, so surfaces facing up, down, or between the cameras stay pure
    black even after the interior is carved away. Black next to painted is far
    more often "unobserved" than "actually black", so borrow locally. If nearly
    nothing is coloured there is nothing worth spreading — the cloud is
    returned untouched rather than painted from noise.
    """
    if pts.shape[0] == 0:
        return cols
    lum = cols.mean(1)
    dark = lum < float(threshold)
    lit = ~dark
    if not dark.any() or float(lit.mean()) < 0.05:
        return cols
    try:
        from scipy.spatial import cKDTree
        idx = cKDTree(pts[lit]).query(pts[dark], k=1)[1]
    except Exception:
        return cols
    out = cols.copy()
    out[dark] = cols[lit][idx]
    return out
