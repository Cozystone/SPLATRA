"""Real multi-view -> 3D via visual-hull carving (shape-from-silhouette).

The honest multi-view pipeline the user asked for ("≥3 views combined to be 3D"):

    text/image
      -> (1) MVDream / ImageDream: 4 *consistent* views (azimuth 0/90/180/270)
      -> (2) rembg cutout -> 4 silhouettes + color
      -> (3) VISUAL HULL voxel carving: a voxel survives only if it projects
             INSIDE every silhouette -> a real asymmetric 3D occupancy (front,
             back, sides all carved), unlike single-view inflation
      -> (4) color each voxel from the view that sees it most frontally, jitter,
             emit a point-cloud GaussianField

Visual hull needs only silhouettes from known cameras — no per-view depth, no
training. With 4 orthogonal views it recovers a genuinely 3D shape. It cannot
carve concavities hidden from all 4 views (that's the method's known limit; more
views or LGM's learned priors fill those in).

GPU-accelerated via torch when available (RTX-class carves a 128^3 grid in ms).
"""

from __future__ import annotations

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return float(np.log(p / (1 - p)))


def carve_visual_hull(
    masks: np.ndarray,          # [V,H,W] in {0,1}
    colors: np.ndarray,         # [V,H,W,3] in [0,1]
    azimuths,                   # [V] radians
    elevations=None,            # [V] radians (default all 0)
    grid: int = 144,
    scale: float = 1.12,        # object half-extent -> image fraction
    max_points: int = 180_000,
    noise: float = 0.006,
    sh_degree: int = 1,
) -> GaussianField:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, H, W = masks.shape
    m = torch.from_numpy(masks.astype(np.float32)).to(dev)
    col = torch.from_numpy(colors.astype(np.float32)).to(dev)
    az = np.asarray(azimuths, np.float32)
    el = np.zeros(V, np.float32) if elevations is None else np.asarray(elevations, np.float32)

    lin = torch.linspace(-1.0, 1.0, grid, device=dev)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    P = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)  # [N,3]
    N = P.shape[0]
    occ = torch.ones(N, dtype=torch.bool, device=dev)
    depth_best = torch.full((N,), -1e9, device=dev)
    color_best = torch.zeros((N, 3), device=dev)

    for v in range(V):
        a, e = float(az[v]), float(el[v])
        # camera direction (origin -> camera) and an orthonormal image basis
        d = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], np.float32)
        right = np.cross([0, 1, 0], d); right /= np.linalg.norm(right) + 1e-8
        up = np.cross(d, right)
        d_t = torch.tensor(d, device=dev)
        r_t = torch.tensor(right.astype(np.float32), device=dev)
        u_t = torch.tensor(up.astype(np.float32), device=dev)
        ix = P @ r_t
        iy = P @ u_t
        depth = P @ d_t                            # toward camera
        u = ix / (2.0 * scale) + 0.5
        vv = 0.5 - iy / (2.0 * scale)
        inside = (u >= 0) & (u <= 1) & (vv >= 0) & (vv <= 1)
        px = (u.clamp(0, 1) * (W - 1)).long()
        py = (vv.clamp(0, 1) * (H - 1)).long()
        insil = (m[v, py, px] > 0.5) & inside
        occ &= insil
        sel = insil & (depth > depth_best)
        depth_best = torch.where(sel, depth, depth_best)
        color_best = torch.where(sel.unsqueeze(1), col[v, py, px], color_best)

    idx = torch.nonzero(occ, as_tuple=False).squeeze(1)
    if idx.numel() < 32:                          # carve failed -> empty
        raise RuntimeError("visual hull empty (silhouettes did not intersect)")
    if idx.numel() > max_points:
        keep = torch.randperm(idx.numel(), device=dev)[:max_points]
        idx = idx[keep]
    pts = P[idx] + torch.randn(idx.numel(), 3, device=dev) * noise
    cols = color_best[idx].clamp(0, 1)

    means = pts.cpu().numpy().astype(np.float32)
    colors_np = cols.cpu().numpy().astype(np.float32)
    n = means.shape[0]
    c = 0.5 * (means.max(0) + means.min(0))
    s = (means.max(0) - means.min(0)).max() * 0.5 + 1e-6
    means = ((means - c) / s).astype(np.float32)

    px_w = 2.4 / grid
    scales = np.log(np.tile(np.array([px_w, px_w, px_w], np.float32), (n, 1)))
    quats = np.zeros((n, 4), np.float32); quats[:, 0] = 1.0
    opacities = np.full((n,), _logit(0.9), np.float32)
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(colors_np)
    return GaussianField(means, scales, quats, opacities, sh, sh_degree=sh_degree)
