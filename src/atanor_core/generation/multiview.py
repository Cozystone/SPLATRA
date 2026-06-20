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

import os
import urllib.request
from pathlib import Path

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


def _quat_from_normal_np(n: np.ndarray) -> np.ndarray:
    """[N,3] normals -> [N,4] quats mapping local +z onto n (oriented surfels)."""
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    N = n.shape[0]
    z = np.array([0.0, 0.0, 1.0], np.float32)
    dotv = np.clip(n[:, 2], -1.0, 1.0)
    axis = np.cross(np.broadcast_to(z, (N, 3)), n)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(an < 1e-6, np.array([1.0, 0.0, 0.0], np.float32), axis / np.maximum(an, 1e-8))
    half = np.arccos(dotv) * 0.5
    s = np.sin(half)
    q = np.empty((N, 4), np.float32)
    q[:, 0] = np.cos(half); q[:, 1] = axis[:, 0] * s
    q[:, 2] = axis[:, 1] * s; q[:, 3] = axis[:, 2] * s
    return q

# Zero123++ diffusers pipeline code (HF custom-code repo is gated; GitHub is open).
_Z123_PIPELINE_URL = (
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/"
    "diffusers-support/pipeline.py"
)
_Z123_MODEL = os.environ.get("SPLATRA_Z123_MODEL", "sudo-ai/zero123plus-v1.2")
# Zero123++ v1.2 fixed output poses (degrees), 3x2 grid, row-major.
_Z123_AZIMUTHS = [30, 90, 150, 210, 270, 330]
_Z123_ELEVATIONS = [20, -10, 20, -10, 20, -10]


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

    if occ.sum() < 32:                            # carve failed -> empty
        raise RuntimeError("visual hull empty (silhouettes did not intersect)")

    import torch.nn.functional as F

    G = grid
    occg = occ.float().view(1, 1, G, G, G)
    # smoothed occupancy field -> sub-voxel surface + stable normals
    occ_soft = occg
    for _ in range(2):
        occ_soft = F.avg_pool3d(occ_soft, 3, stride=1, padding=1)
    # surface shell = occupied voxels that touch empty space (boundary only)
    empty_near = F.max_pool3d(1.0 - occg, 3, stride=1, padding=1)
    surface = (occg > 0.5) & (empty_near > 0.5)
    surf_flat = surface.view(-1)
    # outward normal = -gradient of the smoothed occupancy
    s3 = occ_soft.view(G, G, G)
    gx = torch.zeros_like(s3); gy = torch.zeros_like(s3); gz = torch.zeros_like(s3)
    gx[1:-1] = s3[2:] - s3[:-2]
    gy[:, 1:-1] = s3[:, 2:] - s3[:, :-2]
    gz[:, :, 1:-1] = s3[:, :, 2:] - s3[:, :, :-2]
    normals_grid = -torch.stack([gx, gy, gz], dim=-1).view(-1, 3)  # outward

    idx = torch.nonzero(surf_flat, as_tuple=False).squeeze(1)
    if idx.numel() < 64:                          # too thin -> fall back to solid
        idx = torch.nonzero(occ, as_tuple=False).squeeze(1)
        solid = True
    else:
        solid = False
    if idx.numel() > max_points:
        keep = torch.randperm(idx.numel(), device=dev)[:max_points]
        idx = idx[keep]

    pts = P[idx] + torch.randn(idx.numel(), 3, device=dev) * noise
    cols = color_best[idx].clamp(0, 1)
    nrm = normals_grid[idx]
    nlen = nrm.norm(dim=1, keepdim=True)
    nrm = torch.where(nlen > 1e-4, nrm / (nlen + 1e-8),
                      torch.tensor([0.0, 0.0, 1.0], device=dev))

    means = pts.cpu().numpy().astype(np.float32)
    colors_np = cols.cpu().numpy().astype(np.float32)
    normals_np = nrm.cpu().numpy().astype(np.float32)
    n = means.shape[0]
    c = 0.5 * (means.max(0) + means.min(0))
    s = (means.max(0) - means.min(0)).max() * 0.5 + 1e-6
    means = ((means - c) / s).astype(np.float32)

    vox = 2.0 / G
    if solid:                                     # isotropic points (no surface)
        scales = np.log(np.tile(np.array([vox * 1.2] * 3, np.float32), (n, 1)))
        quats = np.zeros((n, 4), np.float32); quats[:, 0] = 1.0
    else:                                         # oriented surfels: wide tangent, thin normal
        scales = np.log(np.tile(np.array([vox * 1.7, vox * 1.7, vox * 0.5], np.float32), (n, 1)))
        quats = _quat_from_normal_np(normals_np)
    opacities = np.full((n,), _logit(0.92), np.float32)
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(colors_np)
    return GaussianField(means, scales, quats, opacities, sh, sh_degree=sh_degree)


def silhouette_score(field: GaussianField, masks: np.ndarray, azimuths,
                     elevations, scale: float = 1.2, S: int = 128) -> float:
    """Auto quality score (0-100) = mean silhouette IoU of the carved hull
    re-projected into each input view vs that view's silhouette. High = the 3D
    reconstruction is faithful to every generated view."""
    from PIL import Image as I

    m = field.means
    ious = []
    for k in range(masks.shape[0]):
        a, e = float(azimuths[k]), float(elevations[k])
        d = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], np.float32)
        right = np.cross([0, 1, 0], d); right /= np.linalg.norm(right) + 1e-8
        up = np.cross(d, right)
        ix = m @ right; iy = m @ up
        u = np.clip(ix / (2 * scale) + 0.5, 0, 1); v = np.clip(0.5 - iy / (2 * scale), 0, 1)
        proj = np.zeros((S, S), bool)
        proj[(v * (S - 1)).astype(int), (u * (S - 1)).astype(int)] = True
        # dilate the sparse projection a touch to a coverage mask
        for sh in (1, -1):
            proj |= np.roll(proj, sh, 0) | np.roll(proj, sh, 1)
        gt = np.asarray(I.fromarray((masks[k] * 255).astype(np.uint8))
                        .resize((S, S)), np.float32) / 255.0 > 0.5
        inter = float((proj & gt).sum()); union = float((proj | gt).sum()) + 1e-6
        ious.append(inter / union)
    return round(100.0 * float(np.mean(ious)), 1)


def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "splatra" / "zero123plus"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MultiViewGenerator:
    """Real multi-view 3D: text/image -> Zero123++ 6 views -> visual-hull point cloud.

    GPU path (needs the `.[sd]` stack + CUDA). Generates consistent novel views
    with Zero123++ and carves them into a true asymmetric 3D point cloud — the
    honest "≥3 views fused" pipeline. Opt-in via SPLATRA_MV=1.
    """

    def __init__(self, grid: int = 200, scale: float = 1.2, steps: int = 28) -> None:
        self.grid = int(grid)
        self.scale = float(scale)
        self.steps = int(steps)
        self._z123 = None
        self._t2i = None

    def _ensure(self):
        if self._z123 is not None:
            return
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        # fetch the (open) GitHub pipeline.py into a local cache once
        pf = _cache_dir() / "pipeline.py"
        if not pf.exists():
            urllib.request.urlretrieve(_Z123_PIPELINE_URL, pf)
        pipe = DiffusionPipeline.from_pretrained(
            _Z123_MODEL, custom_pipeline=str(_cache_dir()),
            torch_dtype=torch.float16, trust_remote_code=True,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        self._z123 = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # -- core: a single conditioning image -> 3D ------------------------------ #
    def from_cond(self, cond_rgb: np.ndarray) -> GaussianField:
        """[H,W,3|4] in [0,1] -> carved GaussianField via Zero123++ 6 views."""
        from PIL import Image as I

        from .bg import cutout

        self._ensure()
        # isolate the subject on white for Zero123++
        rgba = cutout(cond_rgb)
        if rgba is None:
            rgba = cond_rgb if cond_rgb.shape[-1] == 4 else np.concatenate(
                [cond_rgb, np.ones_like(cond_rgb[..., :1])], -1)
        comp = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])
        cond = I.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).resize((320, 320))

        grid_img = self._z123(cond, num_inference_steps=self.steps).images[0]
        W, H = grid_img.size
        tw, th = W // 2, H // 3
        tiles = [grid_img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                 for r in range(3) for c in range(2)]

        imgs = [cond] + tiles
        azs = [0] + _Z123_AZIMUTHS
        els = [0] + _Z123_ELEVATIONS
        masks, cols = [], []
        for im in imgs:
            a = cutout(np.asarray(im.convert("RGB"), np.float32) / 255.0)
            if a is None:
                continue
            m = I.fromarray(((a[..., 3] > 0.5) * 255).astype(np.uint8)).resize((256, 256))
            c = I.fromarray((np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8)).resize((256, 256))
            masks.append(np.asarray(m, np.float32) / 255.0)
            cols.append(np.asarray(c, np.float32) / 255.0)
        masks = np.stack(masks)
        cols = np.stack(cols)
        azr = np.radians(azs[:len(masks)]).astype(np.float32)
        elr = np.radians(els[:len(masks)]).astype(np.float32)
        field = carve_visual_hull(masks, cols, azr, elr, grid=self.grid, scale=self.scale)
        # automatic quality score = mean silhouette IoU (re-projected hull vs views)
        self.last_score = silhouette_score(field, masks, azr, elr, self.scale)
        return field

    def generate(self, prompt: str, n: int = None) -> GaussianField:
        """Text -> SD-Turbo cond image -> multi-view 3D. best-of-N by auto score."""
        if self._t2i is None:
            from .text_to_3d import TextTo3DGenerator
            self._t2i = TextTo3DGenerator()
        n = int(os.environ.get("SPLATRA_BESTOF", "1")) if n is None else int(n)
        best, best_s = None, -1.0
        for _ in range(max(1, n)):
            f = self.from_cond(self._t2i.image(prompt))
            if self.last_score > best_s:
                best, best_s = f, self.last_score
        self.last_score = best_s
        return best
