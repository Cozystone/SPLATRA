"""Real CPU image -> 3DGS via a 2.5D RGBD lift (no GPU, no weights, no downloads).

This is an HONEST, actually-runnable image-to-Gaussian reconstruction:

    image
      -> (1) foreground/background separation (border-color keying + cleanup)
      -> (2) relief depth estimate (blurred-mask bulge x luminance shaping)
      -> (3) surface normals from the depth gradient
      -> (4) unproject pixels to 3D + a dim back-shell for volume
      -> (5) oriented-surfel GaussianField (anisotropy from the normals)

It is pure numpy + Pillow, so it runs on any machine in milliseconds.

Honesty (PRD §0.3): this is a **2.5D lift**, not novel-view synthesis — it
reconstructs the visible relief of the subject (silhouette-accurate, rounded),
and adds a shallow back-shell so the object has volume when you orbit it. It
does NOT hallucinate unseen geometry. The full novel-view path is the GPU
:class:`atanor_core.generation.lgm.LGMGenerator`. Both are real; they trade
fidelity-of-the-unseen for runnable-anywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


def _box_blur(a: np.ndarray, k: int = 2, iters: int = 3) -> np.ndarray:
    """Cheap separable box blur (numpy only)."""
    out = a.astype(np.float32)
    for _ in range(iters):
        pad = np.pad(out, ((k, k), (k, k)), mode="edge")
        cs = np.cumsum(np.cumsum(pad, axis=0), axis=1)
        cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
        H, W = out.shape
        win = (2 * k + 1)
        s = (cs[win:win + H, win:win + W] - cs[0:H, win:win + W]
             - cs[win:win + H, 0:W] + cs[0:H, 0:W])
        out = s / (win * win)
    return out


def _quat_from_normal(n: np.ndarray) -> np.ndarray:
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    N = n.shape[0]
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dotv = np.clip(n[:, 2], -1.0, 1.0)
    axis = np.cross(np.broadcast_to(z, (N, 3)), n)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(an < 1e-6, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    axis / np.maximum(an, 1e-8))
    half = np.arccos(dotv) * 0.5
    s = np.sin(half)
    q = np.empty((N, 4), dtype=np.float32)
    q[:, 0] = np.cos(half); q[:, 1] = axis[:, 0] * s
    q[:, 2] = axis[:, 1] * s; q[:, 3] = axis[:, 2] * s
    return q


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class Image25DGenerator:
    """Real CPU image -> oriented-surfel GaussianField (2.5D RGBD lift)."""

    def __init__(self, res: int = 168, depth_scale: float = 0.55, sh_degree: int = 1) -> None:
        self.res = int(res)
        self.depth_scale = float(depth_scale)
        self.sh_degree = int(sh_degree)

    def from_image(self, image_rgb: np.ndarray) -> GaussianField:
        img = np.asarray(image_rgb, dtype=np.float32)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        H, W = img.shape[:2]
        alpha = img[..., 3] if img.shape[2] == 4 else None
        rgb = img[..., :3]

        # (1) foreground mask — alpha if it carries a real cutout (transparent
        #     sprites), else key out the border/background color.
        if alpha is not None and float(alpha.std()) > 0.03 and float(alpha.mean()) < 0.97:
            mask = (alpha > 0.4).astype(np.float32)
        else:
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            bg = np.median(border, axis=0)
            diff = np.linalg.norm(rgb - bg[None, None, :], axis=2)
            thr = max(0.12, float(np.percentile(diff, 60)))
            mask = (diff > thr).astype(np.float32)
        mask = (_box_blur(mask, 1, 1) > 0.5).astype(np.float32)
        cov = float(mask.mean())
        if cov < 0.02 or cov > 0.97:     # keying failed or fills frame -> billboard
            mask = np.ones((H, W), np.float32)

        # (2) relief depth = blurred-mask bulge + a center radial DOME (so ANY
        #     image gets volume, not a flat slab) + slight luminance shaping.
        lum = rgb.mean(axis=2)
        bulge = _box_blur(mask, k=max(2, W // 40), iters=3)
        bulge = bulge / (bulge.max() + 1e-6)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        msum = float(mask.sum())
        cy = float((yy * mask).sum() / (msum + 1e-6)) if msum > 0 else H / 2
        cx = float((xx * mask).sum() / (msum + 1e-6)) if msum > 0 else W / 2
        rad = np.sqrt(((yy - cy) / (H * 0.5)) ** 2 + ((xx - cx) / (W * 0.5)) ** 2)
        dome = np.clip(1.0 - rad * rad, 0.0, 1.0)
        depth = ((0.5 * bulge + 0.35 * dome + 0.15 * lum) * mask).astype(np.float32)

        # (3) normals from the depth gradient (real per-pixel orientation).
        gy, gx = np.gradient(depth * self.depth_scale)
        normals_img = np.stack([-gx, gy, np.ones_like(depth)], axis=2)

        # (4) unproject masked pixels to 3D (image plane = xy, relief = +z).
        ys, xs = np.where(mask > 0.5)
        if xs.size == 0:
            ys, xs = np.mgrid[0:H, 0:W].reshape(2, -1)
        # subsample for a light cartridge
        max_pts = self.res * self.res
        if xs.size > max_pts:
            sel = np.random.default_rng(0).choice(xs.size, max_pts, replace=False)
            xs, ys = xs[sel], ys[sel]

        u = (xs / (W - 1) - 0.5) * 2.0
        v = -(ys / (H - 1) - 0.5) * 2.0          # flip y (image down -> world up)
        d = depth[ys, xs]
        col = rgb[ys, xs]
        nrm = normals_img[ys, xs]

        front = np.stack([u, v, d * self.depth_scale], axis=1).astype(np.float32)
        # (4b) thin, dim back-shell so the object has volume when orbited.
        back = np.stack([u, v, -d * self.depth_scale * 0.45], axis=1).astype(np.float32)
        back_col = col * 0.45
        back_nrm = nrm * np.array([1, 1, -1], np.float32)

        means = np.concatenate([front, back], axis=0)
        colors = np.clip(np.concatenate([col, back_col], axis=0), 0.0, 1.0).astype(np.float32)
        normals = np.concatenate([nrm, back_nrm], axis=0).astype(np.float32)
        n = means.shape[0]

        # normalize into the [-1,1] cube the viewer expects
        c = 0.5 * (means.max(0) + means.min(0))
        s = (means.max(0) - means.min(0)).max() * 0.5 + 1e-6
        means = ((means - c) / s).astype(np.float32)

        # (5) oriented surfels: wide in tangent plane, thin along the normal.
        px = 1.6 / max(H, W)                       # ~pixel footprint in world
        scales = np.log(np.tile(
            np.array([px * 1.6, px * 1.6, px * 0.4], np.float32), (n, 1)))
        quats = _quat_from_normal(normals)
        opacities = np.full((n,), _logit(0.92), dtype=np.float32)
        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), dtype=np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(colors)
        return GaussianField(means, scales, quats, opacities, sh, sh_degree=self.sh_degree)

    # GeneratorPort-ish convenience
    def generate(self, mv_images: np.ndarray, cam_rays=None) -> GaussianField:
        mv = np.asarray(mv_images, dtype=np.float32)
        if mv.ndim == 5:
            img = np.transpose(mv[0, 0], (1, 2, 0))
        else:
            img = mv
        return self.from_image(img)
