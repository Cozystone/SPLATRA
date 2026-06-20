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


def _edge_distance(mask: np.ndarray, iters: int = 90) -> np.ndarray:
    """Approx distance-to-silhouette-edge via iterative erosion (vectorized).

    dist[p] = number of 4-neighbour erosion steps pixel p survives ≈ its
    distance from the foreground boundary. Used to inflate the silhouette into a
    rounded closed volume (thick at the core, zero at the rim).
    """
    m = (mask > 0.5).astype(np.float32)
    dist = np.zeros_like(m)
    for _ in range(iters):
        e = m.copy()
        e[1:, :] = np.minimum(e[1:, :], m[:-1, :])
        e[:-1, :] = np.minimum(e[:-1, :], m[1:, :])
        e[:, 1:] = np.minimum(e[:, 1:], m[:, :-1])
        e[:, :-1] = np.minimum(e[:, :-1], m[:, 1:])
        m = e
        dist += m
        if m.sum() == 0:
            break
    return dist


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
    """Real CPU image -> VOLUMETRIC point-cloud GaussianField.

    The silhouette is filled with particles at random depths (within a
    distance-transform thickness) plus noise diffusion — a solid 3D point cloud,
    NOT front/back planes. Honest: depth is still single-view (symmetric about
    the image plane), so it cannot recover an asymmetric pose — that needs the
    multi-view GPU path (LGMGenerator).
    """

    def __init__(self, res: int = 200, inflate: float = 0.95, layers: int = 5,
                 noise: float = 0.012, sh_degree: int = 1) -> None:
        self.res = int(res)
        self.inflate = float(inflate)   # how round the silhouette puffs out
        self.layers = int(layers)       # particles per pixel -> fills the VOLUME
        self.noise = float(noise)       # gaussian jitter (diffusion) per particle
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

        # (2) THICKNESS field from the distance transform: how deep the solid is
        #     at each pixel (thick core, zero rim). NOT a front/back plane — this
        #     is used to FILL a volume with particles.
        dist = _edge_distance(mask, iters=max(40, H // 3))
        dist = _box_blur(dist, k=max(2, W // 50), iters=2)
        t = dist / (dist.max() + 1e-6)
        half = np.sqrt(np.clip(t, 0.0, 1.0)) * self.inflate            # [H,W]

        # (3) sample masked pixels (cap distinct columns for a light cartridge)
        ys, xs = np.where(mask > 0.5)
        if xs.size == 0:
            ys, xs = np.mgrid[0:H, 0:W].reshape(2, -1)
        base = self.res * self.res
        if xs.size > base:
            sel = np.random.default_rng(0).choice(xs.size, base, replace=False)
            xs, ys = xs[sel], ys[sel]

        # (4) VOLUME FILL — K particles per column at random depths in
        #     [-half, +half] (mild surface bias). This is a true point-cloud
        #     volume: no front/back planes, no fake paper structure.
        rng = np.random.default_rng(1)
        K = self.layers
        xs_k = np.repeat(xs, K)
        ys_k = np.repeat(ys, K)
        u = (xs_k / (W - 1) - 0.5) * 2.0
        v = -(ys_k / (H - 1) - 0.5) * 2.0
        halfp = half[ys_k, xs_k]
        a = rng.uniform(-1.0, 1.0, size=u.size).astype(np.float32)
        zt = np.sign(a) * (np.abs(a) ** 0.7) * halfp                   # fill depth

        # (5) NOISE DIFFUSION — jitter every particle so nothing reads as a plane.
        jit = rng.normal(0.0, self.noise, size=(u.size, 3)).astype(np.float32)
        means = (np.stack([u, v, zt], axis=1) + jit).astype(np.float32)

        col = rgb[ys_k, xs_k]
        shade = (0.62 + 0.38 * (1.0 - np.abs(zt) / (halfp + 1e-6))).astype(np.float32)
        colors = np.clip(col * shade[:, None], 0.0, 1.0).astype(np.float32)
        n = means.shape[0]

        # normalize into the [-1,1] cube the viewer expects
        c = 0.5 * (means.max(0) + means.min(0))
        s = (means.max(0) - means.min(0)).max() * 0.5 + 1e-6
        means = ((means - c) / s).astype(np.float32)

        # (6) POINT CLOUD: small ISOTROPIC round particles (no surfels/planes).
        px = 2.0 / max(H, W)
        scales = np.log(np.tile(np.array([px, px, px], np.float32), (n, 1)))
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0
        opacities = np.full((n,), _logit(0.85), dtype=np.float32)
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
