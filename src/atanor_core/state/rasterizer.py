"""Rasterizer adapters.

* :class:`CPURasterizer` — a pure-numpy painter-style splatter. It is **not**
  a quality renderer; it exists so the state machine can be exercised
  end-to-end on CPU with zero heavy dependencies (PRD §7.2).
* :class:`GsplatRasterizer` — the real adapter (gsplat). Lazily imports torch
  and gsplat; the contract is implemented but it is **not** run in the PoC.

Camera convention: ``viewmat`` is a 4x4 world->camera matrix, ``K`` is a 3x3
pinhole intrinsics matrix. Camera looks down +Z in camera space.
"""

from __future__ import annotations

import numpy as np

from ..domain.sgf import GaussianField, sh_dc_to_rgb


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Build a 4x4 world->camera matrix (camera looks toward +Z)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)  # forward (+Z)
    r = np.cross(up, f)
    r = r / (np.linalg.norm(r) + 1e-8)  # right (+X)
    u = np.cross(f, r)  # down-corrected up (+Y)
    rot = np.stack([r, u, f], axis=0)  # rows: camera axes in world
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rot
    viewmat[:3, 3] = -rot @ eye
    return viewmat


def default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics from a vertical field of view."""
    fov = np.deg2rad(fov_deg)
    fy = 0.5 * height / np.tan(0.5 * fov)
    fx = fy
    K = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


class CPURasterizer:
    """Pure-numpy painter splatter. Quality is not the goal; coverage is."""

    def render(
        self,
        field: GaussianField,
        viewmat: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.float32)
        n = field.num_gaussians
        if n == 0:
            return img

        viewmat = np.asarray(viewmat, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]

        # world -> camera
        cam = field.means @ R.T + t  # [N, 3]
        z = cam[:, 2]
        front = z > 1e-3
        if not np.any(front):
            return img

        # perspective projection
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy

        # colors / alphas / radii
        rgb = np.clip(sh_dc_to_rgb(field.sh[:, 0, :]), 0.0, 1.0)  # [N, 3]
        alpha = 1.0 / (1.0 + np.exp(-field.opacities))  # sigmoid
        scale_lin = np.exp(field.scales).mean(axis=1)  # [N]
        radius_px = np.clip(scale_lin * fy / np.maximum(z, 1e-3), 0.6, 0.25 * width)

        # painter: far first
        order = np.argsort(-z)
        for i in order:
            if not front[i]:
                continue
            cxp, cyp = u[i], v[i]
            rad = float(radius_px[i])
            x0 = max(0, int(np.floor(cxp - rad)))
            x1 = min(width - 1, int(np.ceil(cxp + rad)))
            y0 = max(0, int(np.floor(cyp - rad)))
            y1 = min(height - 1, int(np.ceil(cyp + rad)))
            if x1 < x0 or y1 < y0:
                continue
            ys = np.arange(y0, y1 + 1)
            xs = np.arange(x0, x1 + 1)
            gx, gy = np.meshgrid(xs, ys)
            d2 = (gx - cxp) ** 2 + (gy - cyp) ** 2
            sigma2 = max(rad * rad, 1.0)
            g = np.exp(-0.5 * d2 / sigma2) * float(alpha[i])  # [h, w]
            g = g[..., None]  # [h, w, 1]
            patch = img[y0 : y1 + 1, x0 : x1 + 1, :]
            img[y0 : y1 + 1, x0 : x1 + 1, :] = g * rgb[i] + (1.0 - g) * patch

        return np.clip(img, 0.0, 1.0).astype(np.float32)


class GsplatRasterizer:
    """Real gsplat adapter. Lazy torch/gsplat import; NOT run in the PoC.

    Honesty note (PRD §6.6): gsplat / Brush / LGM APIs (especially SH->RGB
    handling) drift between versions. Re-verify against the installed repo's
    docs when actually wiring GPU rendering.
    """

    def __init__(self) -> None:
        self._torch = None
        self._gsplat = None

    def _ensure(self):
        if self._gsplat is None:
            import torch  # lazy, GPU extra
            import gsplat  # lazy, GPU extra

            self._torch = torch
            self._gsplat = gsplat

    def render(
        self,
        field: GaussianField,
        viewmat: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        self._ensure()
        torch = self._torch
        gsplat = self._gsplat

        means = torch.from_numpy(np.ascontiguousarray(field.means)).float()
        quats = torch.from_numpy(np.ascontiguousarray(field.quats)).float()
        scales = torch.exp(torch.from_numpy(np.ascontiguousarray(field.scales)).float())
        opacities = torch.sigmoid(
            torch.from_numpy(np.ascontiguousarray(field.opacities)).float()
        )
        colors = torch.from_numpy(np.ascontiguousarray(field.sh)).float()
        viewmats = torch.from_numpy(np.ascontiguousarray(viewmat)).float()[None]
        Ks = torch.from_numpy(np.ascontiguousarray(K)).float()[None]

        renders, _, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=field.sh_degree,
            radius_clip=0.5,
            packed=True,
        )
        return renders[0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
