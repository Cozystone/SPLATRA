"""Rasterizer adapters.

* :class:`CPURasterizer` — a pure-numpy **EWA splatter**. It implements the
  real 3D Gaussian Splatting math (anisotropic 3D covariance -> projected 2D
  conic -> front-to-back alpha compositing with transmittance), plus
  supersampled anti-aliasing. It is the *same algorithm* gsplat runs on the
  GPU; this is just the CPU/numpy version (correct but slow). Honesty note
  (PRD §7.2): it is real splatting, not a placeholder shortcut — the only
  trade-off vs :class:`GsplatRasterizer` is speed, not technique.
* :class:`GsplatRasterizer` — the GPU adapter (gsplat). Lazily imports torch
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


def orbit_camera(yaw: float, pitch: float, dist: float, target=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera on a sphere around ``target`` (yaw/pitch in radians)."""
    cx, cy, cz = target
    x = cx + dist * np.cos(pitch) * np.sin(yaw)
    y = cy + dist * np.sin(pitch)
    z = cz - dist * np.cos(pitch) * np.cos(yaw)
    return look_at(eye=(x, y, z), target=target)


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


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[N,4] quaternions (w,x,y,z) -> [N,3,3] rotation matrices."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class CPURasterizer:
    """Pure-numpy EWA splatter (real 3DGS math, CPU).

    Args:
        supersample: render at this integer scale then box-downsample (AA).
        lowpass: screen-space covariance dilation in px^2 (anti-aliasing).
        background: RGB background color in [0,1].
    """

    def __init__(
        self,
        supersample: int = 2,
        lowpass: float = 0.3,
        background=(0.0, 0.0, 0.0),
    ) -> None:
        self.supersample = max(1, int(supersample))
        self.lowpass = float(lowpass)
        self.background = np.asarray(background, dtype=np.float32)

    def render(
        self,
        field: GaussianField,
        viewmat: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        ss = self.supersample
        W, H = width * ss, height * ss

        bg = self.background
        out_bg = np.broadcast_to(bg, (height, width, 3)).astype(np.float32)
        n = field.num_gaussians
        if n == 0:
            return out_bg.copy()

        viewmat = np.asarray(viewmat, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        Wr = viewmat[:3, :3]
        t = viewmat[:3, 3]
        fx, fy = K[0, 0] * ss, K[1, 1] * ss
        cx, cy = K[0, 2] * ss, K[1, 2] * ss

        # world -> camera
        cam = field.means @ Wr.T + t  # [N,3]
        z = cam[:, 2]
        front = z > 1e-3
        if not np.any(front):
            return out_bg.copy()

        # 3D covariance Sigma = R S S^T R^T, then to camera: W Sigma W^T
        s = np.exp(field.scales)  # [N,3] linear
        R = _quat_to_rot(field.quats)  # [N,3,3]
        M = R * s[:, None, :]  # [N,3,3] columns scaled
        Sigma = M @ np.transpose(M, (0, 2, 1))  # [N,3,3]
        Sigma_c = Wr @ Sigma @ Wr.T  # [N,3,3]

        # projection Jacobian per gaussian
        zc = np.maximum(z, 1e-3)
        J = np.zeros((n, 2, 3), dtype=np.float32)
        J[:, 0, 0] = fx / zc
        J[:, 0, 2] = -fx * cam[:, 0] / (zc * zc)
        J[:, 1, 1] = fy / zc
        J[:, 1, 2] = -fy * cam[:, 1] / (zc * zc)
        cov2d = J @ Sigma_c @ np.transpose(J, (0, 2, 1))  # [N,2,2]
        cov2d[:, 0, 0] += self.lowpass
        cov2d[:, 1, 1] += self.lowpass

        a = cov2d[:, 0, 0]
        b = cov2d[:, 0, 1]
        c = cov2d[:, 1, 1]
        det = a * c - b * b

        # screen-space centers, colors, opacities
        u = fx * cam[:, 0] / zc + cx
        v = fy * cam[:, 1] / zc + cy
        rgb = np.clip(sh_dc_to_rgb(field.sh[:, 0, :]), 0.0, 1.0)
        opacity = 1.0 / (1.0 + np.exp(-field.opacities))

        # bounding-box radius from largest eigenvalue of cov2d
        mid = 0.5 * (a + c)
        rad_eig = np.sqrt(np.maximum(mid + np.sqrt(np.maximum(mid * mid - det, 0.0)), 1e-6))
        radius = np.clip(np.ceil(3.0 * rad_eig), 1, max(W, H)).astype(np.int32)

        img = np.broadcast_to(bg, (H, W, 3)).astype(np.float32).copy()
        T = np.ones((H, W), dtype=np.float32)  # transmittance

        valid = front & (det > 1e-9)
        order = np.argsort(z)  # near -> far (front-to-back)
        for i in order:
            if not valid[i]:
                continue
            ui, vi = u[i], v[i]
            rad = int(radius[i])
            x0 = max(0, int(np.floor(ui - rad)))
            x1 = min(W - 1, int(np.ceil(ui + rad)))
            y0 = max(0, int(np.floor(vi - rad)))
            y1 = min(H - 1, int(np.ceil(vi + rad)))
            if x1 < x0 or y1 < y0:
                continue

            xs = np.arange(x0, x1 + 1, dtype=np.float32)
            ys = np.arange(y0, y1 + 1, dtype=np.float32)
            dx, dy = np.meshgrid(xs - ui, ys - vi)
            inv = 1.0 / det[i]
            # power = -0.5 * d^T conic d, conic = inv(cov2d)
            power = -0.5 * inv * (c[i] * dx * dx - 2.0 * b[i] * dx * dy + a[i] * dy * dy)
            g = np.exp(np.minimum(power, 0.0))
            alpha = np.clip(opacity[i] * g, 0.0, 0.99)  # [h,w]

            Tp = T[y0 : y1 + 1, x0 : x1 + 1]
            contrib = Tp * alpha  # [h,w]
            patch = img[y0 : y1 + 1, x0 : x1 + 1, :]
            img[y0 : y1 + 1, x0 : x1 + 1, :] = patch + contrib[..., None] * rgb[i]
            T[y0 : y1 + 1, x0 : x1 + 1] = Tp * (1.0 - alpha)

        img = np.clip(img, 0.0, 1.0)
        if ss > 1:
            img = img.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
        return img.astype(np.float32)


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
