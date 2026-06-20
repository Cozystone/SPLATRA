"""3D object generation adapters.

* :class:`MockGenerator` — a deterministic **procedural** generator. It builds
  one of a few parametric shapes (sphere / cube / torus / spiral) with
  Lambert-shaded, average-color Gaussians. This is an explicit **mock**
  (PRD §7.2): it does NOT do real image-to-3D. Its job is to give the state
  machine a plausible, visually-3D GaussianField to converge onto so the
  cache-miss path can be exercised on CPU. The shape is a procedural
  placeholder, not reconstructed geometry.
* :class:`LGMGenerator` — the real adapter slot (LGM / 3DTopia + Zero123++).
  Lazily imports heavy deps; ``_ensure`` raises ``NotImplementedError`` to mark
  exactly where real wiring goes, including the <=4GB sequential-offload plan.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc

_LIGHT = np.array([0.4, 0.7, 0.6], dtype=np.float32)
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _sphere(n: int, r: float):
    i = np.arange(n, dtype=np.float32)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    p = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)
    return (r * p).astype(np.float32), p.astype(np.float32)


def _torus(n: int, r: float):
    """Full donut SURFACE: a (major u) x (minor v) angular grid."""
    R, rr = r * 0.62, r * 0.30
    nv = max(8, int(round(np.sqrt(n * rr / R))))
    nu = max(12, n // nv)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False, dtype=np.float32)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    x = (R + rr * np.cos(vv)) * np.cos(uu)
    y = (R + rr * np.cos(vv)) * np.sin(uu)
    zc = rr * np.sin(vv)
    p = np.stack([x, y, zc], 1).astype(np.float32)
    nrm = np.stack([np.cos(vv) * np.cos(uu), np.cos(vv) * np.sin(uu), np.sin(vv)], 1)
    return p, nrm.astype(np.float32)


def _spiral(n: int, r: float):
    """Coil TUBE: a swept ring along a helix centerline (parallel-transport frame)."""
    ring = 8
    nc = max(24, n // ring)
    t = np.linspace(0.0, 1.0, nc, dtype=np.float32)
    ang = t * 5.0 * np.pi
    rad = r * 0.55
    cx = rad * np.cos(ang)
    cy = (t - 0.5) * 1.5 * r
    cz = rad * np.sin(ang)
    center = np.stack([cx, cy, cz], 1)
    # tangent (finite diff)
    tang = np.gradient(center, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n1 = np.cross(tang, up)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8
    n2 = np.cross(tang, n1)
    tr = r * 0.13
    theta = np.linspace(0, 2 * np.pi, ring, endpoint=False, dtype=np.float32)
    ct, st = np.cos(theta), np.sin(theta)
    # [nc, ring, 3]
    offset = ct[None, :, None] * n1[:, None, :] + st[None, :, None] * n2[:, None, :]
    p = (center[:, None, :] + tr * offset).reshape(-1, 3).astype(np.float32)
    nrm = offset.reshape(-1, 3).astype(np.float32)
    return p, nrm


def _cube(n: int, r: float):
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    face = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    nrm = np.zeros((n, 3), dtype=np.float32)
    for k in range(3):
        m = face == k
        pts[m, k] = sign[m]
        nrm[m, k] = sign[m]
    return (r * 0.8 * pts).astype(np.float32), nrm


_SHAPES = {"sphere": _sphere, "cube": _cube, "torus": _torus, "spiral": _spiral}


class MockGenerator:
    """MOCK generator: procedural shaded shape (deterministic). Not real 3D recon."""

    def __init__(self, n_points: int = 2000, radius: float = 0.6, sh_degree: int = 1) -> None:
        self.n_points = int(n_points)
        self.radius = float(radius)
        self.sh_degree = int(sh_degree)

    def generate(self, mv_images: np.ndarray, cam_rays: Optional[Any] = None) -> GaussianField:
        """Generate a shaded procedural object.

        Args:
            mv_images: [1, V, 3, H, W] multi-view images in [0, 1]; their mean
                       color tints the object.
            cam_rays:  in the real adapter this is a camera ray bundle. The mock
                       repurposes it as an optional hint dict, e.g.
                       ``{"shape": "torus"}`` (defaults to sphere).
        """
        mv = np.asarray(mv_images, dtype=np.float32)
        if mv.ndim == 5 and mv.shape[2] == 3 and mv.size > 0:
            base_rgb = mv.mean(axis=(0, 1, 3, 4))
        else:
            base_rgb = np.array([0.55, 0.55, 0.6], dtype=np.float32)
        base_rgb = np.clip(base_rgb.astype(np.float32), 0.05, 1.0)

        shape = "sphere"
        if isinstance(cam_rays, dict):
            shape = str(cam_rays.get("shape", "sphere")).lower()
        builder = _SHAPES.get(shape, _sphere)

        means, normals = builder(self.n_points, self.radius)
        n = means.shape[0]  # builders may not hit n_points exactly
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        # Lambert shading -> per-point color gives a 3D-looking surface.
        lambert = 0.35 + 0.65 * np.clip(normals @ _LIGHT, 0.0, 1.0)  # [N]
        colors = np.clip(base_rgb[None, :] * lambert[:, None], 0.0, 1.0).astype(np.float32)

        scales = np.log(np.full((n, 3), 0.028, dtype=np.float32))
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0
        opacities = np.full((n,), 3.0, dtype=np.float32)  # logit -> ~0.95

        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), dtype=np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(colors)

        return GaussianField(means, scales, quats, opacities, sh, sh_degree=self.sh_degree)


class LGMGenerator:
    """Real LGM generator slot. Lazy heavy imports; NOT implemented in PoC.

    Wiring plan (<=4GB VRAM, PRD §7.2 / §8):
      1. Load Zero123++ -> synthesize multi-view images from a single view.
      2. Release Zero123++ from VRAM (sequential offload).
      3. Load LGM -> regress a GaussianField from the multi-view images.
    """

    def __init__(self) -> None:
        self._ready = False

    def _ensure(self) -> None:
        raise NotImplementedError(
            "LGMGenerator is a real-adapter slot. Wire Zero123++ -> LGM here "
            "(lazy torch/diffusers import, <=4GB sequential VRAM offload). "
            "Use MockGenerator for the CPU PoC."
        )

    def generate(self, mv_images: np.ndarray, cam_rays: Optional[Any] = None) -> GaussianField:
        self._ensure()
        raise AssertionError("unreachable")  # pragma: no cover
