"""Spectral Gaussian Field (SGF) — pure data model.

This module is the heart of the unified visual representation. It depends on
**numpy only** — no torch, no gsplat, no CUDA. The AI-OS kernel can ``import``
the domain layer without dragging in GPU dependencies.

Theory (see PRD §2): every visual state — a static graph layout, a dynamic
morph, or a freshly generated object — is expressed as a single set of Gaussian
primitives whose spatio-temporal evolution is a spectral series::

    a(t) = a_DC                                  # static (DC term)
         + Σ_n poly_n · t^n                       # polynomial drift
         + anneal(t) · Σ_k [α_k sin + β_k cos]    # Fourier "smoke" turbulence

The DC term is the static field; the AC (higher-order) terms are the morph.
The same Gaussian buffer and the same rasterizer render all three cases.

Honesty note (PRD §2, §7): the *unified SGF framing* is a reasonable but
unpublished frame. Its constituent parts (3DGS rasterization, Gaussian-Flow
DDDM-style deformation, LGM generation, SOG compression, SDS, PSNR
verification) are all published, validated techniques. If SGF-as-a-frame is
wrong, the system still works as the sum of its validated parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Dict, Optional

import numpy as np

# Zeroth-order real spherical harmonic coefficient (the DC band constant).
# Used to convert between linear RGB in [0,1] and the SH DC band.
C0: float = 0.28209479177387814


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB in [0,1] to the spherical-harmonic DC band.

    Inverse of :func:`sh_dc_to_rgb`. ``sh = (rgb - 0.5) / C0``.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb - 0.5) / C0


def sh_dc_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    """Convert the spherical-harmonic DC band back to linear RGB.

    Inverse of :func:`rgb_to_sh_dc`. ``rgb = sh * C0 + 0.5``.
    The result is **not** clipped here; clip downstream if rendering.
    """
    sh_dc = np.asarray(sh_dc, dtype=np.float32)
    return sh_dc * C0 + 0.5


@dataclass
class GaussianField:
    """A set of 3D Gaussian primitives — the renderable buffer.

    Attributes (all numpy, float32):
        means:      [N, 3]      world-space centers
        scales:     [N, 3]      log-space scales (exp() at render time)
        quats:      [N, 4]      rotation quaternions (w, x, y, z)
        opacities:  [N]         logit-space opacity (sigmoid() at render time)
        sh:         [N, K, 3]   spherical-harmonic color coefficients;
                                K = (sh_degree + 1)**2, band 0 == DC color
        sh_degree:  int         SH degree (0 => K == 1, DC only)
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    sh_degree: int = 1

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.quats = np.asarray(self.quats, dtype=np.float32)
        self.opacities = np.asarray(self.opacities, dtype=np.float32)
        self.sh = np.asarray(self.sh, dtype=np.float32)

        n = self.means.shape[0]
        k = (self.sh_degree + 1) ** 2
        assert self.means.shape == (n, 3), f"means must be [N,3], got {self.means.shape}"
        assert self.scales.shape == (n, 3), f"scales must be [N,3], got {self.scales.shape}"
        assert self.quats.shape == (n, 4), f"quats must be [N,4], got {self.quats.shape}"
        assert self.opacities.shape == (n,), f"opacities must be [N], got {self.opacities.shape}"
        assert self.sh.shape == (n, k, 3), (
            f"sh must be [N,{k},3] for sh_degree={self.sh_degree}, got {self.sh.shape}"
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def nbytes(self) -> int:
        """Total raw byte footprint of the buffers (for honest size summaries)."""
        return int(
            self.means.nbytes
            + self.scales.nbytes
            + self.quats.nbytes
            + self.opacities.nbytes
            + self.sh.nbytes
        )

    def copy(self) -> "GaussianField":
        return GaussianField(
            means=self.means.copy(),
            scales=self.scales.copy(),
            quats=self.quats.copy(),
            opacities=self.opacities.copy(),
            sh=self.sh.copy(),
            sh_degree=self.sh_degree,
        )


@dataclass
class DeformationCoeffs:
    """Spectral deformation coefficients for the AC (morphing) terms.

    Position at time ``t`` is::

        pos(t) = origin
               + Σ_n poly[:, :, n] · t^n
               + anneal(t) · Σ_k (fourier sin/cos basis)

    Attributes:
        poly:    [N, 3, n_p]      polynomial coefficients (smooth drift)
        fourier: [N, 3, 2*n_f]    Fourier coefficients [α_1..α_nf, β_1..β_nf]
        n_p:     int              number of polynomial terms
        n_f:     int              number of Fourier harmonics
        period:  float            morph period T
        origin:  [N, 3]           start positions
        target:  [N, 3]           end positions
    """

    poly: np.ndarray
    fourier: np.ndarray
    n_p: int
    n_f: int
    period: float
    origin: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        self.poly = np.asarray(self.poly, dtype=np.float32)
        self.fourier = np.asarray(self.fourier, dtype=np.float32)
        self.origin = np.asarray(self.origin, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        n = self.origin.shape[0]
        assert self.poly.shape == (n, 3, self.n_p)
        assert self.fourier.shape == (n, 3, 2 * self.n_f)
        assert self.origin.shape == (n, 3)
        assert self.target.shape == (n, 3)

    @staticmethod
    def zeros(n: int, n_p: int = 3, n_f: int = 8, period: float = 1.0) -> "DeformationCoeffs":
        """All-zero coefficients with origin == target == zeros."""
        z = np.zeros((n, 3), dtype=np.float32)
        return DeformationCoeffs(
            poly=np.zeros((n, 3, n_p), dtype=np.float32),
            fourier=np.zeros((n, 3, 2 * n_f), dtype=np.float32),
            n_p=n_p,
            n_f=n_f,
            period=float(period),
            origin=z.copy(),
            target=z.copy(),
        )


@dataclass
class Cartridge:
    """A compressed, named, optionally-verified SGF payload.

    A cartridge is the *handle* that travels over the wire — hundreds of KB,
    not the raw multi-MB buffer. The viewer pulls it on a side channel.
    """

    name: str
    field: GaussianField
    meta: Dict[str, Any] = _field(default_factory=dict)
    verified: bool = False
