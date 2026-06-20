"""3D object generation adapters.

* :class:`MockGenerator` — a deterministic average-color spherical blob. This
  is an explicit **mock** (PRD §7.2): it does NOT do real image-to-3D. Its job
  is to give the state machine a plausible GaussianField to converge onto so
  the cache-miss path can be tested on CPU.
* :class:`LGMGenerator` — the real adapter slot (LGM / 3DTopia + Zero123++).
  Lazily imports heavy deps; ``_ensure`` raises ``NotImplementedError`` to mark
  exactly where real wiring goes, including the <=4GB sequential-offload plan.
"""

from __future__ import annotations

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


class MockGenerator:
    """MOCK generator: average-color Fibonacci-sphere blob (deterministic)."""

    def __init__(self, n_points: int = 512, radius: float = 0.6, sh_degree: int = 1) -> None:
        self.n_points = int(n_points)
        self.radius = float(radius)
        self.sh_degree = int(sh_degree)

    def generate(self, mv_images: np.ndarray, cam_rays: np.ndarray) -> GaussianField:
        """Generate a blob whose color is the mean of the input views.

        Args:
            mv_images: [1, V, 3, H, W] multi-view images in [0, 1].
            cam_rays:  camera ray bundle (unused by the mock; kept for the
                       real-adapter signature contract).
        """
        mv = np.asarray(mv_images, dtype=np.float32)
        # Expected layout [B, V, 3, H, W]; mean over everything but the channel.
        if mv.ndim == 5 and mv.shape[2] == 3 and mv.size > 0:
            mean_rgb = mv.mean(axis=(0, 1, 3, 4))
        else:
            mean_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mean_rgb = np.clip(mean_rgb.astype(np.float32), 0.0, 1.0)

        n = self.n_points
        # Deterministic Fibonacci sphere.
        i = np.arange(n, dtype=np.float32)
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        means = (self.radius * np.stack([x, y, z], axis=1)).astype(np.float32)

        scales = np.log(np.full((n, 3), 0.04, dtype=np.float32))
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0
        opacities = np.full((n,), 2.0, dtype=np.float32)  # logit -> ~0.88

        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), dtype=np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(mean_rgb)

        return GaussianField(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh=sh,
            sh_degree=self.sh_degree,
        )


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
        # Mark exactly where real LGM/Zero123++ wiring goes.
        raise NotImplementedError(
            "LGMGenerator is a real-adapter slot. Wire Zero123++ -> LGM here "
            "(lazy torch/diffusers import, <=4GB sequential VRAM offload). "
            "Use MockGenerator for the CPU PoC."
        )

    def generate(self, mv_images: np.ndarray, cam_rays: np.ndarray) -> GaussianField:
        self._ensure()
        raise AssertionError("unreachable")  # pragma: no cover
