"""Learned single-image -> 3D via TripoSR (triplane transformer), as a point cloud.

The "learned full-3D" path (vs the geometric visual hull): TripoSR reconstructs a
**learned density+color field** from one image — it hallucinates the unseen sides
with a trained prior, so it fills geometry the visual hull can't. We query its
field on a grid and threshold into a colored point cloud (so we skip TripoSR's
``torchmcubes`` CUDA marching-cubes dependency entirely — we want points, not a
mesh).

    image -> TripoSR encoder -> triplane scene code
          -> query_triplane(grid) -> density + color
          -> threshold -> colored 3D point cloud -> GaussianField

GPU path (needs `.[sd]` + the TripoSR repo on PYTHONPATH + CUDA). On RTX 5080:
encode ~2.5s, field query ~0.1s. Honest: single-view, so the back is a learned
guess; quality is far above the silhouette hull but the unseen side can be soft.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc

# TripoSR repo location (cloned). Override with SPLATRA_TRIPOSR_DIR.
_TRIPOSR_DIR = os.environ.get("SPLATRA_TRIPOSR_DIR", "")


class TripoSRGenerator:
    def __init__(self, grid: int = 192, threshold: float = 25.0,
                 n_points: int = 170_000, sh_degree: int = 0) -> None:
        self.grid = int(grid)
        self.threshold = float(threshold)
        self.n_points = int(n_points)
        self.sh_degree = int(sh_degree)
        self._model = None
        self._t2i = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch

        if not torch.cuda.is_available():
            raise NotImplementedError("TripoSR needs CUDA; not available here.")
        if _TRIPOSR_DIR and _TRIPOSR_DIR not in sys.path:
            sys.path.insert(0, _TRIPOSR_DIR)
        # Stub torchmcubes: we sample the density field, never call marching cubes.
        if "torchmcubes" not in sys.modules:
            mc = types.ModuleType("torchmcubes")
            mc.marching_cubes = lambda *a, **k: (None, None)
            sys.modules["torchmcubes"] = mc
        try:
            from tsr.system import TSR
        except Exception as exc:  # pragma: no cover
            raise NotImplementedError(
                "TripoSR code not importable. Clone github.com/VAST-AI-Research/"
                "TripoSR and set SPLATRA_TRIPOSR_DIR to it."
            ) from exc
        model = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(131072)
        self._model = model.to("cuda").eval()
        self._torch = torch

    def from_image(self, image_rgb: np.ndarray) -> GaussianField:
        self._ensure()
        torch = self._torch
        from PIL import Image

        from .bg import cutout

        rgba = cutout(image_rgb)
        if rgba is None:
            rgba = image_rgb if image_rgb.shape[-1] == 4 else np.concatenate(
                [image_rgb, np.ones_like(image_rgb[..., :1])], -1)
        comp = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1 - rgba[..., 3:4])  # gray bg
        img = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8))

        m = self._model
        with torch.no_grad():
            scene = m([img], device="cuda")
            r = float(m.renderer.cfg.radius)
            N = self.grid
            lin = torch.linspace(-r, r, N, device="cuda")
            P = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
            dens = [m.renderer.query_triplane(m.decoder, ch, scene[0])["density_act"].squeeze(-1)
                    for ch in P.split(262144)]
            keep = torch.cat(dens) > self.threshold
            kpts = P[keep]
            if kpts.shape[0] < 64:
                raise RuntimeError("TripoSR produced an empty volume")
            cols = torch.cat([m.renderer.query_triplane(m.decoder, ch, scene[0])["color"]
                              for ch in kpts.split(262144)]).clamp(0, 1)
            kpts = kpts.detach().cpu().numpy()
            colors = cols.detach().cpu().numpy()

        if kpts.shape[0] > self.n_points:
            s = np.random.default_rng(0).choice(kpts.shape[0], self.n_points, replace=False)
            kpts, colors = kpts[s], colors[s]

        # TripoSR frame -> viewer frame (+y up, front toward camera):
        # rotate so the object stands upright (TripoSR is +z-up / lying).
        x, y, z = kpts[:, 0], kpts[:, 1], kpts[:, 2]
        kpts = np.stack([x, z, -y], axis=1).astype(np.float32)

        c = 0.5 * (kpts.max(0) + kpts.min(0))
        sc = (kpts.max(0) - kpts.min(0)).max() * 0.5 + 1e-6
        means = ((kpts - c) / sc).astype(np.float32)
        n = means.shape[0]
        px = 2.2 / self.grid
        scales = np.log(np.tile(np.array([px, px, px], np.float32), (n, 1)))
        quats = np.zeros((n, 4), np.float32); quats[:, 0] = 1.0
        opacities = np.full((n,), 2.2, np.float32)
        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(colors.astype(np.float32))
        return GaussianField(means, scales, quats, opacities, sh, sh_degree=self.sh_degree)

    def generate(self, prompt: str) -> GaussianField:
        if self._t2i is None:
            from .text_to_3d import TextTo3DGenerator
            self._t2i = TextTo3DGenerator()
        return self.from_image(self._t2i.image(prompt))
