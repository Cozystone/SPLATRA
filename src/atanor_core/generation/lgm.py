"""Real image -> 3DGS generation via LGM (Large Multi-view Gaussian Model).

End-to-end feed-forward pipeline (PRD §3.3 / §8 "real adapter"):

    single image
      -> (1) multi-view diffusion  (ImageDream / MVDream, image-conditioned)
            : 1 image  ->  4 orthogonal RGB views @ 256x256
      -> (2) LGM asymmetric U-Net
            : [1,4,9,256,256] (rgb + Plücker ray embedding)  ->  per-pixel
              14-channel Gaussians (xyz, opacity, scale, rotation, rgb)
      -> (3) pack into a SGF GaussianField (the cartridge the viewer renders)

VRAM ≤ 4GB strategy (PRD §2.2):
  * **Sequential offload** — load + run ImageDream, free it from VRAM, *then*
    load + run LGM. The two big models never co-reside on the GPU.
  * **Quantization** — INT8 (bitsandbytes) by default, optional INT4, applied to
    the diffusion UNet and the LGM backbone. fp16 fallback if bnb is absent.
  * **Attention/VAE slicing + xformers** to cap the diffusion activation peak.

HONESTY (PRD §0.3, §7): this is **real wiring**, not a mock. It calls the actual
published models. But it requires the `gen` extra (torch/diffusers/…), a CUDA
GPU, and a one-time weight download — none of which exist in the CPU/numpy PoC
test environment, so this exact path is **not executed by `pytest`** and has not
been run here. ``_ensure()`` raises a clear, actionable error when the stack is
missing; the API falls back to the procedural ``MockGenerator`` with an explicit
note so the product never silently pretends. Model IDs / activation conventions
must be re-verified against the installed repos at wiring time (PRD §7.6).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc

# Published weights (re-verify at wiring time — PRD §7.6).
MVDREAM_ID = os.environ.get("SPLATRA_MVDREAM_ID", "ashawkey/imagedream-ipmv-diffusers")
LGM_ID = os.environ.get("SPLATRA_LGM_ID", "ashawkey/LGM")
LGM_CKPT = os.environ.get("SPLATRA_LGM_CKPT", "model_fp16.safetensors")

# The four canonical LGM views (azimuth°, elevation°).
LGM_VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class LGMGenerator:
    """Feed-forward image->3DGS adapter. Lazy heavy imports; GPU + weights req'd.

    Args:
        device: torch device ("cuda" expected; "cpu" works but is very slow).
        quant: "int8" | "int4" | "none" — backbone quantization for VRAM budget.
        fp16: run diffusion/LGM in half precision.
        n_points_cap: optional cap on returned Gaussians (downsample for the
            lightweight cartridge / viewer; LGM emits ~250k by default).
    """

    def __init__(
        self,
        device: str = "cuda",
        quant: str = "int8",
        fp16: bool = True,
        n_points_cap: Optional[int] = 120_000,
        sh_degree: int = 1,
    ) -> None:
        self.device = device
        self.quant = quant
        self.fp16 = fp16
        self.n_points_cap = n_points_cap
        self.sh_degree = int(sh_degree)
        self._torch = None
        self._mv = None      # ImageDream pipeline (loaded on demand, then freed)
        self._lgm = None     # LGM model (loaded on demand)

    # -- lazy stack -------------------------------------------------------- #
    def _ensure(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            import kiui  # noqa: F401  # LGM utilities (kiuikit)
        except Exception as exc:  # pragma: no cover - gen extra not installed
            raise NotImplementedError(
                "LGMGenerator needs the GPU generation stack. Install it on a "
                "CUDA box: `pip install -e \".[gen]\"` (torch, diffusers, "
                "transformers, accelerate, kiui, bitsandbytes, safetensors). "
                "This path is intentionally NOT part of the CPU/numpy PoC."
            ) from exc

        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise NotImplementedError(
                "CUDA not available. LGM image->3D is a GPU path; the CPU PoC "
                "uses MockGenerator. Set device='cpu' only for slow debugging."
            )
        self._torch = torch

    # -- (0) bitsandbytes / fp16 quant config ------------------------------ #
    def _dtype(self):
        return self._torch.float16 if self.fp16 else self._torch.float32

    def _maybe_quantize(self, module):
        """Apply INT8/INT4 to a backbone in-place where supported (bnb)."""
        if self.quant in ("int8", "int4"):
            try:
                import bitsandbytes as bnb  # noqa: F401

                # In production: replace nn.Linear with bnb.nn.Linear8bitLt /
                # Linear4bit, or load via BitsAndBytesConfig. Kept explicit so
                # the VRAM strategy is visible, not hidden behind "trust me".
                from .quant import quantize_linears  # local helper

                quantize_linears(module, bits=4 if self.quant == "int4" else 8)
            except Exception:
                # bnb missing -> fp16 only (still fits on most 6GB+; 4GB tight).
                pass
        return module

    # -- (1) multi-view diffusion: 1 image -> 4 views ---------------------- #
    def _diffuse_views(self, image_rgb: np.ndarray) -> np.ndarray:
        """[H,W,3] in [0,1] -> [4,3,256,256] multi-view tensor (numpy)."""
        torch = self._torch
        from diffusers import DiffusionPipeline

        self._mv = DiffusionPipeline.from_pretrained(
            MVDREAM_ID,
            custom_pipeline=MVDREAM_ID,  # repo ships its MV pipeline
            torch_dtype=self._dtype(),
            trust_remote_code=True,
        ).to(self.device)
        self._mv = self._maybe_quantize(self._mv)
        # cap activation peak for the 4GB budget
        try:
            self._mv.enable_attention_slicing()
            self._mv.enable_vae_slicing()
            self._mv.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        import kiui

        img = kiui.op.recenter(image_rgb, np.ones_like(image_rgb[..., :1]), border_ratio=0.2)
        views = self._mv(
            prompt="", image=img, guidance_scale=5.0, num_inference_steps=30, elevation=0
        )  # repo returns [4,256,256,3] in [0,1]
        views = np.asarray(views, dtype=np.float32)
        if views.ndim == 4 and views.shape[-1] == 3:
            views = np.transpose(views, (0, 3, 1, 2))  # -> [4,3,H,W]

        # SEQUENTIAL OFFLOAD: free the diffusion model before loading LGM.
        del self._mv
        self._mv = None
        torch.cuda.empty_cache()
        return views

    # -- (2) Plücker ray embeddings for the 4 fixed cameras ---------------- #
    def _plucker(self, h: int, w: int):
        """[4,6,H,W] Plücker ray embedding for the canonical LGM cameras."""
        import kiui
        from kiui.cam import orbit_camera

        torch = self._torch
        rays = []
        for az, el in LGM_VIEWS:
            c2w = torch.from_numpy(orbit_camera(el, az, radius=1.5)).to(self.device)
            # ray origins + directions -> Plücker (o x d, d)
            embed = kiui.cam.get_plucker_embedding(c2w, h, w, fovy=49.1)
            rays.append(embed)
        return torch.stack(rays, dim=0).to(self._dtype())

    # -- (3) LGM forward: views -> Gaussians ------------------------------- #
    def _run_lgm(self, views: np.ndarray) -> "Any":
        torch = self._torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Build the LGM model (asymmetric U-Net). The class ships in the LGM
        # repo / kiui; import is wrapped so the missing-dep error is actionable.
        try:
            from core.models import LGM  # LGM repo layout
            from core.options import config_defaults
            opt = config_defaults["big"]
        except Exception as exc:  # pragma: no cover
            raise NotImplementedError(
                "LGM model code not importable. Clone github.com/3DTopia/LGM "
                "(or `pip install lgm`) so `core.models.LGM` is on the path."
            ) from exc

        model = LGM(opt)
        ckpt = load_file(hf_hub_download(LGM_ID, LGM_CKPT))
        model.load_state_dict(ckpt, strict=False)
        model = self._maybe_quantize(model).half().to(self.device).eval()
        self._lgm = model

        v = torch.from_numpy(views).to(self.device, self._dtype())  # [4,3,H,W]
        v = torch.nn.functional.interpolate(v, size=(256, 256), mode="bilinear", align_corners=False)
        rays = self._plucker(256, 256)                               # [4,6,256,256]
        inp = torch.cat([v, rays], dim=1).unsqueeze(0)               # [1,4,9,256,256]
        with torch.no_grad():
            gaussians = model.forward_gaussians(inp)                 # [1,N,14]
        return gaussians[0].float().cpu().numpy()

    # -- public API -------------------------------------------------------- #
    def from_image(self, image_rgb: np.ndarray) -> GaussianField:
        """Single RGB image [H,W,3] in [0,1] -> reconstructed GaussianField."""
        self._ensure()
        views = self._diffuse_views(image_rgb)     # (1) MV diffusion + offload
        g = self._run_lgm(views)                    # (2)+(3) LGM forward
        return self._to_field(g)

    def generate(self, mv_images: np.ndarray, cam_rays: Optional[Any] = None) -> GaussianField:
        """GeneratorPort contract. If 4-view images are supplied, skip diffusion.

        ``mv_images``: [1,4,3,H,W] in [0,1] (already multi-view) OR a single
        image routed through diffusion.
        """
        self._ensure()
        mv = np.asarray(mv_images, dtype=np.float32)
        if mv.ndim == 5 and mv.shape[1] >= 4:
            views = mv[0, :4]                        # [4,3,H,W] given directly
            g = self._run_lgm(views)
            return self._to_field(g)
        # fall back to treating it as one image -> diffuse
        img = mv.reshape(-1, mv.shape[-2], mv.shape[-1])[:3]
        return self.from_image(np.transpose(img, (1, 2, 0)))

    # -- LGM gaussians [N,14] -> SGF GaussianField ------------------------- #
    def _to_field(self, g: np.ndarray) -> GaussianField:
        g = np.asarray(g, dtype=np.float32)
        xyz = g[:, 0:3]
        opacity = g[:, 3:4]                  # already sigmoid-activated in LGM
        scale = g[:, 4:7]                    # already exp/softplus-activated
        rot = g[:, 7:11]                     # LGM order (x,y,z,w) — reorder
        rgb = g[:, 11:14]                    # already sigmoid-activated

        # optional downsample for the lightweight cartridge
        n = xyz.shape[0]
        if self.n_points_cap and n > self.n_points_cap:
            idx = np.random.default_rng(0).choice(n, self.n_points_cap, replace=False)
            xyz, opacity, scale, rot, rgb = xyz[idx], opacity[idx], scale[idx], rot[idx], rgb[idx]
            n = self.n_points_cap

        # normalize into the [-1,1] cube the viewer expects
        c = 0.5 * (xyz.max(0) + xyz.min(0))
        s = (xyz.max(0) - xyz.min(0)).max() * 0.5 + 1e-6
        means = ((xyz - c) / s).astype(np.float32)
        scales = np.log(np.clip(scale / s, 1e-4, None)).astype(np.float32)  # -> log space
        quats = rot[:, [3, 0, 1, 2]].astype(np.float32)                     # (w,x,y,z)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
        opacities = _logit(opacity[:, 0]).astype(np.float32)                # -> logit space

        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), dtype=np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(np.clip(rgb, 0.0, 1.0))
        return GaussianField(means, scales, quats, opacities, sh, sh_degree=self.sh_degree)
