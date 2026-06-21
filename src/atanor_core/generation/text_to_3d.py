"""Text -> 3DGS via a tiny Stable Diffusion model + the 2.5D lift (CPU).

Pipeline (the user's suggested route, made real and runnable):

    text prompt
      -> (1) tiny Stable Diffusion  (segmind/tiny-sd, distilled SD1.5)
            : prompt -> a single isolated RGB image @ 256x256
      -> (2) rembg cutout            (U²-Net background removal -> clean alpha)
      -> (3) Image25DGenerator       (silhouette inflation -> closed 3DGS volume)

This lets "사과" / "pikachu" / "a teapot" become the *actual* object instead of
a generic procedural sphere. tiny-sd runs on CPU in a few seconds per image
(~4s at 12 steps / 256px here), so it is feasible without a GPU.

Honesty (PRD §0.3): the result is a **single-view** reconstruction — SD imagines
one canonical view and we inflate its silhouette into a closed volume. It is not
multi-view-consistent novel-view synthesis (that's the GPU LGM path). Opt-in via
``SPLATRA_SD=1`` because it needs the diffusers stack + a one-time ~1.7GB
weight download. Model weights / APIs should be re-verified at wiring time.
"""

from __future__ import annotations

import os

import numpy as np

from ..domain.sgf import GaussianField
from .image_lift import Image25DGenerator

def _pick_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _default_model(device: str) -> str:
    if os.environ.get("SPLATRA_SD_MODEL"):
        return os.environ["SPLATRA_SD_MODEL"]
    # On a GPU, use SD-Turbo (1-4 steps, high quality, fits 16GB easily). On CPU,
    # fall back to the distilled tiny-sd (low quality but ~4s/image).
    return "stabilityai/sd-turbo" if device == "cuda" else "segmind/tiny-sd"


SD_MODEL = os.environ.get("SPLATRA_SD_MODEL", "")

# tiny-SD uses an English CLIP text encoder — Korean prompts produce washed-out
# blobs. A small noun map keeps the common Korean demo words working; unmapped
# text passes through. (For arbitrary languages, set SPLATRA_SD_MODEL to a
# multilingual SD, or type English.)
_KO_EN = {
    "사과": "red apple", "바나나": "banana", "딸기": "strawberry", "오렌지": "orange fruit",
    "피카츄": "pikachu, yellow pokemon", "포켓몬": "pokemon", "강아지": "puppy dog",
    "고양이": "cat", "토끼": "rabbit", "곰": "teddy bear", "공룡": "dinosaur",
    "자동차": "car", "비행기": "airplane", "로켓": "rocket", "배": "ship",
    "집": "house", "나무": "tree", "꽃": "flower", "별": "star", "하트": "red heart",
    "컵": "coffee cup", "책": "book", "시계": "clock", "축구공": "soccer ball",
    "버섯": "mushroom", "케이크": "cake", "도넛": "donut", "햄버거": "hamburger",
    "우산": "umbrella", "달": "moon", "지구": "planet earth", "왕관": "golden crown",
}


def _translate(prompt: str) -> str:
    out = prompt
    for ko, en in _KO_EN.items():
        if ko in out:
            out = out.replace(ko, en)
    return out.strip() or "object"


class TextTo3DGenerator:
    """Lazy SD text->image (GPU SD-Turbo / CPU tiny-sd), then cutout + lift."""

    def __init__(self, model: str = "", steps: int = 0, size: int = 0) -> None:
        self.device = _pick_device()
        self.model = model or _default_model(self.device)
        self._turbo = "turbo" in self.model.lower()
        # turbo: 1-4 steps, no CFG; regular SD: more steps + CFG.
        self.steps = steps or (3 if self._turbo else (16 if self.device == "cpu" else 25))
        self.size = size or (512 if self.device == "cuda" else 256)
        self._pipe = None
        self._lift = Image25DGenerator()

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model, torch_dtype=dtype, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(self.device)

    def image(self, prompt: str) -> np.ndarray:
        """prompt -> [H,W,3] float image (isolated subject on plain background)."""
        self._ensure()
        prompt = _translate(prompt)
        full = (f"a single {prompt}, the entire object fully visible and centered, "
                "not cropped, isolated on a plain solid white background, vivid "
                "saturated colors, sharp detail, clean studio product photo, even lighting")
        kw = dict(num_inference_steps=self.steps, height=self.size, width=self.size)
        if self._turbo:
            kw["guidance_scale"] = 0.0           # turbo models are CFG-free
        else:
            kw["guidance_scale"] = 7.0
            kw["negative_prompt"] = "multiple, cropped, text, watermark, busy background, shadow"
        out = self._pipe(full, **kw).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0

    def generate(self, prompt: str) -> GaussianField:
        rgb = self.image(prompt)
        from .bg import cutout

        rgba = cutout(rgb)                      # clean alpha if rembg present
        return self._lift.from_image(rgba if rgba is not None else rgb)
