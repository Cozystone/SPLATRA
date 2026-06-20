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

SD_MODEL = os.environ.get("SPLATRA_SD_MODEL", "segmind/tiny-sd")

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
    """Lazy tiny-SD text->image, then cutout + inflation lift to a GaussianField."""

    def __init__(self, model: str = SD_MODEL, steps: int = 16, size: int = 256) -> None:
        self.model = model
        self.steps = int(steps)
        self.size = int(size)
        self._pipe = None
        self._lift = Image25DGenerator()

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model, torch_dtype=torch.float32, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to("cpu")

    def image(self, prompt: str) -> np.ndarray:
        """prompt -> [H,W,3] float image (isolated subject on plain background)."""
        self._ensure()
        prompt = _translate(prompt)
        full = (f"a single {prompt}, centered, isolated on a plain solid white "
                "background, full object in frame, vivid saturated colors, detailed, "
                "product photo, soft studio lighting")
        out = self._pipe(
            full,
            negative_prompt="multiple, cropped, text, watermark, busy background, shadow",
            num_inference_steps=self.steps,
            guidance_scale=7.0,
            height=self.size,
            width=self.size,
        ).images[0]
        return np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0

    def generate(self, prompt: str) -> GaussianField:
        rgb = self.image(prompt)
        from .bg import cutout

        rgba = cutout(rgb)                      # clean alpha if rembg present
        return self._lift.from_image(rgba if rgba is not None else rgb)
