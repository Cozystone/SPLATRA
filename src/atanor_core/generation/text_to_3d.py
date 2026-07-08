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
import re

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


_HYPER_REPO = "ByteDance/Hyper-SD"
_HYPER_FILE = "Hyper-SDXL-8steps-CFG-lora.safetensors"


def _hyper_lora_path():
    """Local path to the cached Hyper-SDXL 8-step CFG LoRA, or None if not present."""
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(_HYPER_REPO, _HYPER_FILE)
        return hit if isinstance(hit, str) else None
    except Exception:
        return None


def _sdxl_cached() -> bool:
    """True if SDXL-base weights are already downloaded (so we don't block a CPU
    box or a fresh GPU on a 6GB pull — fall back to sd-turbo until it's present)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo = "stabilityai/stable-diffusion-xl-base-1.0"
        # require the heavy UNet weight, not just the index — otherwise we'd select
        # SDXL mid-download and fail to load it.
        for f in ("unet/diffusion_pytorch_model.fp16.safetensors",
                  "unet/diffusion_pytorch_model.safetensors"):
            if isinstance(try_to_load_from_cache(repo, f), str):
                return True
        return False
    except Exception:
        return False


def _default_model(device: str) -> str:
    if os.environ.get("SPLATRA_SD_MODEL"):
        return os.environ["SPLATRA_SD_MODEL"]
    if device != "cuda":
        return "segmind/tiny-sd"            # CPU: distilled, low quality but ~4s
    # GPU interactive default: SD-Turbo reliably completes local text->image->2.5D
    # jobs in tens of seconds. SDXL can be higher quality, but it is heavy enough
    # to look like an indefinite pending job in the ATANOR dashboard, so require an
    # explicit opt-in or model override for that path.
    if os.environ.get("SPLATRA_SD_USE_SDXL", "0") == "1" and _sdxl_cached():
        return "stabilityai/stable-diffusion-xl-base-1.0"
    return "stabilityai/sd-turbo"


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

_KO_VISUAL_TERMS = [
    ("사실적인", "realistic"),
    ("고품질", "high quality"),
    ("입체적인", "volumetric"),
    ("유리", "glass"),
    ("투명", "transparent"),
    ("금속", "metal"),
    ("금속성", "metallic"),
    ("크롬", "chrome"),
    ("빛나는", "glowing"),
    ("황금색", "golden"),
    ("금색", "gold"),
    ("은색", "silver"),
    ("빨간", "red"),
    ("붉은", "red"),
    ("파란", "blue"),
    ("푸른", "blue"),
    ("초록", "green"),
    ("녹색", "green"),
    ("보라색", "purple"),
    ("검은", "black"),
    ("흰", "white"),
    ("하얀", "white"),
    ("사과", "apple"),
    ("찻잔", "tea cup"),
    ("컵", "cup"),
    ("로봇", "robot"),
    ("우주선", "spaceship"),
    ("자동차", "car"),
    ("비행기", "airplane"),
    ("고양이", "cat"),
    ("강아지", "dog"),
    ("나무", "tree"),
    ("꽃", "flower"),
    ("책", "book"),
    ("열린 책", "open book"),
    ("조각상", "statue"),
    ("구슬", "glass orb"),
    ("구체", "sphere"),
]

_KO_INSTRUCTION_TERMS = [
    "SPLATRA",
    "파티클",
    "입자",
    "직접",
    "생성",
    "만들어",
    "보여줘",
    "보여",
    "모델",
    "3D",
]


def _translate(prompt: str) -> str:
    source = prompt
    out = prompt
    for ko, en in _KO_EN.items():
        if ko in out:
            out = out.replace(ko, en)
    if re.search(r"[가-힣]", source):
        recognized = []
        for ko, en in _KO_VISUAL_TERMS:
            if ko in source and en not in recognized:
                recognized.append(en)
        ascii_tail = re.sub(r"[^\x00-\x7f]+", " ", out)
        for term in _KO_INSTRUCTION_TERMS:
            ascii_tail = ascii_tail.replace(term, " ")
        ascii_tail = re.sub(r"\s+", " ", ascii_tail).strip()
        if recognized:
            out = " ".join(recognized + ([ascii_tail] if ascii_tail else []))
        else:
            out = ascii_tail
    return out.strip() or "object"
_KO_EN_CLEAN = {
    "\uc720\ub9ac \uad6c\uc2ac": "translucent glass marble sphere with visible rim",
    "\uc720\ub9ac\uad6c\uc2ac": "translucent glass marble sphere with visible rim",
    "\uad6c\uc2ac": "translucent glass marble sphere with visible rim",
    "\uad6c\uccb4": "sphere",
    "\uacf5": "sphere",
    "\uc0ac\uacfc": "red apple",
    "\ube68\uac04 \uc0ac\uacfc": "red apple",
    "\ub85c\ubd07": "robot",
    "\ub098\ubb34": "tree",
    "\ucc45": "book",
    "\uc5f4\ub9b0 \ucc45": "open book",
    "빨간 사과": "red apple",
    "유리 구슬": "glass orb",
    "유리구슬": "glass orb",
    "사과": "red apple",
    "바나나": "banana",
    "딸기": "strawberry",
    "오렌지": "orange fruit",
    "고양이": "cat",
    "강아지": "puppy dog",
    "토끼": "rabbit",
    "곰": "teddy bear",
    "공룡": "dinosaur",
    "자동차": "car",
    "비행기": "airplane",
    "로켓": "rocket",
    "배": "ship",
    "집": "house",
    "나무": "tree",
    "꽃": "flower",
    "별": "star",
    "하트": "red heart",
    "커피잔": "coffee cup",
    "찻잔": "tea cup",
    "컵": "cup",
    "책": "book",
    "열린 책": "open book",
    "시계": "clock",
    "축구공": "soccer ball",
    "버섯": "mushroom",
    "케이크": "cake",
    "도넛": "donut",
    "햄버거": "hamburger",
    "우산": "umbrella",
    "달": "moon",
    "지구": "planet earth",
    "왕관": "golden crown",
    "로봇": "robot",
    "우주선": "spaceship",
    "조각상": "statue",
    "구슬": "glass orb",
    "구체": "sphere",
    "구": "sphere",
}

_KO_EN_CLEAN.update({
    "\uc720\ub9ac \uad6c\uc2ac": "translucent glass marble sphere with visible rim",
    "\uc720\ub9ac\uad6c\uc2ac": "translucent glass marble sphere with visible rim",
    "\uad6c\uc2ac": "translucent glass marble sphere with visible rim",
})

_KO_VISUAL_TERMS_CLEAN = [
    ("사실적인", "realistic"),
    ("실사", "photorealistic"),
    ("고품질", "high quality"),
    ("입체적인", "volumetric"),
    ("정교한", "detailed"),
    ("선명한", "sharp"),
    ("유리", "glass"),
    ("투명", "transparent"),
    ("반투명", "translucent"),
    ("금속", "metal"),
    ("크롬", "chrome"),
    ("빛나는", "glowing"),
    ("황금", "golden"),
    ("금색", "gold"),
    ("은색", "silver"),
    ("빨간", "red"),
    ("붉은", "red"),
    ("파란", "blue"),
    ("푸른", "blue"),
    ("초록", "green"),
    ("녹색", "green"),
    ("보라", "purple"),
    ("검은", "black"),
    ("하얀", "white"),
    ("흰", "white"),
]

_KO_INSTRUCTION_TERMS_CLEAN = [
    "SPLATRA",
    "스플라트라",
    "파티클",
    "입자",
    "직접",
    "생성",
    "만들어",
    "만들어줘",
    "보여줘",
    "보여",
    "모델",
    "홀로그램",
    "3D",
]

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def _translate_clean(prompt: str) -> str:
    """Translate Korean visual prompts before they reach English SD text encoders."""
    source = prompt.strip()
    if not _HANGUL_RE.search(source):
        return source or "object"
    recognized: list[str] = []
    normalized = source
    for ko, en in sorted(_KO_EN_CLEAN.items(), key=lambda item: len(item[0]), reverse=True):
        if ko in normalized and en not in recognized:
            recognized.append(en)
            normalized = normalized.replace(ko, " ")
    for ko, en in _KO_VISUAL_TERMS_CLEAN:
        if ko in source and en not in recognized:
            recognized.append(en)
            normalized = normalized.replace(ko, " ")
    ascii_tail = re.sub(r"[^\x00-\x7f]+", " ", normalized)
    for term in _KO_INSTRUCTION_TERMS_CLEAN:
        ascii_tail = ascii_tail.replace(term, " ")
    ascii_tail = re.sub(r"\s+", " ", ascii_tail).strip()
    if recognized:
        return " ".join(recognized + ([ascii_tail] if ascii_tail else []))
    return ascii_tail or "object"


def _expand_material_prompt(prompt: str) -> str:
    """Add generator-facing visual constraints for materials that SD under-specifies."""
    core = re.sub(r"\s+", " ", str(prompt or "").strip())
    low = core.lower()
    if (
        ("glass" in low or "transparent" in low or "translucent" in low)
        and ("orb" in low or "sphere" in low or "marble" in low)
    ):
        return (
            "translucent blue glass marble sphere, visible circular rim, glossy "
            "specular highlights, refractive caustic glow, round silhouette, "
            "single centered product render"
        )
    return core


class TextTo3DGenerator:
    """Lazy SD text->image (GPU SD-Turbo / CPU tiny-sd), then cutout + lift."""

    def __init__(self, model: str = "", steps: int = 0, size: int = 0) -> None:
        self.device = _pick_device()
        self.model = model or _default_model(self.device)
        self._turbo = "turbo" in self.model.lower()
        self._sdxl = "xl" in self.model.lower()
        # FAST MODE: a Hyper-SDXL *CFG* LoRA distills SDXL to ~8 steps while KEEPING
        # classifier-free guidance, so composition control (front view / white bg /
        # single object) survives. ~2x fewer steps than the 18-step base → big SD
        # speedup. Auto-on for SDXL on GPU when the LoRA is cached; off otherwise.
        # Hyper-SD few-step is OFF by default: measured ~8% faster at best (8 vs 18
        # steps) because the diffusion step is NOT the pipeline bottleneck here —
        # TripoSR dominates — and its slightly softer images trip best-of-N more
        # often, netting slower. Opt in with SPLATRA_FAST=1 on a VRAM-free GPU.
        self._fast = (self._sdxl and self.device == "cuda"
                      and os.environ.get("SPLATRA_FAST", "0") == "1"
                      and _hyper_lora_path() is not None)
        # LOW-VRAM: stream submodels CPU<->GPU so only the active one is resident.
        # Cuts our footprint ~9GB -> ~4-5GB so we coexist with Docker/Chrome on a
        # 16GB card without thrashing. Costs ~1-2s/gen in transfers but kills the
        # 60s VRAM-spill. Opt in with SPLATRA_LOWVRAM=1.
        self._lowvram = (self.device == "cuda"
                         and os.environ.get("SPLATRA_LOWVRAM", "0") == "1")
        # turbo: 1-4 steps, no CFG. Fast(Hyper-CFG): 8. Base SDXL/SD: 16-18 + CFG.
        self.steps = steps or (3 if self._turbo else
                               (8 if self._fast else
                                (16 if self.device == "cpu" else 18)))
        # SDXL: 768 instead of native 1024 — ~2x faster and far less VRAM (the 1024
        # activations nearly filled 16GB), still clearly better than SD1.x@512.
        self.size = size or (768 if self._sdxl else (512 if self.device == "cuda" else 256))
        self._pipe = None
        self._lift = Image25DGenerator()
        import threading
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def _ensure(self):
        if self._pipe is not None:
            return
        # The prewarm thread and the first request can both reach here (TripoSR
        # shares this generator), and two concurrent loads corrupt the pipe into
        # meta tensors. Serialize the load.
        with self._load_lock:
            if self._pipe is not None:
                return
            self._load()

    def _load(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        if self.device == "cuda":
            # Inference-speed flags: TF32 matmuls + cuDNN autotuning. Safe for
            # generation quality, ~10-20% faster on Ampere+/Blackwell.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kw = dict(torch_dtype=dtype, safety_checker=None)
        if self.device == "cuda":
            # we only fetch the fp16 weights — tell diffusers to load that variant
            # (otherwise it expects fp32 files, gets meta tensors, and .to() fails).
            kw["variant"] = "fp16"
            kw["use_safetensors"] = True
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(self.model, **kw)
        except Exception:
            kw.pop("variant", None)           # model without an fp16 variant
            kw.pop("use_safetensors", None)   # model may only ship .bin weights
            pipe = AutoPipelineForText2Image.from_pretrained(self.model, **kw)
        # Keep SDXL's shipped scheduler (EulerDiscrete). A DPM++/Karras override
        # tripped a last-step `sigmas[step_index+1]` IndexError in this diffusers
        # build; the default schedule is stable. Speed comes from 768px + fewer steps.
        pipe.set_progress_bar_config(disable=True)

        def _fuse_fast():
            # Fuse the Hyper-SDXL 8-step CFG LoRA, then recast to fp16 (fusing the
            # fp32 LoRA otherwise leaves mixed dtypes -> "Half vs float" crash).
            if not self._fast:
                return
            try:
                pipe.load_lora_weights(_hyper_lora_path())
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                pipe.to(dtype=dtype)
            except Exception:
                self._fast = False
                self.steps = 18

        if self._lowvram:
            # accelerate streams each submodel to the GPU only while it runs, then
            # back to CPU — peak GPU ~= the UNet (~5GB) instead of the whole ~9GB
            # pipeline, so we fit alongside Docker/Chrome. Do NOT .to(device) first.
            _fuse_fast()
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
            _fuse_fast()
        self._pipe = pipe
        # Snapshot the scheduler config so we can hand each generation a FRESH
        # scheduler — the EulerDiscrete scheduler is stateful and, once a call is
        # interrupted, leaves _step_index/sigmas corrupted so every later call
        # throws `sigmas[step_index+1]` IndexError until restart. Resetting per call
        # makes generation deterministic and self-healing.
        self._sched_cls = type(pipe.scheduler)
        self._sched_cfg = dict(pipe.scheduler.config)
        if self.device == "cuda":
            try:
                pipe.vae.enable_tiling()      # cut VAE-decode VRAM spikes (non-deprecated API)
            except Exception:
                pass
            if not self._lowvram:                 # offload juggles device; don't pin format
                try:
                    import torch
                    # channels-last speeds fp16 convolutions on the UNet/VAE.
                    pipe.unet.to(memory_format=torch.channels_last)
                    pipe.vae.to(memory_format=torch.channels_last)
                except Exception:
                    pass

    def warmup(self):
        """Run one throwaway generation so cuDNN autotuning / lazy CUDA kernels are
        paid during prewarm, not on the user's first request."""
        try:
            self._ensure()
            if self.device == "cuda":
                self._images("a sphere", 1)
                self._free_cache()
        except Exception:
            pass

    def _reset_scheduler(self):
        try:
            self._pipe.scheduler = self._sched_cls.from_config(self._sched_cfg)
        except Exception:
            pass

    def _frame_score(self, rgb: np.ndarray) -> float:
        """Heuristic 0-1 score of how well an image is framed for TripoSR: a single
        centered subject, fully visible (silhouette not touching the border), on a
        clean light background, occupying a reasonable fraction of the frame. Used
        to pick the best of N samples — SD-Turbo's framing varies a lot by seed."""
        H, W = rgb.shape[:2]
        # Gate on a genuinely light, uniform background. A frame-filling macro shot
        # has no clean border, so the cutout is unreliable there — reject it up front
        # instead of trusting a brightness key that the dark fruit body fools.
        c = np.concatenate([rgb[:10, :10], rgb[:10, -10:], rgb[-10:, :10], rgb[-10:, -10:]])
        bg_clean = float(np.clip(c.mean() * 1.4 - c.std() * 3.0 - 0.2, 0, 1))
        if c.mean() < 0.55:                   # background not light -> macro/busy
            return 0.05 * bg_clean
        from .bg import cutout
        rgba = cutout(rgb)
        if rgba is None:
            fg = rgb.mean(-1) < (c.mean() - 0.12)   # darker than the light bg
        else:
            fg = rgba[..., 3] > 0.5
        area = fg.mean()
        if area < 0.04:                       # nothing / tiny -> bad
            return 0.0
        ys, xs = np.where(fg)
        # 1) not cropped: silhouette should not hug the frame border
        border = ((xs < 2).mean() + (xs > W - 3).mean()
                  + (ys < 2).mean() + (ys > H - 3).mean())
        crop_pen = min(1.0, border * 6.0)
        # 2) good size: subject ~12-55% of frame (penalize both tiny and frame-filling)
        size_score = float(np.clip((area - 0.04) / 0.16, 0, 1)
                           * np.clip((0.65 - area) / 0.20, 0, 1))
        # 3) centered
        cy, cx = ys.mean() / H, xs.mean() / W
        center = 1.0 - min(1.0, (abs(cy - 0.5) + abs(cx - 0.5)) * 1.6)
        # 4) SINGLE object: a duplicated "two-apple" image splits into >1 big blob.
        # Penalize so best-of-N rejects it (TripoSR fuses multiple blobs into a mess).
        single = self._single_object_score(fg)
        return (1 - crop_pen) * single * (0.40 * size_score + 0.25 * center
                                          + 0.20 * bg_clean + 0.15)

    @staticmethod
    def _single_object_score(fg: np.ndarray) -> float:
        """1.0 for one connected blob, lower when the foreground splits into several
        large components (SDXL sometimes draws 2-3 copies despite 'a single ...')."""
        try:
            from scipy import ndimage
            small = fg[::4, ::4]                  # downscale for speed
            lab, k = ndimage.label(small)
            if k <= 1:
                return 1.0
            sizes = np.bincount(lab.ravel())[1:]
            sizes.sort()
            big = sizes[sizes > sizes.max() * 0.18]   # components within ~5x of largest
            return 1.0 if len(big) <= 1 else 0.45     # multi-object -> strong penalty
        except Exception:
            return 1.0

    def image(self, prompt: str, n: int = 0) -> np.ndarray:
        """prompt -> [H,W,3] float image. Generates up to ``n`` candidates ONE AT A
        TIME (batch=1) and returns the best-framed one. Sequential (not batched) so
        peak VRAM stays at single-image level — batching N at 768px doubled the
        activation memory and, with other GPU apps co-resident, spilled past 16GB
        and ground to a halt. Early-outs as soon as a clean single-object candidate
        appears, so the common case costs one generation."""
        self._ensure()
        if n <= 0:
            n = int(os.environ.get("SPLATRA_SD_BESTOF",
                                   "2" if self.device == "cuda" else "1"))
        n = max(1, n)
        best, best_s, last_err = None, -1.0, None
        for _ in range(n):
            try:
                cand = self._images(prompt, 1)[0]
            except Exception as exc:
                last_err = exc
                continue
            s = self._frame_score(cand)
            if s > best_s:
                best, best_s = cand, s
            if best_s >= 0.6:                 # good enough — skip remaining candidates
                break
        self._free_cache()
        if best is None:
            raise last_err or RuntimeError("image generation produced no candidates")
        return best

    def _free_cache(self):
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()      # release peak activation cache
            except Exception:
                pass

    def _images(self, prompt: str, n: int) -> list:
        """prompt -> list of n [H,W,3] float images (isolated subject), one batched call.

        Prompt kept under CLIP's 77-token limit — a longer one gets silently
        truncated, dropping the trailing 'white background / lighting' cues and
        degrading framing quality."""
        prompt = _translate_clean(prompt)
        # Accept sentence-form prompts ("a girl wearing a hat") — drop a leading
        # article so "a single ..." reads cleanly, and say "subject" not "object"
        # so it isn't odd for people/animals.
        core = _expand_material_prompt(prompt)
        for art in ("a ", "an ", "the "):
            if core.lower().startswith(art):
                core = core[len(art):]
                break
        full = (f"a single {core}, three-quarter front view, upright, the full "
                "subject fully visible and centered with margin, isolated on a "
                "plain white background, sharp, even studio lighting")
        kw = dict(num_inference_steps=self.steps, height=self.size, width=self.size,
                  num_images_per_prompt=n)
        if self._turbo:
            kw["guidance_scale"] = 0.0           # turbo models are CFG-free
        else:
            kw["guidance_scale"] = 7.0
            kw["negative_prompt"] = ("two, pair, multiple, several, group, duplicate, "
                                     "collage, top-down, from above, overhead, "
                                     "bird's eye, bottom view, tilted, close-up, "
                                     "macro, extreme zoom, cropped, partial, "
                                     "text, watermark, busy background, shadow")
        if self.device == "cuda" and not self._turbo:
            self._reset_scheduler()           # fresh scheduler state per generation
        with self._gen_lock:
            imgs = self._pipe(full, **kw).images
        return [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in imgs]

    def generate(self, prompt: str) -> GaussianField:
        rgb = self.image(prompt)
        from .bg import cutout

        rgba = cutout(rgb)                      # clean alpha if rembg present
        return self._lift.from_image(rgba if rgba is not None else rgb)
