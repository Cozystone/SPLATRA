"""Part understanding for a reconstructed shell — every point learns what it is.

Bolting separately generated parts onto a reconstructed body was always a losing
fight: the planner guesses positions blind, the reconstruction already contains
its own wheels, and the two never quite agree (the snap/carve seating in the API
is damage control for exactly that). This module goes at the problem from the
other end: generate ONE crisp shell, then teach each of its points what it is.

Primary path — ask the image. CLIPSeg is prompted with the planner's own part
vocabulary ("wheel", "window", "headlight"...) against the very photograph the
shell was lifted from; measured on a car it puts wheels exactly on the tyres and
lamps exactly on the lamps. Points are projected back onto that image to pick up
seed labels, and the labels are then grown through 3D space by neighbourhood
majority until every point belongs to a part. Costs about a second and adds no
new model — the same CLIPSeg that already does materials.

Experimental alternative (SPLATRA_PARTCRAFTER=1) — PartCrafter (NeurIPS 2025,
MIT) jointly generates K aligned part meshes from the image and the shell points
take the label of the nearest part surface. Measured honestly: ~96s on the 5080
and on our photoreal SD inputs it splits objects into halves rather than into
wheels-and-body, so it stays off by default. The alignment (chamfer-scored yaw +
per-axis affine) and nearest-surface labelling live here and work; the weak link
is the decomposition itself on out-of-distribution imagery.
"""

from __future__ import annotations

import os
import sys
import types
from typing import List, Optional, Tuple

import numpy as np

_PC_DIR = os.environ.get(
    "SPLATRA_PARTCRAFTER_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "splatra", "PartCrafter"))


def available() -> bool:
    return os.path.isdir(_PC_DIR)


class PartLabeler:
    def __init__(self, num_tokens: int = 1024, steps: int = 50,
                 guidance: float = 7.0) -> None:
        self.num_tokens = int(num_tokens)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self._pipe = None
        self._torch = None

    # ------------------------------------------------------------------ setup
    def _ensure(self):
        if self._pipe is not None:
            return
        import torch

        if not torch.cuda.is_available():
            raise NotImplementedError("PartCrafter needs CUDA")
        if "torch_cluster" not in sys.modules:   # encoder-only dep; we decode
            tc = types.ModuleType("torch_cluster")

            def _fps(*a, **k):
                raise RuntimeError("PartCrafter encoder path is not used here")

            tc.fps = _fps
            sys.modules["torch_cluster"] = tc
        if _PC_DIR not in sys.path:
            sys.path.insert(0, _PC_DIR)
        from huggingface_hub import snapshot_download
        from transformers import BitImageProcessor, Dinov2Model

        from src.models.autoencoders import TripoSGVAEModel
        from src.models.transformers import PartCrafterDiTModel
        from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
        from src.schedulers.scheduling_rectified_flow import RectifiedFlowScheduler

        d = snapshot_download("wgsxm/PartCrafter")
        # assembled by hand: recent diffusers refuses to auto-resolve the
        # repo-local scheduler module named in model_index.json
        self._pipe = PartCrafterPipeline(
            vae=TripoSGVAEModel.from_pretrained(d, subfolder="vae"),
            transformer=PartCrafterDiTModel.from_pretrained(d, subfolder="transformer"),
            scheduler=RectifiedFlowScheduler.from_pretrained(d, subfolder="scheduler"),
            image_encoder_dinov2=Dinov2Model.from_pretrained(d, subfolder="image_encoder_dinov2"),
            feature_extractor_dinov2=BitImageProcessor.from_pretrained(d, subfolder="feature_extractor_dinov2"),
        ).to("cuda", torch.float16)
        self._torch = torch

    def park(self) -> None:
        """Lend the GPU out between calls; weights stay warm in system RAM."""
        if self._pipe is not None:
            try:
                self._pipe.to("cpu")
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------- generation
    def part_meshes(self, image_rgb: np.ndarray, num_parts: int,
                    seed: int = 0) -> List[np.ndarray]:
        """One image -> K jointly generated part surfaces, as point arrays.

        Returns a list of [Ni,3] float32 surface samples, one per part, in
        PartCrafter's own canonical frame. Parts that fail to decode are
        dropped rather than faked.
        """
        self._ensure()
        torch = self._torch
        from PIL import Image

        try:
            if next(self._pipe.transformer.parameters()).device.type != "cuda":
                self._pipe.to("cuda")
        except Exception:
            pass

        n = int(max(2, min(16, num_parts)))
        img = Image.fromarray((np.clip(image_rgb, 0, 1) * 255).astype(np.uint8))
        with torch.no_grad():
            meshes = self._pipe(
                image=[img] * n,
                attention_kwargs={"num_parts": n},
                num_tokens=self.num_tokens,
                generator=torch.Generator(device="cuda").manual_seed(int(seed)),
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                max_num_expanded_coords=int(1e9),
                use_flash_decoder=False,
            ).meshes
        import trimesh

        out: List[np.ndarray] = []
        for m in meshes:
            if m is None or len(m.faces) < 8:
                continue
            k = int(max(1500, min(20000, m.area * 40000)))
            pts, _ = trimesh.sample.sample_surface(m, k)
            out.append(np.asarray(pts, np.float32))
        return out


# ---------------------------------------------------------------- pure labelling
def _normalize(pts: np.ndarray) -> np.ndarray:
    c = 0.5 * (pts.max(0) + pts.min(0))
    s = float((pts.max(0) - pts.min(0)).max()) * 0.5 + 1e-9
    return ((pts - c) / s).astype(np.float32)


def _yaw(pts: np.ndarray, k: int) -> np.ndarray:
    """Rotate k*90 degrees about +y."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], 1).astype(np.float32)


def _chamfer(a: np.ndarray, b: np.ndarray, cap: int = 4000,
             seed: int = 0) -> float:
    """Symmetric mean nearest-neighbour distance between two surface clouds.

    Voxel IoU was tried first and is the wrong instrument here: both clouds are
    thin surfaces, so even a good alignment shares few voxels and every yaw
    scores in the noise. Chamfer distance measures how far each surface sits
    from the other, which is exactly the thing alignment changes.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    sa = a[rng.choice(a.shape[0], min(cap, a.shape[0]), replace=False)]
    sb = b[rng.choice(b.shape[0], min(cap, b.shape[0]), replace=False)]
    da = cKDTree(sb).query(sa, k=1)[0].mean()
    db = cKDTree(sa).query(sb, k=1)[0].mean()
    return float(0.5 * (da + db))


def align_parts(shell_pts: np.ndarray,
                part_pts: List[np.ndarray]) -> Tuple[List[np.ndarray], int, float]:
    """Bring PartCrafter parts into the shell's frame.

    Both clouds were made from the same photograph but live in different
    canonical frames, and PartCrafter's proportions drift (a soft model makes a
    stubbier car). So: normalise, pick the yaw whose chamfer distance to the
    shell is smallest, then stretch each axis so the part cloud's bounding box
    matches the shell's exactly — the same object may not disagree about its own
    wheelbase. Returns (aligned parts, yaw index, chamfer in shell units).
    """
    union = np.concatenate(part_pts, 0)
    c = 0.5 * (union.max(0) + union.min(0))
    s = float((union.max(0) - union.min(0)).max()) * 0.5 + 1e-9
    shell_n = _normalize(shell_pts)
    union_n = ((union - c) / s).astype(np.float32)

    best_k, best_d = 0, np.inf
    for k in range(4):
        d = _chamfer(shell_n, _yaw(union_n, k))
        if d < best_d:
            best_k, best_d = k, d

    # per-axis affine: after the yaw, make the part cloud occupy the shell's
    # exact bounding box, axis by axis
    ry = _yaw(union_n, best_k)
    lo_p, hi_p = ry.min(0), ry.max(0)
    sc = 0.5 * (shell_pts.max(0) + shell_pts.min(0))
    lo_s = shell_pts.min(0) - sc
    hi_s = shell_pts.max(0) - sc
    scale = (hi_s - lo_s) / np.maximum(hi_p - lo_p, 1e-6)
    shift = lo_s - lo_p * scale
    aligned = []
    for p in part_pts:
        q = _yaw(((p - c) / s).astype(np.float32), best_k)
        aligned.append((q * scale + shift + sc).astype(np.float32))
    return aligned, best_k, best_d


def label_points(points: np.ndarray, aligned_parts: List[np.ndarray],
                 chunk: int = 65536, smooth_k: int = 12) -> np.ndarray:
    """Assign every shell point to its nearest part surface, then let each
    point take the majority label of its neighbourhood — a lone point deep in
    "body" territory that grazed a wheel sample is noise, not a wheel."""
    from scipy.spatial import cKDTree

    trees = [cKDTree(p) for p in aligned_parts]
    n = points.shape[0]
    dist = np.full((len(trees), n), np.inf, np.float32)
    for i, t in enumerate(trees):
        for j in range(0, n, chunk):
            d, _ = t.query(points[j:j + chunk], k=1)
            dist[i, j:j + chunk] = d
    lab = np.argmin(dist, axis=0).astype(np.int32)
    if smooth_k > 1 and n > smooth_k:
        tree = cKDTree(points)
        _, nb = tree.query(points, k=smooth_k)
        votes = lab[nb]                                    # [N, k]
        K = len(aligned_parts)
        counts = np.zeros((n, K), np.int32)
        for j in range(smooth_k):
            np.add.at(counts, (np.arange(n), votes[:, j]), 1)
        lab = counts.argmax(1).astype(np.int32)
    return lab


# ------------------------------------------------------- image-driven labelling
def label_from_image(image_rgb: np.ndarray, means: np.ndarray,
                     prompts: List[str], threshold: float = 0.30,
                     frame_ratio: float = 0.85) -> Optional[np.ndarray]:
    """Seed part labels by asking the source photograph.

    ``prompts[0]`` must be the whole-object/shell phrase — it is the fallback
    label everywhere nothing more specific wins. Points are projected onto the
    image with the same frontal mapping the material pass uses; a pixel where
    some part's confidence clears ``threshold`` seeds that label, everything
    else stays -1 for :func:`propagate_labels` to fill. Returns [N] int32 with
    -1 for unseeded, or None when the segmenter is unavailable.
    """
    if len(prompts) < 2:
        return None
    try:
        from ..vision.segment import segment
        # the shell phrase is deliberately NOT segmented: "a car" lights up the
        # entire subject and would out-vote "wheel" on the very tyres it names.
        # The shell is the default — it wins wherever nothing specific does.
        masks = segment(image_rgb, list(prompts[1:]))
    except Exception:
        return None
    h, w = masks.shape[1:3]
    k = 0.5 * float(frame_ratio)
    u = np.clip(((0.5 + k * means[:, 0]) * w).astype(np.int32), 0, w - 1)
    v = np.clip(((0.5 - k * means[:, 1]) * h).astype(np.int32), 0, h - 1)
    per_point = masks[:, v, u]                    # [P-1, N]
    lab = (per_point.argmax(0) + 1).astype(np.int32)
    conf = per_point.max(0)
    lab[conf < threshold] = -1
    return lab


def propagate_labels(points: np.ndarray, labels: np.ndarray,
                     k: int = 14, rounds: int = 3) -> np.ndarray:
    """Grow seed labels through space; whatever stays unclaimed is the shell.

    Each round every point takes the majority label among its k nearest
    neighbours, counting only neighbours that have a label. Seeds never flip.
    Points no round can reach — the whole unlabelled hull of the object — get
    label 0, the shell.
    """
    from scipy.spatial import cKDTree

    lab = labels.astype(np.int32).copy()
    n = points.shape[0]
    if n == 0:
        return lab
    nb = cKDTree(points).query(points, k=min(int(k), n))[1]
    seeds = lab >= 0
    K = int(lab.max()) + 2
    for _ in range(max(1, int(rounds))):
        neigh = lab[nb]                                   # [N, k]
        counts = np.zeros((n, max(K, 1)), np.int32)
        for j in range(neigh.shape[1]):
            good = neigh[:, j] >= 0
            np.add.at(counts, (np.nonzero(good)[0], neigh[good, j]), 1)
        best = counts.argmax(1).astype(np.int32)
        has = counts.max(1) > 0
        grow = (~seeds) & has
        lab[grow] = best[grow]
    lab[lab < 0] = 0
    return lab


_FRONT_WORDS = ("headlight", "grille", "windshield", "windscreen", "face",
                "nose", "mouth", "eyes", "beak", "door", "screen")


def forward_yaw_from_labels(means: np.ndarray, labels: np.ndarray,
                            names: List[str]) -> Optional[float]:
    """Which way does the object actually face? Ask its own labelled anatomy.

    Interior parts are placed and oriented assuming the object faces +z, but a
    3/4-view source image gives the reconstruction a diagonal forward. Labels
    that are semantically front-mounted — headlights, a grille, a face — sit on
    the true front, so the horizontal direction from the cloud's centre to such
    a group IS the forward axis. Returns the yaw (radians about +y, 0 = +z) or
    None when nothing front-mounted was labelled confidently.
    """
    if labels is None or not names:
        return None
    centre = means.mean(0)
    for i, name in enumerate(names):
        low = str(name).lower()
        if not any(w in low for w in _FRONT_WORDS):
            continue
        m = labels == i
        if int(m.sum()) < 800:
            continue
        v = means[m].mean(0) - centre
        if float(np.hypot(v[0], v[2])) < 0.12:      # front-ish but not lateral
            continue
        return float(np.arctan2(v[0], v[2]))
    return None
