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
    def __init__(self, grid: int = 256, threshold: float = 27.0,
                 n_points: int = 300_000, sh_degree: int = 0) -> None:
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

        from .bg import cutout, reframe_foreground

        rgba = cutout(image_rgb)
        if rgba is None:
            rgba = image_rgb if image_rgb.shape[-1] == 4 else np.concatenate(
                [image_rgb, np.ones_like(image_rgb[..., :1])], -1)
        # Recenter+rescale to TripoSR's expected framing (subject ~85% of frame,
        # centered) — frame-filling/cropped SD shots otherwise deform the volume.
        rgba = reframe_foreground(rgba, ratio=0.85, size=512)
        comp = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1 - rgba[..., 3:4])  # gray bg
        img = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8))

        m = self._model
        # the multi-view path parks this model on the CPU to lend out the GPU;
        # make sure every parameter is back before running the encoder
        try:
            if next(m.parameters()).device.type != "cuda":
                m.to("cuda")
        except Exception:
            pass
        with torch.no_grad():
            scene = m([img], device="cuda")
            r = float(m.renderer.cfg.radius)
            N = self.grid
            lin = torch.linspace(-r, r, N, device="cuda")
            # Walk the volume a slab at a time. Building all N^3 sample points at
            # once was the real ceiling on resolution — at N=384 that is 57M
            # points held alongside their densities, before a single one has been
            # thrown away. Only a few percent survive the threshold, so query a
            # slab, keep its survivors, and let the rest go: peak memory follows
            # the slab rather than the grid, and the grid is free to get finer.
            rows = max(1, int(4_000_000 // max(1, N * N)))
            pk, ck = [], []
            for i0 in range(0, N, rows):
                gx, gy, gz = torch.meshgrid(lin[i0:i0 + rows], lin, lin, indexing="ij")
                P = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
                del gx, gy, gz
                d = torch.cat([m.renderer.query_triplane(m.decoder, ch, scene[0])
                               ["density_act"].squeeze(-1) for ch in P.split(262144)])
                hit = P[d > self.threshold]
                del P, d
                if hit.shape[0]:
                    c = torch.cat([m.renderer.query_triplane(m.decoder, ch, scene[0])["color"]
                                   for ch in hit.split(262144)]).clamp(0, 1)
                    pk.append(hit.detach().cpu().numpy())
                    ck.append(c.detach().cpu().numpy())
                    del c
                del hit
            torch.cuda.empty_cache()
            if not pk or sum(a.shape[0] for a in pk) < 64:
                raise RuntimeError("TripoSR produced an empty volume")
            kpts = np.concatenate(pk, 0)
            colors = np.concatenate(ck, 0)
            del pk, ck

        if os.environ.get("SPLATRA_CLEAN") != "0":   # on by default (~0.5s):
            # stray disconnected specks read as flicker and colour noise the
            # moment the model turns; SPLATRA_CLEAN=0 opts back out
            kpts, colors = self._remove_floaters(kpts, colors)
        if os.environ.get("SPLATRA_SHELL", "1") == "1":
            kpts, colors = self._surface_shell(kpts, colors, step=2.0 * r / (N - 1))
        kept_before = kpts.shape[0]

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
        px = self._splat_sigma(means)
        scales = np.log(np.tile(np.array([px, px, px], np.float32), (n, 1)))
        quats = np.zeros((n, 4), np.float32); quats[:, 0] = 1.0
        opacities = np.full((n,), 2.2, np.float32)

        # See-through material. A reconstruction has no idea which part of the crust
        # is a windscreen, so ask a vision-language model where the glass is in the
        # very image this volume was lifted from, then project the points back onto
        # that image and thin out whatever lands on it. The cabin we assembled
        # inside then actually reads through the windows.
        opacities, colors = self._apply_materials(comp, means, opacities, colors)
        self._label_parts(comp, means)
        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), np.float32)
        sh[:, 0, :] = rgb_to_sh_dc(colors.astype(np.float32))
        return GaussianField(means, scales, quats, opacities, sh, sh_degree=self.sh_degree)


    def _splat_sigma(self, means: np.ndarray) -> float:
        """Splat radius, measured from how far apart the points actually ended up.

        Deriving it from the sampling grid was wrong in both directions: the cloud
        gets carved to a shell and then thinned to ``n_points``, so the real gaps
        between neighbours are nothing like one grid cell. Measure the median
        nearest-neighbour distance instead and size each splat to cover it. Below
        about 0.7x that distance the object fills with holes and the near-black
        background reads through as noise; far above it the surface turns to mush.
        """
        n = means.shape[0]
        if n < 32:
            return 2.2 / self.grid
        try:
            from scipy.spatial import cKDTree
            probe = means[np.random.default_rng(0).choice(n, min(4096, n), replace=False)]
            d = cKDTree(means).query(probe, k=2)[0][:, 1]
            nn = float(np.median(d[np.isfinite(d) & (d > 0)]))
        except Exception:
            nn = 2.2 / self.grid
        return float(np.clip(nn * 0.75, 1.5 / self.grid, 0.06))

    def _surface_shell(self, pts: np.ndarray, cols: np.ndarray, step: float,
                       depth: int = 2) -> tuple:
        """Keep the crust of the volume and drop what is buried inside it.

        Thresholding the density field returns a *solid*: every voxel the object
        occupies, not just the ones you can see. But TripoSR only ever learns
        colour where a camera ray stops, so the colour it reports for a buried
        voxel is unconstrained — muddy and dark. Those buried points outnumber the
        visible surface several times over, and because a splat only a grid step
        wide does not fully occlude what sits behind it, the view ends up looking
        straight through the skin into that mush. That is the washed-out, dark,
        noisy reading: the shell is right and everything behind it is garbage.

        So erode the occupancy by ``depth`` voxels and keep only what the erosion
        removes — a shell that still has thickness (so it survives being seen
        edge-on) but no unsupervised interior. Hollowing also frees most of the
        point budget for the surface, which is the only part anyone sees, and it
        leaves the inside genuinely empty so composed interior parts show through
        glass instead of being buried in filler.
        """
        if pts.shape[0] < 1000 or step <= 0:
            return pts, cols
        # index on the lattice the samples actually came from. Rescaling by the
        # bounding box instead spreads neighbouring samples across several voxels,
        # nothing ends up 6-connected, and every point looks like surface.
        lo = pts.min(0)
        vox = np.rint((pts - lo) / step).astype(np.int64)
        K = int(vox.max()) + 1
        vox = np.clip(vox, 0, K - 1) + 1
        occ = np.zeros((K + 2, K + 2, K + 2), bool)
        occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True
        inner = occ
        for _ in range(max(1, int(depth))):          # 6-neighbour erosion
            e = inner.copy()
            for ax in (0, 1, 2):
                for sh in (-1, 1):
                    e &= np.roll(inner, sh, ax)
            inner = e
        keep = ~inner[vox[:, 0], vox[:, 1], vox[:, 2]]
        # a thin or small object can be all surface (nothing to hollow) — and if
        # the shell came out implausibly sparse, trust the solid rather than ship
        # an object with holes in it
        if keep.sum() < max(2000, pts.shape[0] * 0.02):
            return pts, cols
        return pts[keep], cols[keep]

    def _label_parts(self, image_rgb, means) -> None:
        """Teach every point which part of the object it belongs to.

        The caller may set ``part_prompts`` (first entry = the whole object)
        before generating; the planner's own part vocabulary is asked of the
        source image via CLIPSeg, seeds are projected onto the cloud, and grown
        through 3D space until every point belongs to a part. The result lands
        on ``last_part_labels``/``last_part_names`` — one generation, one shell,
        and the structure comes as labels instead of as bolted-on duplicates.
        """
        self.last_part_labels = None
        self.last_part_names = []
        self.last_forward_yaw = None
        prompts = list(getattr(self, "part_prompts", []) or [])
        if len(prompts) < 2:
            return
        try:
            from ..structure.partlabel import label_from_image, propagate_labels
            seeds = label_from_image(image_rgb, means, prompts)
            if seeds is None or not (seeds >= 0).any():
                return
            self.last_part_labels = propagate_labels(means, seeds)
            self.last_part_names = prompts
            from ..structure.partlabel import forward_yaw_from_labels
            self.last_forward_yaw = forward_yaw_from_labels(
                means, self.last_part_labels, prompts)
        except Exception:
            self.last_part_labels = None
            self.last_part_names = []


    def _apply_materials(self, image_rgb, means, opacities, colors):
        """Let a vision-language model decide what the surface is made of, then
        render it that way.

        The reconstruction hands back one uniform crust: every point equally
        opaque, glass and chrome and tyre rubber all treated the same. CLIPSeg is
        asked which pixels are which material, the points are projected back onto
        that image, and each material's optical response is applied — glass thins
        out and cools, water half-clears, rubber darkens, metal picks up a sheen.
        The observed mix is also kept on the generator so the physics layer can be
        grounded in what was actually seen rather than in the prompt alone.
        """
        self.last_materials = []
        if os.environ.get("SPLATRA_MATERIALS", "1") == "0":
            return opacities, colors
        try:
            from ..vision.segment import (MATERIAL_OPTICS, MATERIAL_PHYSICS,
                                          material_map)
            mm = material_map(image_rgb)
        except Exception:
            return opacities, colors
        if not mm:
            return opacities, colors

        h, w = next(iter(mm.values())).shape[:2]
        # the lift is frontal, so the source image plane is the viewer's xy-plane
        # seen from +z; reframe_foreground put the subject at 85% of the frame
        k = 0.5 * 0.85
        u = np.clip(((0.5 + k * means[:, 0]) * w).astype(np.int32), 0, w - 1)
        v = np.clip(((0.5 - k * means[:, 1]) * h).astype(np.int32), 0, h - 1)

        for name, mask in mm.items():
            weight = mask[v, u].astype(np.float32)
            if float(weight.max()) <= 0.0:
                continue
            alpha, tint, tw = MATERIAL_OPTICS.get(name, (1.0, (0.0, 0.0, 0.0), 0.0))
            if alpha != 1.0:
                opacities = opacities * (1.0 - (1.0 - alpha) * weight)
            if tw > 0.0:
                t = np.asarray(tint, np.float32)
                colors = (colors * (1.0 - tw * weight[:, None])
                          + t[None, :] * (tw * weight[:, None])).astype(np.float32)
            self.last_materials.append({
                "material": name,
                "coverage": round(float((weight > 0.1).mean()), 4),
                "physics": MATERIAL_PHYSICS.get(name, "soft"),
            })
        self.last_materials.sort(key=lambda d: -d["coverage"])
        return opacities.astype(np.float32), colors.astype(np.float32)

    def _remove_floaters(self, pts: np.ndarray, cols: np.ndarray,
                         min_neighbors: int = 4) -> tuple:
        """Drop isolated stray points (faint boundary 'floaters') that make the
        cloud look fuzzy. Voxelize at the sampling resolution and keep only points
        whose 3x3x3 voxel neighborhood holds >= min_neighbors occupied cells — a
        conservative O(N) filter (27 array rolls) that leaves solid surfaces and
        even thin sheets intact while removing specks."""
        if pts.shape[0] < 1000:
            return pts, cols
        K = self.grid
        lo = pts.min(0)
        span = float((pts.max(0) - lo).max()) + 1e-9
        vox = np.clip(np.floor((pts - lo) / span * K).astype(np.int64), 0, K - 1) + 1
        occ = np.zeros((K + 2, K + 2, K + 2), bool)
        occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True
        cnt = np.zeros_like(occ, np.int16)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cnt += np.roll(np.roll(np.roll(occ, dx, 0), dy, 1), dz, 2)
        keep = cnt[vox[:, 0], vox[:, 1], vox[:, 2]] >= min_neighbors
        if keep.sum() < pts.shape[0] * 0.5:      # never nuke more than half (safety)
            return pts, cols
        return pts[keep], cols[keep]

    def generate(self, prompt: str) -> GaussianField:
        if self._t2i is None:
            from .text_to_3d import TextTo3DGenerator
            self._t2i = TextTo3DGenerator()
        return self.from_image(self._t2i.image(prompt))
