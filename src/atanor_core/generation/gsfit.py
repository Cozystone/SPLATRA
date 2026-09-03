"""Our own reconstruction engine: fit Gaussians to views by optimisation.

Every other image-to-3D system in reach is a *pretrained* feed-forward model —
TripoSR, TRELLIS, Hunyuan3D. Adopting one makes the core of SPLATRA somebody
else's weights, and training a replacement needs Objaverse-scale data and a GPU
cluster. Neither is what we want.

But 3D Gaussian Splatting (Kerbl et al. 2023) is an *algorithm*, not a model:
given images of an object you can recover its Gaussians by gradient descent
through a differentiable renderer, with no pretrained 3D weights anywhere. That we
can own outright, and it beats the visual hull it replaces — carving can only
intersect silhouettes, so it yields a blunt convex-ish shell with smeared colour,
while optimisation is driven by every pixel of every view.

This implements the three things that separate a toy fit from the real method:

* **anisotropic Gaussians** — each one carries a 3D scale and a rotation, so a
  surface is covered by flat ellipses lying along it instead of round blobs.
  The 2D footprint comes from projecting the full 3D covariance, exactly as EWA
  splatting does.
* **perspective projection** — the views are rendered from a finite distance, so
  fitting them with an orthographic camera bakes in a systematic error.
* **adaptive densification** — clone/split the Gaussians whose position gradient
  stays large (under-reconstructed regions) and prune the transparent ones, so
  detail appears where the images demand it rather than where we happened to
  initialise.

Still deliberately pure PyTorch: a bounded per-Gaussian footprint composited with
scatter-add, no custom CUDA, no build step.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc, sh_dc_to_rgb

_OFFSETS: dict = {}


def _view_basis(azimuth: float, elevation: float):
    """Orthonormal (right, up, forward) for a camera looking at the origin."""
    import torch

    d = torch.tensor([math.cos(elevation) * math.sin(azimuth),
                      math.sin(elevation),
                      math.cos(elevation) * math.cos(azimuth)])
    d = d / d.norm()
    up_w = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(up_w, d, dim=0)
    if right.norm() < 1e-6:
        right = torch.tensor([1.0, 0.0, 0.0])
    right = right / right.norm()
    up = torch.cross(d, right, dim=0)
    return right, up, d


def _quat_to_rot(q):
    """[N,4] (w,x,y,z) -> [N,3,3]; normalised internally so the optimiser is free."""
    import torch

    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def _offsets(dev, foot):
    key = (str(dev), foot)
    got = _OFFSETS.get(key)
    if got is None:
        import torch

        off = torch.arange(-foot, foot + 1, device=dev)
        gv, gu = torch.meshgrid(off, off, indexing="ij")
        got = (gu.reshape(-1), gv.reshape(-1))
        _OFFSETS[key] = got
    return got


def _splat(means, colors, opacity, log_scale, quat, right, up, fwd, S, scale,
           cam_dist=2.6, foot=3):
    """Differentiable anisotropic splatting -> (image [S,S,3], alpha [S,S]).

    The 3D covariance R S S^T R^T is projected onto the view's image plane and
    inverted to give the elliptical falloff, which is what lets a Gaussian lie flat
    along a surface instead of sitting on it as a bead.
    """
    import torch

    dev = means.device
    N = means.shape[0]

    ix = means @ right
    iy = means @ up
    depth = means @ fwd                              # + = toward the camera

    # perspective: the camera sits at cam_dist, so things further away shrink
    zc = (cam_dist - depth).clamp(min=0.35)
    persp = cam_dist / zc
    u = (ix * persp / (2.0 * scale) + 0.5) * (S - 1)
    v = (0.5 - iy * persp / (2.0 * scale)) * (S - 1)

    # 3D covariance -> 2D image covariance
    R = _quat_to_rot(quat)                           # [N,3,3]
    sc = torch.exp(log_scale).clamp(1e-3, 0.35)      # [N,3]
    M = torch.stack([right, up], 0)                  # [2,3] image basis
    RS = R * sc.unsqueeze(1)                         # columns scaled
    A = torch.einsum("ij,njk->nik", M, RS)           # [N,2,3]
    cov2 = torch.einsum("nik,njk->nij", A, A)        # [N,2,2]
    px = (S - 1) / (2.0 * scale)                     # world -> pixels
    cov2 = cov2 * (px * persp.view(-1, 1, 1)) ** 2
    cov2 = cov2 + torch.eye(2, device=dev).unsqueeze(0) * 0.35   # AA / min size

    det = (cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]).clamp(min=1e-6)
    inv00, inv11 = cov2[:, 1, 1] / det, cov2[:, 0, 0] / det
    inv01 = -cov2[:, 0, 1] / det

    order = torch.argsort(depth)                     # far -> near
    u, v = u[order], v[order]
    col = colors[order]
    alpha = opacity[order].clamp(1e-4, 0.999)
    inv00, inv11, inv01 = inv00[order], inv11[order], inv01[order]

    du, dv = _offsets(dev, foot)
    K = du.numel()
    u0 = u.round().long().unsqueeze(1) + du.unsqueeze(0)
    v0 = v.round().long().unsqueeze(1) + dv.unsqueeze(0)
    ok = (u0 >= 0) & (u0 < S) & (v0 >= 0) & (v0 < S)

    dx = u.unsqueeze(1) - u0.float()
    dy = v.unsqueeze(1) - v0.float()
    q = (inv00.unsqueeze(1) * dx * dx
         + 2.0 * inv01.unsqueeze(1) * dx * dy
         + inv11.unsqueeze(1) * dy * dy)
    w = torch.exp(-0.5 * q.clamp(0, 30.0)) * alpha.unsqueeze(1)
    w = torch.where(ok, w, torch.zeros_like(w)).reshape(-1)

    idx = (v0.clamp(0, S - 1) * S + u0.clamp(0, S - 1)).reshape(-1)
    img = torch.zeros(S * S, 3, device=dev)
    acc = torch.zeros(S * S, device=dev)
    img.index_add_(0, idx, w.unsqueeze(1) * col.repeat_interleave(K, dim=0))
    acc.index_add_(0, idx, w)
    # Premultiplied, NOT normalised by the accumulated weight. Dividing by acc
    # discards exactly the quantity the display renderer composites with, so
    # opacities tuned against a normalised image do not survive the round trip
    # and the result draws wispy. Premultiplied means the parameters we optimise
    # are the parameters that get drawn.
    return img.view(S, S, 3), acc.view(S, S).clamp(0.0, 1.0)


def fit_gaussians(masks: np.ndarray, colors: np.ndarray, azimuths, elevations,
                  init: Optional[GaussianField] = None, scale: float = 1.2,
                  iters: int = 300, n_points: int = 40_000, S: int = 160,
                  sh_degree: int = 0, densify_every: int = 60,
                  max_points: int = 160_000, log=None) -> Tuple[GaussianField, dict]:
    """Recover a GaussianField that reproduces the given views."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V = masks.shape[0]
    tgt_a = torch.tensor(masks, dtype=torch.float32, device=dev)
    tgt_c = torch.tensor(colors, dtype=torch.float32, device=dev)
    if tgt_a.shape[-1] != S:
        tgt_a = torch.nn.functional.interpolate(
            tgt_a[:, None], size=(S, S), mode="bilinear", align_corners=False)[:, 0]
        tgt_c = torch.nn.functional.interpolate(
            tgt_c.permute(0, 3, 1, 2), size=(S, S), mode="bilinear",
            align_corners=False).permute(0, 2, 3, 1)

    if init is not None and init.means.shape[0] > 32:
        pts = init.means.astype(np.float32)
        rgb0 = np.clip(sh_dc_to_rgb(init.sh[:, 0, :]), 0.0, 1.0).astype(np.float32)
        if pts.shape[0] > n_points:
            sel = np.random.default_rng(0).choice(pts.shape[0], n_points, replace=False)
            pts, rgb0 = pts[sel], rgb0[sel]
    else:
        rng = np.random.default_rng(0)
        pts = rng.uniform(-0.6, 0.6, (n_points, 3)).astype(np.float32)
        rgb0 = np.full((pts.shape[0], 3), 0.5, np.float32)

    n0 = pts.shape[0]
    xyz = torch.tensor(pts, device=dev).requires_grad_(True)
    col = torch.tensor(rgb0, device=dev).requires_grad_(True)
    col_anchor = torch.tensor(rgb0, device=dev)
    logit_a = torch.full((n0,), 1.2, device=dev).requires_grad_(True)
    log_s = torch.full((n0, 3), math.log(0.012), device=dev).requires_grad_(True)
    quat = torch.zeros((n0, 4), device=dev)
    quat[:, 0] = 1.0
    quat = quat.requires_grad_(True)

    def make_opt(params):
        return torch.optim.Adam([
            {"params": [params[0]], "lr": 3.0e-3},   # xyz
            {"params": [params[1]], "lr": 6.0e-3},   # colour
            {"params": [params[2]], "lr": 2.5e-2},   # opacity
            {"params": [params[3]], "lr": 8.0e-3},   # log scale
            {"params": [params[4]], "lr": 6.0e-3},   # rotation
        ])

    opt = make_opt([xyz, col, logit_a, log_s, quat])
    bases = [tuple(t.to(dev) for t in _view_basis(float(azimuths[v]),
                                                  float(elevations[v])))
             for v in range(V)]

    grad_accum = torch.zeros(n0, device=dev)
    seen = 0
    hist = []

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for v in range(V):
            r, u_, f = bases[v]
            img, alpha = _splat(xyz, col.clamp(0, 1), torch.sigmoid(logit_a),
                                log_s, quat, r, u_, f, S, scale)
            m = tgt_a[v]
            l_a = torch.abs(alpha - m).mean()
            # compare like with like: the target is premultiplied too
            l_c = torch.abs(img - tgt_c[v] * m.unsqueeze(-1)).mean() * 3.0
            total = total + l_a + 0.8 * l_c
        total = total + 0.35 * torch.abs(col - col_anchor).mean()
        total.backward()
        opt.step()
        with torch.no_grad():
            xyz.clamp_(-1.5, 1.5)
            col.clamp_(0.0, 1.0)
            if xyz.grad is not None:
                grad_accum += xyz.grad.norm(dim=1)
                seen += 1

        # ── adaptive density control ─────────────────────────────────────────
        if densify_every and (it + 1) % densify_every == 0 and it < iters - densify_every:
            with torch.no_grad():
                a = torch.sigmoid(logit_a)
                keep = a > 0.03                      # prune what contributes nothing
                if int(keep.sum()) < 128:
                    keep = torch.ones_like(a, dtype=torch.bool)
                g = (grad_accum / max(1, seen))[keep]
                x_, c_, la_, ls_, q_ = (xyz[keep], col[keep], logit_a[keep],
                                        log_s[keep], quat[keep])
                room = max_points - int(keep.sum())
                if room > 0 and g.numel():
                    k = min(room, max(0, int(0.15 * g.numel())))
                    if k > 0:
                        # split the worst-reconstructed ones: two smaller children
                        # offset along the Gaussian's own extent
                        pick = torch.topk(g, k).indices
                        jitter = (torch.randn(k, 3, device=dev)
                                  * torch.exp(ls_[pick]).mean(1, keepdim=True))
                        x_ = torch.cat([x_, x_[pick] + jitter], 0)
                        c_ = torch.cat([c_, c_[pick]], 0)
                        la_ = torch.cat([la_, la_[pick]], 0)
                        ls_ = torch.cat([ls_, ls_[pick] - math.log(1.6)], 0)
                        q_ = torch.cat([q_, q_[pick]], 0)
                        ls_[pick] = ls_[pick] - math.log(1.6)
                xyz = x_.detach().requires_grad_(True)
                col = c_.detach().requires_grad_(True)
                logit_a = la_.detach().requires_grad_(True)
                log_s = ls_.detach().requires_grad_(True)
                quat = q_.detach().requires_grad_(True)
                col_anchor = col.detach().clone()
                grad_accum = torch.zeros(xyz.shape[0], device=dev)
                seen = 0
            opt = make_opt([xyz, col, logit_a, log_s, quat])

        if it % max(1, iters // 8) == 0:
            hist.append(round(float(total.item()), 4))
            if log:
                log("fit %3d/%d n=%d loss %.4f"
                    % (it, iters, xyz.shape[0], float(total.item())))

    with torch.no_grad():
        a = torch.sigmoid(logit_a)
        keep = a > 0.05
        if int(keep.sum()) < 64:
            keep = a > 0.0
        P = xyz[keep].detach().cpu().numpy().astype(np.float32)
        C = col[keep].clamp(0, 1).detach().cpu().numpy().astype(np.float32)
        A = a[keep].detach().cpu().numpy().astype(np.float32)
        Sc = torch.exp(log_s[keep]).clamp(1e-3, 0.35).detach().cpu().numpy().astype(np.float32)
        Q = quat[keep]
        Q = (Q / (Q.norm(dim=1, keepdim=True) + 1e-8)).detach().cpu().numpy().astype(np.float32)

    n = P.shape[0]
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(C)
    field = GaussianField(P, np.log(Sc), Q, (np.clip(A, 0.02, 0.999) * 2.4).astype(np.float32),
                          sh, sh_degree=sh_degree)
    return field, {"points": int(n), "iters": iters, "loss": hist,
                   "final_loss": hist[-1] if hist else None}
