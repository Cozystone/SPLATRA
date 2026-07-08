# -*- coding: utf-8 -*-
"""Learned rig predictor — generalises rigging to ANY generated model.

Geometric auto-rig (farthest-point sampling) finds only the TIPS. A learned
predictor also finds INTERNAL joints (knees, spine) because it learns, from a
distribution of shapes, WHERE things bend — a prior geometry alone can't give.

Architecture: per-point geometry features -> small MLP -> per-point `jointness`
(+ bone class). Trained on synthetic rigged creatures (synth_rigs) whose internal
joints are known exactly. Joint positions are recovered by weighted mean-shift on
the high-jointness points. Same means[N,3] interface as a real field; a real
rigged-mesh dataset can later replace/augment the synthetic teacher unchanged.

Torch for training (GPU-optional); tiny net so it runs anywhere.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


def features(P: np.ndarray, k: int = 16) -> np.ndarray:
    """Per-point geometry features (canonicalised).

    - position (3), radius (1), height (1): coarse where-in-the-body.
    - local density at two radii (2): thin limb vs thick body vs converging hub.
    - local PCA shape (linearity/planarity/sphericity, 3): the joint signal.
      A straight tube is *linear* (one dominant eigenvalue); at a joint two bones
      meet at an angle so the k-neighbourhood bends -> linearity drops. This is
      what a knee/spine has and a mid-limb point does not, so it's learnable.
    """
    P = np.asarray(P, dtype=np.float32)
    P = P - P.mean(0)
    P = P / (np.abs(P).max() + 1e-6)
    n = len(P)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    r = np.linalg.norm(P, axis=1, keepdims=True)
    y = P[:, 1:2]
    dens1 = (D < 0.12).sum(1, keepdims=True) / n
    dens2 = (D < 0.25).sum(1, keepdims=True) / n
    kk = min(k, n - 1)
    idx = np.argpartition(D, kk, axis=1)[:, :kk + 1]          # k nearest (incl. self)
    nb = P[idx] - P[:, None, :]                               # [n,kk+1,3]
    cov = np.einsum("nki,nkj->nij", nb, nb) / (kk + 1)        # [n,3,3]
    w = np.linalg.eigvalsh(cov)[:, ::-1]                      # descending l0>=l1>=l2
    l0 = w[:, 0:1] + 1e-9
    lin = (w[:, 0:1] - w[:, 1:2]) / l0
    pla = (w[:, 1:2] - w[:, 2:3]) / l0
    sph = w[:, 2:3] / l0
    return np.concatenate([P, r, y, dens1, dens2, lin, pla, sph],
                          axis=1).astype(np.float32)          # [N,10]


_FIN = 10

if _TORCH:
    class _Net(nn.Module):
        def __init__(self, fin=_FIN, hid=96):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(fin, hid), nn.ReLU(),
                                      nn.Linear(hid, hid), nn.ReLU(),
                                      nn.Linear(hid, hid), nn.ReLU())
            self.jointness = nn.Linear(hid, 1)

        def forward(self, x):
            return self.jointness(self.body(x)).squeeze(-1)


class RigPredictor:
    def __init__(self):
        self.net = _Net() if _TORCH else None
        self.mu = None
        self.sd = None

    def _norm(self, X):
        return (X - self.mu) / self.sd

    def train(self, shapes, epochs: int = 400, lr: float = 2e-3,
              batch: int = 16384, device: str = "cpu"):
        assert _TORCH, "torch required to train"
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.net.to(dev)
        Xn = np.concatenate([features(s["points"]) for s in shapes])
        Yn = np.concatenate([s["jointness"] for s in shapes])
        self.mu = Xn.mean(0, keepdims=True)
        self.sd = Xn.std(0, keepdims=True) + 1e-6
        X = torch.tensor(self._norm(Xn), device=dev)
        Y = torch.tensor(Yn, device=dev)
        opt = torch.optim.Adam(self.net.parameters(), lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.MSELoss()
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, batch):
                bi = perm[s:s + batch]
                opt.zero_grad()
                loss = lossf(self.net(X[bi]), Y[bi])
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    @classmethod
    def load(cls, path: str) -> "RigPredictor":
        rp = cls()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        rp.net.load_state_dict(blob["net"])
        rp.net.eval()
        rp.mu, rp.sd = blob["mu"], blob["sd"]
        return rp

    def jointness(self, points: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            f = torch.tensor(self._norm(features(points)))
            return self.net(f.to(next(self.net.parameters()).device)).cpu().numpy()

    def predict_joints(self, points: np.ndarray, floor: float = 0.15,
                       rel: float = 0.5, merge: float = 0.12,
                       max_joints: int = 20) -> np.ndarray:
        """High-jointness points -> joint centres via greedy weighted mean-shift.
        Cut adapts to the net's confidence on THIS shape — rel*peak — with a small
        absolute floor so pure noise (peak~0) still yields nothing. A hard high
        threshold would return zero joints on any off-distribution shape where
        the net is under-confident but still correctly ranked."""
        P = np.asarray(points, dtype=np.float32)
        Pc = (P - P.mean(0)); Pc /= (np.abs(Pc).max() + 1e-6)
        j = self.jointness(P)
        cut = max(floor, rel * float(j.max()))
        hot, wj = Pc[j > cut], j[j > cut]
        joints, used = [], np.zeros(len(hot), bool)
        for idx in np.argsort(-wj):
            if used[idx]:
                continue
            grp = np.linalg.norm(hot - hot[idx], axis=1) < merge
            used |= grp
            joints.append(np.average(hot[grp], axis=0, weights=wj[grp]))
            if len(joints) >= max_joints:
                break
        return np.array(joints, dtype=np.float32)
