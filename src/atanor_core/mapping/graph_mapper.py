"""Knowledge-graph -> GaussianField mapping (closed-form, no neural net).

Given a graph of nodes (each with an embedding, centrality, importance and
category) we deterministically place one Gaussian per node:

* **position**  -> 3D via UMAP if installed, else a numpy SVD/PCA fallback
                   (zero extra deps). Normalized into the [-1, 1] cube.
* **scale**     -> log-mapped from centrality, stored in log-space.
* **rotation**  -> identity unit quaternion (1, 0, 0, 0).
* **opacity**   -> importance clipped to (0.01, 0.99), stored as a logit.
* **color**     -> category % 12 -> HSV(s=0.8, v=0.9) -> RGB -> SH DC band.

There is no training and no inference here; the mapping is a pure function of
the graph. UMAP is strictly optional (lazy, guarded by try/except).
"""

from __future__ import annotations

import colorsys
from typing import Any, Dict, List

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _pca_3d(emb: np.ndarray) -> np.ndarray:
    """SVD-based PCA to 3 dims. Pads with zeros if fewer than 3 components."""
    x = emb - emb.mean(axis=0, keepdims=True)
    # Economy SVD: columns of Vt are principal directions.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    n_comp = min(3, vt.shape[0])
    coords = x @ vt[:n_comp].T  # [N, n_comp]
    if n_comp < 3:
        pad = np.zeros((coords.shape[0], 3 - n_comp), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords.astype(np.float32)


def _normalize_cube(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates into the [-1, 1] cube, axis-wise."""
    lo = coords.min(axis=0, keepdims=True)
    hi = coords.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, 1e-8)
    return (2.0 * (coords - lo) / span - 1.0).astype(np.float32)


class GraphMapper:
    """Deterministic graph -> GaussianField mapper.

    Args:
        s_min, s_max: scale range (linear) before log-storage.
        use_umap: if True and umap-learn is importable, use UMAP for layout;
                  otherwise fall back to numpy PCA (no extra deps).
        sh_degree: SH degree of the produced field (default 1 -> K=4 bands).
    """

    def __init__(
        self,
        s_min: float = 0.02,
        s_max: float = 0.12,
        use_umap: bool = True,
        sh_degree: int = 1,
    ) -> None:
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.use_umap = bool(use_umap)
        self.sh_degree = int(sh_degree)

    # -- layout ------------------------------------------------------------ #
    def _layout(self, emb: np.ndarray) -> np.ndarray:
        n = emb.shape[0]
        if self.use_umap and n >= 4:
            try:
                import umap  # type: ignore  # optional, lazy

                reducer = umap.UMAP(n_components=3, random_state=42)
                coords = reducer.fit_transform(emb)
                return _normalize_cube(np.asarray(coords, dtype=np.float32))
            except Exception:
                # UMAP not installed or failed -> deterministic PCA fallback.
                pass
        return _normalize_cube(_pca_3d(emb))

    # -- color ------------------------------------------------------------- #
    @staticmethod
    def _category_rgb(category: int) -> np.ndarray:
        hue = (int(category) % 12) / 12.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return np.array([r, g, b], dtype=np.float32)

    def map(self, graph: Dict[str, Any]) -> GaussianField:
        nodes: List[Dict[str, Any]] = list(graph.get("nodes", []))
        if not nodes:
            raise ValueError("graph has no nodes")
        n = len(nodes)

        # Embeddings: stack, default to a small random-but-deterministic vector
        # if a node lacks one (keeps mapping total).
        dim = 0
        for nd in nodes:
            e = nd.get("embedding")
            if e is not None:
                dim = max(dim, len(e))
        dim = max(dim, 3)
        emb = np.zeros((n, dim), dtype=np.float32)
        for i, nd in enumerate(nodes):
            e = nd.get("embedding")
            if e is not None:
                emb[i, : len(e)] = np.asarray(e, dtype=np.float32)

        means = self._layout(emb)

        centrality = np.array(
            [float(nd.get("centrality", 0.0)) for nd in nodes], dtype=np.float32
        )
        importance = np.array(
            [float(nd.get("importance", 0.5)) for nd in nodes], dtype=np.float32
        )
        category = np.array(
            [int(nd.get("category", 0)) for nd in nodes], dtype=np.int64
        )

        # scale: log-mapped centrality, stored in log-space.
        c_max = float(np.log1p(centrality.max())) if centrality.max() > 0 else 1.0
        c_max = max(c_max, 1e-8)
        s_lin = self.s_min + (self.s_max - self.s_min) * (np.log1p(centrality) / c_max)
        s_lin = np.clip(s_lin, 1e-4, None)
        scales = np.log(np.repeat(s_lin[:, None], 3, axis=1)).astype(np.float32)

        # rotation: identity unit quaternion.
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0

        # opacity: importance clipped, stored as logit.
        opacities = _logit(np.clip(importance, 0.01, 0.99)).astype(np.float32)

        # color: category -> RGB -> SH DC band; higher bands stay zero.
        k = (self.sh_degree + 1) ** 2
        sh = np.zeros((n, k, 3), dtype=np.float32)
        for i in range(n):
            rgb = self._category_rgb(int(category[i]))
            sh[i, 0, :] = rgb_to_sh_dc(rgb)

        return GaussianField(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh=sh,
            sh_degree=self.sh_degree,
        )
