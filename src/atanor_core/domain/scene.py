"""Multi-object spatial scene (foundation for the real-time LLM explainer).

A :class:`Scene` holds many :class:`SceneObject` s — each its own Gaussian field
placed in a shared world with a transform (position / rotation / scale / opacity).
:meth:`Scene.flatten` composes them into a single :class:`GaussianField`, so a
multi-object scene renders through the *existing* single-field cartridge + viewer
today (objects placed apart, no viewer rewrite needed). Pure numpy.

See ``docs/REALTIME_EXPLAINER.md`` for how the streaming LLM authors this scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sgf import GaussianField


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [N,4] (w,x,y,z) quaternions by a single [4]."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1).astype(np.float32)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


@dataclass
class SceneObject:
    id: str
    field: GaussianField
    position: np.ndarray = _field(default_factory=lambda: np.zeros(3, np.float32))
    rotation: np.ndarray = _field(default_factory=lambda: np.array([1, 0, 0, 0], np.float32))
    scale: float = 1.0
    opacity: float = 1.0
    label: Optional[str] = None
    meta: Dict[str, Any] = _field(default_factory=dict)

    def world_field(self) -> GaussianField:
        """This object's particles transformed into world space."""
        f = self.field
        R = _quat_to_rot(np.asarray(self.rotation, np.float32))
        means = (f.means * self.scale) @ R.T + np.asarray(self.position, np.float32)
        scales = (f.scales + np.log(max(self.scale, 1e-6))).astype(np.float32)
        quats = _quat_mul(f.quats, np.asarray(self.rotation, np.float32))
        op = f.opacities.copy()
        if self.opacity < 0.999:  # fade the whole object via a logit shift
            op = op + float(np.log(max(self.opacity, 1e-4) / (1 - min(self.opacity, 0.9999))))
        return GaussianField(means.astype(np.float32), scales, quats, op.astype(np.float32),
                             f.sh.copy(), sh_degree=f.sh_degree)


@dataclass
class Scene:
    objects: Dict[str, SceneObject] = _field(default_factory=dict)
    links: List[Tuple[str, str, Dict[str, Any]]] = _field(default_factory=list)
    camera: Dict[str, Any] = _field(default_factory=dict)
    version: int = 0

    def add(self, obj: SceneObject) -> "Scene":
        self.objects[obj.id] = obj
        self.version += 1
        return self

    def move(self, oid: str, position) -> "Scene":
        self.objects[oid].position = np.asarray(position, np.float32)
        self.version += 1
        return self

    def remove(self, oid: str) -> "Scene":
        self.objects.pop(oid, None)
        self.links = [l for l in self.links if oid not in (l[0], l[1])]
        self.version += 1
        return self

    def link(self, src: str, dst: str, **style) -> "Scene":
        self.links.append((src, dst, style))
        self.version += 1
        return self

    def auto_layout(self, radius: float = 2.0) -> "Scene":
        """Place positionless objects evenly on a circle (LLM may override)."""
        ids = list(self.objects)
        n = max(len(ids), 1)
        for i, oid in enumerate(ids):
            a = 2 * np.pi * i / n
            self.objects[oid].position = np.array(
                [radius * np.cos(a), 0.0, radius * np.sin(a)], np.float32)
        self.version += 1
        return self

    def flatten(self) -> GaussianField:
        """Compose every object (+ link strands) into one world GaussianField."""
        if not self.objects:
            raise ValueError("empty scene")
        fields = [o.world_field() for o in self.objects.values()]
        # link strands: dim particles sampled along each src->dst centroid line
        for src, dst, style in self.links:
            if src in self.objects and dst in self.objects:
                fields.append(self._link_field(src, dst, style))
        return GaussianField(
            means=np.concatenate([f.means for f in fields], 0),
            scales=np.concatenate([f.scales for f in fields], 0),
            quats=np.concatenate([f.quats for f in fields], 0),
            opacities=np.concatenate([f.opacities for f in fields], 0),
            sh=np.concatenate([f.sh for f in fields], 0),
            sh_degree=fields[0].sh_degree,
        )

    def _link_field(self, src: str, dst: str, style: Dict[str, Any]) -> GaussianField:
        a = np.asarray(self.objects[src].position, np.float32)
        b = np.asarray(self.objects[dst].position, np.float32)
        m = int(style.get("samples", 40))
        t = np.linspace(0.05, 0.95, m, dtype=np.float32)[:, None]
        pts = a[None] * (1 - t) + b[None] * t
        k = self.objects[src].field.sh.shape[1]
        sh = np.zeros((m, k, 3), np.float32)
        sh[:, 0, :] = np.asarray(style.get("color", (0.5, 0.8, 1.0)), np.float32)
        quats = np.tile([1, 0, 0, 0], (m, 1)).astype(np.float32)
        return GaussianField(pts, np.log(np.full((m, 3), 0.02, np.float32)), quats,
                             np.full((m,), 0.2, np.float32), sh,
                             sh_degree=self.objects[src].field.sh_degree)
