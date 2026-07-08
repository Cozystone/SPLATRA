"""Material-specific particle generators for SPLATRA.

These generators are local geometry/material generators, not language models.
They are used when a prompt asks for a material whose silhouette is poorly
served by single-view 2.5D image lifting, such as transparent glass.
"""

from __future__ import annotations

import numpy as np

from ..domain.sgf import GaussianField, rgb_to_sh_dc


def _quat_from_normal(normals: np.ndarray) -> np.ndarray:
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    n = normals.shape[0]
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.cross(np.broadcast_to(z, (n, 3)), normals)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(axis_norm < 1e-6, np.array([1.0, 0.0, 0.0], dtype=np.float32), axis / np.maximum(axis_norm, 1e-8))
    dotv = np.clip(normals[:, 2], -1.0, 1.0)
    half = np.arccos(dotv) * 0.5
    s = np.sin(half)
    quat = np.empty((n, 4), dtype=np.float32)
    quat[:, 0] = np.cos(half)
    quat[:, 1] = axis[:, 0] * s
    quat[:, 2] = axis[:, 1] * s
    quat[:, 3] = axis[:, 2] * s
    return quat


def _sphere_points(n: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n, dtype=np.float32)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    normals = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)
    return (radius * normals).astype(np.float32), normals.astype(np.float32)


def _ring_points(n: int, radius: float, axis: str, width: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    jitter = rng.normal(0.0, width, size=(n,)).astype(np.float32)
    r = radius + jitter
    if axis == "z":
        pts = np.stack([r * np.cos(t), r * np.sin(t), rng.normal(0.0, width * 0.35, size=n)], axis=1)
    elif axis == "x":
        pts = np.stack([rng.normal(0.0, width * 0.35, size=n), r * np.cos(t), r * np.sin(t)], axis=1)
    else:
        pts = np.stack([r * np.cos(t), rng.normal(0.0, width * 0.35, size=n), r * np.sin(t)], axis=1)
    normals = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
    return pts.astype(np.float32), normals.astype(np.float32)


def _ribbon_points(n: int, radius: float, phase: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    band = 0.11 * np.sin(3.0 * t + phase) + rng.normal(0.0, 0.012, size=n).astype(np.float32)
    r = radius * (0.78 + 0.10 * np.sin(2.0 * t + phase))
    pts = np.stack([
        r * np.cos(t),
        band,
        r * np.sin(t),
    ], axis=1)
    tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phase), -np.sin(phase)],
        [0.0, np.sin(phase), np.cos(phase)],
    ], dtype=np.float32)
    pts = pts @ tilt.T
    normals = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
    return pts.astype(np.float32), normals.astype(np.float32)


def glass_orb_field(n_points: int = 200_000, radius: float = 0.72, seed: int = 44017) -> GaussianField:
    """Generate a translucent glass marble as dense particle-only geometry."""
    rng = np.random.default_rng(seed)
    shell_n = int(n_points * 0.58)
    haze_n = int(n_points * 0.17)
    rim_n = int(n_points * 0.13)
    ribbon_n = max(0, n_points - shell_n - haze_n - rim_n)

    shell, shell_normals = _sphere_points(shell_n, radius)
    shell += shell_normals * rng.normal(0.0, 0.006, size=(shell_n, 1)).astype(np.float32)

    haze_dir, haze_normals = _sphere_points(haze_n, radius * 0.93)
    haze = haze_dir * rng.uniform(0.30, 0.95, size=(haze_n, 1)).astype(np.float32)

    rings: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []
    for axis in ("z", "x", "y"):
        pts, nrm = _ring_points(rim_n // 3, radius * 1.005, axis, 0.006, rng)
        rings.append(pts)
        ring_normals.append(nrm)

    ribbons: list[np.ndarray] = []
    ribbon_normals: list[np.ndarray] = []
    for phase in (0.2, 2.25, 4.15):
        pts, nrm = _ribbon_points(ribbon_n // 3, radius * 0.72, phase, rng)
        ribbons.append(pts)
        ribbon_normals.append(nrm)

    means = np.concatenate([shell, haze, *rings, *ribbons], axis=0)
    normals = np.concatenate([shell_normals, haze_normals, *ring_normals, *ribbon_normals], axis=0)
    n = means.shape[0]

    colors = np.empty((n, 3), dtype=np.float32)
    shell_end = shell_n
    haze_end = shell_end + haze_n
    rim_end = haze_end + sum(arr.shape[0] for arr in rings)

    colors[:shell_end] = np.array([0.46, 0.86, 1.0], dtype=np.float32)
    colors[shell_end:haze_end] = np.array([0.20, 0.42, 0.58], dtype=np.float32)
    colors[haze_end:rim_end] = np.array([0.86, 0.98, 1.0], dtype=np.float32)
    colors[rim_end:] = np.array([0.98, 0.44, 0.92], dtype=np.float32)

    light = np.array([-0.28, 0.52, 0.81], dtype=np.float32)
    light = light / np.linalg.norm(light)
    lambert = 0.45 + 0.55 * np.clip(normals @ light, 0.0, 1.0)
    colors = np.clip(colors * lambert[:, None] + 0.08, 0.0, 1.0)

    scales = np.empty((n, 3), dtype=np.float32)
    scales[:shell_end] = np.log(np.array([0.0065, 0.0065, 0.0028], dtype=np.float32))
    scales[shell_end:haze_end] = np.log(np.array([0.008, 0.008, 0.006], dtype=np.float32))
    scales[haze_end:rim_end] = np.log(np.array([0.010, 0.010, 0.003], dtype=np.float32))
    scales[rim_end:] = np.log(np.array([0.012, 0.012, 0.004], dtype=np.float32))

    opacities = np.empty((n,), dtype=np.float32)
    opacities[:shell_end] = 0.15
    opacities[shell_end:haze_end] = -0.45
    opacities[haze_end:rim_end] = 1.25
    opacities[rim_end:] = 0.95

    sh_degree = 1
    sh = np.zeros((n, (sh_degree + 1) ** 2, 3), dtype=np.float32)
    sh[:, 0, :] = rgb_to_sh_dc(colors)
    return GaussianField(means, scales, _quat_from_normal(normals), opacities, sh, sh_degree=sh_degree)


def looks_like_glass_orb(prompt: str) -> bool:
    low = str(prompt or "").lower()
    return (
        ("glass" in low or "transparent" in low or "translucent" in low or "\uc720\ub9ac" in low)
        and ("orb" in low or "sphere" in low or "marble" in low or "\uad6c\uc2ac" in low or "\uad6c\uccb4" in low)
    )
