"""The reconstruction keeps a surface, and sizes its splats to fit it.

Thresholding TripoSR's density field returns a *solid* — every voxel the object
occupies. But the colour field is only ever supervised where a camera ray stops,
so the colour reported for a buried voxel is unconstrained, and buried voxels
outnumber the visible surface several times over. Rendered, you look straight
through the skin into that mush: dark, washed out, noisy. These tests pin the two
pieces that fix it — carve the volume down to its shell, then size each splat to
the spacing the points actually ended up with.
"""

from __future__ import annotations

import numpy as np

from atanor_core.generation.triposr import TripoSRGenerator


def _lattice(n: int, step: float = 0.01) -> np.ndarray:
    a = np.arange(n) * step
    g = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    return g.astype(np.float32)


def test_surface_shell_drops_the_buried_interior() -> None:
    pts = _lattice(20)
    cols = np.zeros_like(pts)
    gen = TripoSRGenerator(grid=64)

    kept, kept_cols = gen._surface_shell(pts, cols, step=0.01, depth=2)

    # a 20^3 solid eroded by 2 leaves a 16^3 core buried: 8000 of 8000 gone
    assert kept.shape[0] == pts.shape[0] - 16 ** 3
    assert kept_cols.shape[0] == kept.shape[0]
    # every survivor is within 2 lattice steps of a face
    lo, hi = pts.min(0), pts.max(0)
    edge = ((kept <= lo + 0.0201) | (kept >= hi - 0.0201)).any(axis=1)
    assert edge.all()


def test_surface_shell_keeps_a_thin_sheet_whole() -> None:
    """A sheet is all surface — there is nothing buried to carve away, and
    hollowing it further would delete the object."""
    a = np.arange(40) * 0.01
    x, y = np.meshgrid(a, a, indexing="ij")
    pts = np.stack([x, y, np.zeros_like(x)], -1).reshape(-1, 3).astype(np.float32)
    gen = TripoSRGenerator(grid=64)

    kept, _ = gen._surface_shell(pts, np.zeros_like(pts), step=0.01, depth=2)

    assert kept.shape[0] == pts.shape[0]


def test_surface_shell_indexes_the_sampling_lattice_not_the_bounding_box() -> None:
    """Rescaling points by their bounding box instead of their own lattice step
    spreads neighbours across several voxels, nothing reads as 6-connected, and
    every point survives as 'surface' — the carve silently does nothing."""
    pts = _lattice(20, step=0.01)          # spans 0.19, far short of any bbox norm
    gen = TripoSRGenerator(grid=256)       # deliberately unrelated to the lattice

    kept, _ = gen._surface_shell(pts, np.zeros_like(pts), step=0.01, depth=1)

    assert kept.shape[0] < pts.shape[0] * 0.6


def test_splat_size_follows_the_point_spacing() -> None:
    """The cloud is carved and then thinned, so the gaps between points are
    nothing like one grid cell. Splats sized to the cell leave holes; measure the
    real spacing instead, and a cloud twice as sparse gets splats twice as wide."""
    gen = TripoSRGenerator(grid=256)
    tight = _lattice(16, step=0.01)
    loose = _lattice(16, step=0.02)

    s_tight = gen._splat_sigma(tight)
    s_loose = gen._splat_sigma(loose)

    assert np.isclose(s_loose / s_tight, 2.0, rtol=0.05)
    assert 0.5 < s_tight / 0.01 < 1.0        # covers the gap without turning to mush
