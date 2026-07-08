# -*- coding: utf-8 -*-
"""Material regimes: water flows flat, soil holds a thicker fractured mass,
flesh keeps its shape. Same integrator, different constraint semantics."""

from __future__ import annotations

import numpy as np

from atanor_core.motion.pbd import SoftBody, _grid_pairs


def _ball(n=1500, r=0.5, h=1.0, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
    p *= rng.uniform(0, 1, (n, 1)) ** (1 / 3) * r
    p[:, 1] += h
    return p


def _settle(material, frames=150, **kw):
    blob = _ball()
    sb = SoftBody(blob, material=material, floor=0.0, **kw)
    x = sb.settle(blob if material == "flesh" else None, frames=frames)
    return sb, x, blob


def test_grid_pairs_find_dense_neighbours():
    blob = _ball(800)
    i, j = _grid_pairs(blob, 0.08)
    assert len(i) > 800                      # way beyond ~1/particle
    d = np.linalg.norm(blob[i] - blob[j], axis=1)
    assert (d < 0.08 * np.sqrt(3) * 2).all() # candidates are grid-local


def test_water_spreads_flat():
    sb, x, blob = _settle("water")
    spread0 = float(np.std(np.linalg.norm(blob[:, [0, 2]], axis=1)))
    spread = float(np.std(np.linalg.norm(x[:, [0, 2]], axis=1)))
    assert spread > spread0 * 1.4            # flowed outward
    assert float(np.quantile(x[:, 1], 0.5)) < 0.1   # flat puddle
    assert np.isfinite(x).all()


def test_soil_holds_more_than_water():
    _, xw, _ = _settle("water")
    sb_s, xs, _ = _settle("soil")
    spread_w = float(np.std(np.linalg.norm(xw[:, [0, 2]], axis=1)))
    spread_s = float(np.std(np.linalg.norm(xs[:, [0, 2]], axis=1)))
    assert spread_w > spread_s               # water flows wider
    # soil keeps a thicker body of mass off the floor
    assert float(np.quantile(xs[:, 1], 0.5)) > float(np.quantile(xw[:, 1], 0.5))
    # soil fractures under the fall; water's topology just rebuilds
    assert sb_s.n_constraints < len(sb_s.rest)


def test_flesh_keeps_shape_under_no_gravity():
    sb, x, blob = _settle("flesh", frames=60)
    assert np.linalg.norm(x - blob, axis=1).max() < 0.05


def test_gravity_materials_never_explode():
    for mat in ("water", "soil"):
        sb, x, _ = _settle(mat, frames=80)
        assert np.isfinite(x).all()
        assert np.abs(x).max() < 10.0
        assert x[:, 1].min() >= -1e-4        # nothing below the floor
