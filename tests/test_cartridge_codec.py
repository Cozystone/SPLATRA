from __future__ import annotations

import struct

import numpy as np

from apps.plugin_api import _apply_cartridge_budget, _pack_cartridge, _pack_cartridge_spl3


def _fixture_arrays(n: int = 32):
    rng = np.random.default_rng(123)
    pos = rng.uniform(-2.0, 3.0, size=(n, 3)).astype(np.float32)
    col = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    scale = rng.uniform(0.01, 0.25, size=(n, 3)).astype(np.float32)
    quat = rng.normal(size=(n, 4)).astype(np.float32)
    opa = rng.uniform(0.05, 1.0, size=n).astype(np.float32)
    return pos, col, scale, quat, opa


def test_spl3_is_quantized_and_smaller_than_spl2() -> None:
    arrays = _fixture_arrays(64)
    spl2 = _pack_cartridge(*arrays)
    spl3 = _pack_cartridge_spl3(*arrays)

    assert spl2[:4] == b"SPL2"
    assert spl3[:4] == b"SPL3"
    assert len(spl2) == 8 + 64 * 56
    assert len(spl3) == 32 + 64 * 20
    assert len(spl3) < len(spl2) * 0.4


def test_spl3_header_and_position_roundtrip_are_bounded() -> None:
    pos, col, scale, quat, opa = _fixture_arrays(10)
    spl3 = _pack_cartridge_spl3(pos, col, scale, quat, opa)
    n = struct.unpack_from("<I", spl3, 4)[0]
    bbox_min = np.frombuffer(spl3, dtype="<f4", count=3, offset=8)
    bbox_max = np.frombuffer(spl3, dtype="<f4", count=3, offset=20)
    pos_q = np.frombuffer(spl3, dtype="<i2", count=n * 3, offset=32).reshape(n, 3)
    decoded = bbox_min + ((pos_q.astype(np.float32) + 32768.0) / 65535.0) * (bbox_max - bbox_min)
    max_axis_error = float(np.max((bbox_max - bbox_min) / 65535.0))

    assert n == 10
    assert np.allclose(decoded, pos, atol=max_axis_error * 1.1)


def test_cartridge_budget_keeps_the_most_important_splats() -> None:
    pos, col, scale, quat, opa = _fixture_arrays(12)
    scale[:] = 0.01
    scale[-3:] = 1.0
    opa[:] = 0.1
    opa[-3:] = 1.0

    b_pos, b_col, b_scale, b_quat, b_opa = _apply_cartridge_budget(pos, col, scale, quat, opa, budget=3)

    assert b_pos.shape == (3, 3)
    assert np.allclose(b_pos, pos[-3:])
    assert np.allclose(b_col, col[-3:])
    assert np.allclose(b_scale, scale[-3:])
    assert np.allclose(b_quat, quat[-3:])
    assert np.allclose(b_opa, opa[-3:])
