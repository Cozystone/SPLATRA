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


def test_cartridge_budget_keeps_every_part_not_just_the_biggest() -> None:
    """A budget must thin the whole object, never delete parts of it.

    Every splat within one generated part shares a single size, so ranking by
    ``opacity * scale`` scores a part rather than a splat: the largest part takes
    every slot and the others vanish. A composed car handed a small budget used to
    keep its body shell and lose the wheels, seats and steering wheel outright.
    """
    n = 1000
    pos, col, scale, quat, opa = _fixture_arrays(n)
    scale[:800] = 0.05          # "body": big splats, most of the cloud
    scale[800:] = 0.005         # "wheels": small splats, the rest
    opa[:] = 0.9

    b_pos, b_col, b_scale, b_quat, b_opa = _apply_cartridge_budget(
        pos, col, scale, quat, opa, budget=200)

    assert b_pos.shape == (200, 3)
    small = b_scale[:, 0] < 0.02 * (n / 200.0) ** (1.0 / 3.0)
    assert small.any(), "the smaller part was deleted entirely"
    assert 0.1 < small.mean() < 0.3, "parts kept out of proportion: %.2f" % small.mean()


def test_cartridge_budget_grows_splats_to_cover_the_gaps_it_opens() -> None:
    """Thinning a 3D cloud to a fraction f widens the gaps by f**(-1/3). Splats
    left at their old size no longer meet, and the near-black background reads
    through the holes — the object looks loose, dark and broken up. So the
    survivors have to grow by the same cube root."""
    n, budget = 8000, 1000
    pos, col, scale, quat, opa = _fixture_arrays(n)
    scale[:] = 0.02

    _, _, b_scale, _, _ = _apply_cartridge_budget(pos, col, scale, quat, opa, budget=budget)

    assert np.allclose(b_scale, 0.02 * (n / budget) ** (1.0 / 3.0), rtol=1e-5)


def test_cartridge_budget_is_a_no_op_when_it_fits() -> None:
    arrays = _fixture_arrays(32)
    out = _apply_cartridge_budget(*arrays, budget=64)
    for a, b in zip(out, arrays):
        assert np.allclose(a, b)


def test_lod_shuffle_keeps_rows_together_and_makes_prefixes_fair() -> None:
    """The viewer draws the first K instances as its level of detail, so any
    prefix must be a uniform sample of the whole object — not 'the shell first,
    then the seats' as the arrays are built. And a shuffle that permuted the
    attribute arrays differently would scramble the model."""
    from apps.plugin_api import _shuffle_for_lod

    n = 30000
    pos, col, scale, quat, opa = _fixture_arrays(n)
    pos[:, 0] = np.arange(n)              # row identity riding along in x
    col[:, 0] = np.arange(n) / n
    part = np.zeros(n); part[20000:] = 1  # a "seats" block at the end
    opa[:] = part * 0.5 + 0.25

    s_pos, s_col, s_scale, s_quat, s_opa = _shuffle_for_lod(pos, col, scale, quat, opa)

    assert not np.allclose(s_pos[:100, 0], pos[:100, 0]), "nothing was shuffled"
    assert np.allclose(s_col[:, 0], s_pos[:, 0] / n), "rows were torn apart"
    frac = (s_opa[:6000] > 0.5).mean()    # prefix share of the 1/3-sized block
    assert 0.28 < frac < 0.39, "a prefix is not a fair sample: %.3f" % frac


def test_lod_shuffle_is_deterministic() -> None:
    arrays = _fixture_arrays(500)
    a = _shuffle_for_lod_import()(*arrays)
    b = _shuffle_for_lod_import()(*arrays)
    for x, y in zip(a, b):
        assert np.allclose(x, y)


def _shuffle_for_lod_import():
    from apps.plugin_api import _shuffle_for_lod
    return _shuffle_for_lod


def test_detail_ladder_is_clamped_to_what_the_gpu_can_hold() -> None:
    """One build serves the 5080 and a 4GB laptop: the slider still reads 1-5,
    it just quietly tops out at the finest step that fits in VRAM."""
    from apps.plugin_api import _max_detail_for_vram

    assert _max_detail_for_vram(16.0) == 5
    assert _max_detail_for_vram(10.0) == 4
    assert _max_detail_for_vram(7.5) == 3
    assert _max_detail_for_vram(5.0) == 2
    assert _max_detail_for_vram(3.5) == 1
    assert _max_detail_for_vram(None) == 5     # unknown GPU: do not punish it


def test_spl4_interleaves_the_same_bytes_spl3_lays_out_planar() -> None:
    """SPL4 exists so a partially downloaded cartridge is drawable: same 20
    bytes per splat as SPL3, grouped per splat instead of per attribute. Any
    record prefix must decode back to the first splats, exactly."""
    from apps.plugin_api import _pack_cartridge_spl4

    pos, col, scale, quat, opa = _fixture_arrays(257)   # crosses a texture row
    blob = _pack_cartridge_spl4(pos, col, scale, quat, opa)

    assert blob[:4] == b"SPL4"
    n = struct.unpack("<I", blob[4:8])[0]
    assert n == 257
    assert len(blob) == 32 + n * 20
    bb_min = np.frombuffer(blob, "<f4", 3, 8)
    bb_max = np.frombuffer(blob, "<f4", 3, 20)
    rec = np.frombuffer(blob, np.uint8, n * 20, 32).reshape(n, 20)

    span = np.maximum(bb_max - bb_min, 1e-8)
    pos_d = bb_min + ((rec[:, 0:6].copy().view("<i2").astype(np.float32) + 32768)
                      / 65535.0) * span
    col_d = rec[:, 6:9].astype(np.float32) / 255.0
    opa_d = rec[:, 9].astype(np.float32) / 255.0
    scale_d = rec[:, 10:16].copy().view("<f2").astype(np.float32)
    quat_d = np.maximum(-1.0, rec[:, 16:20].copy().view(np.int8).astype(np.float32) / 127.0)

    assert np.allclose(pos_d, pos, atol=2e-4)
    assert np.allclose(col_d, col, atol=1/254)
    assert np.allclose(opa_d, opa, atol=1/254)
    assert np.allclose(scale_d, scale, rtol=1e-3, atol=1e-4)
    qn = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    assert np.allclose(quat_d, qn, atol=1.2/127)

    # a prefix of records IS the first k splats — the streaming contract
    k = 40
    prefix = np.frombuffer(blob, np.uint8, k * 20, 32).reshape(k, 20)
    assert np.array_equal(prefix, rec[:k])


def test_spl4_is_no_bigger_than_spl3() -> None:
    from apps.plugin_api import _pack_cartridge_spl3, _pack_cartridge_spl4

    arrays = _fixture_arrays(1000)
    assert len(_pack_cartridge_spl4(*arrays)) == len(_pack_cartridge_spl3(*arrays))
