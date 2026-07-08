# -*- coding: utf-8 -*-
"""Inner state -> motion: hormones shape the body's movement, deterministically."""

from __future__ import annotations

from atanor_core.rigging.mood_drive import motion_params, chain_drives


def _state(cort=0.0, dopa=0.0, nora=0.0, energy=0.6, valence=0.4, curiosity=0.5):
    return {"homeostasis": {"hormones": {"cortisol": cort, "dopamine": dopa,
                                         "noradrenaline": nora}},
            "vitals": {"energy": energy, "valence": valence, "curiosity": curiosity}}


def test_dopamine_lifts_amp_and_tempo():
    calm = motion_params(_state())
    rewarded = motion_params(_state(dopa=0.6, valence=0.7))
    assert rewarded["amp"] > calm["amp"]
    assert rewarded["tempo"] > calm["tempo"]


def test_cortisol_shrinks_motion_and_adds_tremor_and_droop():
    calm = motion_params(_state())
    stressed = motion_params(_state(cort=0.7))
    assert stressed["amp"] < calm["amp"]
    assert stressed["jitter"] > calm["jitter"]
    assert stressed["droop"] > calm["droop"]


def test_noradrenaline_sharpens_tempo():
    assert motion_params(_state(nora=0.5))["tempo"] > motion_params(_state())["tempo"]


def test_low_energy_slows():
    assert motion_params(_state(energy=0.1))["tempo"] < \
           motion_params(_state(energy=0.9))["tempo"]


def test_params_and_drives_bounded_and_deterministic():
    for s in (_state(), _state(cort=1, dopa=1, nora=1, energy=1, valence=1),
              _state(cort=1, energy=0, valence=0)):
        p = motion_params(s)
        assert 0.04 <= p["amp"] <= 0.9 and 0.2 <= p["tempo"] <= 2.5
        d1 = chain_drives(p, [0, 3, 5], 1.25)
        d2 = chain_drives(p, [0, 3, 5], 1.25)
        assert d1 == d2                          # pure function of state + t
        assert all(abs(v) < 1.6 for v in d1.values())


def test_accepts_flat_hormone_dict_too():
    p = motion_params({"hormones": {"dopamine": 0.5}, "vitals": {}})
    assert p["amp"] > motion_params({"hormones": {}, "vitals": {}})["amp"]
