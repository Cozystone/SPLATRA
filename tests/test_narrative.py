# -*- coding: utf-8 -*-
"""SPLATRA point-cloud detail: a concept's structural FORM by graph type, a bounded LoD budget so a
2D canvas never drowns, and an importance decimator a real 3DGS asset flows through."""
from atanor_core import narrative as scene_compiler
from atanor_core.narrative.cloud import (
    CEILING,
    SUBJECT_BASE,
    decimate_importance,
    form_for,
    lod_budget,
    scene_point_estimate,
    shape_spec,
)


def test_form_by_graph_type_generalizes():
    assert form_for({"label": "나무", "desc": "키 큰 식물"}) == "branching"
    assert form_for({"label": "소나무", "type": "plant"}) == "branching"      # generalizes by type
    assert form_for({"label": "물병", "type": "container"}) == "vessel"
    assert form_for({"label": "사장님", "type": "person"}) == "humanoid"
    assert form_for({"label": "지구", "desc": "행성"}) == "orb"
    assert form_for({"label": "블라블라"}) == "orb"                            # unknown → clean solid


def test_shape_spec_is_stable_per_concept():
    a = shape_spec({"concept_id": "c_tree", "label": "나무"})
    b = shape_spec({"concept_id": "c_tree", "label": "나무"})
    assert a["form"] == "branching" and a["seed"] == b["seed"]                # same tree, same seed


def test_lod_budget_falls_off_with_depth_and_role():
    near_subj = lod_budget(salience=1.0, depth=0.0, is_subject=True)
    far_sat = lod_budget(salience=0.5, depth=1.5, is_subject=False)
    assert near_subj == SUBJECT_BASE and far_sat < near_subj                  # farther/lesser → fewer
    assert lod_budget(salience=0.01, depth=5.0) >= 24                         # never below a legible core


def test_scene_stays_under_the_ceiling():
    # a crowded scene (1 subject + many satellites) must still fit the per-frame budget
    objs = [{"role": "subject", "weight": 1.0, "pos": [0, 0, 0]}]
    objs += [{"role": "satellite", "weight": 0.6, "pos": [1, 0, 0.4]} for _ in range(12)]
    assert scene_point_estimate(objs) <= CEILING


def test_decimate_keeps_the_important_gaussians():
    pts = [{"x": i, "w": i / 100.0, "scale": 0.0} for i in range(1000)]        # importance rises with i
    out = decimate_importance(pts, 50)
    assert len(out) == 50 and min(p["w"] for p in out) > 0.9                   # kept the top by importance
    assert decimate_importance(pts, 5000) == pts                              # budget ≥ source → untouched


def test_scene_objects_carry_a_shape_spec():
    scene = scene_compiler.compile_scene(
        [{"id": "t", "label": "나무", "desc": "식물"}], [], subject_id="t")
    o = scene["objects"][0]
    assert o["shape"]["form"] == "branching" and isinstance(o["shape"]["seed"], int)
