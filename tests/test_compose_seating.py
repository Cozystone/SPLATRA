"""Composed parts join the shell instead of sitting on top of it.

The planner places a part from a description, never having seen the object it is
going on: it says a wheel belongs at [-0.45, -0.45, 0.45]. But the shell was
reconstructed from a photograph of a whole car, so it already has wheels there.
Bolting four more on leaves two copies of the same geometry a few centimetres
apart, which reads as loose lumps stuck to the surface rather than as one object.
These tests pin the repair: slide each part onto the host geometry it was aimed
at, then delete that geometry so the part replaces it.
"""

from __future__ import annotations

import numpy as np

from apps.plugin_api import (_compose_state, _scene, _seat_exterior_parts)
from atanor_core.domain.scene import SceneObject
from atanor_core.domain.sgf import GaussianField


def _blob(n: int, centre=(0.0, 0.0, 0.0), radius: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = (rng.normal(size=(n, 3)) * 0.3).astype(np.float32)
    m = np.clip(m, -1.0, 1.0) * radius + np.asarray(centre, np.float32)
    return GaussianField(
        m.astype(np.float32),
        np.full((n, 3), np.log(0.01), np.float32),
        np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        np.full((n,), 2.2, np.float32),
        np.zeros((n, 1, 3), np.float32),
        sh_degree=0,
    )


def _scene_with(host_pts, part_at, layers):
    _scene.objects.clear()
    _scene.links.clear()
    _compose_state["layers"] = layers
    _scene.add(SceneObject(id="shell", field=host_pts,
                           position=np.zeros(3, np.float32), scale=1.0))
    _scene.add(SceneObject(id="wheel", field=_blob(2000, radius=1.0, seed=7),
                           position=np.asarray(part_at, np.float32), scale=0.22))


def test_part_snaps_onto_the_host_geometry_it_was_aimed_at() -> None:
    """The planner's coordinate is a guess. The shell knows where its own wheel
    is, so the part should slide onto it rather than hang beside it."""
    host = _blob(4000, centre=(0.60, -0.50, 0.0), radius=0.22, seed=1)
    _scene_with(host, part_at=(0.45, -0.45, 0.0),
                layers={"shell": "exterior", "wheel": "exterior"})

    out = _seat_exterior_parts()

    assert out["snapped"] == 1
    moved = np.asarray(_scene.objects["wheel"].position, np.float32)
    assert np.linalg.norm(moved - np.array([0.60, -0.50, 0.0])) < 0.08


def test_the_host_geometry_a_part_replaces_is_carved_away() -> None:
    """Two copies of a wheel in nearly the same place is the overlap. Once the
    part is seated, the shell's own copy has to go — and only that copy."""
    body = _blob(4000, centre=(0.0, 0.0, 0.0), radius=1.0, seed=2)
    wheel = _blob(900, centre=(0.45, -0.45, 0.0), radius=0.07, seed=6)
    host = GaussianField(
        np.concatenate([body.means, wheel.means]),
        np.concatenate([body.scales, wheel.scales]),
        np.concatenate([body.quats, wheel.quats]),
        np.concatenate([body.opacities, wheel.opacities]),
        np.concatenate([body.sh, wheel.sh]), sh_degree=0)
    _scene_with(host, part_at=(0.45, -0.45, 0.0),
                layers={"shell": "exterior", "wheel": "exterior"})

    out = _seat_exterior_parts()

    assert out["carved"] > 0
    kept = _scene.objects["shell"].field.means
    at = np.array([0.45, -0.45, 0.0], np.float32)
    assert ((np.abs(kept - at) <= 0.07).all(1)).sum() < 900 * 0.2,         "the duplicated wheel survived the carve"
    assert kept.shape[0] > 4000 * 0.9, "the carve ate into the body"


def test_interior_parts_are_left_where_the_plan_put_them() -> None:
    """The shell is hollow, so a seat has no host geometry to seat against and
    nothing to replace — snapping it would drag it into the bodywork."""
    host = _blob(4000, centre=(0.0, 0.0, 0.0), radius=1.0, seed=3)
    _scene_with(host, part_at=(0.22, 0.0, 0.0),
                layers={"shell": "exterior", "wheel": "interior"})

    out = _seat_exterior_parts()

    assert out == {"snapped": 0, "carved": 0}
    assert np.allclose(_scene.objects["wheel"].position, [0.22, 0.0, 0.0])


def test_a_part_with_nothing_under_it_keeps_the_planners_position() -> None:
    host = _blob(4000, centre=(0.0, 0.9, 0.0), radius=0.15, seed=4)
    _scene_with(host, part_at=(-0.8, -0.8, 0.0),
                layers={"shell": "exterior", "wheel": "exterior"})

    out = _seat_exterior_parts()

    assert out["snapped"] == 0
    assert np.allclose(_scene.objects["wheel"].position, [-0.8, -0.8, 0.0])


def test_the_carve_can_never_gouge_the_body() -> None:
    """A part that somehow covers most of the shell must not delete it."""
    host = _blob(4000, centre=(0.0, 0.0, 0.0), radius=0.2, seed=5)
    _scene_with(host, part_at=(0.0, 0.0, 0.0),
                layers={"shell": "exterior", "wheel": "exterior"})

    out = _seat_exterior_parts()

    assert out["carved"] == 0
    assert _scene.objects["shell"].field.num_gaussians == 4000


def test_the_shell_is_prompted_with_the_whole_object() -> None:
    """"car body" renders a bare stamped panel about as often as it renders a car,
    and a flat panel makes a flat reconstruction. The exterior shell of a car is
    just a car."""
    from atanor_core.structure.decompose import _name_the_shell

    parts = [{"name": "body shell", "prompt": "car body", "scale": 1.0, "layer": "exterior"},
             {"name": "wheel", "prompt": "wheel", "scale": 0.22, "layer": "exterior"},
             {"name": "seat", "prompt": "car seat", "scale": 0.2, "layer": "interior"}]

    _name_the_shell(parts, "a red sports car")

    assert parts[0]["prompt"] == "a red sports car"
    assert parts[1]["prompt"] == "wheel"      # only the shell is renamed
    assert parts[2]["prompt"] == "car seat"


def test_prose_part_prompts_fall_back_to_the_part_name() -> None:
    """Asked for one concrete object the planner sometimes answers with a
    definition, and diffusion renders the definition."""
    from atanor_core.structure.decompose import _tighten

    assert _tighten("round components at the bottom corners for a car to move on",
                    "wheels1") == "wheels"
    assert _tighten("steering wheel", "steering wheel") == "steering wheel"
    assert _tighten("wheel", "wheels2") == "wheel"


def test_exterior_parts_collapse_into_label_prompts() -> None:
    """Wheels are not generated and bolted on any more — the shell that already
    contains them gets labelled with their names instead. Interior parts still
    generate; the hollow shell genuinely lacks them."""
    from apps.plugin_api import _collapse_exterior

    parts = [
        {"name": "body shell", "prompt": "a car", "scale": 1.0, "layer": "exterior"},
        {"name": "wheels1", "prompt": "wheel", "scale": 0.22, "layer": "exterior"},
        {"name": "wheels2", "prompt": "wheel", "scale": 0.22, "layer": "exterior"},
        {"name": "headlights", "prompt": "headlight", "scale": 0.1, "layer": "exterior"},
        {"name": "seats1", "prompt": "car seat", "scale": 0.2, "layer": "interior"},
    ]

    keep, prompts = _collapse_exterior(parts)

    assert [q["name"] for q in keep] == ["body shell", "seats1"]
    assert prompts == ["a car", "wheel", "headlight"]      # deduped, shell first


def test_a_single_exterior_part_is_left_alone() -> None:
    from apps.plugin_api import _collapse_exterior

    parts = [{"name": "whole", "prompt": "an apple", "scale": 1.0, "layer": "exterior"}]

    keep, prompts = _collapse_exterior(parts)

    assert keep == parts and prompts == []


def test_labels_grow_from_seeds_and_the_rest_stays_shell() -> None:
    """Seeded points keep their part; unlabelled space between them takes the
    neighbourhood majority; whatever no seed can reach is the shell (label 0)."""
    from atanor_core.structure.partlabel import propagate_labels

    rng = np.random.default_rng(0)
    wheel = rng.normal((0.5, -0.5, 0.0), 0.05, (300, 3)).astype(np.float32)
    body = rng.normal((0.0, 0.3, 0.0), 0.25, (2000, 3)).astype(np.float32)
    pts = np.concatenate([wheel, body])
    seeds = np.full(pts.shape[0], -1, np.int32)
    seeds[:150] = 1                        # half the wheel points are seeded

    lab = propagate_labels(pts, seeds)

    assert (lab[:300] == 1).mean() > 0.85, "the wheel did not become a wheel"
    assert (lab[300:] == 0).mean() > 0.9, "the body should default to shell"
    assert (lab >= 0).all()


def test_part_alignment_recovers_a_yaw_and_the_shells_proportions() -> None:
    """PartCrafter's clouds arrive in their own frame; alignment must find the
    right quarter-turn by measurement and then adopt the shell's own bounding
    box, axis by axis."""
    from atanor_core.structure.partlabel import align_parts

    rng = np.random.default_rng(3)
    shell = rng.uniform((-1.0, -0.4, -0.5), (1.0, 0.4, 0.5), (4000, 3)).astype(np.float32)
    # the "generated" cloud: the same box but long axis on z (yaw 90) and squashed
    part_a = rng.uniform((-0.4, -0.3, -0.9), (0.4, 0.3, 0.0), (1500, 3)).astype(np.float32)
    part_b = rng.uniform((-0.4, -0.3, 0.0), (0.4, 0.3, 0.9), (1500, 3)).astype(np.float32)

    aligned, yaw_k, dist = align_parts(shell, [part_a, part_b])

    assert yaw_k in (1, 3), "the quarter-turn was not recovered"
    union = np.concatenate(aligned)
    assert np.allclose(union.min(0), shell.min(0), atol=0.08)
    assert np.allclose(union.max(0), shell.max(0), atol=0.08)
