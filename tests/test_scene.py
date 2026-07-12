"""Multi-object scene foundation (the LLM-explainer Phase 1)."""

import numpy as np

from atanor_core.domain.scene import Scene, SceneObject
from atanor_core.domain.sgf import GaussianField
from atanor_core.generation.generator import MockGenerator


def _obj(oid, shape, pos):
    f = MockGenerator(n_points=300).generate(
        np.full((1, 4, 3, 8, 8), 0.6, np.float32), cam_rays={"shape": shape})
    return SceneObject(id=oid, field=f, position=np.array(pos, np.float32))


def _field(n, sh_degree):
    k = (sh_degree + 1) ** 2
    return GaussianField(
        means=np.zeros((n, 3), np.float32),
        scales=np.full((n, 3), -3.0, np.float32),
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.full((n,), 0.5, np.float32),
        sh=np.full((n, k, 3), 0.3, np.float32),
        sh_degree=sh_degree,
    )


def test_flatten_composes_mixed_sh_degrees():
    """A procedural object (K=4) and a TripoSR/flat-color object (K=1) must compose in one
    scene. Raw np.concatenate of their SH used to ValueError and 500 the whole scene
    ('지구와 달' = procedural 지구 + TripoSR 달); flatten now zero-pads SH to the max K."""
    scene = Scene()
    scene.add(SceneObject(id="a", field=_field(10, 1), position=np.array([-2, 0, 0], np.float32)))
    scene.add(SceneObject(id="b", field=_field(12, 0), position=np.array([2, 0, 0], np.float32)))
    scene.link("a", "b")
    field = scene.flatten()                      # must not raise
    assert field.sh.shape[1] == 4                # padded up to max K
    assert field.sh_degree == 1
    assert field.num_gaussians >= 22             # both objects + link strands
    # the K=1 object's color (band 0) is preserved after padding
    assert np.allclose(field.sh[15, 0, :], 0.3)
    assert np.allclose(field.sh[15, 1:, :], 0.0)  # padded higher bands are zero


def test_scene_flatten_places_objects_apart():
    scene = Scene()
    scene.add(_obj("a", "sphere", [-2, 0, 0]))
    scene.add(_obj("b", "cube", [2, 0, 0]))
    scene.link("a", "b", color=(0.4, 0.9, 1.0))

    field = scene.flatten()
    na = scene.objects["a"].field.num_gaussians
    nb = scene.objects["b"].field.num_gaussians
    assert field.num_gaussians >= na + nb           # both objects + link strands
    # spatially separated: x spans roughly [-2-r, 2+r]
    assert field.means[:, 0].min() < -1.5
    assert field.means[:, 0].max() > 1.5
    assert scene.version >= 3                        # 2 adds + 1 link


def test_scene_move_and_remove():
    scene = Scene().add(_obj("a", "sphere", [0, 0, 0]))
    scene.move("a", [0, 3, 0])
    assert scene.objects["a"].position[1] == 3.0
    f = scene.flatten()
    assert f.means[:, 1].mean() > 1.0                # shifted up
    scene.remove("a")
    assert not scene.objects
