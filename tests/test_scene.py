"""Multi-object scene foundation (the LLM-explainer Phase 1)."""

import numpy as np

from atanor_core.domain.scene import Scene, SceneObject
from atanor_core.generation.generator import MockGenerator


def _obj(oid, shape, pos):
    f = MockGenerator(n_points=300).generate(
        np.full((1, 4, 3, 8, 8), 0.6, np.float32), cam_rays={"shape": shape})
    return SceneObject(id=oid, field=f, position=np.array(pos, np.float32))


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
