"""Focused cut-volume checks. Kept for later regression runs."""

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.cut_volume import combine_parts, source_parts
from scissors.core.mesh import face_components, topology_mesh


def box(minimum=(0, -2, -2), maximum=(2, 2, 2)):
    return {"type": "box", "min": list(minimum), "max": list(maximum)}


def test_multi_target_cut_keeps_identity_attributes_and_dry_run(app):
    scene = app.scenes.create()
    objects = [app.meshes.create_cube(scene_id=scene.id) for _ in range(2)]
    group = app.call(
        "scene.group", scene_id=scene.id, name="Assembly", object_ids=[obj.id for obj in objects]
    )
    for obj in objects:
        app.call("object.parent", object_id=obj.id, parent_id=group["id"])
        app.call("uv.generate", object_id=obj.id, method="box")
    before = app.engine.snapshot()
    args = {"scene_id": scene.id, "target_ids": [group["id"], objects[0].id], "volume": box()}
    app.call("scene.cut_volume", **args, dry_run=True)
    assert app.engine.snapshot() == before
    result = app.call("scene.cut_volume", **args)
    assert set(result["changed_ids"]) == {obj.id for obj in objects}
    assert len(app.engine.scene(scene.id).objects) == 3
    for ref in objects:
        obj = app.engine.object(ref.id)[1]
        assert obj.parent_id == group["id"]
        assert obj.geometry().volume == pytest.approx(0.5)
        assert topology_mesh(obj.geometry()).is_volume
        assert len(obj.uv) == len(obj.vertices)
        np.testing.assert_allclose(np.linalg.norm(obj.normals, axis=1), 1)


def test_material_parts_can_be_cut_again_and_deleted_without_leaks(app):
    scene = app.scenes.create()
    obj = app.meshes.create_cube(scene_id=scene.id)
    mat = app.call("material.create", scene_id=scene.id)
    app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[obj.id],
        volume=box(),
        cut_material_id=mat["id"],
    )
    root = app.engine.object(obj.id)[1]
    assert root.kind == "group"
    parts = source_parts(app.engine, scene.id, root)
    assert {part.material_id for part in parts} == {None, mat["id"]}
    result = app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[obj.id, parts[0].id],
        volume=box((-2, 0, -2)),
    )
    assert result["changed_ids"] == [obj.id]
    combined, _ = combine_parts(source_parts(app.engine, scene.id, app.engine.object(obj.id)[1]), 1)
    assert topology_mesh(combined.geometry()).is_volume
    assert combined.geometry().volume == pytest.approx(0.25)
    app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[obj.id],
        volume=box((-3, -3, -3), (3, 3, 3)),
        empty_policy="delete_object",
    )
    assert not app.engine.scene(scene.id).objects


def test_hollow_wall_and_disconnected_volumes(app):
    scene = app.scenes.create()
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
    inner.invert()
    wall = trimesh.util.concatenate([outer, inner])
    obj = app.call(
        "mesh.create", scene_id=scene.id, vertices=wall.vertices.tolist(), faces=wall.faces.tolist()
    )
    result = app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[obj["id"]],
        volume=box((0, -0.4, -0.4), (2, 0.4, 0.4)),
    )
    assert result["changed_ids"] == [obj["id"]]
    assert topology_mesh(app.engine.object(obj["id"])[1].geometry()).is_volume
    assert result["targets"][0]["opening"]["contours"]
    cubes = [trimesh.creation.box(), trimesh.creation.box()]
    cubes[1].apply_translation([2, 0, 0])
    mesh = trimesh.util.concatenate(cubes)
    separate = app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )
    app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[separate["id"]],
        volume=box((-1, -1, -1), (1, 1, 1)),
    )
    assert app.engine.object(separate["id"])[1].geometry().volume == pytest.approx(1)


def test_failure_ownership_and_noop_preserve_state(app):
    scene = app.scenes.create()
    cubes = [app.meshes.create_cube(scene_id=scene.id) for _ in range(2)]
    first, second = sorted(ref.id for ref in cubes)
    app.call("mesh.translate", object_id=second, x=2)
    before = app.engine.snapshot()
    # One planned partial cut and one complete removal must not publish either edit.
    with pytest.raises(ScissorsError):
        app.call(
            "scene.cut_volume",
            scene_id=scene.id,
            target_ids=[first, second],
            volume=box((0, -2, -2), (3, 2, 2)),
        )
    assert app.engine.snapshot() == before
    result = app.call(
        "scene.cut_volume",
        scene_id=scene.id,
        target_ids=[first, second],
        volume=box((10, 10, 10), (11, 11, 11)),
    )
    assert result["unchanged_ids"] == [first, second]
    assert app.engine.snapshot() == before
    other = app.scenes.create()
    with pytest.raises(ScissorsError):
        app.call("scene.cut_volume", scene_id=other.id, target_ids=[first], volume=box())


def test_default_topology_components_need_no_optional_graph_backend():
    groups = face_components(5, np.asarray([[0, 1], [1, 2], [3, 4]]))
    assert [group.tolist() for group in groups] == [[0, 1, 2], [3, 4]]
