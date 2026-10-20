"""Dimension, anchor, hierarchy, and rollback contracts for mesh.resize."""

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError


def box(app, scene_id, size, center):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return app.call(
        "mesh.create", scene_id=scene_id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )["id"]


def test_resize_pedestal_keeps_base_and_unspecified_dimensions(app):
    sid = app.scenes.create().id
    oid = box(app, sid, [0.35, 0.3, 0.48], [0, 0, 0.54])
    app.call("uv.generate", object_id=oid, method="box")
    app.call("mesh.split_normals", object_id=oid)
    before = app.engine.object(oid)[1].model_copy(deep=True)
    result = app.call("mesh.resize", object_id=oid, dimensions={"z": 0.29}, anchor="min_z")
    np.testing.assert_allclose(result["new_bounds"], [[-0.175, -0.15, 0.3], [0.175, 0.15, 0.59]])
    assert result["new_dimensions"]["z"] == pytest.approx(0.29)
    assert result["new_dimensions"]["x"] == pytest.approx(0.35)
    obj = app.engine.object(oid)[1]
    assert obj.faces == before.faces and obj.uv == before.uv
    assert obj.geometry_revision != before.geometry_revision
    assert result["changed_ids"] == [oid]


def test_resize_nested_subtree_uses_one_anchor_and_excludes_marker_bounds(app):
    sid = app.scenes.create().id
    first = box(app, sid, [1, 1, 1], [-2, 0, 2])
    second = box(app, sid, [1, 1, 1], [2, 0, 4])
    marker = app.call("scene.marker", scene_id=sid, position=[10, 20, 30])["id"]
    inner = app.call("scene.group", scene_id=sid, object_ids=[first, marker])["id"]
    root = app.call("scene.group", scene_id=sid, object_ids=[inner, second])["id"]
    app.call("mesh.hide", object_id=second)
    result = app.call("mesh.resize", object_id=root, dimensions={"z": 6})
    assert result["old_dimensions"] == {"x": 5, "y": 1, "z": 3}
    np.testing.assert_allclose(
        app.engine.object(first)[1].geometry().centroid, [-2, 0, 1], atol=1e-12
    )
    np.testing.assert_allclose(
        app.engine.object(second)[1].geometry().centroid, [2, 0, 5], atol=1e-12
    )
    np.testing.assert_allclose(app.engine.object(marker)[1].pivot, [10, 20, 57])
    assert app.engine.object(marker)[1].parent_id == inner
    assert app.engine.object(second)[1].hidden
    assert len(result["changed_ids"]) == len(set(result["changed_ids"])) == 5


def test_world_dimensions_after_placement_and_explicit_anchor(app):
    sid = app.scenes.create().id
    oid = app.meshes.create_cube(scene_id=sid).id
    app.call(
        "mesh.place", object_id=oid, position=[2, 3, 4], rotation=[10, 20, 30], scale=[2, 1, 3]
    )
    source = app.engine.object(oid)[1].model_copy(deep=True)
    extents = source.geometry().extents
    anchor = np.array([0.25, -0.5, 1])
    result = app.call(
        "mesh.resize",
        object_id=oid,
        dimensions={"x": float(extents[0] * 0.6)},
        anchor={"type": "point", "position": anchor.tolist()},
    )
    np.testing.assert_allclose(
        app.engine.object(oid)[1].vertices,
        (np.asarray(source.vertices) - anchor) * [0.6, 1, 1] + anchor,
    )
    np.testing.assert_allclose(np.asarray(result["delta_matrix"]) @ [*anchor, 1], [*anchor, 1])
    app.call("mesh.set_transform", object_id=oid, matrix=source.transform)
    np.testing.assert_allclose(app.engine.object(oid)[1].vertices, source.vertices)


def test_uniform_dimensions_must_agree_and_noop_retains_revision(app):
    sid = app.scenes.create().id
    oid = box(app, sid, [1, 2, 3], [0, 0, 0])
    result = app.call("mesh.resize", object_id=oid, dimensions={"y": 4}, aspect_policy="uniform")
    assert result["new_dimensions"] == {"x": 2, "y": 4, "z": 6}
    state = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="same scale ratio"):
        app.call("mesh.resize", object_id=oid, dimensions={"x": 4, "z": 9}, aspect_policy="uniform")
    assert state == app.engine.snapshot()
    unchanged = app.call("mesh.resize", object_id=oid, dimensions={"x": 2, "y": 4})
    assert unchanged["changed_ids"] == [] and state == app.engine.snapshot()


def test_small_matrix_deltas_reach_mesh_and_marker(app):
    sid = app.scenes.create().id
    oid = box(app, sid, [1, 1, 1], [0, 0, 0])
    marker = app.call("scene.marker", scene_id=sid, position=[0, 0, 0], parent_id=oid)["id"]
    result = app.call("mesh.resize", object_id=oid, dimensions={"x": 1 + 1e-10}, anchor="min_x")
    assert abs(result["new_dimensions"]["x"] - (1 + 1e-10)) < 1e-15
    assert app.engine.object(marker)[1].pivot[0] == pytest.approx(5e-11, abs=1e-15)


@pytest.mark.parametrize("dimensions", [{}, {"z": None}, {"x": 0}, {"x": -1}])
def test_invalid_dimension_requests_are_atomic(app, dimensions):
    oid = app.meshes.create_cube(scene_id=app.scenes.create().id).id
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.resize", object_id=oid, dimensions=dimensions)
    assert app.engine.snapshot() == before


def test_geometry_failure_after_mutation_restores_group_and_children(app):
    sid = app.scenes.create().id
    child = app.meshes.create_cube(scene_id=sid).id
    group = app.call("scene.group", scene_id=sid, object_ids=[child])["id"]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.resize", object_id=group, dimensions={"x": 1e-8}, aspect_policy="uniform")
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


def test_zero_source_extent_cannot_create_thickness(app):
    sid = app.scenes.create().id
    oid = app.call(
        "mesh.create", scene_id=sid, vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    )["id"]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="zero or nonfinite source extent"):
        app.call("mesh.resize", object_id=oid, dimensions={"z": 1})
    assert app.engine.snapshot() == before
    result = app.call("mesh.resize", object_id=oid, dimensions={"x": 2}, aspect_policy="uniform")
    assert result["new_dimensions"] == {"x": 2, "y": 2, "z": 0}
