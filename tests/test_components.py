"""Whole-component geometry, render preservation, and atomic deletion contracts."""

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.face_queries import selection_indices


def merged_cubes(app):
    sid = app.scenes.create().id
    cubes = []
    for x in [-2, 0, 2]:
        mesh = trimesh.creation.box()
        mesh.apply_translation([x, 0, 0])
        cubes.append(mesh)
    mesh = trimesh.util.concatenate(cubes)
    oid = app.call(
        "mesh.create", scene_id=sid, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )["id"]
    return sid, oid


def test_remove_whole_seamed_component_and_preserve_render_buffers(app):
    _, oid = merged_cubes(app)
    app.call("uv.generate", object_id=oid, method="box")
    app.call("mesh.split_normals", object_id=oid)
    before = app.engine.object(oid)[1].model_copy(deep=True)
    faces = app.call("mesh.select_faces", object_id=oid)
    args = dict(object_id=oid, region={"bounds": [[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]})
    state = app.engine.snapshot()
    plan = app.call("mesh.delete_components", **args, dry_run=True)
    assert app.engine.snapshot() == state
    result = app.call("mesh.delete_components", **args)
    assert result["selected_components"] == plan["selected_components"]
    assert result["before"]["component_count"] == 3
    assert result["after"]["component_count"] == 2
    assert result["after"]["face_count"] == 24
    np.testing.assert_allclose(result["selected_components"][0]["centroid"], [0, 0, 0], atol=1e-12)
    current = app.engine.object(oid)[1]
    indices = result["index_maps"]["new_to_old_vertices"]
    assert current.uv == [before.uv[index] for index in indices]
    assert current.normals == [before.normals[index] for index in indices]
    assert all(abs(vertex[0]) >= 1.5 for vertex in current.vertices)
    assert app.call("mesh.validate", object_id=oid, require_closed=True)["valid"]
    with pytest.raises(ScissorsError, match="another geometry revision"):
        selection_indices(current, faces)


def test_point_contact_does_not_connect_faces(app):
    sid = app.scenes.create().id
    oid = app.call(
        "mesh.create",
        scene_id=sid,
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 3, 4]],
    )["id"]
    result = app.call(
        "mesh.delete_components",
        object_id=oid,
        region={"bounds": [[0.2, 0.2, -0.1], [0.5, 0.5, 0.1]]},
        selection_rule="centroid_inside",
    )
    assert result["before"]["component_count"] == 2
    assert result["after"]["face_count"] == 1
    assert result["warnings"]


def test_region_tolerance_does_not_weld_nearby_edges(app):
    sid = app.scenes.create().id
    oid = app.call(
        "mesh.create",
        scene_id=sid,
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1e-10], [1, 0, 1e-10], [0, -1, 1e-10]],
        faces=[[0, 1, 2], [3, 5, 4]],
    )["id"]
    result = app.call(
        "mesh.delete_components",
        object_id=oid,
        region={"bounds": [[0.2, 0.2, -0.1], [0.5, 0.5, 0.1]]},
        selection_rule="centroid_inside",
        tolerance=1e-6,
    )
    assert result["before"]["component_count"] == 2
    assert result["after"]["component_count"] == 1


def test_intersection_uses_triangle_geometry_and_removes_complete_component(app):
    sid = app.scenes.create().id
    oid = app.call(
        "mesh.create", scene_id=sid, vertices=[[0, 0, 0], [2, 0, 0], [0, 2, 0]], faces=[[0, 1, 2]]
    )["id"]
    miss = app.call(
        "mesh.delete_components",
        object_id=oid,
        region={"bounds": [[1.4, 1.4, -0.1], [1.6, 1.6, 0.1]]},
        selection_rule="intersects",
        dry_run=True,
    )
    assert miss["selected_component_count"] == 0
    hit = app.call(
        "mesh.delete_components",
        object_id=oid,
        region={"bounds": [[0.1, 0.1, 0], [0.2, 0.2, 0]]},
        selection_rule="intersects",
        dry_run=True,
    )
    assert hit["selected_component_count"] == 1 and not hit["allowed"]
    assert hit["planned_after"]["face_count"] == 0


def test_centroid_uses_surface_area_weights(app):
    sid = app.scenes.create().id
    oid = app.call(
        "mesh.create",
        scene_id=sid,
        vertices=[[0, 0, 0], [4, 0, 0], [0, 1, 0], [4, 0.5, 0]],
        faces=[[0, 1, 2], [1, 3, 2]],
    )["id"]
    result = app.call(
        "mesh.delete_components",
        object_id=oid,
        region={"bounds": [[1.7, 0.35, -0.1], [1.85, 0.45, 0.1]]},
        selection_rule="centroid_inside",
        dry_run=True,
    )
    assert result["selected_component_count"] == 1
    np.testing.assert_allclose(result["selected_components"][0]["centroid"], [16 / 9, 7 / 18, 0])


def test_empty_policy_and_child_reparenting(app):
    sid, oid = merged_cubes(app)
    group = app.call("scene.group", scene_id=sid, object_ids=[oid])["id"]
    marker = app.call("scene.marker", scene_id=sid, position=[4, 5, 6], parent_id=oid)["id"]
    args = dict(object_id=oid, region={"bounds": [[-10, -10, -10], [10, 10, 10]]})
    before = app.engine.snapshot()
    plan = app.call("mesh.delete_components", **args, dry_run=True)
    assert not plan["allowed"] and plan["planned_action"] == "reject_empty"
    assert before == app.engine.snapshot()
    with pytest.raises(ScissorsError, match="All components selected"):
        app.call("mesh.delete_components", **args)
    assert before == app.engine.snapshot()
    result = app.call("mesh.delete_components", **args, empty_policy="delete_object")
    assert result["object_deleted"]
    assert app.engine.object(marker)[1].parent_id == group
    assert app.engine.object(marker)[1].pivot == [4, 5, 6]
    assert set(result["index_maps"]["old_to_new_faces"]) == {-1}


def test_noop_retains_revision_and_output_limit_rolls_back(app):
    _, oid = merged_cubes(app)
    state = app.engine.snapshot()
    result = app.call(
        "mesh.delete_components", object_id=oid, region={"bounds": [[10, 10, 10], [11, 11, 11]]}
    )
    assert result["planned_action"] == "unchanged" and state == app.engine.snapshot()
    limit = app.engine.settings.max_file_bytes
    app.engine.settings.max_file_bytes = 512
    try:
        with pytest.raises(ScissorsError) as error:
            app.call(
                "mesh.delete_components",
                object_id=oid,
                region={"bounds": [[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]},
            )
        assert error.value.code == "RESOURCE_LIMIT"
        assert state == app.engine.snapshot()
    finally:
        app.engine.settings.max_file_bytes = limit
