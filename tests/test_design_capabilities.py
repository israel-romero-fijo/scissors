"""Regression cases for native scene design. Do not run under the no-tests request."""

import json
from pathlib import Path

import numpy as np
import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


def test_group_parent_cycle_and_world_preservation(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.meshes.data(object_id=cube.id).vertices
    group = app.scenes.group(scene_id=scene.id, object_ids=[cube.id])
    assert app.meshes.data(object_id=cube.id).vertices == before
    with pytest.raises(ScissorsError):
        app.objects.parent(object_id=group.id, parent_id=cube.id)
    app.meshes.translate(object_id=group.id, x=2)
    np.testing.assert_allclose(
        app.meshes.inspect(object_id=cube.id).centroid, [2, 0, 0], atol=1e-12
    )


def test_group_with_child_does_not_transform_twice(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    marker = app.scenes.marker(scene_id=scene.id, position=[0, 0, 2])
    app.objects.parent(object_id=cube.id, parent_id=marker.id)
    group = app.scenes.group(scene_id=scene.id, object_ids=[marker.id, cube.id])
    assert group.children == [marker.id]
    app.meshes.translate(object_id=group.id, x=1)
    np.testing.assert_allclose(
        app.meshes.inspect(object_id=cube.id).centroid, [1, 0, 0], atol=1e-12
    )


def test_parent_rejects_other_scene_and_delete_reparents(app):
    first, second = app.scenes.create(), app.scenes.create()
    cube = app.meshes.create_cube(scene_id=first.id)
    marker = app.scenes.marker(scene_id=second.id)
    with pytest.raises(ScissorsError):
        app.objects.parent(object_id=cube.id, parent_id=marker.id)
    group = app.scenes.group(scene_id=first.id, object_ids=[cube.id])
    app.meshes.delete(object_id=group.id)
    assert app.meshes.inspect(object_id=cube.id).parent_id is None


def test_marker_and_visibility_persist(tmp_path):
    with Scissors(workspace=tmp_path) as app:
        scene = app.scenes.create()
        marker = app.scenes.marker(scene_id=scene.id, position=[1, 2, 3])
        cube = app.meshes.create_cube(scene_id=scene.id)
        group = app.scenes.group(scene_id=scene.id, object_ids=[cube.id])
        app.meshes.hide(object_id=group.id)
    with Scissors(workspace=tmp_path) as app:
        np.testing.assert_allclose(app.meshes.inspect(object_id=marker.id).pivot, [1, 2, 3])
        assert cube.id in app.meshes.hidden(scene_id=scene.id).object_ids


def test_explicit_uv_rectangle_and_flip(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id)
    app.uv.planar(object_id=plane.id, u0=0.25, v0=0.5, u1=0.5, v1=0.75)
    uv = np.asarray(app.meshes.data(object_id=plane.id).uv)
    np.testing.assert_allclose(uv.min(axis=0), [0.25, 0.5])
    np.testing.assert_allclose(uv.max(axis=0), [0.5, 0.75])
    app.uv.flip_v(object_id=plane.id)
    flipped = np.asarray(app.meshes.data(object_id=plane.id).uv)
    np.testing.assert_allclose(flipped[:, 1], 1 - uv[:, 1])


def test_edit_failure_restores_every_step(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[
                {"operation": "mesh.translate", "arguments": {"object_id": cube.id, "x": 5}},
                {"operation": "mesh.delete", "arguments": {"object_id": "missing"}},
            ],
        )
    assert app.engine.snapshot() == before


def test_transaction_final_check_failure_restores(app, monkeypatch):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()

    def fail():
        raise ScissorsError("RESOURCE_LIMIT", "Injected final check failure")

    monkeypatch.setattr(app.engine, "check_state", fail)
    with pytest.raises(ScissorsError):
        with app.transaction():
            app.meshes.translate(object_id=cube.id, x=5)
    assert app.engine.snapshot() == before


def test_group_export_and_real_instance_mesh_sharing(app):
    scene = app.scenes.create(name="Assembly")
    cube = app.meshes.create_cube(scene_id=scene.id, name="Bolt")
    other = app.scenes.instance(object_id=cube.id, name="Other bolt")
    app.meshes.translate(object_id=other.id, x=2)
    app.scenes.group(scene_id=scene.id, object_ids=[cube.id, other.id], name="Bolts")
    exported = app.export.glb(scene.id)
    data = Path(exported.path).read_bytes()
    length = int.from_bytes(data[12:16], "little")
    document = json.loads(data[20 : 20 + length])
    assert len(document["meshes"]) == 1
    nodes = {node["name"]: node for node in document["nodes"]}
    assert len(nodes["Bolts"]["children"]) == 2


def test_short_prism_is_closed_and_faceted(app):
    app.scenes.create()
    prism = app.meshes.create_prism(6, 0.1, 0.3)
    assert prism.watertight
    normals = np.asarray(app.meshes.data(object_id=prism.id).normals)
    for start in range(0, len(normals), 3):
        np.testing.assert_allclose(normals[start], normals[start + 1])
        np.testing.assert_allclose(normals[start], normals[start + 2])


def test_profile_hole_counts_and_extrusion_volume(app):
    scene = app.scenes.create()
    outer = [[0, 0], [2, 0], [2, 2], [0, 2]]
    hole = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
    profile = app.meshes.create_profile(scene_id=scene.id, points=outer, holes=[hole])
    assert profile.face_count == 8
    extrusion = app.meshes.extrude(object_id=profile.id, distance=2)
    assert extrusion.watertight
    assert extrusion.volume == pytest.approx(6)


def test_profile_rejects_crossing_and_outside_hole(app):
    scene = app.scenes.create()
    with pytest.raises(ScissorsError) as crossed:
        app.meshes.create_profile(scene_id=scene.id, points=[[0, 0], [1, 1], [0, 1], [1, 0]])
    assert crossed.value.code == "GEOMETRY_ERROR"
    with pytest.raises(ScissorsError):
        app.meshes.create_profile(
            scene_id=scene.id,
            points=[[0, 0], [1, 0], [1, 1], [0, 1]],
            holes=[[[2, 2], [3, 2], [3, 3]]],
        )


def test_profile_determinism_and_duplicate_warning(app):
    scene = app.scenes.create()
    points = [[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]]
    first = app.meshes.create_profile(scene_id=scene.id, points=points)
    second = app.meshes.create_profile(scene_id=scene.id, points=points)
    assert first.warnings
    a = app.meshes.data(object_id=first.id)
    b = app.meshes.data(object_id=second.id)
    assert a.vertices == b.vertices and a.faces == b.faces


def test_surface_distance_and_contained_solid(app):
    scene = app.scenes.create()
    outer = app.meshes.create_cube(scene_id=scene.id, size=2)
    inner = app.meshes.create_cube(scene_id=scene.id, size=1)
    result = app.call("mesh.intersect_test", object_id=outer.id, other_object_id=inner.id)
    assert result["penetration"]
    assert not result["surface_contact"]
    assert result["surface_distance"] == pytest.approx(0.5)
    app.meshes.translate(object_id=inner.id, x=3)
    result = app.call("scene.distance", object_id=outer.id, other_object_id=inner.id)
    assert not result["intersects"]
    assert result["surface_distance"] == pytest.approx(1.5)


def test_render_sheet_is_labeled_and_bounded(app):
    from PIL import Image

    scene = app.scenes.create()
    app.meshes.create_cube(scene_id=scene.id)
    result = app.call("scene.render_sheet", scene_id=scene.id, size=64)
    with Image.open(result["path"]) as image:
        assert image.size == (192, 176)
    assert result["view_count"] == 6


def test_weld_preserves_uv_seams_by_default(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.uv.generate(object_id=cube.id, method="box")
    before = app.meshes.inspect(object_id=cube.id)
    result = app.meshes.weld(object_id=cube.id)
    assert result.has_uv and result.watertight
    assert result.vertex_count >= 8 and result.vertex_count <= before.vertex_count
    merged = app.meshes.weld(object_id=cube.id, preserve_uv=False)
    assert merged.vertex_count == 8


def test_convex_collider_keeps_source_and_rejects_invalid_budget(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    collider = app.meshes.collider(object_id=cube.id, max_faces=12)
    assert collider.watertight and collider.face_count <= 12
    assert app.meshes.inspect(object_id=cube.id).face_count == 12
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.meshes.collider(object_id=cube.id, max_faces=3)
    assert app.engine.snapshot() == before


def test_fill_hole_after_missing_cube_face(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    data = app.meshes.data(object_id=cube.id)
    faces = [face for face in data.faces if not all(data.vertices[index][2] > 0 for index in face)]
    opened = app.call("mesh.create", scene_id=scene.id, vertices=data.vertices, faces=faces)
    filled = app.meshes.fill_holes(object_id=opened["object_id"], max_edges=4)
    assert filled.filled_holes == 1 and filled.watertight


def test_smoothed_render_seams_remain_coincident(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.facet(object_id=cube.id)
    result = app.meshes.smooth(object_id=cube.id, factor=0.1)
    assert result.watertight
    assert len(np.unique(app.meshes.data(object_id=cube.id).vertices, axis=0)) == 8


def test_default_rotation_keeps_legacy_world_origin_and_explicit_pivot_moves(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=cube.id, x=2)
    rotated = app.meshes.rotate(object_id=cube.id, angle_degrees=90)
    np.testing.assert_allclose(rotated.centroid, [0, 2, 0], atol=1e-12)
    app.meshes.pivot(object_id=cube.id, position=[0, 2, 0])
    app.meshes.translate(object_id=cube.id, x=1)
    rotated = app.meshes.rotate(object_id=cube.id, angle_degrees=90)
    np.testing.assert_allclose(rotated.centroid, [1, 2, 0], atol=1e-12)
