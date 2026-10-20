"""Placement and deformation regression cases; unrun under the user request."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError


def test_linear_array_copies_subtrees_and_remaps_parents(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    group = app.scenes.group(scene_id=scene.id, object_ids=[cube.id])
    result = app.call("mesh.array", object_id=group.id, count=3, offset=[2, 0, 0])
    assert result["created_count"] == 2 and result["object_ids"][0] == group.id
    for index, root in enumerate(result["object_ids"]):
        parent = app.engine.object(root)[1]
        children = app.engine.descendants(app.engine.scene(scene.id), root)
        assert len(children) == 1 and children[0].parent_id == parent.id
        np.testing.assert_allclose(children[0].summary()["centroid"], [2 * index, 0, 0], atol=1e-12)


def test_full_polar_array_does_not_repeat_source_angle(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=cube.id, x=2)
    result = app.call("mesh.array", object_id=cube.id, count=4, mode="polar", rotate=False)
    for oid, center in zip(
        result["object_ids"], [[2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]], strict=True
    ):
        obj = app.engine.object(oid)[1]
        np.testing.assert_allclose(obj.summary()["centroid"], center, atol=1e-12)
        np.testing.assert_allclose(np.ptp(obj.vertices, axis=0), [1, 1, 1])


def test_arbitrary_mirror_records_transform_and_preserves_uv(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=cube.id, x=2)
    before = app.engine.object(cube.id)[1].model_copy(deep=True)
    result = app.call("mesh.mirror", object_id=cube.id, normal=[1, 1, 0], point=[1, 0, 0])
    obj = app.engine.object(result["object_id"])[1]
    expected = np.asarray(before.vertices).copy()
    expected[:, 0], expected[:, 1] = 1 - expected[:, 1].copy(), 1 - expected[:, 0].copy()
    np.testing.assert_allclose(obj.vertices, expected, atol=1e-12)
    assert obj.uv == before.uv and obj.geometry().is_winding_consistent
    assert obj.geometry().volume > 0
    np.testing.assert_allclose(app.engine.object(cube.id)[1].vertices, before.vertices)


def test_warp_preserves_attributes_and_failed_bend_restores_state(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.object(cube.id)[1].model_copy(deep=True)
    with pytest.raises(ScissorsError):
        app.call("deform.warp", object_id=cube.id, bend=300)
    assert app.engine.object(cube.id)[1].vertices == before.vertices
    app.call("deform.warp", object_id=cube.id, twist=90, taper=0.5)
    obj = app.engine.object(cube.id)[1]
    assert obj.uv == before.uv and obj.material_id == before.material_id
    assert obj.faces == before.faces and obj.vertices != before.vertices
    assert obj.metadata["last_warp"]["twist_degrees"] == 90


@pytest.mark.parametrize("operation", ["mesh.align", "mesh.snap"])
def test_surface_placement_respects_gap_and_rejects_moving_target(app, operation):
    scene = app.scenes.create()
    source = app.meshes.create_cube(scene_id=scene.id)
    target = app.meshes.create_cube(scene_id=scene.id)
    mesh = app.engine.object(target.id)[1].geometry()
    face = int(np.argmax(mesh.face_normals[:, 2]))
    args = {"target_face": face} if operation == "mesh.align" else {"point": [0, 0, 2]}
    app.call(operation, object_id=source.id, target_object_id=target.id, gap=0.1, **args)
    bounds = app.engine.object(source.id)[1].summary()["bounds"]
    assert bounds[0][2] == pytest.approx(0.6)
    app.objects.parent(object_id=target.id, parent_id=source.id)
    with pytest.raises(ScissorsError):
        app.call(operation, object_id=source.id, target_object_id=target.id, **args)


def test_symmetry_skips_existing_counterpart_and_handles_unknown_selection(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    assert app.call("scene.symmetrize", scene_id=scene.id)["created_object_ids"] == []
    app.meshes.translate(object_id=cube.id, x=2)
    result = app.call("scene.symmetrize", scene_id=scene.id)
    assert len(result["created_object_ids"]) == 1
    assert app.call("scene.symmetrize", scene_id=scene.id)["created_object_ids"] == []
    with pytest.raises(ScissorsError) as failure:
        app.call("scene.symmetrize", scene_id=scene.id, object_ids=["missing"])
    assert failure.value.code == "OBJECT_NOT_FOUND"


def test_measure_face_distance_and_opposing_normals(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    normals = app.engine.object(cube.id)[1].geometry().face_normals
    first, second = int(np.argmax(normals[:, 2])), int(np.argmin(normals[:, 2]))
    distance = app.call(
        "mesh.measure", object_id=cube.id, mode="distance", face_a=first, face_b=second
    )
    assert distance["distance"] == pytest.approx(1)
    angle = app.call("mesh.measure", object_id=cube.id, mode="angle", face_a=first, face_b=second)
    assert angle["angle_degrees"] == pytest.approx(180)
    assert angle["acute_plane_angle_degrees"] == pytest.approx(0)


def test_measure_hole_uses_current_transformed_geometry(app):
    scene = app.scenes.create()
    plate = app.call(
        "mesh.create_plate",
        scene_id=scene.id,
        size=[2, 2],
        thickness=0.1,
        holes=[{"center": [0, 0], "radius": 0.2}],
    )
    oid = plate["object_id"]
    app.meshes.translate(object_id=oid, x=3)
    mesh = app.engine.object(oid)[1].geometry()
    face = int(np.argmax(mesh.face_normals[:, 2]))
    result = app.call("mesh.measure", object_id=oid, mode="diameter", face_a=face)
    assert result["diameter"] == pytest.approx(0.4)
    np.testing.assert_allclose(result["center"][:2], [3, 0], atol=1e-12)
    revision = app.registry.revision
    app.call("mesh.measure", object_id=oid, mode="diameter", face_a=face)
    assert app.registry.revision == revision


def test_distribute_builds_a_fibonacci_cutter_sphere(app):
    import trimesh

    scene = app.scenes.create()
    ball = app.meshes.create_sphere(scene_id=scene.id, radius=1.0, subdivisions=2)
    result = app.call(
        "mesh.distribute",
        scene_id=scene.id,
        object_id=ball.id,
        count=60,
        hole_radius=0.07,
        pattern="fibonacci",
        orient="normal",
    )
    assert result["count"] == 60
    assert len(result["object_ids"]) == 60
    matrices = np.asarray(result["matrices"])
    assert matrices.shape == (60, 4, 4)
    axes = matrices[:, :3, 2]
    assert np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-9)
    np.testing.assert_allclose(matrices[:, :3, 3], 0.0, atol=1e-9)
    assert len({tuple(np.round(axis, 6)) for axis in axes}) == 60
    expected = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axes[0])
    np.testing.assert_allclose(matrices[0][:3, :3], expected[:3, :3], atol=1e-12)
    assert result["length"] == pytest.approx(2.4)
    for object_id in result["object_ids"]:
        assert app.engine.object(object_id)[1].kind == "mesh"

    axial = app.call(
        "mesh.distribute",
        scene_id=scene.id,
        object_id=ball.id,
        count=4,
        hole_radius=0.05,
        orient="z",
    )
    np.testing.assert_allclose(
        np.asarray(axial["matrices"])[:, :3, :3], np.tile(np.eye(3), (4, 1, 1))
    )
