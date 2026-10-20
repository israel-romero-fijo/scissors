"""Regression cases prepared for the cab contracts; execution is user-controlled."""

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.face_queries import triangle_box
from scissors.core.mesh import MeshObject
from scissors.core.normals import split_normals
from scissors.core.validation import validate_mesh


def underside(app, object_id):
    return app.call(
        "mesh.select_faces",
        object_id=object_id,
        normal={"direction": [0, 0, -1], "angular_tolerance_degrees": 0.1},
    )


def test_selected_shell_is_one_closed_wall_with_an_open_underside(app):
    scene = app.scenes.create()
    body = app.meshes.create_cube(scene_id=scene.id, size=2)
    selection = underside(app, body.id)
    assert selection["count"] == 2
    app.call("mesh.shell", object_id=body.id, opening=selection, thickness=0.1)
    report = app.call(
        "mesh.validate",
        object_id=body.id,
        require_closed=True,
        require_connected=True,
        check_self_intersections=True,
    )
    assert report["valid"]
    assert report["properties"]["surface_components"] == 1
    assert report["properties"]["boundary_edges"] == 0
    assert report["properties"]["self_intersections"]["status"] == "passed"
    obj = app.engine.object(body.id)[1]
    # There is a rim, but no cap covering the center of the underside.
    horizontal = np.abs(obj.geometry().face_normals[:, 2]) > 0.99
    centers = obj.geometry().triangles_center
    assert not np.any(
        horizontal & (centers[:, 2] < -0.8) & (np.linalg.norm(centers[:, :2], axis=1) < 0.5)
    )


def test_stale_opening_rolls_back_after_world_transform(app):
    scene = app.scenes.create()
    body = app.meshes.create_cube(scene_id=scene.id)
    selection = underside(app, body.id)
    app.call("mesh.translate", object_id=body.id, z=2)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="revision"):
        app.call("mesh.shell", object_id=body.id, opening=selection)
    assert app.engine.snapshot() == before


@pytest.mark.parametrize("minimum,status", [(0.05, "pass"), (0.15, "fail"), (0.1, "inconclusive")])
def test_rectangular_shell_thickness_uses_bounds(app, minimum, status):
    scene = app.scenes.create()
    body = app.meshes.create_cube(scene_id=scene.id, size=2)
    app.call("mesh.shell", object_id=body.id, opening=underside(app, body.id), thickness=0.1)
    result = app.call("mesh.check_thickness", object_id=body.id, minimum=minimum, tolerance=1e-5)
    assert result["status"] == status
    assert result["coverage"]["complete"]
    assert result["excluded_rim_regions"]
    assert result["coverage"]["certified_faces"] > 0
    assert result["coverage"]["measured_area"] is None


def test_open_surface_is_inconclusive_and_closed_curved_surface_can_pass_thickness(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id)
    sphere = app.meshes.create_sphere(scene_id=scene.id, subdivisions=0)
    result = app.call("mesh.check_thickness", object_id=plane.id, minimum=0.001, tolerance=1e-5)
    assert result["status"] == "inconclusive"
    assert result["unresolved_regions"]
    result = app.call("mesh.check_thickness", object_id=sphere.id, minimum=0.001, tolerance=1e-5)
    assert result["status"] == "pass"
    assert result["coverage"]["complete"]
    assert result["achieved_error_bounds"]["global_lower"] > 0


def test_exact_triangle_box_rule_rejects_overlapping_bounds_without_contact():
    triangle = np.asarray([[[2, 0, 0], [0, 2, 0], [2, 2, 0]]], dtype=float)
    assert not triangle_box(triangle, np.array([0, 0, -1]), np.array([0.5, 0.5, 1]))[0]


def test_edge_connectedness_does_not_join_point_contact(app):
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([1, 1, 1])
    joined = trimesh.util.concatenate([first, second])
    obj = MeshObject(
        id="point-contact",
        name="Point contact",
        vertices=joined.vertices.tolist(),
        faces=joined.faces.tolist(),
    )
    report = validate_mesh(obj, app.engine.settings, require_closed=True, require_connected=True)
    assert not report["valid"]
    assert report["properties"]["surface_components"] == 2
    assert "NOT_CONNECTED" in report["errors"]


def test_normal_splitting_retains_skinny_valid_corner(app):
    obj = MeshObject(
        id="skinny", name="Skinny", vertices=[[0, 0, 0], [1, 0, 0], [1, 1e-8, 0]], faces=[[0, 1, 2]]
    )
    split_normals(obj, 0, app.engine.settings)
    np.testing.assert_allclose(obj.normals, [[0, 0, 1]] * 3)


@pytest.mark.parametrize("operation", ["sphere", "capsule", "polyhedron"])
def test_added_capped_false_creates_an_open_surface(app, operation):
    scene = app.scenes.create()
    result = app.call(f"mesh.create_{operation}", scene_id=scene.id, capped=False)
    assert not result["watertight"]
    assert result["face_count"] > 0


def test_even_torus_twist_produces_two_closed_link_components(app):
    scene = app.scenes.create()
    result = app.call(
        "mesh.create_torus", scene_id=scene.id, twist=2, major_sections=64, minor_radius=0.05
    )
    report = app.call("mesh.validate", object_id=result["object_id"], require_closed=True)
    assert report["valid"]
    assert report["properties"]["surface_components"] == 2
