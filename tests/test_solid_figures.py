"""Native figure contracts. These regression cases remain unrun by user request."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("mesh.create_capsule", {"radius": 0.1, "height": 0.6, "segments": 16}),
        ("mesh.create_capsule", {"radius": 0.1, "height": 0.2, "segments": 16}),
        ("mesh.create_hemisphere", {"radius": 0.2, "segments": 16}),
        ("mesh.create_dome", {"radius": 0.3, "height": 0.1, "segments": 16}),
        ("mesh.create_wedge", {"size": [0.3, 0.2, 0.1]}),
        ("mesh.create_wedge", {"size": [0.3, 0.2, 0.1], "taper": 0.5}),
        ("mesh.create_wedge", {"size": [0.3, 0.2, 0.1], "taper": 1}),
        ("mesh.create_gear", {"teeth": 12, "radius": 0.2, "thickness": 0.03, "bore_radius": 0.04}),
        ("mesh.create_spring", {"radius": 0.2, "wire_radius": 0.01, "height": 0.3, "coils": 4}),
    ],
)
def test_native_solid_is_closed_outward_and_deterministic(app, operation, arguments):
    scene = app.scenes.create()
    a = app.call(operation, scene_id=scene.id, **arguments)
    b = app.call(operation, scene_id=scene.id, **arguments)
    first, second = app.engine.object(a["object_id"])[1], app.engine.object(b["object_id"])[1]
    assert first.vertices == second.vertices and first.faces == second.faces
    assert a["watertight"] and a["volume"] > 0
    assert first.geometry().is_winding_consistent


@pytest.mark.parametrize(
    "kind,triangles",
    [
        ("tetrahedron", 4),
        ("octahedron", 8),
        ("dodecahedron", 36),
        ("icosahedron", 20),
    ],
)
def test_platonic_faces_radius_and_normals(app, kind, triangles):
    scene = app.scenes.create()
    result = app.call("mesh.create_polyhedron", scene_id=scene.id, kind=kind, radius=0.5)
    obj = app.engine.object(result["object_id"])[1]
    assert result["face_count"] == triangles and result["watertight"]
    np.testing.assert_allclose(np.linalg.norm(obj.vertices, axis=1), 0.5, atol=1e-6)
    for face in obj.faces:
        normals = np.asarray(obj.normals)[face]
        np.testing.assert_allclose(normals, np.repeat(normals[:1], 3, axis=0))


def test_capsule_total_height_and_axis(app):
    scene = app.scenes.create()
    result = app.call(
        "mesh.create_capsule", scene_id=scene.id, radius=0.1, height=0.6, axis="x", segments=16
    )
    np.testing.assert_allclose(result["bounds"], [[0, -0.1, -0.1], [0.6, 0.1, 0.1]], atol=1e-12)


def test_open_dome_has_only_base_boundary(app):
    scene = app.scenes.create()
    result = app.call(
        "mesh.create_dome", scene_id=scene.id, radius=0.2, height=0.1, capped=False, segments=16
    )
    validation = app.call("mesh.validate", object_id=result["object_id"])
    assert validation["valid"] and not result["watertight"]
    assert validation["properties"]["boundary_edges"] == 16


def test_plate_through_hole_volume_material_uv_and_error_rollback(app):
    scene = app.scenes.create()
    material = app.materials.create(scene_id=scene.id)
    result = app.call(
        "mesh.create_plate",
        scene_id=scene.id,
        size=[1, 1],
        thickness=0.1,
        holes=[{"center": [0, 0], "radius": 0.2, "sides": 24}],
        material_id=material.id,
    )
    expected = (1 - 24 * np.sin(2 * np.pi / 24) * 0.2**2 / 2) * 0.1
    assert result["volume"] == pytest.approx(expected)
    assert result["watertight"] and result["has_uv"] and result["material_id"] == material.id
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "mesh.create_plate",
            scene_id=scene.id,
            size=[1, 1],
            holes=[{"center": [0.49, 0], "radius": 0.2}],
        )
    assert app.engine.snapshot() == before


@pytest.mark.parametrize("kind", ["cross", "arrow"])
def test_star_family_variants_create_closed_shapes(app, kind):
    scene = app.scenes.create()
    result = app.call("mesh.create_star", scene_id=scene.id, kind=kind, thickness=0.1)
    assert result["watertight"] and result["volume"] > 0


@pytest.mark.parametrize(
    "arc,capped,closed", [(180, True, True), (180, False, False), (360, True, True)]
)
def test_torus_arc_caps(app, arc, capped, closed):
    scene = app.scenes.create()
    result = app.call("mesh.create_torus", scene_id=scene.id, arc=arc, capped=capped)
    assert result["watertight"] is closed
    assert app.call("mesh.validate", object_id=result["object_id"])["valid"]


def test_torus_knot_has_one_closed_shell(app):
    scene = app.scenes.create()
    result = app.call("mesh.create_torus", scene_id=scene.id, twist=3, minor_radius=0.15)
    assert result["watertight"] and result["volume"] > 0


def test_angle_normals_keep_hard_cube_edges_and_smooth_uv_seams(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.uv.generate(object_id=cube.id, method="box")
    hard = app.call("mesh.split_normals", object_id=cube.id, angle=30)
    assert hard["watertight"] and hard["has_uv"]
    obj = app.engine.object(cube.id)[1]
    for face in obj.faces:
        normals = np.asarray(obj.normals)[face]
        np.testing.assert_allclose(normals, np.repeat(normals[:1], 3, axis=0))
    app.call("mesh.split_normals", object_id=cube.id, angle=180)
    obj = app.engine.object(cube.id)[1]
    for point in np.unique(obj.vertices, axis=0):
        indices = np.flatnonzero(np.all(np.asarray(obj.vertices) == point, axis=1))
        normals = np.asarray(obj.normals)[indices]
        np.testing.assert_allclose(normals, np.repeat(normals[:1], len(indices), axis=0))
    app.meshes.scale(object_id=cube.id, x=2)
    obj = app.engine.object(cube.id)[1]
    np.testing.assert_allclose(np.linalg.norm(obj.normals, axis=1), 1)
    assert "INCORRECT_NORMALS" not in app.call("mesh.validate", object_id=cube.id)["warnings"]
