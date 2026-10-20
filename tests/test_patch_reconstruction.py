"""Constrained patch reconstruction with fixed borders, features and outside faces."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def panel(app, seam=False):
    scene = app.scenes.create()
    n = 5
    vertices = [[i, j, 0.15 if i == j == 2 else 0] for j in range(n) for i in range(n)]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.extend([[a, a + 1, a + n + 1], [a, a + n + 1, a + n]])
    obj = app.call("mesh.create", scene_id=scene.id, vertices=vertices, faces=faces)
    if seam:
        app.call("mesh.facet", object_id=obj["id"])
    app.call("uv.planar", object_id=obj["id"])
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=obj["id"], material_id=material["id"])
    source = app.engine.object(obj["id"])[1]
    source.vertex_colors = [[30, 90, 120, 255]] * len(source.vertices)
    selected = app.call(
        "mesh.select_faces",
        object_id=obj["id"],
        region={"bounds": [[1, 1, -1], [3, 3, 1]], "rule": "fully_inside"},
    )
    return obj["id"], selected


@pytest.mark.parametrize("seam", [False, True])
def test_reconstruction_fixes_boundary_and_outside_positions_and_preserves_attributes(app, seam):
    oid, selected = panel(app, seam)
    before = app.engine.object(oid)[1].model_dump()
    result = app.call("mesh.reconstruct_patch", object_id=oid, selection=selected)
    obj = app.engine.object(oid)[1]
    original_triangles = np.asarray(before["vertices"])[before["faces"]]
    triangles = np.asarray(obj.vertices)[obj.faces]
    keep = [i for i in range(len(before["faces"])) if i not in selected["face_indices"]]
    np.testing.assert_array_equal(triangles[keep], original_triangles[keep])
    protected = result["protected_vertex_indices"]
    np.testing.assert_array_equal(
        np.asarray(obj.vertices)[protected], np.asarray(before["vertices"])[protected]
    )
    centers = [i for i, point in enumerate(before["vertices"]) if point[:2] == [2, 2]]
    assert all(obj.vertices[i][2] < 0.15 for i in centers)
    assert len(result["selection"]["face_indices"]) > len(selected["face_indices"])
    assert obj.material_id == before["material_id"] and obj.transform == before["transform"]
    assert all(color == [30, 90, 120, 255] for color in obj.vertex_colors)
    assert len(obj.uv) == len(obj.vertices)
    assert topology_mesh(obj.geometry()).is_winding_consistent
    assert len(app.call("mesh.boundaries", object_id=oid)["loops"]) == 1


def test_sharp_cube_features_survive_local_refinement(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    selected = app.call("mesh.select_faces", object_id=cube.id)
    before = app.engine.object(cube.id)[1].model_dump()
    result = app.call(
        "mesh.reconstruct_patch", object_id=cube.id, selection=selected, sharp_angle_degrees=45
    )
    obj = app.engine.object(cube.id)[1]
    assert result["watertight"] and result["volume"] == pytest.approx(1)
    np.testing.assert_array_equal(obj.vertices[: len(before["vertices"])], before["vertices"])
    assert len(result["protected_edges"]) == 12
    assert result["face_count"] > len(before["faces"])


def test_named_patch_selection_and_reopened_workspace(app):
    oid, selected = panel(app)
    app.call("mesh.selection_save", name="Panel", selection=selected)
    selected = app.call("mesh.selection_get", object_id=oid, name="Panel")
    result = app.call("mesh.reconstruct_patch", object_id=oid, selection=selected, iterations=0)
    with pytest.raises(ScissorsError, match="stale"):
        app.call("mesh.selection_get", object_id=oid, name="Panel")
    app.call("mesh.selection_save", name="Refined", selection=result["selection"])
    saved = app.engine.object(oid)[1].model_dump()
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.object(oid)[1].model_dump() == saved


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_faces", 32),
        ("max_vertices", 25),
        ("max_memory_bytes", 1),
        ("max_file_bytes", 1),
        ("max_geometry_pairs", 1),
    ],
)
def test_patch_budgets_fail_atomically(app, field, value):
    oid, selected = panel(app)
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.reconstruct_patch", object_id=oid, selection=selected)
    assert error.value.code == "RESOURCE_LIMIT" and app.engine.snapshot() == before


def test_patch_selection_validation_and_schema(app):
    oid, selected = panel(app)
    before = app.engine.snapshot()
    for invalid in [
        dict(selected, geometry_revision="old"),
        dict(selected, face_indices=[]),
        dict(selected, object_id="other"),
        dict(selected, face_indices=[0, 0]),
    ]:
        with pytest.raises(ScissorsError):
            app.call("mesh.reconstruct_patch", object_id=oid, selection=invalid)
        assert app.engine.snapshot() == before
    described = app.registry.describe_operation("mesh.reconstruct_patch")
    assert described["side_effects"] and described["permissions"] == ["scene.write"]
    assert described["parameters"]["properties"]["refinement"]["maximum"] == 3


def test_patch_checkpoint_batch_timeout_and_export(app, monkeypatch):
    from scissors.core import patches

    oid, selected = panel(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "operations.batch",
            operations=[
                {
                    "operation": "mesh.reconstruct_patch",
                    "arguments": {"object_id": oid, "selection": selected},
                },
                {"operation": "mesh.flip", "arguments": {"object_id": oid, "selection": selected}},
            ],
        )
    assert app.engine.snapshot() == before

    def failed_commit():
        raise OSError("Checkpoint unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(app.engine, "commit", failed_commit)
        with pytest.raises(ScissorsError):
            app.call("mesh.reconstruct_patch", object_id=oid, selection=selected)
    assert app.engine.snapshot() == before
    ticks = iter([0, 31])
    with monkeypatch.context() as patch:
        patch.setattr(patches, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.reconstruct_patch", object_id=oid, selection=selected)
        assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before
    result = app.call("mesh.reconstruct_patch", object_id=oid, selection=selected)
    exported = app.call(
        "scene.export",
        scene_id=app.engine.object(oid)[0].id,
        output="exports/patch.glb",
        format="glb",
    )
    loaded = trimesh.load_scene(exported["path"])
    assert sum(len(mesh.faces) for mesh in loaded.geometry.values()) == result["face_count"]


def test_point_contact_with_unselected_surface_is_pinned(app):
    scene = app.scenes.create()
    cube = trimesh.creation.box()
    vertices = [*cube.vertices.tolist(), [-2, -0.5, -0.5], [-2, -1.5, -0.5]]
    faces = [*cube.faces.tolist(), [0, 8, 9]]
    obj = app.call("mesh.create", scene_id=scene.id, vertices=vertices, faces=faces)
    selected = {
        "object_id": obj["id"],
        "geometry_revision": obj["geometry_revision"],
        "face_indices": list(range(12)),
    }
    result = app.call(
        "mesh.reconstruct_patch",
        object_id=obj["id"],
        selection=selected,
        sharp_angle_degrees=180,
        iterations=1,
        strength=0.1,
    )
    edited = app.engine.object(obj["id"])[1]
    assert 0 in result["protected_vertex_indices"]
    np.testing.assert_array_equal(
        np.asarray(edited.vertices)[edited.faces[12]], np.asarray(vertices)[faces[12]]
    )
