"""Targeted hole filling, measured smooth joins and state invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def open_cube(app, facet=False):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    faces = app.call("mesh.select_faces", object_id=cube.id)
    obj = app.engine.object(cube.id)[1]
    sides = [i for i, normal in enumerate(obj.geometry().face_normals) if abs(normal[2]) < 0.1]
    faces["face_indices"] = sorted(set(range(len(obj.faces))) - set(sides))
    app.call("mesh.delete_faces", object_id=cube.id, selection=faces)
    if facet:
        app.call("mesh.facet", object_id=cube.id)
    app.call("uv.planar", object_id=cube.id)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=cube.id, material_id=material["id"])
    obj = app.engine.object(cube.id)[1]
    obj.vertex_colors = [[10, 30, 90, 255] for _ in obj.vertices]
    return cube.id


def curved_panel(app):
    scene = app.scenes.create()
    n = 9
    coordinates = np.linspace(-2, 2, n)
    vertices = [[x, y, 0.2 * (x * x + y * y)] for y in coordinates for x in coordinates]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            if i in (3, 4) and j in (3, 4):
                continue
            a = j * n + i
            faces.extend([[a, a + 1, a + n + 1], [a, a + n + 1, a + n]])
    obj = app.call("mesh.create", scene_id=scene.id, vertices=vertices, faces=faces)
    app.call("uv.planar", object_id=obj["id"])
    app.engine.object(obj["id"])[1].vertex_colors = [[20, 60, 120, 255]] * len(vertices)
    selected = app.call(
        "mesh.select_boundaries",
        object_id=obj["id"],
        region={"bounds": [[-0.6, -0.6, -1], [0.6, 0.6, 1]], "rule": "fully_inside"},
    )
    assert len(selected["loop_indices"]) == 1
    return obj["id"], selected


@pytest.mark.parametrize("facet", [False, True])
def test_only_selected_hole_is_closed_and_original_attributes_are_exact(app, facet):
    oid = open_cube(app, facet)
    before = app.engine.object(oid)[1].model_dump()
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    other = app.call("mesh.boundaries", object_id=oid)["loops"][1]["vertex_indices"]
    result = app.call("mesh.fill_holes", object_id=oid, selection=selected)
    edited = app.engine.object(oid)[1]
    assert result["filled_holes"] == 1 and result["filled_loop_indices"] == [0]
    assert len(result["patch_face_indices"]) == 2
    remaining = app.call("mesh.boundaries", object_id=oid)["loops"]
    assert len(remaining) == 1 and set(remaining[0]["vertex_indices"]) == set(other)
    assert (
        edited.vertices == before["vertices"]
        and edited.faces[: len(before["faces"])] == before["faces"]
    )
    assert edited.uv == before["uv"] and edited.vertex_colors == before["vertex_colors"]
    assert edited.material_id == before["material_id"] and edited.transform == before["transform"]
    assert topology_mesh(edited.geometry()).is_winding_consistent


def test_automatic_fill_and_no_holes_are_supported(app):
    oid = open_cube(app)
    result = app.call("mesh.fill_holes", object_id=oid, max_edges=4)
    assert result["filled_holes"] == 2 and result["watertight"]
    before = app.engine.snapshot()
    result = app.call("mesh.fill_holes", object_id=oid)
    assert result["filled_holes"] == 0 and result["patch_face_indices"] == []
    assert app.engine.snapshot() == before


def test_selected_edge_limit_is_explicit_and_old_automatic_limit_skips(app):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    before = app.engine.snapshot()
    assert app.call("mesh.fill_holes", object_id=oid, max_edges=3)["filled_holes"] == 0
    with pytest.raises(ScissorsError, match="max_edges"):
        app.call("mesh.fill_holes", object_id=oid, selection=selected, max_edges=3)
    assert app.engine.snapshot() == before


def test_smooth_patch_improves_join_and_preserves_source_geometry(app):
    oid, selected = curved_panel(app)
    before = app.engine.snapshot()
    projected = app.call("mesh.fill_holes", object_id=oid, selection=selected)
    projected_angle = projected["patches"][0]["max_join_angle_degrees"]
    app.engine.restore(before)
    original = app.engine.object(oid)[1].model_dump()
    result = app.call(
        "mesh.fill_holes",
        object_id=oid,
        selection=selected,
        method="smooth",
        tangent_tolerance_degrees=20,
    )
    edited = app.engine.object(oid)[1]
    measured = result["patches"][0]["max_join_angle_degrees"]
    assert measured < projected_angle and measured <= 20
    topology = topology_mesh(edited.geometry())
    adjacency = topology.face_adjacency
    seams = adjacency[
        (adjacency[:, 0] < len(original["faces"])) & (adjacency[:, 1] >= len(original["faces"]))
    ]
    normals = topology.face_normals
    actual_angles = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", normals[seams[:, 0]], normals[seams[:, 1]]), -1, 1))
    )
    assert len(actual_angles) == 8 and actual_angles.max() == pytest.approx(measured)
    assert len(result["patch_face_indices"]) > 6
    np.testing.assert_array_equal(
        edited.vertices[: len(original["vertices"])], original["vertices"]
    )
    assert edited.faces[: len(original["faces"])] == original["faces"]
    assert edited.uv[: len(original["uv"])] == original["uv"]
    assert edited.vertex_colors[: len(original["vertex_colors"])] == original["vertex_colors"]
    assert all(color == [20, 60, 120, 255] for color in edited.vertex_colors)
    assert len(app.call("mesh.boundaries", object_id=oid)["loops"]) == 1
    assert topology_mesh(edited.geometry()).is_winding_consistent


def test_smooth_tolerance_failure_is_measured_and_atomic(app):
    oid, selected = curved_panel(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="tangent tolerance") as error:
        app.call(
            "mesh.fill_holes",
            object_id=oid,
            selection=selected,
            method="smooth",
            tangent_tolerance_degrees=0.01,
        )
    assert error.value.code == "GEOMETRY_ERROR"
    assert error.value.details["max_join_angle_degrees"] > 0.01
    assert app.engine.snapshot() == before


def test_corner_with_vertical_collar_rejects_smooth_instead_of_flat_fallback(app):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="height chart"):
        app.call("mesh.fill_holes", object_id=oid, selection=selected, method="smooth")
    assert app.engine.snapshot() == before


def test_bad_boundary_selections_are_rejected_without_changing_state(app):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    before = app.engine.snapshot()
    for invalid in [
        dict(selected, geometry_revision="old"),
        dict(selected, object_id="other"),
        dict(selected, loop_indices=[]),
        dict(selected, loop_indices=[0, 0]),
        dict(selected, loop_indices=[99]),
    ]:
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.fill_holes", object_id=oid, selection=invalid)
        assert error.value.code == "INVALID_PARAMETER"
        assert app.engine.snapshot() == before


def test_named_boundaries_patch_selection_and_restart(app):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    app.call("mesh.selection_save", name="Roof", selection=selected)
    selected = app.call("mesh.selection_get", object_id=oid, name="Roof")
    result = app.call("mesh.fill_holes", object_id=oid, selection=selected)
    app.call("mesh.selection_save", name="Patch", selection=result["selection"])
    with pytest.raises(ScissorsError, match="stale"):
        app.call("mesh.selection_get", object_id=oid, name="Roof")
    saved = app.engine.object(oid)[1].model_dump()
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.object(oid)[1].model_dump() == saved
        assert (
            reopened.call("mesh.selection_get", object_id=oid, name="Patch")["face_indices"]
            == result["patch_face_indices"]
        )


@pytest.mark.parametrize(
    "field,value",
    [("max_faces", 8), ("max_memory_bytes", 1), ("max_file_bytes", 1), ("max_geometry_pairs", 1)],
)
def test_hole_filling_budgets_restore_state(app, field, value):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.fill_holes", object_id=oid, selection=selected)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_smooth_patch_vertex_budget_and_timeout(app, monkeypatch):
    from scissors.core import hole_filling

    oid, selected = curved_panel(app)
    before = app.engine.snapshot()
    app.engine.settings.max_vertices = len(app.engine.object(oid)[1].vertices)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.fill_holes", object_id=oid, selection=selected, method="smooth")
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before
    app.engine.settings.max_vertices = 200_000
    ticks = iter([0, 31])
    monkeypatch.setattr(hole_filling, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.fill_holes", object_id=oid, selection=selected)
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before


def test_hole_filling_checkpoint_and_atomic_batch_rollback(app, monkeypatch):
    oid = open_cube(app)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[0])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call(
            "operations.batch",
            operations=[
                {
                    "operation": "mesh.fill_holes",
                    "arguments": {"object_id": oid, "selection": selected},
                },
                {
                    "operation": "mesh.fill_holes",
                    "arguments": {"object_id": oid, "selection": selected},
                },
            ],
        )
    assert error.value.code == "BATCH_FAILED"
    assert app.engine.snapshot() == before

    def fail_commit():
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(app.engine, "commit", fail_commit)
    with pytest.raises(ScissorsError):
        app.call("mesh.fill_holes", object_id=oid, selection=selected)
    assert app.engine.snapshot() == before


def test_hole_filling_schema_and_permissions(app):
    description = app.registry.describe_operation("mesh.fill_holes")
    assert description["side_effects"] and description["permissions"] == ["scene.write"]
    properties = description["parameters"]["properties"]
    assert properties["method"]["enum"] == ["projected", "smooth"]
    assert properties["refinement"]["maximum"] == 4
    oid = open_cube(app)
    before = app.engine.snapshot()
    for kwargs in [dict(method="fake"), dict(refinement=0), dict(tangent_tolerance_degrees=0)]:
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.fill_holes", object_id=oid, **kwargs)
        assert error.value.code == "INVALID_PARAMETER"
    assert app.engine.snapshot() == before


def test_smooth_patch_fits_exact_output_budgets_and_reopens_glb(app):
    oid, selected = curved_panel(app)
    before = app.engine.snapshot()
    result = app.call("mesh.fill_holes", object_id=oid, selection=selected, method="smooth")
    app.engine.restore(before)
    app.engine.settings.max_vertices = result["vertex_count"]
    app.engine.settings.max_faces = result["face_count"]
    result = app.call("mesh.fill_holes", object_id=oid, selection=selected, method="smooth")
    scene = app.engine.object(oid)[0]
    exported = app.call(
        "scene.export", scene_id=scene.id, output="exports/patched.glb", format="glb"
    )
    reopened = trimesh.load_scene(exported["path"])
    assert sum(len(mesh.faces) for mesh in reopened.geometry.values()) == result["face_count"]
    assert all(mesh.visual.uv is not None for mesh in reopened.geometry.values())


def test_smooth_patch_uses_world_geometry_after_rotation(app):
    oid, _ = curved_panel(app)
    app.call("mesh.rotate", object_id=oid, axis="x", angle_degrees=47)
    app.call("mesh.rotate", object_id=oid, axis="z", angle_degrees=23)
    loops = app.call("mesh.boundaries", object_id=oid)
    loop_index = next(loop["loop_index"] for loop in loops["loops"] if loop["edge_count"] == 8)
    selected = app.call("mesh.select_boundaries", object_id=oid, loop_indices=[loop_index])
    before = app.engine.object(oid)[1].model_dump()
    result = app.call("mesh.fill_holes", object_id=oid, selection=selected, method="smooth")
    edited = app.engine.object(oid)[1]
    assert result["patches"][0]["max_join_angle_degrees"] <= 20
    assert edited.transform == before["transform"]
    np.testing.assert_array_equal(edited.vertices[: len(before["vertices"])], before["vertices"])
