"""Selected solid creases, finite scope and attribute preservation."""

from types import SimpleNamespace

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def selected_cube_edge(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.call("uv.generate", object_id=cube.id, method="box")
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=cube.id, material_id=material["id"])
    obj = app.engine.object(cube.id)[1]
    obj.vertex_colors = [[40, 80, 120, 255]] * len(obj.vertices)
    selection = app.call(
        "mesh.select_edges",
        object_id=cube.id,
        region={"bounds": [[0.5, 0.5, -0.5], [0.5, 0.5, 0.5]], "rule": "fully_inside"},
    )
    assert selection["count"] == 1
    return cube.id, selection


@pytest.mark.parametrize("operation", ["mesh.bevel", "mesh.chamfer", "mesh.fillet"])
def test_only_selected_cube_edge_is_treated(app, operation):
    oid, selected = selected_cube_edge(app)
    before = app.engine.object(oid)[1].model_dump()
    kwargs = {"radius": 0.1, "segments": 8} if operation == "mesh.fillet" else {"width": 0.1}
    result = app.call(operation, object_id=oid, selection=selected, **kwargs)
    obj = app.engine.object(oid)[1]
    assert result["method"] == "selected_edge_prisms" and result["watertight"]
    assert topology_mesh(obj.geometry()).is_winding_consistent
    assert obj.material_id == before["material_id"] and obj.transform == before["transform"]
    assert len(obj.uv) == len(obj.vertices) and len(obj.vertex_colors) == len(obj.vertices)
    np.testing.assert_allclose(result["bounds"], [[-0.5] * 3, [0.5] * 3])
    assert any(np.allclose(vertex, [-0.5, -0.5, -0.5]) for vertex in obj.vertices)
    assert not any(np.allclose(vertex, [0.5, 0.5, -0.5]) for vertex in obj.vertices)
    if operation == "mesh.fillet":
        assert 0.995 < result["volume"] < 1
    else:
        assert result["volume"] == pytest.approx(0.995)


def test_selection_validation_and_copy_source_policy(app):
    oid, selected = selected_cube_edge(app)
    before = app.engine.snapshot()
    for invalid in [
        dict(selected, geometry_revision="old"),
        dict(selected, edge_indices=[]),
        dict(selected, edge_indices=[99]),
        dict(selected, edge_indices=[0, 0]),
    ]:
        with pytest.raises(ScissorsError):
            app.call("mesh.chamfer", object_id=oid, selection=invalid)
        assert app.engine.snapshot() == before
    result = app.call(
        "mesh.chamfer",
        object_id=oid,
        selection=selected,
        width=0.1,
        replace=False,
        name="Local chamfer",
    )
    assert result["object_id"] != oid
    assert (
        app.engine.object(oid)[1].model_dump()
        == before["scenes"][result.get("scene_id", next(iter(before["scenes"])))]["objects"][oid]
    )


def test_concave_selected_edge_adds_only_local_fillet(app):
    scene = app.scenes.create()
    profile = app.call(
        "mesh.create_profile",
        scene_id=scene.id,
        points=[[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]],
    )
    solid = app.call("mesh.extrude", object_id=profile["id"], distance=1)
    selected = app.call(
        "mesh.select_edges",
        object_id=solid["id"],
        region={"bounds": [[0, 0, 0], [0, 0, 1]], "rule": "fully_inside"},
    )
    assert selected["count"] == 1
    result = app.call(
        "mesh.chamfer", object_id=solid["id"], selection=selected, width=0.1, mode="concave"
    )
    assert result["watertight"] and result["volume"] == pytest.approx(3.005)


def test_multiple_touching_edges_have_a_valid_local_junction(app):
    oid, _ = selected_cube_edge(app)
    selected = app.call(
        "mesh.select_edges",
        object_id=oid,
        region={"bounds": [[0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], "rule": "fully_inside"},
    )
    # Remove coplanar diagonals; the four perimeter edges meet at cube corners.
    obj = app.engine.object(oid)[1]
    normals = obj.geometry().face_normals
    positions = np.asarray(obj.vertices)
    crease_indices = []
    for index, edge in zip(selected["edge_indices"], selected["edges"], strict=True):
        a, b = positions[edge]
        if np.count_nonzero(a != b) == 1:
            crease_indices.append(index)
    selected["edge_indices"] = crease_indices
    assert len(crease_indices) == 4 and len(normals)
    result = app.call("mesh.bevel", object_id=oid, selection=selected, width=0.1)
    assert result["watertight"] and 0.97 < result["volume"] < 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_faces", 1),
        ("max_vertices", 1),
        ("max_memory_bytes", 1),
        ("max_geometry_pairs", 1),
        ("max_file_bytes", 1),
    ],
)
def test_selected_bevel_budgets_restore_state(app, field, value):
    oid, selected = selected_cube_edge(app)
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.chamfer", object_id=oid, selection=selected)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_selected_bevel_rejects_flat_diagonals_and_mode_mismatch(app):
    oid, selected = selected_cube_edge(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="mode"):
        app.call("mesh.chamfer", object_id=oid, selection=selected, mode="concave")
    all_edges = app.call("mesh.select_edges", object_id=oid)
    positions = np.asarray(app.engine.object(oid)[1].vertices)
    diagonal = next(
        index
        for index, edge in zip(all_edges["edge_indices"], all_edges["edges"], strict=True)
        if np.count_nonzero(positions[edge[0]] != positions[edge[1]]) == 2
    )
    all_edges["edge_indices"] = [diagonal]
    with pytest.raises(ScissorsError, match="coplanar"):
        app.call("mesh.chamfer", object_id=oid, selection=all_edges)
    assert app.engine.snapshot() == before


def test_selected_bevel_timeout_and_checkpoint_restore_state(app, monkeypatch):
    from scissors.core import local_bevel

    oid, selected = selected_cube_edge(app)
    before = app.engine.snapshot()
    ticks = iter([0, 31])
    with monkeypatch.context() as patch:
        patch.setattr(local_bevel, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.chamfer", object_id=oid, selection=selected)
        assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before

    def failed_commit():
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(app.engine, "commit", failed_commit)
    with pytest.raises(ScissorsError):
        app.call("mesh.chamfer", object_id=oid, selection=selected)
    assert app.engine.snapshot() == before


def test_selected_round_bevel_uses_rotated_world_geometry(app):
    oid, selected = selected_cube_edge(app)
    endpoints = selected["edges"][0]
    app.call("mesh.rotate", object_id=oid, axis="z", angle_degrees=25)
    positions = np.asarray(app.engine.object(oid)[1].vertices)
    wanted = {tuple(point) for point in positions[endpoints]}
    all_edges = app.call("mesh.select_edges", object_id=oid)
    index = next(
        i
        for i, edge in zip(all_edges["edge_indices"], all_edges["edges"], strict=True)
        if {tuple(point) for point in positions[edge]} == wanted
    )
    all_edges["edge_indices"] = [index]
    result = app.call("mesh.bevel", object_id=oid, selection=all_edges, width=0.1, segments=4)
    assert result["watertight"] and 0.995 < result["volume"] < 1
