"""Local face editing: geometry, attributes, public selections and rollback."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def cube_top(app, facet=False):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    if facet:
        app.call("mesh.facet", object_id=cube.id)
    app.call("uv.planar", object_id=cube.id)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=cube.id, material_id=material["id"])
    obj = app.engine.object(cube.id)[1]
    obj.vertex_colors = [[i % 256, 40, 80, 255] for i in range(len(obj.vertices))]
    selected = app.call(
        "mesh.select_faces",
        object_id=cube.id,
        normal={"direction": [0, 0, 1], "angular_tolerance_degrees": 1},
    )
    assert len(selected["face_indices"]) == 2
    return obj, selected


def assert_attributes(obj, before, origins):
    np.testing.assert_array_equal(obj.vertex_colors, np.asarray(before["vertex_colors"])[origins])
    assert obj.material_id == before["material_id"]


@pytest.mark.parametrize("facet", [False, True])
@pytest.mark.parametrize("detach", [False, True])
def test_move_faces_shared_or_detached_and_preserve_seams(app, facet, detach):
    obj, selected = cube_top(app, facet)
    before = obj.model_dump()
    result = app.call(
        "mesh.move_faces", object_id=obj.id, selection=selected, offset=[0, 0, 1], detach=detach
    )
    edited = app.engine.object(obj.id)[1]
    triangles = np.asarray(edited.vertices)[edited.faces]
    original = np.asarray(before["vertices"])[before["faces"]]
    chosen = selected["face_indices"]
    np.testing.assert_allclose(triangles[chosen], original[chosen] + [0, 0, 1])
    if detach:
        keep = [i for i in range(len(original)) if i not in chosen]
        np.testing.assert_array_equal(triangles[keep], original[keep])
        assert not result["watertight"]
    else:
        assert result["watertight"] and result["volume"] == pytest.approx(2)
    origins = result["new_to_old_vertices"]
    assert_attributes(edited, before, origins)
    np.testing.assert_array_equal(edited.uv, np.asarray(before["uv"])[origins])


def test_delete_faces_compacts_and_explicit_empty_policy(app):
    obj, selected = cube_top(app, True)
    before = obj.model_dump()
    result = app.call("mesh.delete_faces", object_id=obj.id, selection=selected)
    edited = app.engine.object(obj.id)[1]
    assert len(edited.faces) == 10 and len(edited.vertices) == 30
    assert_attributes(edited, before, result["new_to_old_vertices"])
    assert set(result["new_to_old_faces"]) == set(range(12)) - set(selected["face_indices"])
    all_faces = app.call("mesh.select_faces", object_id=obj.id)
    snapshot = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="empty_policy"):
        app.call("mesh.delete_faces", object_id=obj.id, selection=all_faces)
    assert app.engine.snapshot() == snapshot
    result = app.call(
        "mesh.delete_faces", object_id=obj.id, selection=all_faces, empty_policy="delete_object"
    )
    assert result["source_deleted"]
    with pytest.raises(ScissorsError) as error:
        app.engine.object(obj.id)
    assert error.value.code == "OBJECT_NOT_FOUND"


@pytest.mark.parametrize("copy", [False, True])
def test_separate_preserves_world_geometry_and_source_policy(app, copy):
    obj, selected = cube_top(app)
    app.call("mesh.translate", object_id=obj.id, x=3)
    selected = app.call(
        "mesh.select_faces",
        object_id=obj.id,
        normal={"direction": [0, 0, 1], "angular_tolerance_degrees": 1},
    )
    before = app.engine.object(obj.id)[1].model_dump()
    result = app.call(
        "mesh.separate_faces", object_id=obj.id, selection=selected, name="Panel", keep_source=copy
    )
    extracted = app.engine.object(result["object_id"])[1]
    assert extracted.id != obj.id and extracted.name == "Panel"
    assert len(extracted.faces) == 2
    np.testing.assert_array_equal(extracted.transform, np.eye(4))
    np.testing.assert_array_equal(
        np.asarray(extracted.vertices)[extracted.faces],
        np.asarray(before["vertices"])[np.asarray(before["faces"])[selected["face_indices"]]],
    )
    assert_attributes(extracted, before, result["new_to_old_vertices"])
    remaining = app.engine.object(obj.id)[1]
    if copy:
        assert remaining.model_dump() == before
    else:
        assert len(remaining.faces) == 10 and not result["source_deleted"]
        assert len(result["source"]["new_to_old_faces"]) == 10


def test_separate_all_deletes_source_and_reparents_children(app):
    obj, _ = cube_top(app)
    scene = app.engine.object(obj.id)[0]
    child = app.meshes.create_cube(scene_id=scene.id)
    app.engine.object(child.id)[1].parent_id = obj.id
    result = app.call(
        "mesh.separate_faces",
        object_id=obj.id,
        selection=app.call("mesh.select_faces", object_id=obj.id),
    )
    assert result["source_deleted"] and result["watertight"]
    assert obj.id not in scene.objects
    assert app.engine.object(child.id)[1].parent_id is None


@pytest.mark.parametrize("facet", [False, True])
def test_extrude_cube_top_closes_walls_and_preserves_old_faces(app, facet):
    obj, selected = cube_top(app, facet)
    before = obj.model_dump()
    result = app.call("mesh.extrude_faces", object_id=obj.id, selection=selected, offset=[0, 0, 1])
    edited = app.engine.object(obj.id)[1]
    topo = topology_mesh(edited.geometry())
    assert topo.is_watertight and topo.is_winding_consistent
    assert result["volume"] == pytest.approx(2)
    assert result["side_face_count"] == 8 and result["face_count"] == 20
    np.testing.assert_allclose(result["bounds"], [[-0.5, -0.5, -0.5], [0.5, 0.5, 1.5]])
    assert_attributes(edited, before, result["new_to_old_vertices"])
    for i, old in enumerate(before["faces"]):
        points = np.asarray(edited.vertices)[edited.faces[i]]
        expected = np.asarray(before["vertices"])[old]
        if i in selected["face_indices"]:
            expected = expected + [0, 0, 1]
        np.testing.assert_array_equal(points, expected)
        np.testing.assert_array_equal(
            np.asarray(edited.uv)[edited.faces[i]], np.asarray(before["uv"])[old]
        )
    assert result["selection"]["face_indices"] == selected["face_indices"]
    assert result["new_to_old_faces"][-8:] == [-1] * 8


def test_flip_selection_preserves_attributes_and_global_flip_still_works(app):
    obj, selected = cube_top(app, True)
    before = obj.model_dump()
    result = app.call("mesh.flip", object_id=obj.id, selection=selected)
    edited = app.engine.object(obj.id)[1]
    for i, face in enumerate(before["faces"]):
        assert edited.faces[i] == (face[::-1] if i in selected["face_indices"] else face)
    assert edited.uv == before["uv"] and edited.vertex_colors == before["vertex_colors"]
    assert not topology_mesh(edited.geometry()).is_winding_consistent
    app.call("mesh.flip", object_id=obj.id, selection=result["selection"])
    assert topology_mesh(app.engine.object(obj.id)[1].geometry()).is_winding_consistent
    full = app.call("mesh.flip", object_id=obj.id)
    assert full["volume"] == pytest.approx(-1)


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("mesh.move_faces", {"offset": [0, 0, 1]}),
        ("mesh.delete_faces", {}),
        ("mesh.separate_faces", {}),
        ("mesh.extrude_faces", {"offset": [0, 0, 1]}),
        ("mesh.flip", {}),
    ],
)
def test_invalid_selections_fail_without_changing_state(app, operation, kwargs):
    obj, selected = cube_top(app)
    before = app.engine.snapshot()
    for invalid in [
        dict(selected, geometry_revision="old"),
        dict(selected, face_indices=[]),
        dict(selected, face_indices=[0, 0]),
        dict(selected, face_indices=[100]),
        dict(selected, object_id="other"),
    ]:
        with pytest.raises(ScissorsError) as error:
            app.call(operation, object_id=obj.id, selection=invalid, **kwargs)
        assert error.value.code == "INVALID_PARAMETER"
        assert app.engine.snapshot() == before
    description = app.registry.describe_operation(operation)
    assert description["side_effects"] and description["permissions"] == ["scene.write"]


def test_saved_selection_chaining_persistence_and_export_reopening(app):
    obj, selected = cube_top(app)
    app.call("mesh.selection_save", name="Top", selection=selected)
    selection = app.call("mesh.selection_get", object_id=obj.id, name="Top")
    result = app.call("mesh.extrude_faces", object_id=obj.id, selection=selection, offset=[0, 0, 1])
    with pytest.raises(ScissorsError, match="stale"):
        app.call("mesh.selection_get", object_id=obj.id, name="Top")
    result = app.call(
        "mesh.move_faces",
        object_id=obj.id,
        selection=result["selection"],
        offset=[0, 0, 0.5],
        detach=False,
    )
    assert result["volume"] == pytest.approx(2.5)
    scene = app.engine.object(obj.id)[0]
    exported = app.call(
        "scene.export", scene_id=scene.id, output="exports/extruded.glb", format="glb"
    )
    loaded = trimesh.load_scene(Path(exported["path"]))
    assert sum(len(mesh.faces) for mesh in loaded.geometry.values()) == result["face_count"]
    saved = app.engine.object(obj.id)[1].model_dump()
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.object(obj.id)[1].model_dump() == saved


def test_extrude_region_with_hole_keeps_both_contours_and_no_internal_walls(app):
    scene = app.scenes.create()
    outer = [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]
    inner = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, j + 4], [i, j + 4, i + 4]])
    obj = app.call("mesh.create", scene_id=scene.id, vertices=outer + inner, faces=faces)
    selected = app.call("mesh.select_faces", object_id=obj["id"])
    result = app.call(
        "mesh.extrude_faces", object_id=obj["id"], selection=selected, offset=[0, 0, 1]
    )
    assert result["side_face_count"] == 16 and result["face_count"] == 24
    boundaries = app.call("mesh.boundaries", object_id=obj["id"])
    assert len(boundaries["loops"]) == 2
    edited = app.engine.object(obj["id"])[1]
    for loop in boundaries["loops"]:
        assert all(edited.vertices[i][2] == 0 for i in loop["vertex_indices"])
    assert topology_mesh(edited.geometry()).is_winding_consistent


def test_disconnected_patches_extrude_and_closed_component_is_rejected(app):
    scene = app.scenes.create()
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
        faces=[[0, 1, 2], [3, 4, 5]],
    )
    selected = app.call("mesh.select_components", object_id=obj["id"])
    result = app.call(
        "mesh.extrude_faces", object_id=obj["id"], selection=selected, offset=[0, 0, 1]
    )
    assert result["side_face_count"] == 12
    assert len(app.call("mesh.select_components", object_id=obj["id"])["components"]) == 2
    cube, _ = cube_top(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="open boundary"):
        app.call(
            "mesh.extrude_faces",
            object_id=cube.id,
            selection=app.call("mesh.select_faces", object_id=cube.id),
            offset=[0, 0, 1],
        )
    assert app.engine.snapshot() == before


@pytest.mark.parametrize(
    "operation,offset",
    [
        ("mesh.move_faces", [0, 0, 0]),
        ("mesh.extrude_faces", [0, 0, 0]),
        ("mesh.extrude_faces", [1, 0, 0]),
    ],
)
def test_zero_and_degenerate_offsets_restore_scene(app, operation, offset):
    obj, selected = cube_top(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(operation, object_id=obj.id, selection=selected, offset=offset)
    assert app.engine.snapshot() == before


@pytest.mark.parametrize(
    "limit,value",
    [("max_vertices", 9), ("max_faces", 12), ("max_memory_bytes", 1), ("max_file_bytes", 1)],
)
def test_edit_budgets_restore_geometry_and_selection_metadata(app, limit, value):
    obj, selected = cube_top(app)
    app.call("mesh.selection_save", name="Top", selection=selected)
    before = app.engine.snapshot()
    setattr(app.engine.settings, limit, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.extrude_faces", object_id=obj.id, selection=selected, offset=[0, 0, 1])
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_face_edit_checkpoint_failure_and_atomic_batch_restore_state(app, monkeypatch):
    obj, selected = cube_top(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "operations.batch",
            operations=[
                {
                    "operation": "mesh.delete_faces",
                    "arguments": {"object_id": obj.id, "selection": selected},
                },
                {
                    "operation": "mesh.flip",
                    "arguments": {"object_id": obj.id, "selection": selected},
                },
            ],
        )
    assert app.engine.snapshot() == before

    def failed_commit():
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(app.engine, "commit", failed_commit)
    with pytest.raises(ScissorsError):
        app.call("mesh.separate_faces", object_id=obj.id, selection=selected)
    assert app.engine.snapshot() == before


def test_face_edit_timeout_does_not_publish_meshes(app, monkeypatch):
    from scissors.core import face_editing

    obj, selected = cube_top(app)
    before = app.engine.snapshot()
    ticks = iter([0, 31])
    monkeypatch.setattr(face_editing, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.move_faces", object_id=obj.id, selection=selected, offset=[0, 0, 1])
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before
