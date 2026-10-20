"""Automatic winding repair constraints, scope and attribute invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def cube(app, seam=False):
    scene = app.scenes.create()
    obj = app.meshes.create_cube(scene_id=scene.id)
    if seam:
        app.call("mesh.facet", object_id=obj.id)
    app.call("uv.planar", object_id=obj.id)
    material = app.call("material.create", scene_id=scene.id)
    app.call("material.assign", object_id=obj.id, material_id=material["id"])
    app.engine.object(obj.id)[1].vertex_colors = [[40, 80, 120, 255]] * len(
        app.engine.object(obj.id)[1].vertices
    )
    return obj.id


def selection(app, oid, faces):
    return {
        "object_id": oid,
        "geometry_revision": app.engine.object(oid)[1].geometry_revision,
        "face_indices": faces,
    }


@pytest.mark.parametrize("seam", [False, True])
def test_single_inverted_face_is_repaired_and_attributes_remain_exact(app, seam):
    oid = cube(app, seam)
    correct = app.engine.object(oid)[1].model_dump()
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    result = app.call("mesh.repair_winding", object_id=oid)
    obj = app.engine.object(oid)[1]
    assert result["flipped_face_indices"] == [0]
    assert result["scope_conflicts_before"] == 3 and result["scope_conflicts_after"] == 0
    assert topology_mesh(obj.geometry()).is_winding_consistent and result[
        "volume"
    ] == pytest.approx(1)
    for key in ("vertices", "faces", "uv", "vertex_colors", "material_id", "transform", "pivot"):
        assert getattr(obj, key) == correct[key]


def test_selection_is_anchored_to_unchanged_outside_faces(app):
    oid = cube(app)
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    before = app.engine.object(oid)[1].model_dump()
    result = app.call("mesh.repair_winding", object_id=oid, selection=selection(app, oid, [0]))
    assert result["regions"][0]["anchored"]
    assert result["flipped_face_indices"] == [0]
    assert app.engine.object(oid)[1].faces[1:] == before["faces"][1:]


def test_too_small_selection_reports_conflicting_fixed_faces(app):
    oid = cube(app)
    top = app.call(
        "mesh.select_faces",
        object_id=oid,
        normal={"direction": [0, 0, 1], "angular_tolerance_degrees": 1},
    )
    result = app.call("mesh.flip", object_id=oid, selection=top)
    too_small = dict(result["selection"], face_indices=top["face_indices"][:1])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="fixed outside faces") as error:
        app.call("mesh.repair_winding", object_id=oid, selection=too_small)
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before
    repaired = app.call("mesh.repair_winding", object_id=oid, selection=result["selection"])
    assert repaired["scope_conflicts_after"] == 0 and repaired["flipped_face_count"] == 2


def test_preserve_keeps_inward_component_and_absolute_modes_are_explicit(app):
    oid = cube(app)
    app.call("mesh.flip", object_id=oid)
    before = app.engine.snapshot()
    result = app.call("mesh.repair_winding", object_id=oid)
    assert not result["changed"] and result["volume"] == pytest.approx(-1)
    assert app.engine.snapshot() == before
    result = app.call("mesh.repair_winding", object_id=oid, orientation="outward")
    assert result["flipped_face_count"] == 12 and result["volume"] == pytest.approx(1)
    result = app.call("mesh.repair_winding", object_id=oid, orientation="inward")
    assert result["flipped_face_count"] == 12 and result["volume"] == pytest.approx(-1)


def test_component_selection_leaves_other_component_conflicts_unmodified(app):
    scene = app.scenes.create()
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([first, second])
    obj = app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )
    oid = obj["id"]
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0, 12]))
    selected = app.call("mesh.select_components", object_id=oid, seed_face_indices=[0])
    app.call("mesh.selection_save", name="Component", selection=selected)
    selected = app.call("mesh.selection_get", object_id=oid, name="Component")
    before = app.engine.object(oid)[1].model_dump()
    result = app.call("mesh.repair_winding", object_id=oid, selection=selected)
    assert result["flipped_face_indices"] == [0]
    assert result["scope_conflicts_after"] == 0 and result["global_conflicts_after"] == 3
    assert app.engine.object(oid)[1].faces[12:] == before["faces"][12:]
    with pytest.raises(ScissorsError, match="stale"):
        app.call("mesh.selection_get", object_id=oid, name="Component")


def test_closed_component_orientation_does_not_reorient_other_components(app):
    scene = app.scenes.create()
    a, b = trimesh.creation.box(), trimesh.creation.box()
    b.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([a, b])
    obj = app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )
    selected = app.call("mesh.select_components", object_id=obj["id"], seed_face_indices=[0])
    before = app.engine.object(obj["id"])[1].model_dump()
    result = app.call(
        "mesh.repair_winding", object_id=obj["id"], selection=selected, orientation="inward"
    )
    assert result["flipped_face_indices"] == list(range(12))
    assert app.engine.object(obj["id"])[1].faces[12:] == before["faces"][12:]


def test_open_patch_repair_uses_majority_and_rejects_absolute_orientation(app):
    scene = app.scenes.create()
    # Three triangles in a strip; only the middle triangle starts inverted.
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]],
        faces=[[0, 1, 2], [1, 2, 3], [1, 4, 3]],
    )
    result = app.call("mesh.repair_winding", object_id=obj["id"])
    assert result["flipped_face_indices"] == [1]
    assert not result["regions"][0]["closed"]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="complete closed"):
        app.call("mesh.repair_winding", object_id=obj["id"], orientation="outward")
    assert app.engine.snapshot() == before


def test_dry_run_reports_actual_and_planned_conflicts_without_mutation(app):
    oid = cube(app)
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    before = app.engine.snapshot()
    plan = app.call("mesh.repair_winding", object_id=oid, dry_run=True)
    assert not plan["changed"] and plan["flipped_face_count"] == 0
    assert plan["planned_flip_face_indices"] == [0]
    assert plan["scope_conflicts_after"] == 3 and plan["planned_scope_conflicts_after"] == 0
    assert app.engine.snapshot() == before
    result = app.call("mesh.repair_winding", object_id=oid)
    assert result["flipped_face_indices"] == plan["planned_flip_face_indices"]


def test_nonorientable_strip_reports_constraint_contradiction(app):
    scene = app.scenes.create()
    n = 9
    vertices = []
    for angle in np.arange(n) * 2 * np.pi / n:
        for t in [-0.2, 0.2]:
            radius = 2 + t * np.cos(angle / 2)
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), t * np.sin(angle / 2)])
    faces = []
    for i in range(n):
        a, d = 2 * i, 2 * i + 1
        b, c = (2 * (i + 1), 2 * (i + 1) + 1) if i < n - 1 else (1, 0)
        faces.extend([[a, b, c], [a, c, d]])
    obj = app.call("mesh.create", scene_id=scene.id, vertices=vertices, faces=faces)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="contradictory winding") as error:
        app.call("mesh.repair_winding", object_id=obj["id"])
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


def test_repaired_workspace_restart_and_unused_vertices_survive(app):
    oid = cube(app)
    obj = app.engine.object(oid)[1]
    scene = app.engine.object(oid)[0]
    copied = app.call(
        "mesh.create", scene_id=scene.id, vertices=[*obj.vertices, [20, 20, 20]], faces=obj.faces
    )
    oid = copied["id"]
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    result = app.call("mesh.repair_winding", object_id=oid)
    saved = app.engine.object(oid)[1].model_dump()
    assert saved["vertices"][-1] == [20, 20, 20] and result["vertex_count"] == 9
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.object(oid)[1].model_dump() == saved


def test_preserve_repairs_cavity_wall_without_filling_the_cavity(app):
    scene = app.scenes.create()
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box()
    inner.invert()
    mesh = trimesh.util.concatenate([outer, inner])
    obj = app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )
    app.call("mesh.flip", object_id=obj["id"], selection=selection(app, obj["id"], [12]))
    result = app.call("mesh.repair_winding", object_id=obj["id"])
    assert result["flipped_face_indices"] == [12]
    assert result["volume"] == pytest.approx(7)
    assert topology_mesh(app.engine.object(obj["id"])[1].geometry()).is_winding_consistent


def test_invalid_selection_and_absolute_partial_scope_are_atomic(app):
    oid = cube(app)
    valid = selection(app, oid, [0])
    before = app.engine.snapshot()
    for invalid in [
        dict(valid, geometry_revision="old"),
        dict(valid, object_id="other"),
        dict(valid, face_indices=[]),
        dict(valid, face_indices=[0, 0]),
        dict(valid, face_indices=[99]),
    ]:
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.repair_winding", object_id=oid, selection=invalid)
        assert error.value.code == "INVALID_PARAMETER"
        assert app.engine.snapshot() == before
    with pytest.raises(ScissorsError, match="complete closed"):
        app.call("mesh.repair_winding", object_id=oid, selection=valid, orientation="outward")
    assert app.engine.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_memory_bytes", 1),
        ("max_file_bytes", 1),
        ("max_geometry_pairs", 1),
        ("max_vertices", 1),
        ("max_faces", 1),
    ],
)
def test_winding_repair_resource_errors_restore_state(app, field, value):
    oid = cube(app)
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.repair_winding", object_id=oid)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_winding_timeout_checkpoint_and_batch_failure_restore_state(app, monkeypatch):
    from scissors.core import winding

    oid = cube(app)
    app.call("mesh.flip", object_id=oid, selection=selection(app, oid, [0]))
    stale_after_edit = selection(app, oid, [0])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call(
            "operations.batch",
            operations=[
                {"operation": "mesh.repair_winding", "arguments": {"object_id": oid}},
                {
                    "operation": "mesh.repair_winding",
                    "arguments": {"object_id": oid, "selection": stale_after_edit},
                },
            ],
        )
    assert error.value.code == "BATCH_FAILED" and app.engine.snapshot() == before

    def fail_commit():
        raise OSError("Checkpoint unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(app.engine, "commit", fail_commit)
        with pytest.raises(ScissorsError):
            app.call("mesh.repair_winding", object_id=oid)
        assert app.engine.snapshot() == before
    ticks = iter([0, 31])
    monkeypatch.setattr(winding, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.repair_winding", object_id=oid)
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before


def test_winding_repair_schema_and_explicit_flip_contract(app):
    description = app.registry.describe_operation("mesh.repair_winding")
    assert description["side_effects"] and description["permissions"] == ["scene.write"]
    assert description["parameters"]["properties"]["orientation"]["enum"] == [
        "preserve",
        "outward",
        "inward",
    ]
    oid = cube(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.repair_winding", object_id=oid, orientation="guess")
    assert error.value.code == "INVALID_PARAMETER" and app.engine.snapshot() == before
    explicit = app.registry.describe_operation("mesh.flip")
    assert "selection" in explicit["parameters"]["properties"]


def test_equal_flip_counts_keep_lowest_face_unchanged(app):
    scene = app.scenes.create()
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[2, 1, 0], [0, 2, 3]],
    )
    result = app.call("mesh.repair_winding", object_id=obj["id"])
    assert result["flipped_face_indices"] == [1]
    assert app.engine.object(obj["id"])[1].faces[0] == [2, 1, 0]


def test_absolute_orientation_rejects_zero_volume_closed_component(app):
    scene = app.scenes.create()
    # A tetrahedral surface folded into a plane has valid individual triangles
    # and manifold edges, but no volume sign that can define its exterior.
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    )
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="ambiguous") as error:
        app.call("mesh.repair_winding", object_id=obj["id"], orientation="outward")
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before
