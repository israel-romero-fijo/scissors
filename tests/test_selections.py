"""Geometry selection semantics, named storage and registry invariants."""

from types import SimpleNamespace

import numpy as np
import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


def square(app):
    scene = app.scenes.create()
    return app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
    )["object_id"]


def box(low, high, rule="fully_inside"):
    return {"bounds": [low, high], "rule": rule}


def test_render_vertices_and_index_box_intersection(app):
    oid = square(app)
    app.call("mesh.facet", object_id=oid)
    before = app.engine.snapshot()
    all_vertices = app.call("mesh.select_vertices", object_id=oid)
    assert all_vertices["count"] == 6
    result = app.call(
        "mesh.select_vertices", object_id=oid, region=box([0, 0, 0], [0, 0, 0]), tolerance=0
    )
    assert len(result["vertex_indices"]) == 2
    one = app.call(
        "mesh.select_vertices",
        object_id=oid,
        vertex_indices=result["vertex_indices"][:1],
        region=box([0, 0, 0], [0, 0, 0]),
        tolerance=0,
    )
    assert one["count"] == 1
    assert app.engine.snapshot() == before


def test_edge_table_welds_seams_and_actual_segment_intersection(app):
    oid = square(app)
    original = app.call("mesh.select_edges", object_id=oid)
    assert original["count"] == 5
    app.call("mesh.facet", object_id=oid)
    result = app.call("mesh.select_edges", object_id=oid)
    assert result["count"] == 5
    boundary = app.call("mesh.select_edges", object_id=oid, boundary_only=True)
    assert boundary["count"] == 4
    center = app.call(
        "mesh.select_edges",
        object_id=oid,
        region=box([0.9, 0.9, 0], [1.1, 1.1, 0], "intersects"),
        tolerance=0,
    )
    assert center["count"] == 1
    coords = np.asarray(app.engine.object(oid)[1].vertices)[center["edges"][0]]
    np.testing.assert_array_equal(coords, [[0, 0, 0], [2, 2, 0]])
    empty = app.call(
        "mesh.select_edges", object_id=oid, boundary_only=True, edge_indices=center["edge_indices"]
    )
    assert empty["empty"] and empty["edges"] == [] and empty["bounds"] is None
    missed = app.call(
        "mesh.select_edges",
        object_id=oid,
        region=box([0.2, 1.2, 0], [0.4, 1.4, 0], "intersects"),
        tolerance=0,
    )
    assert missed["empty"]  # The diagonal's bounding box overlaps this region.


def test_boundary_selection_is_whole_loop_and_matches_bridge_query(app):
    oid = square(app)
    result = app.call("mesh.select_boundaries", object_id=oid)
    boundaries = app.call("mesh.boundaries", object_id=oid)
    assert result["count"] == 1 and result["loops"] == boundaries["loops"]
    assert result["loop_indices"] == [0]
    empty = app.call(
        "mesh.select_boundaries",
        object_id=oid,
        region=box([0.5, 0.5, 0], [1.5, 1.5, 0], "intersects"),
        tolerance=0,
    )
    assert empty["empty"]  # Boundary contour does not cross the interior box.
    intersects = app.call(
        "mesh.select_boundaries",
        object_id=oid,
        region=box([0.5, 0, 0], [1.5, 0, 0], "intersects"),
        tolerance=0,
    )
    assert intersects["loop_indices"] == [0] and intersects["vertex_indices"] == [0, 1, 2, 3]


def test_components_use_edges_not_point_contact_and_return_faces(app):
    scene = app.scenes.create()
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
        faces=[[0, 1, 2], [0, 3, 4]],
    )
    result = app.call("mesh.select_components", object_id=obj["id"])
    assert result["count"] == 2
    seeded = app.call("mesh.select_components", object_id=obj["id"], seed_face_indices=[1])
    assert seeded["component_indices"] == [1] and seeded["face_indices"] == [1]
    assert app.call(
        "mesh.select_components", object_id=obj["id"], component_indices=[0], seed_face_indices=[1]
    )["empty"]


def test_component_box_rules_test_surface_and_area_centroid(app):
    oid = square(app)
    region = box([0.9, 0.9, 0], [1.1, 1.1, 0], "intersects")
    result = app.call("mesh.select_components", object_id=oid, region=region)
    assert result["face_indices"] == [0, 1]
    region["rule"] = "fully_inside"
    assert app.call("mesh.select_components", object_id=oid, region=region)["empty"]
    region["rule"] = "centroid_inside"
    assert app.call("mesh.select_components", object_id=oid, region=region)["count"] == 1
    # Welded seams must not split a component.
    app.call("mesh.facet", object_id=oid)
    assert app.call("mesh.select_components", object_id=oid)["count"] == 1


@pytest.mark.parametrize(
    "operation",
    ["select_faces", "select_vertices", "select_edges", "select_boundaries", "select_components"],
)
def test_named_selection_round_trip_restart_and_staleness(app, operation):
    oid = square(app)
    selected = app.call(f"mesh.{operation}", object_id=oid)
    revision = selected["geometry_revision"]
    saved = app.call("mesh.selection_save", name="Region", selection=selected)
    assert saved["selection"]["geometry_revision"] == revision
    assert app.call("mesh.selection_get", object_id=oid, name="Region") == saved["selection"]
    workspace = app.engine.root
    app.close()
    with Scissors(workspace) as reopened:
        result = reopened.call("mesh.selection_get", object_id=oid, name="Region")
        assert result == saved["selection"]
        assert not reopened.call("mesh.selection_list", object_id=oid)["selections"][0]["stale"]
        reopened.call("mesh.translate", object_id=oid, x=1)
        assert reopened.call("mesh.selection_list", object_id=oid)["selections"][0]["stale"]
        with pytest.raises(ScissorsError, match="stale"):
            reopened.call("mesh.selection_get", object_id=oid, name="Region")
        reopened.call("mesh.selection_delete", object_id=oid, name="Region")
        assert reopened.call("mesh.selection_list", object_id=oid)["selections"] == []


def test_named_selection_overwrite_empty_and_invalid_payloads(app):
    oid = square(app)
    selected = app.call("mesh.select_vertices", object_id=oid, vertex_indices=[])
    app.call("mesh.selection_save", name="Empty", selection=selected)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="exists"):
        app.call("mesh.selection_save", name="Empty", selection=selected)
    assert app.engine.snapshot() == before
    app.call("mesh.selection_save", name="Empty", selection=selected, overwrite=True)
    assert app.call("mesh.selection_get", object_id=oid, name="Empty")["empty"]
    for invalid in [
        dict(selected, vertex_indices=[99]),
        dict(selected, vertex_indices=[0, 0]),
        dict(selected, geometry_revision="old"),
        dict(selected, vertex_indices=None),
    ]:
        with pytest.raises(ScissorsError):
            app.call("mesh.selection_save", name="Bad", selection=invalid)
    assert app.engine.snapshot() == before


@pytest.mark.parametrize(
    "operation,field",
    [
        ("select_vertices", "vertex_indices"),
        ("select_edges", "edge_indices"),
        ("select_boundaries", "loop_indices"),
        ("select_components", "component_indices"),
    ],
)
def test_query_errors_are_atomic_and_schemas_are_discoverable(app, operation, field):
    oid = square(app)
    before = app.engine.snapshot()
    for values in [[0, 0], [-1], [999]]:
        with pytest.raises(ScissorsError) as error:
            app.call(f"mesh.{operation}", object_id=oid, **{field: values})
        assert error.value.code == "INVALID_PARAMETER"
    with pytest.raises(ScissorsError, match="inverted"):
        app.call(f"mesh.{operation}", object_id=oid, region=box([2, 2, 0], [0, 0, 0]))
    assert app.engine.snapshot() == before
    description = app.registry.describe_operation(f"mesh.{operation}")
    assert not description["side_effects"] and description["permissions"] == ["scene.read"]
    assert field in description["parameters"]["properties"]


def test_selection_resource_errors_and_atomic_batch_rollback(app):
    oid = square(app)
    selected = app.call("mesh.select_edges", object_id=oid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "operations.batch",
            operations=[
                {
                    "operation": "mesh.selection_save",
                    "arguments": {"name": "Edges", "selection": selected},
                },
                {
                    "operation": "mesh.selection_get",
                    "arguments": {"object_id": oid, "name": "Missing"},
                },
            ],
        )
    assert app.engine.snapshot() == before


def test_query_resource_limits(app):
    oid = square(app)
    before = app.engine.snapshot()
    app.engine.settings.max_geometry_pairs = 1
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.select_edges", object_id=oid, region=box([0, 0, 0], [1, 1, 0], "intersects"))
    assert error.value.code == "RESOURCE_LIMIT"
    app.engine.settings.max_memory_bytes = 1
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.select_vertices", object_id=oid)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_saved_face_selection_can_drive_shell_and_snapshot_reload(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    faces = app.call(
        "mesh.select_faces",
        object_id=cube.id,
        normal={"direction": [0, 0, -1], "angular_tolerance_degrees": 1},
    )
    app.call("mesh.selection_save", name="Opening", selection=faces)
    app.call("scene.save", scene_id=scene.id, output="scenes/selected.json")
    opening = app.call("mesh.selection_get", object_id=cube.id, name="Opening")
    assert opening["face_indices"] == faces["face_indices"]
    result = app.call("mesh.shell", object_id=cube.id, thickness=0.1, opening=opening)
    assert result["watertight"]
    assert app.call("mesh.selection_list", object_id=cube.id)["selections"][0]["stale"]
    app.call("scene.load", path="scenes/selected.json")
    assert app.call("mesh.selection_get", object_id=cube.id, name="Opening") == opening


def test_named_selection_limits_and_checkpoint_failure_restore_state(app, monkeypatch):
    oid = square(app)
    selected = app.call("mesh.select_vertices", object_id=oid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        with app.transaction():
            for i in range(129):
                app.call("mesh.selection_save", name=f"Selection {i}", selection=selected)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before

    def fail_commit():
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(app.engine, "commit", fail_commit)
    with pytest.raises(ScissorsError):
        app.call("mesh.selection_save", name="Vertices", selection=selected)
    assert app.engine.snapshot() == before


def test_timeout_and_finite_region_errors_are_explicit(app, monkeypatch):
    from scissors.core import selections

    oid = square(app)
    with pytest.raises(ScissorsError, match="finite"):
        app.call("mesh.select_vertices", object_id=oid, region=box([-1e308, 0, 0], [1e308, 1, 1]))
    before = app.engine.snapshot()
    ticks = iter([0, 31])
    monkeypatch.setattr(selections, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.select_vertices", object_id=oid)
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before


@pytest.mark.parametrize("limit,value", [("max_file_bytes", 1024), ("max_memory_bytes", 16384)])
def test_named_selection_byte_limits(app, limit, value):
    oid = square(app)
    selected = app.call("mesh.select_vertices", object_id=oid)
    before = app.engine.snapshot()
    setattr(app.engine.settings, limit, value)
    with pytest.raises(ScissorsError) as error:
        with app.transaction():
            for i in range(20):
                app.call("mesh.selection_save", name=f"Set {i}", selection=selected)
    assert error.value.code == "RESOURCE_LIMIT"
    assert "storage budget" in str(error.value)
    assert app.engine.snapshot() == before


def test_overwrite_replaces_indices_and_does_not_change_geometry_revision(app):
    oid = square(app)
    first = app.call("mesh.select_vertices", object_id=oid, vertex_indices=[0])
    second = app.call("mesh.select_vertices", object_id=oid, vertex_indices=[1, 2])
    app.call("mesh.selection_save", name="Part", selection=first)
    app.call("mesh.selection_save", name="Part", selection=second, overwrite=True)
    selected = app.call("mesh.selection_get", object_id=oid, name="Part")
    assert selected["vertex_indices"] == [1, 2]
    assert selected["geometry_revision"] == first["geometry_revision"]
    before = app.engine.snapshot()
    selected["vertex_indices"].clear()
    assert app.call("mesh.selection_get", object_id=oid, name="Part")["vertex_indices"] == [1, 2]
    assert app.engine.snapshot() == before


@pytest.mark.parametrize("name", ["", "   ", "X" * 129])
def test_invalid_names_do_not_change_state(app, name):
    oid = square(app)
    selected = app.call("mesh.select_vertices", object_id=oid)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.selection_save", name=name, selection=selected)
    assert error.value.code == "INVALID_PARAMETER"
    assert app.engine.snapshot() == before
