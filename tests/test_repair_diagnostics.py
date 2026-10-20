"""Validation selections and shared renderer diagnostic overlays."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject
from scissors.core.validation import validate_mesh


def plane(app):
    scene = app.scenes.create()
    obj = app.meshes.create_plane(scene_id=scene.id, size=2)
    return scene.id, obj.id


def test_boundary_diagnostics_are_revision_bound_and_saveable(app):
    _, oid = plane(app)
    before = app.engine.snapshot()
    report = app.call("mesh.validate", object_id=oid, include_selections=True)
    assert report["selection_status"] == "complete"
    selected = report["selections"]["boundary_edges"]
    assert selected["count"] == report["properties"]["boundary_edges"] == 4
    assert len(selected["edges"]) == 4 and len(selected["vertex_indices"]) == 4
    canonical = app.call("mesh.select_edges", object_id=oid, boundary_only=True)
    assert canonical["edge_indices"] == selected["edge_indices"]
    assert selected["geometry_revision"] == canonical["geometry_revision"]
    assert selected["preview"]["highlights"][0]["edge_indices"] == selected["edge_indices"]
    assert app.engine.snapshot() == before
    app.call("mesh.selection_save", name="Boundary", selection=selected)
    assert (
        app.call("mesh.selection_get", object_id=oid, name="Boundary")["edge_indices"]
        == selected["edge_indices"]
    )
    assert "selections" not in app.call("mesh.validate", object_id=oid)


def test_duplicate_nonmanifold_and_degenerate_selections_match_counts():
    obj = MeshObject(
        id="damaged",
        name="Damaged",
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [2, 0, 0]],
        faces=[[0, 1, 2], [2, 1, 0], [0, 1, 3], [0, 1, 4]],
    )
    report = validate_mesh(obj, Settings(), include_selections=True)
    selected = report["selections"]
    assert selected["duplicate_faces"]["face_indices"] == [1]
    assert selected["duplicate_faces"]["count"] == report["properties"]["duplicate_faces"]
    assert selected["degenerate_faces"]["face_indices"] == [3]
    assert selected["non_manifold_edges"]["count"] == report["properties"]["non_manifold_edges"]
    assert selected["non_manifold_edges"]["face_indices"] == [0, 1, 2, 3]
    assert selected["non_manifold_edges"]["vertex_indices"] == [0, 1]
    assert selected["incorrect_normals"]["vertex_indices"] == list(range(5))


def test_winding_diagnostics_can_drive_existing_repair(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    selected = app.call("mesh.select_faces", object_id=cube.id)
    selected["face_indices"] = [0]
    app.call("mesh.flip", object_id=cube.id, selection=selected)
    report = app.call("mesh.validate", object_id=cube.id, include_selections=True)
    assert report["selections"]["winding_edges"]["count"] == 3
    faces = report["selections"]["inconsistent_winding"]
    assert 0 in faces["face_indices"]
    result = app.call("mesh.repair_winding", object_id=cube.id, selection=faces)
    assert result["flipped_face_indices"] == [0]


def test_self_intersection_selections_reuse_completed_query(app):
    scene = app.scenes.create()
    obj = app.call(
        "mesh.create",
        scene_id=scene.id,
        vertices=[[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]],
        faces=[[0, 1, 2], [3, 4, 5]],
    )
    unrequested = app.call("mesh.validate", object_id=obj["id"], include_selections=True)
    assert "self_intersections" not in unrequested["selections"]
    assert unrequested["selection_statuses"]["self_intersections"] == "not_requested"
    checked = app.call(
        "mesh.validate", object_id=obj["id"], include_selections=True, check_self_intersections=True
    )
    assert checked["selections"]["self_intersections"]["face_indices"] == [0, 1]
    assert len(checked["components"]) == 2


def test_uv_color_and_unused_vertex_diagnostics():
    mesh = trimesh.creation.box()
    obj = MeshObject(
        id="attributes",
        name="Attributes",
        vertices=[*mesh.vertices.tolist(), [3, 3, 3]],
        faces=mesh.faces.tolist(),
        uv=[[0, 0]] * 9,
        vertex_colors=[[255, 0, 0, 255]] * 9,
    )
    obj.uv[1] = [2, 0]
    obj.vertex_colors[2] = [300, 0, 0, 255]
    report = validate_mesh(obj, Settings(), include_selections=True)
    assert report["selections"]["uv_outside_unit_square"]["vertex_indices"] == [1]
    assert report["selections"]["invalid_vertex_colors"]["vertex_indices"] == [2]
    assert report["selections"]["unused_vertices"]["vertex_indices"] == [8]


def test_invalid_base_arrays_do_not_produce_fake_selections():
    obj = MeshObject(id="invalid", name="Invalid", vertices=[[0, 0, 0]], faces=[[0, 1, 2]])
    report = validate_mesh(obj, Settings(), include_selections=True)
    assert report["selection_status"] == "unavailable" and not report["selections"]


@pytest.mark.parametrize("kind", ["faces", "edges", "vertices", "boundaries", "components"])
def test_preview_renders_visible_highlight_pixels_and_keeps_scene_unchanged(app, kind):
    scene, oid = plane(app)
    selected = app.call(f"mesh.select_{kind}", object_id=oid)
    if kind == "faces":
        selected["face_indices"] = selected["face_indices"][:1]
    before = app.engine.snapshot()
    plain = app.call(
        "scene.preview",
        scene_id=scene,
        output="previews/plain.png",
        width=96,
        height=96,
        renderer="pillow",
        elevation_degrees=90,
    )
    marked = app.call(
        "scene.preview",
        scene_id=scene,
        output="previews/marked.png",
        width=96,
        height=96,
        renderer="pillow",
        elevation_degrees=90,
        highlights=[selected],
    )
    a = np.asarray(Image.open(plain["path"]).convert("RGB"))
    b = np.asarray(Image.open(marked["path"]).convert("RGB"))
    assert np.any(a != b)
    assert np.any(np.all(b == [255, 80, 0], axis=-1))
    assert marked["highlight_mode"] == "xray_overlay"
    assert marked["highlights"][0]["drawn_primitives"] > 0
    assert app.engine.snapshot() == before


def test_stale_foreign_and_filtered_highlights_fail_before_writing(app):
    scene, oid = plane(app)
    selected = app.call("mesh.select_faces", object_id=oid)
    output = "previews/rejected.png"
    for arguments in [
        dict(highlights=[dict(selected, object_id="another_scene_object")]),
        dict(highlights=[dict(selected, geometry_revision="old")]),
        dict(highlights=[dict(selected, face_indices=[99])]),
        dict(highlights=[selected], exclude_object_ids=[oid]),
        dict(highlights=[selected], render_pass="normal"),
    ]:
        with pytest.raises(ScissorsError):
            app.call("scene.preview", scene_id=scene, output=output, renderer="pillow", **arguments)
        assert not (app.engine.root / output).exists()
    app.call("mesh.translate", object_id=oid, x=1)
    with pytest.raises(ScissorsError, match="revision"):
        app.call(
            "scene.preview", scene_id=scene, output=output, highlights=[selected], renderer="pillow"
        )
    assert not Path(app.engine.root / output).exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_memory_bytes", 1),
        ("max_geometry_pairs", 1),
        ("max_file_bytes", 1500),
        ("max_faces", 1),
    ],
)
def test_diagnostic_resource_limits_preserve_state(app, field, value):
    _, oid = plane(app)
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.validate", object_id=oid, include_selections=True)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before


def test_diagnostic_timeout_and_default_validation_avoid_selection_work(app, monkeypatch):
    from scissors.core import diagnostics

    _, oid = plane(app)
    ticks = iter([0, 31])
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert app.call("mesh.validate", object_id=oid)["valid"]
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.validate", object_id=oid, include_selections=True)
    assert error.value.code == "OPERATION_TIMEOUT"


def test_highlight_section_and_near_clipping_remove_hidden_targets(app):
    scene, oid = plane(app)
    selected = app.call("mesh.select_faces", object_id=oid)
    result = app.call(
        "scene.preview",
        scene_id=scene,
        output="previews/section.png",
        renderer="pillow",
        width=64,
        height=64,
        highlights=[selected],
        clip_axis="z",
        clip_position=1,
    )
    assert result["highlights"][0]["drawn_primitives"] == 0
    # Look away from the object. Targets behind the eye cannot draw an overlay.
    result = app.call(
        "scene.preview",
        scene_id=scene,
        output="previews/behind.png",
        renderer="pillow",
        width=64,
        height=64,
        highlights=[selected],
        projection="perspective",
        camera_position=[0, 0, 3],
        camera_target=[0, 0, 6],
    )
    assert result["highlights"][0]["drawn_primitives"] == 0


def test_highlight_work_limit_and_projection_timeout_do_not_publish(app, monkeypatch):
    from scissors.core import highlighting

    scene, oid = plane(app)
    selected = app.call("mesh.select_faces", object_id=oid)
    app.engine.settings.max_geometry_pairs = 1
    with pytest.raises(ScissorsError) as error:
        app.call(
            "scene.preview",
            scene_id=scene,
            output="previews/limit.png",
            highlights=[selected],
            renderer="pillow",
        )
    assert error.value.code == "RESOURCE_LIMIT"
    assert not (app.engine.root / "previews/limit.png").exists()
    app.engine.settings.max_geometry_pairs = 2_000_000
    ticks = iter([0, 31])
    monkeypatch.setattr(highlighting, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ScissorsError) as error:
        app.call(
            "scene.preview",
            scene_id=scene,
            output="previews/timeout.png",
            highlights=[selected],
            renderer="pillow",
        )
    assert error.value.code == "OPERATION_TIMEOUT"
    assert not (app.engine.root / "previews/timeout.png").exists()


def test_highlight_with_vtk_uses_the_same_overlay_layer(app):
    from scissors.core.rendering_fast import vtk_capability

    if not vtk_capability()[0]:
        pytest.skip("VTK OpenGL context is unavailable")
    scene, oid = plane(app)
    selected = app.call("mesh.select_edges", object_id=oid, boundary_only=True)
    result = app.call(
        "scene.preview",
        scene_id=scene,
        output="previews/vtk-highlights.png",
        renderer="vtk",
        width=64,
        height=64,
        elevation_degrees=90,
        highlights=[selected],
    )
    pixels = np.asarray(Image.open(result["path"]).convert("RGB"))
    assert result["renderer"] == "vtk" and result["highlight_mode"] == "xray_overlay"
    assert np.any(np.all(pixels == [255, 80, 0], axis=-1))


def test_invalid_topology_does_not_claim_clean_intersection_selection():
    obj = MeshObject(
        id="duplicate",
        name="Duplicate",
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 1, 2]],
    )
    report = validate_mesh(obj, Settings(), include_selections=True, check_self_intersections=True)
    assert report["selection_status"] == "partial"
    assert report["selection_statuses"]["self_intersections"] == "uncompleted"
    assert "self_intersections" not in report["selections"]


@pytest.mark.parametrize("kind", ["group", "marker"])
def test_nonmesh_nodes_have_no_element_selections(kind):
    obj = MeshObject(id="node", name="Node", kind=kind, vertices=[], faces=[])
    report = validate_mesh(obj, Settings(), include_selections=True)
    assert report["selection_status"] == "not_applicable" and report["selections"] == {}
