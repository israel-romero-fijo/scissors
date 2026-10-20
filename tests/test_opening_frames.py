"""Opening frame invariants for future regression runs."""

import copy

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh


def hollow_wall(app, material_split=False):
    sid = app.scenes.create().id
    outer = trimesh.creation.box(extents=[2, 2, 2])
    inner = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
    inner.invert()
    wall = trimesh.util.concatenate([outer, inner])
    source = app.call(
        "mesh.create", scene_id=sid, vertices=wall.vertices.tolist(), faces=wall.faces.tolist()
    )
    material = app.call("material.create", scene_id=sid)["id"] if material_split else None
    cut = app.call(
        "scene.cut_volume",
        scene_id=sid,
        target_ids=[source["id"]],
        volume={"min": [0, -0.4, -0.4], "max": [2, 0.4, 0.4]},
        cut_material_id=material,
    )["targets"][0]
    return sid, source["id"], cut


@pytest.mark.parametrize("material_split", [False, True])
def test_closed_wall_rim_can_frame_without_mesh_boundary_edges(app, material_split):
    _, oid, cut = hollow_wall(app, material_split)
    before = app.engine.object(oid)[1].model_dump()
    result = app.call(
        "mesh.frame_opening",
        object_id=oid,
        opening={"type": "cut", "geometry_revision": cut["geometry_revision"]},
        contour_side="outer",
        normal=[1, 0, 0],
        section={"type": "circle", "radius": 0.019},
    )
    assert result["watertight"]
    assert all(point[0] == pytest.approx(1) for point in result["resolved_contour"]["points"])
    assert result["checks"]["non_adjacent_self_intersections"]["status"] == "passed"
    assert app.engine.object(oid)[1].model_dump() == before
    # Public results must not alias the stored cut selection.
    cut["opening"]["contours"].clear()
    assert app.engine.object(oid)[1].metadata["_last_cut_opening"]["contours"]


def test_rim_faces_and_ambiguity_are_explicit(app):
    _, oid, cut = hollow_wall(app)
    args = dict(
        object_id=oid,
        opening={"type": "cut", "geometry_revision": cut["geometry_revision"]},
        section={"type": "rectangle", "width": 0.04, "depth": 0.02},
    )
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="Choose contour_index"):
        app.call("mesh.frame_opening", **args)
    assert before == app.engine.snapshot()
    selected = app.call(
        "mesh.select_faces",
        object_id=oid,
        tolerance=1e-8,
        region={"bounds": [[0.75, -0.4, -0.4], [1, 0.4, 0.4]], "rule": "fully_inside"},
    )
    args["opening"] = {
        "type": "rim_faces",
        "geometry_revision": selected["geometry_revision"],
        "face_indices": selected["face_indices"],
        "contour_index": 0,
    }
    result = app.call("mesh.frame_opening", **args)
    assert result["watertight"]
    app.call("mesh.translate", object_id=oid, x=0.1)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="geometry revision"):
        app.call("mesh.frame_opening", **args)
    assert before == app.engine.snapshot()


def test_section_clearance_and_offset_match_actual_world_geometry(app):
    sid = app.scenes.create().id
    source = app.call("mesh.create_cube", scene_id=sid)
    result = app.call(
        "mesh.frame_opening",
        object_id=source["id"],
        opening={
            "type": "points",
            "geometry_revision": source["geometry_revision"],
            "points": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        },
        section={"type": "rectangle", "width": 0.1, "depth": 0.04},
        normal=[0, 0, 1],
        placement="outside",
        clearance=0.02,
        offset=0.1,
    )
    mesh = app.engine.object(result["id"])[1].geometry()
    np.testing.assert_allclose(mesh.bounds, [[-1.12, -1.12, 0.08], [1.12, 1.12, 0.12]])
    midpoint_section = (mesh.vertices[:4] + mesh.vertices[4:8]) / 2
    np.testing.assert_allclose(np.ptp(midpoint_section, axis=0)[[1, 2]], [0.1, 0.04])
    assert midpoint_section[:, 1].max() == pytest.approx(-1.02)


def test_crossings_degenerate_corners_and_invalid_options_leave_no_frame(app):
    sid = app.scenes.create().id
    source = app.call("mesh.create_cube", scene_id=sid)
    args = dict(
        object_id=source["id"],
        opening={
            "type": "points",
            "geometry_revision": source["geometry_revision"],
            "points": [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]],
        },
        section={"type": "circle", "radius": 0.02},
        normal=[0, 0, 1],
    )
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.frame_opening", **args)
    assert before == app.engine.snapshot()
    invalid = copy.deepcopy(args)
    invalid["normal"] = [0, 0, 0]
    with pytest.raises(ScissorsError):
        app.call("mesh.frame_opening", **invalid)
    assert before == app.engine.snapshot()


def test_spatial_path_and_explicit_open_ends(app):
    sid = app.scenes.create().id
    source = app.call("mesh.create_cube", scene_id=sid)
    args = dict(
        object_id=source["id"],
        opening={
            "type": "points",
            "geometry_revision": source["geometry_revision"],
            "points": [[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0]],
        },
        section={"type": "circle", "radius": 0.02},
        joint_policy="transport",
    )
    closed = app.call("mesh.frame_opening", **args)
    assert closed["watertight"] and not closed["planar"]
    args["opening"]["closed"] = False
    opened = app.call("mesh.frame_opening", **args, end_policy="open")
    assert not topology_mesh(app.engine.object(opened["id"])[1].geometry()).is_watertight
    assert opened["end_policy"] == "open"


def test_resource_limits_never_count_as_successful_checks(app):
    sid = app.scenes.create().id
    source = app.call("mesh.create_cube", scene_id=sid)
    args = dict(
        object_id=source["id"],
        opening={
            "type": "points",
            "geometry_revision": source["geometry_revision"],
            "points": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        },
        section={"type": "rectangle", "width": 0.1, "depth": 0.04},
    )
    limit = app.engine.settings.max_geometry_pairs
    app.engine.settings.max_geometry_pairs = 100
    before = app.engine.snapshot()
    try:
        with pytest.raises(ScissorsError) as error:
            app.call("mesh.frame_opening", **args)
        assert error.value.code == "RESOURCE_LIMIT"
        assert before == app.engine.snapshot()
        result = app.call("mesh.frame_opening", **args, allow_incomplete_checks=True)
        assert result["checks"]["non_adjacent_self_intersections"]["status"] == "uncompleted"
        assert result["warnings"]
    finally:
        app.engine.settings.max_geometry_pairs = limit
