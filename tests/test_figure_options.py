"""Figure placement, cap schemas and rounded-pole openings."""

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import face_components, topology_mesh


@pytest.mark.parametrize("kind", ["cube", "cylinder", "cone", "sphere"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_base_figure_options_place_axial_minimum_at_zero(app, kind, axis):
    scene = app.scenes.create()
    result = app.call(
        f"mesh.create_{kind}", scene_id=scene.id, axis=axis, pivot="base", name="Base figure"
    )
    index = "xyz".index(axis)
    assert result["name"] == "Base figure" and result["watertight"]
    assert result["bounds"][0][index] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("kind", ["cube", "cylinder", "cone", "rounded_box"])
def test_uncapped_figures_have_open_boundaries(app, kind):
    scene = app.scenes.create()
    parameters = {"size": [1, 2, 0.2]} if kind == "rounded_box" else {}
    result = app.call(
        f"mesh.create_{kind}", scene_id=scene.id, capped=False, axis="x", **parameters
    )
    assert not result["watertight"] and result["face_count"] > 0


def test_cylinder_segments_alias_and_conflict_rollback(app):
    scene = app.scenes.create()
    original = app.call("mesh.create_cylinder", scene_id=scene.id, sections=7)
    alias = app.call("mesh.create_cylinder", scene_id=scene.id, segments=7)
    np.testing.assert_allclose(
        app.engine.object(original["object_id"])[1].vertices,
        app.engine.object(alias["object_id"])[1].vertices,
    )
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("mesh.create_cylinder", scene_id=scene.id, sections=8, segments=7)
    assert app.engine.snapshot() == before


def test_sphere_segment_mode_is_distinct_from_icosphere(app):
    scene = app.scenes.create()
    sphere = app.call("mesh.create_sphere", scene_id=scene.id, segments=8)
    assert sphere["watertight"] and sphere["volume"] > 0
    with pytest.raises(ScissorsError):
        app.call("mesh.create_sphere", scene_id=scene.id, segments=8, subdivisions=1)


def test_profile_axis_and_centered_extruded_polygon(app):
    scene = app.scenes.create()
    points = [[0, 0], [2, 0], [2, 1], [0, 1]]
    profile = app.call("mesh.create_profile", scene_id=scene.id, points=points, axis="x")
    extruded = app.call("mesh.extrude", object_id=profile["object_id"], distance=0.3)
    np.testing.assert_allclose(np.diff(extruded["bounds"], axis=0)[0], [0.3, 1, 2])
    polygon = app.call(
        "mesh.create_polygon",
        scene_id=scene.id,
        points=points,
        thickness=0.3,
        axis="x",
        pivot="center",
        capped=False,
    )
    assert not polygon["watertight"]
    np.testing.assert_allclose(np.mean(polygon["bounds"], axis=0), 0, atol=1e-10)


def test_tube_cap_alias_retains_input_path_by_default(app):
    scene = app.scenes.create()
    tube = app.call(
        "mesh.create_tube",
        scene_id=scene.id,
        points=[[2, 0, 1], [2, 0, 3]],
        radius=0.1,
        capped=False,
    )
    assert not tube["watertight"]
    np.testing.assert_allclose(np.mean(tube["bounds"], axis=0), [2, 0, 2])


def test_figure_discovery_exposes_applicable_common_option_schemas(app):
    catalog = {figure["kind"]: figure for figure in app.call("mesh.figures")["figures"]}
    for figure in catalog.values():
        assert {"axis", "pivot", "name"} <= figure["common_options"].keys()
    assert "capped" in catalog["tube"]["common_options"]
    assert "segments" in catalog["sphere"]["common_options"]
    for kind in ("sphere", "capsule"):
        described = app.registry.describe_operation(f"mesh.create_{kind}")
        assert (
            catalog[kind]["common_options"]["capped"]
            == described["parameters"]["properties"]["capped"]
        )
        assert catalog[kind]["common_options"]["capped"]["type"] == "boolean"


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("sphere", {"subdivisions": 1}),
        ("sphere", {"segments": 8}),
        ("capsule", {"segments": 8}),
    ],
)
def test_rounded_figure_cap_switch_opens_two_poles(app, kind, parameters):
    scene = app.scenes.create()
    closed = app.call(f"mesh.create_{kind}", scene_id=scene.id, **parameters)
    opened = app.call(f"mesh.create_{kind}", scene_id=scene.id, capped=False, **parameters)
    assert closed["watertight"] and not opened["watertight"]
    assert 0 < opened["face_count"] < closed["face_count"]
    mesh = topology_mesh(app.engine.object(opened["object_id"])[1].geometry())
    edges = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse) == 1]
    vertices, adjacency = np.unique(edges, return_inverse=True)
    assert len(face_components(len(vertices), adjacency.reshape(-1, 2))) == 2
