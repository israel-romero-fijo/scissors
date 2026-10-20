"""Whole-domain thickness bounds, curved walls, explicit scope and read-only limits."""

from fractions import Fraction

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.thickness import bounds, clip, difference


def create(app, mesh):
    scene = app.scenes.create()
    return app.call(
        "mesh.create", scene_id=scene.id, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
    )["object_id"]


def inspect(app, oid, minimum, **kwargs):
    return app.call(
        "mesh.check_thickness", object_id=oid, minimum=minimum, tolerance=1e-9, **kwargs
    )


def face_direction(app, oid, direction):
    return app.call(
        "mesh.select_faces",
        object_id=oid,
        normal={"direction": direction, "angular_tolerance_degrees": 0.01},
    )


@pytest.mark.parametrize("minimum,status", [(0.1, "pass"), (0.2, "fail")])
def test_curved_hollow_wall_has_global_bounds(app, minimum, status):
    outer = trimesh.creation.icosphere(subdivisions=0)
    inner = trimesh.creation.icosphere(subdivisions=0, radius=0.8)
    inner.invert()
    oid = create(app, trimesh.util.concatenate([outer, inner]))
    before = app.engine.snapshot()
    checkpoint = (app.engine.root / "state.json").read_bytes()
    result = inspect(app, oid, minimum)
    assert result["status"] == status
    assert result["coverage"]["complete"], result["unresolved_regions"]
    measured = result["achieved_error_bounds"]
    expected = 0.2 * abs(outer.face_normals[0] @ outer.triangles[0, 0])
    assert measured["global_lower"] == pytest.approx(expected, abs=1e-12)
    assert measured["global_upper"] - measured["global_lower"] <= 1e-9
    assert result["coverage"]["sampled_faces"] == 40
    assert app.engine.snapshot() == before
    assert (app.engine.root / "state.json").read_bytes() == checkpoint


def test_tapered_panel_detects_thin_corner_between_samples(app):
    mesh = trimesh.creation.box()
    corner = np.flatnonzero(np.all(mesh.vertices == [0.5, 0.5, 0.5], axis=1))[0]
    mesh.vertices[corner, 2] = -0.49
    oid = create(app, mesh)
    selection = face_direction(app, oid, [0, 0, -1])
    result = inspect(app, oid, 0.05, selection=selection)
    assert result["status"] == "fail"
    assert result["coverage"]["complete"]
    assert result["achieved_error_bounds"]["global_upper"] == pytest.approx(0.01)
    assert result["violations"]
    assert result["coverage"]["measured_area"] == pytest.approx(1)


def test_annular_curved_panel_and_world_transform(app):
    mesh = trimesh.creation.annulus(r_min=1, r_max=1.2, height=1, sections=8)
    oid = create(app, mesh)
    app.call("mesh.rotate", object_id=oid, axis="x", angle_degrees=27)
    app.call("mesh.translate", object_id=oid, x=4, y=-2, z=3)
    axis = [0, -np.sin(np.radians(27)), np.cos(np.radians(27))]
    rims = face_direction(app, oid, axis)
    opposite = face_direction(app, oid, [-c for c in axis])
    rims["face_indices"] += opposite["face_indices"]
    result = inspect(app, oid, 0.15, rim_selection=rims)
    assert result["status"] == "pass", result["unresolved_regions"]
    assert result["coverage"]["complete"]
    assert result["achieved_error_bounds"]["global_lower"] > 0.15


def test_explicit_rims_report_excluded_first_exit_area(app):
    oid = create(app, trimesh.creation.box())
    top = face_direction(app, oid, [0, 0, 1])
    bottom = face_direction(app, oid, [0, 0, -1])
    bottom["face_indices"] = bottom["face_indices"][:1]
    result = inspect(app, oid, 0.5, selection=top, rim_selection=bottom)
    assert result["status"] == "pass"
    assert result["coverage"]["complete"]
    assert result["coverage"]["measured_area"] == pytest.approx(0.5)
    assert result["coverage"]["excluded_exit_area"] == pytest.approx(0.5)
    assert all(
        region["opposing_face_index"] == bottom["face_indices"][0]
        for region in result["excluded_rim_exit_regions"]
    )
    assert result["excluded_rim_regions"] == bottom["face_indices"]


def test_empty_domain_and_stale_selections_do_not_pass(app):
    oid = create(app, trimesh.creation.box())
    top = face_direction(app, oid, [0, 0, 1])
    result = inspect(app, oid, 0.1, selection=top, rim_selection=top)
    assert result["status"] == "inconclusive"
    assert not result["coverage"]["complete"]
    empty_rims = {**top, "face_indices": []}
    assert inspect(app, oid, 0.1, selection=top, rim_selection=empty_rims)["status"] == "pass"
    app.call("mesh.translate", object_id=oid, x=1)
    with pytest.raises(ScissorsError, match="revision"):
        inspect(app, oid, 0.1, selection=top)


@pytest.mark.parametrize(
    "setting,value",
    [("max_geometry_pairs", 1), ("max_memory_bytes", 1), ("operation_timeout", 1e-12)],
)
def test_uncompleted_checks_do_not_produce_pass(app, monkeypatch, setting, value):
    oid = create(app, trimesh.creation.box())
    before = app.engine.snapshot()
    monkeypatch.setattr(app.engine.settings, setting, value)
    result = inspect(app, oid, 0.1)
    assert result["status"] == "inconclusive"
    assert not result["coverage"]["complete"]
    assert result["unresolved_regions"]
    assert app.engine.snapshot() == before


def test_nested_same_orientation_and_intersecting_boundaries_are_unresolved(app):
    first = trimesh.creation.box(extents=[2, 2, 2])
    second = trimesh.creation.box()
    oid = create(app, trimesh.util.concatenate([first, second]))
    result = inspect(app, oid, 0.01)
    assert result["status"] == "inconclusive"
    assert "orientation" in str(result["unresolved_regions"])
    second.apply_translation([0.8, 0.2, 0.1])
    oid = create(app, trimesh.util.concatenate([first, second]))
    result = inspect(app, oid, 0.01)
    assert result["status"] == "inconclusive"
    assert result["unresolved_regions"]


def test_float_bounds_enclose_exact_square_roots_and_tiny_values():
    for squared in (Fraction(2), Fraction(1, 10**650), Fraction(10**600), Fraction(0)):
        low, high = bounds(squared)
        assert Fraction(low) ** 2 <= squared <= Fraction(high) ** 2


def test_exact_clipping_retains_tiny_regions_and_does_not_duplicate_domains():
    domain = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    tiny = Fraction(1, 10**30)
    polygon = clip(domain, (tiny, Fraction(-1), Fraction(-1)))
    assert len(polygon) == 3
    outside, inside = difference(domain, [(Fraction(0), Fraction(0), Fraction(0))])
    assert outside == []
    assert inside == domain


def test_catalog_declares_query_scope_and_read_only_permission(app):
    op = app.registry.operations["mesh.check_thickness"]
    assert not op.side_effects
    assert op.permissions == ["scene.read"]
    assert {"selection", "rim_selection"} <= set(op.parameters["properties"])


@pytest.mark.parametrize(
    "second,valid",
    [
        ([[0, 0, 0], [1, 1, -1], [1, 1, 1]], False),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], False),
        ([[2, 0, 0], [2, 2, 0], [0, 2, 0]], True),
        ([[3, 0, 0], [4, 0, 0], [3, 1, 0]], True),
    ],
)
def test_exact_contact_check_allows_only_shared_topology(tmp_path, second, valid):
    from scissors.config import Settings
    from scissors.core.thickness import Budget, cross, exact_contacts, sub

    triangles = np.asarray([[[0, 0, 0], [2, 0, 0], [0, 2, 0]], second], dtype=float)
    points = [[tuple(Fraction(float(c)) for c in vertex) for vertex in face] for face in triangles]
    data = [(a, sub(b, a), sub(c, a)) for a, b, c in points]
    normals = [cross(a, b) for _, a, b in data]
    budget = Budget(Settings(workspace=tmp_path))
    if valid:
        exact_contacts(data, normals, triangles, budget)
    else:
        with pytest.raises(ScissorsError):
            exact_contacts(data, normals, triangles, budget)
