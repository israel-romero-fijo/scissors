"""Cockpit geometry cases migrated to the native registry. Tests remain unrun."""

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core import mechanical as module

SCENE = None
APP = None
TOGGLE_TILT = math.radians(20.0)
TOGGLE_RADIUS = 0.0035


@pytest.fixture(autouse=True)
def native_scene(app, monkeypatch):
    scene = app.scenes.create(name="Native mechanical cases")
    monkeypatch.setitem(globals(), "APP", app)
    monkeypatch.setitem(globals(), "SCENE", scene.id)


def native_arguments(operation, arguments):
    arguments = dict(arguments)
    if operation == "cockpit.rounded_box":
        arguments["size"] = [
            arguments.pop("width"),
            arguments.pop("depth"),
            arguments.pop("height"),
        ]
    return "mesh.create_" + operation.split(".")[1], arguments


# --------------------------------------------------------------------------- helpers


def staged(operation, **arguments):
    """Create through the native registry and return geometry plus deterministic summary."""
    name, arguments = native_arguments(operation, arguments)
    result = APP.call(name, scene_id=SCENE, **arguments)
    obj = APP.engine.object(result["object_id"])[1]
    geometry = {"vertices": obj.vertices, "faces": obj.faces}
    summary = {key: result[key] for key in ("name", "vertex_count", "face_count", "bounds")}
    return geometry, summary


def geometry(operation, **arguments):
    """Read arrays created by a native operation."""
    created, _ = staged(operation, **arguments)
    return np.asarray(created["vertices"], dtype=float), np.asarray(created["faces"], dtype=int)


def assert_rejected(operation, **arguments):
    """Invalid native calls must raise a structured error and restore scene state."""
    before = APP.engine.snapshot()
    name, arguments = native_arguments(operation, arguments)
    with pytest.raises(ScissorsError):
        APP.call(name, scene_id=SCENE, **arguments)
    assert APP.engine.snapshot() == before


def shells(faces):
    """Group triangles into connected shells that share undirected edges."""
    parent = list(range(len(faces)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner = {}
    for index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            if key in owner:
                left, right = find(index), find(owner[key])
                if left != right:
                    parent[max(left, right)] = min(left, right)
            else:
                owner[key] = index
    groups = {}
    for index in range(len(faces)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def edge_counts(faces):
    counts = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def count_boundary_edges(faces):
    """Edges used by a single face belong to an open border."""
    return int(sum(1 for value in edge_counts(faces).values() if value == 1))


def signed_volume(vertices, faces):
    """Signed volume of a face subset, positive when the surface faces outward."""
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", first, np.cross(second, third)).sum() / 6.0)


def assert_closed_and_outward(vertices, faces):
    """Every shell must be watertight, consistently wound and positively oriented."""
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1)
    assert areas.min() > 1e-14, "the engine rejects degenerate faces"
    groups = shells(faces)
    assert groups, "geometry must contain faces"
    for group in groups:
        part = faces[group]
        counts = {}
        directed = set()
        for a, b, c in part:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(min(u, v)), int(max(u, v)))
                counts[key] = counts.get(key, 0) + 1
                directed.add((int(u), int(v)))
        assert set(counts.values()) == {2}, "every edge of a closed shell is shared by two faces"
        assert all((v, u) in directed for u, v in directed), "faces must be wound consistently"
        assert signed_volume(vertices, part) > 0, "every shell must face outward"
    return groups


def _segment_triangle(start, end, triangle, tolerance=1e-12):
    """Moeller-Trumbore test: does the segment [start, end] cross the triangle?"""
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    direction = end - start
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    scale = float(np.linalg.norm(edge1) * np.linalg.norm(edge2) * np.linalg.norm(direction))
    if scale == 0.0 or abs(determinant) <= tolerance * scale:
        return False
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -tolerance or u > 1 + tolerance:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -tolerance or u + v > 1 + tolerance:
        return False
    t = float(np.dot(edge2, qvec)) * inverse
    return -tolerance <= t <= 1 + tolerance


def _convex_overlap_2d(first, second, tolerance=1e-12):
    """Separating-axis test for two convex polygons in 2D."""
    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.array([-edge[1], edge[0]])
            length = float(np.linalg.norm(axis))
            if length == 0.0:
                continue
            axis = axis / length
            one, other = first @ axis, second @ axis
            if one.max() < other.min() - tolerance or other.max() < one.min() - tolerance:
                return False
    return True


def _triangles_intersect(first, second, tolerance=1e-12):
    """Conservative triangle-triangle overlap test; touching counts as intersecting."""
    normal_first = np.cross(first[1] - first[0], first[2] - first[0])
    normal_second = np.cross(second[1] - second[0], second[2] - second[0])
    normals = float(np.linalg.norm(normal_first) * np.linalg.norm(normal_second))
    distance = float(np.dot(normal_first, second[0] - first[0]))
    parallel = float(np.linalg.norm(np.cross(normal_first, normal_second))) <= tolerance * normals
    if parallel and abs(distance) <= tolerance * float(np.linalg.norm(normal_first)):
        drop = int(np.argmax(np.abs(normal_first)))
        keep = [axis for axis in range(3) if axis != drop]
        return _convex_overlap_2d(first[:, keep], second[:, keep])
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_triangle(triangle[index], triangle[(index + 1) % 3], other):
                return True
    return False


def _boxes_overlap(first, second, tolerance=1e-12):
    low_first, high_first = first.reshape(-1, 3).min(axis=0), first.reshape(-1, 3).max(axis=0)
    low_second, high_second = second.reshape(-1, 3).min(axis=0), second.reshape(-1, 3).max(axis=0)
    return not (
        np.any(low_first > high_second + tolerance) or np.any(low_second > high_first + tolerance)
    )


def assert_shells_do_not_intersect(vertices, faces):
    """No two separate shells may overlap or touch."""
    triangles = vertices[faces]
    groups = [triangles[group] for group in shells(faces)]
    for index, one in enumerate(groups):
        for other in groups[index + 1 :]:
            if not _boxes_overlap(one, other):
                continue
            for triangle in one:
                for candidate in other:
                    assert not _triangles_intersect(triangle, candidate), (
                        "separate shells must not intersect or touch"
                    )


def assert_deterministic(operation, **arguments):
    """Two identical calls must stage identical arrays and summaries."""
    first, result_first = staged(operation, **arguments)
    second, result_second = staged(operation, **arguments)
    assert first["vertices"] == second["vertices"]
    assert first["faces"] == second["faces"]
    assert result_first == result_second


def assert_bounds(vertices, low, high):
    np.testing.assert_allclose(vertices.min(axis=0), np.asarray(low, dtype=float), atol=1e-12)
    np.testing.assert_allclose(vertices.max(axis=0), np.asarray(high, dtype=float), atol=1e-12)


# ------------------------------------------------------------------- helper self-tests


def test_triangle_helper_matches_known_cases():
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    crossing = np.array([[0.2, 0.2, -1.0], [0.2, 0.2, 1.0], [0.6, 0.6, 0.0]])
    assert _triangles_intersect(flat, crossing)
    assert not _triangles_intersect(flat, crossing + np.array([5.0, 0.0, 0.0]))
    coplanar = np.array([[0.1, 0.1, 0.0], [0.6, 0.1, 0.0], [0.1, 0.6, 0.0]])
    assert _triangles_intersect(flat, coplanar)
    assert not _triangles_intersect(flat, coplanar + np.array([5.0, 0.0, 0.0]))
    touching = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 1.0]])
    assert _triangles_intersect(flat, touching)


def test_shell_helper_groups_one_box():
    vertices, faces = module._oriented_box(
        np.zeros(3),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
    )
    assert len(vertices) == 8
    assert len(shells(np.asarray(faces))) == 1
    assert count_boundary_edges(np.asarray(faces)) == 0


# --------------------------------------------------------------- cockpit.rounded_box


def test_rounded_box_counts_bounds_and_volume():
    vertices, faces = geometry(
        "cockpit.rounded_box", width=2, depth=1.5, height=0.4, corner_radius=0.2, segments=4
    )
    # 4 corners x (4 chords + 1 tangent end) per ring, two rings and two cap centers.
    assert vertices.shape == (42, 3)
    assert faces.shape == (80, 3)
    assert_bounds(vertices, [-1.0, -0.75, 0.0], [1.0, 0.75, 0.4])
    assert_closed_and_outward(vertices, faces)
    step = math.radians(90.0 / 4)
    inscribed = 0.4 * (2 * 1.5 - 4 * 0.2**2 + 4 * 4 * 0.5 * 0.2**2 * math.sin(step))
    assert signed_volume(vertices, faces) == pytest.approx(inscribed, abs=1e-12)
    ideal = 0.4 * (2 * 1.5 - (4 - math.pi) * 0.2**2)
    assert signed_volume(vertices, faces) == pytest.approx(ideal, abs=2e-3)


def test_rounded_box_zero_radius_is_a_plain_box():
    vertices, faces = geometry(
        "cockpit.rounded_box", width=2, depth=1.5, height=0.4, corner_radius=0, segments=16
    )
    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert_bounds(vertices, [-1.0, -0.75, 0.0], [1.0, 0.75, 0.4])
    assert_closed_and_outward(vertices, faces)
    assert signed_volume(vertices, faces) == pytest.approx(1.2, abs=1e-12)


def test_rounded_box_full_radius_collapses_to_a_regular_polygon():
    vertices, faces = geometry(
        "cockpit.rounded_box", width=2, depth=2, height=0.4, corner_radius=1.0, segments=4
    )
    # The four corner centers coincide, so the four repeated points are dropped.
    assert vertices.shape == (34, 3)
    assert faces.shape == (64, 3)
    assert_bounds(vertices, [-1.0, -1.0, 0.0], [1.0, 1.0, 0.4])
    assert_closed_and_outward(vertices, faces)
    regular = 0.4 * 0.5 * 16 * math.sin(2 * math.pi / 16)
    assert signed_volume(vertices, faces) == pytest.approx(regular, abs=1e-12)


def test_rounded_box_accepts_the_radius_limit_exactly():
    vertices, faces = geometry(
        "cockpit.rounded_box", width=2, depth=1.5, height=0.4, corner_radius=0.75, segments=2
    )
    assert_closed_and_outward(vertices, faces)
    assert_bounds(vertices, [-1.0, -0.75, 0.0], [1.0, 0.75, 0.4])


def test_rounded_box_is_deterministic():
    assert_deterministic(
        "cockpit.rounded_box", width=1.2, depth=0.8, height=0.5, corner_radius=0.15, segments=6
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"width": 0, "depth": 1, "height": 1},
        {"width": 1, "depth": -1, "height": 1},
        {"width": 1, "depth": 1, "height": 0},
        {"width": 1, "depth": float("nan"), "height": 1},
        {"width": float("inf"), "depth": 1, "height": 1},
        {"width": 2, "depth": 1.5, "height": 1, "corner_radius": 0.76},
        {"width": 2, "depth": 1.5, "height": 1, "corner_radius": -0.1},
        {"width": 1, "depth": 1, "height": 1, "segments": 0},
        {"width": 1, "depth": 1, "height": 1, "segments": 17},
        {"width": 1, "depth": 1, "height": 1, "segments": 4.0},
        {"width": 1, "depth": 1, "height": 1, "name": "   "},
        {"width": "1", "depth": 1, "height": 1},
    ],
)
def test_rounded_box_rejects_invalid_arguments(arguments):
    assert_rejected("cockpit.rounded_box", **arguments)


# ---------------------------------------------------------------------- cockpit.tube


def test_tube_straight_counts_bounds_and_volume():
    vertices, faces = geometry("cockpit.tube", points=[[0, 0, 0], [0, 0, 1]], radius=0.1, sides=8)
    assert vertices.shape == (2 * 8 + 2, 3)
    assert faces.shape == (4 * 8, 3)
    assert_bounds(vertices, [-0.1, -0.1, 0.0], [0.1, 0.1, 1.0])
    assert_closed_and_outward(vertices, faces)
    octagon = 0.5 * 8 * 0.1**2 * math.sin(2 * math.pi / 8)
    assert signed_volume(vertices, faces) == pytest.approx(octagon, abs=1e-12)


def test_tube_mitered_bend_is_watertight_and_circumferential():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    vertices, faces = geometry("cockpit.tube", points=path.tolist(), radius=0.05, sides=6)
    assert vertices.shape == (3 * 6 + 2, 3)
    assert faces.shape == (2 * 2 * 6 + 2 * 6, 3)
    assert count_boundary_edges(faces) == 0
    assert_closed_and_outward(vertices, faces)
    # Every cross-section vertex is exactly one radius away from its path point.
    np.testing.assert_allclose(vertices[-2:], [path[0], path[-1]], atol=1e-12)
    distances = np.linalg.norm(vertices[:-2, None, :] - path[None, :, :], axis=2).min(axis=1)
    np.testing.assert_allclose(distances, 0.05, atol=1e-12)
    # A planar path keeps every section in the path plane.
    assert np.abs(vertices[:, 2]).max() == pytest.approx(0.05, abs=1e-12)
    assert np.all(np.abs(vertices[:, 2]) <= 0.05 + 1e-12)


def test_tube_closed_ring_is_watertight_and_has_positive_volume():
    square = [[0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0]]
    vertices, faces = geometry("cockpit.tube", points=square, radius=0.1, sides=6, closed=True)
    assert vertices.shape == (4 * 6, 3)
    assert faces.shape == (2 * 4 * 6, 3)
    assert count_boundary_edges(faces) == 0
    groups = assert_closed_and_outward(vertices, faces)
    assert len(groups) == 1
    corner = 0.5 + 0.1 / math.sqrt(2)
    np.testing.assert_allclose(vertices[:, :2].min(axis=0), [-corner, -corner], atol=1e-12)
    np.testing.assert_allclose(vertices[:, :2].max(axis=0), [corner, corner], atol=1e-12)
    assert np.abs(vertices[:, 2]).max() <= 0.1 + 1e-12
    assert signed_volume(vertices, faces) > 0


def test_tube_without_caps_is_open_and_allowed():
    vertices, faces = geometry(
        "cockpit.tube", points=[[0, 0, 0], [1, 0, 0], [1, 0, 1]], radius=0.05, sides=6, caps=False
    )
    assert vertices.shape == (3 * 6, 3)
    assert faces.shape == (2 * 2 * 6, 3)
    # An open tube keeps the two end rings as boundary loops.
    assert count_boundary_edges(faces) == 2 * 6
    assert all(value <= 2 for value in edge_counts(faces).values())


def test_tube_with_caps_is_closed():
    vertices, faces = geometry(
        "cockpit.tube", points=[[0, 0, 0], [1, 0, 0], [1, 0, 1]], radius=0.05, sides=6
    )
    assert vertices.shape == (3 * 6 + 2, 3)
    assert count_boundary_edges(faces) == 0
    assert_closed_and_outward(vertices, faces)


def test_tube_is_deterministic():
    assert_deterministic(
        "cockpit.tube",
        points=[[0, 0, 0], [0.5, 0.2, 0.3], [0.5, 0.9, 0.6], [1.2, 0.9, 0.6]],
        radius=0.04,
        sides=10,
    )


def test_tube_rejects_sharp_reversal_with_a_clear_message():
    before = APP.engine.snapshot()
    with pytest.raises(ScissorsError) as failure:
        APP.call(
            "mesh.create_tube",
            scene_id=SCENE,
            points=[[0, 0, 0], [1, 0, 0], [0.5, 0, 0]],
            radius=0.05,
        )
    assert "reverses direction" in failure.value.details["reason"]
    assert APP.engine.snapshot() == before


@pytest.mark.parametrize(
    "arguments",
    [
        {"points": [[0, 0, 0]], "radius": 0.1},
        {"points": [[float(i), 0, 0] for i in range(257)], "radius": 0.1},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": -0.1},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0.1, "sides": 2},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0.1, "sides": 33},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0.1, "caps": "yes"},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0.1, "closed": 1},
        {"points": [[0, 0], [0, 0, 1]], "radius": 0.1},
        {"points": [[0, 0, 0], [0, 0, float("nan")]], "radius": 0.1},
        {"points": [[0, 0, 0], [0, 0, 0]], "radius": 0.1},
        {"points": {"a": 1}, "radius": 0.1},
        {"points": [[0, 0, 0], [0, 0, 1]], "radius": 0.1, "name": ""},
    ],
)
def test_tube_rejects_invalid_arguments(arguments):
    assert_rejected("cockpit.tube", **arguments)


# -------------------------------------------------------------------- cockpit.louver


def test_louver_counts_bounds_and_shells():
    vertices, faces = geometry("cockpit.louver", width=0.2, height=0.1, blade_count=8)
    assert vertices.shape == (8 * 8 + 16, 3)
    assert faces.shape == (8 * 12 + 32, 3)
    # The frame defines the outer bounds; the blades stay inside it.
    assert_bounds(vertices, [-0.1, -0.05, 0.0], [0.1, 0.05, 0.02])
    groups = assert_closed_and_outward(vertices, faces)
    assert len(groups) == 9
    assert_shells_do_not_intersect(vertices, faces)


def test_louver_without_frame_fills_the_opening():
    vertices, faces = geometry(
        "cockpit.louver",
        width=0.1,
        height=0.1,
        blade_count=4,
        blade_angle_degrees=0,
        frame_thickness=0,
        depth=0.01,
    )
    assert vertices.shape == (4 * 8, 3)
    assert faces.shape == (4 * 12, 3)
    # Flat blades: 2 % of the opening stays free on every side.
    assert_bounds(vertices, [-0.049, -0.048, 0.003], [0.049, 0.048, 0.007])
    groups = assert_closed_and_outward(vertices, faces)
    assert len(groups) == 4
    assert_shells_do_not_intersect(vertices, faces)


@pytest.mark.parametrize("angle", [0, 30, -45, 80, -80])
def test_louver_blades_never_touch_for_documented_angles(angle):
    vertices, faces = geometry(
        "cockpit.louver", width=0.16, height=0.08, blade_count=6, blade_angle_degrees=angle
    )
    assert_closed_and_outward(vertices, faces)
    assert_shells_do_not_intersect(vertices, faces)


def test_louver_flat_blades_keep_a_gap_in_y():
    vertices, faces = geometry(
        "cockpit.louver",
        width=0.16,
        height=0.08,
        blade_count=6,
        blade_angle_degrees=0,
        frame_thickness=0,
    )
    groups = shells(faces)
    assert len(groups) == 6
    spans = []
    for group in groups:
        part = vertices[faces[group]].reshape(-1, 3)
        spans.append((part[:, 1].min(), part[:, 1].max()))
    spans.sort()
    for (_, upper), (lower, _) in zip(spans, spans[1:], strict=False):
        assert lower > upper, "flat blades must stay separated"


def test_louver_is_deterministic():
    assert_deterministic("cockpit.louver", width=0.24, height=0.12, blade_count=5, depth=0.014)


@pytest.mark.parametrize(
    "arguments",
    [
        {"width": 0, "height": 0.1, "blade_count": 4},
        {"width": 0.1, "height": -0.1, "blade_count": 4},
        {"width": 0.1, "height": 0.1, "blade_count": 0},
        {"width": 0.1, "height": 0.1, "blade_count": 41},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "blade_angle_degrees": 81},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "blade_angle_degrees": -90},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "blade_thickness": 0},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "frame_thickness": -0.001},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "frame_thickness": 0.05},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "frame_depth": 0},
        {"width": 0.1, "height": 0.1, "blade_count": 4, "depth": 0},
        {
            "width": 0.1,
            "height": 0.001,
            "blade_count": 4,
            "frame_thickness": 0,
            "blade_thickness": 0.01,
        },
    ],
)
def test_louver_rejects_invalid_arguments(arguments):
    assert_rejected("cockpit.louver", **arguments)


# --------------------------------------------------------------- cockpit.switch_grid


def test_switch_grid_counts_bounds_and_shells():
    vertices, faces = geometry(
        "cockpit.switch_grid", columns=3, rows=2, spacing=0.012, base_width=0.06, base_height=0.04
    )
    assert vertices.shape == (8 + 6 * (2 * 12 + 2), 3)
    assert faces.shape == (12 + 6 * 4 * 12, 3)
    tip = 0.016 * math.cos(TOGGLE_TILT) + TOGGLE_RADIUS * math.sin(TOGGLE_TILT)
    assert_bounds(vertices, [-0.03, -0.02, -0.004], [0.03, 0.02, tip])
    groups = assert_closed_and_outward(vertices, faces)
    assert len(groups) == 7


def test_switch_grid_toggles_stand_on_the_top_surface():
    vertices, faces = geometry(
        "cockpit.switch_grid",
        columns=3,
        rows=2,
        spacing=0.012,
        base_width=0.06,
        base_height=0.04,
    )
    lowest = vertices[
        np.isclose(vertices[:, 2], -TOGGLE_RADIUS * math.sin(TOGGLE_TILT), atol=1e-15)
    ]
    assert lowest.shape[0] == 6  # one lowest base vertex per toggle
    for column in range(3):
        for row in range(2):
            x = (column - 1) * 0.012
            y = (row - 0.5) * 0.012
            # The base circle is centered on the grid point at z = 0 and tilted
            # around +X, so its lowest vertex is r * (-cos(tilt), -sin(tilt)).
            expected = np.array(
                [
                    x,
                    y - TOGGLE_RADIUS * math.cos(TOGGLE_TILT),
                    -TOGGLE_RADIUS * math.sin(TOGGLE_TILT),
                ]
            )
            assert any(np.allclose(vertex, expected, atol=1e-12) for vertex in lowest)
    plate = vertices[:8]
    assert int(np.isclose(plate[:, 2], 0.0).sum()) == 4
    assert int(np.isclose(plate[:, 2], -0.004).sum()) == 4


def test_switch_grid_toggles_are_disjoint_and_only_pierce_the_plate():
    vertices, faces = geometry(
        "cockpit.switch_grid", columns=3, rows=2, spacing=0.012, base_width=0.06, base_height=0.04
    )
    groups = shells(faces)
    triangles = [vertices[faces[group]] for group in groups]
    volumes = [abs(signed_volume(vertices, faces[group])) for group in groups]
    plate = volumes.index(max(volumes))
    assert plate == 0, "the plate is built first and is the largest shell"
    for index, one in enumerate(triangles):
        for other_index, other in enumerate(triangles):
            if other_index <= index or plate in {index, other_index}:
                continue
            for first in one:
                for second in other:
                    assert not _triangles_intersect(first, second), "toggles must not touch"
    # The documented penetration is real, and it stays inside the plate thickness.
    assert any(
        _triangles_intersect(first, second) for first in triangles[plate] for second in triangles[1]
    )
    assert vertices[:, 2].min() == pytest.approx(-0.004, abs=1e-12)


def test_switch_grid_is_deterministic():
    assert_deterministic(
        "cockpit.switch_grid", columns=2, rows=3, spacing=0.02, base_width=0.08, base_height=0.1
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"columns": 0, "rows": 1, "spacing": 0.01, "base_width": 0.05, "base_height": 0.05},
        {"columns": 13, "rows": 1, "spacing": 0.01, "base_width": 0.05, "base_height": 0.05},
        {"columns": 1, "rows": 0, "spacing": 0.01, "base_width": 0.05, "base_height": 0.05},
        {"columns": 1, "rows": 13, "spacing": 0.01, "base_width": 0.05, "base_height": 0.05},
        {"columns": 1, "rows": 1, "spacing": 0, "base_width": 0.05, "base_height": 0.05},
        {"columns": 1, "rows": 1, "spacing": 0.01, "base_width": 0, "base_height": 0.05},
        {"columns": 1, "rows": 1, "spacing": 0.01, "base_width": 0.05, "base_height": -0.05},
        {
            "columns": 1,
            "rows": 1,
            "spacing": 0.01,
            "base_width": 0.05,
            "base_height": 0.05,
            "base_thickness": 0,
        },
        {
            "columns": 1,
            "rows": 1,
            "spacing": 0.01,
            "base_width": 0.05,
            "base_height": 0.05,
            "toggle_length": 0,
        },
        {
            "columns": 1,
            "rows": 1,
            "spacing": 0.01,
            "base_width": 0.05,
            "base_height": 0.05,
            "toggle_radius": -1,
        },
        {"columns": 2.5, "rows": 1, "spacing": 0.01, "base_width": 0.05, "base_height": 0.05},
    ],
)
def test_switch_grid_rejects_invalid_arguments(arguments):
    assert_rejected("cockpit.switch_grid", **arguments)


# ---------------------------------------------------- package contract (manifest/code)


def test_native_catalog_matches_the_code():
    for name in ("rounded_box", "tube", "louver", "switch_grid"):
        operation = APP.registry.describe_operation(f"mesh.create_{name}")
        assert operation["plugin"] is None
        assert operation["side_effects"] is True
        assert operation["permissions"] == ["scene.write"]
        schema = operation["parameters"]
        assert schema["additionalProperties"] is False
        executor = APP.registry.operations[operation["name"]].executor
        for parameter, value in inspect.signature(executor).parameters.items():
            assert parameter in schema["properties"]
            if value.default is inspect.Parameter.empty:
                assert parameter in schema["required"]
            else:
                assert schema["properties"][parameter].get("default") == value.default


def test_native_geometry_has_no_plugin_dependency():
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and name.startswith("scissors.plugins") for name in imports)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"eval", "exec", "compile", "__import__", "open"}


@pytest.mark.parametrize("radius", [0, 0.1, 0.5])
def test_native_rounded_box_edge_bevel_closes_when_corner_collapses(radius):
    result = APP.call(
        "mesh.create_rounded_box",
        scene_id=SCENE,
        size=[1, 1.5, 0.6],
        corner_radius=radius,
        edge_bevel=0.15,
    )
    obj = APP.engine.object(result["object_id"])[1]
    assert_closed_and_outward(np.asarray(obj.vertices), np.asarray(obj.faces))
    assert_bounds(np.asarray(obj.vertices), [-0.5, -0.75, 0], [0.5, 0.75, 0.6])


@pytest.mark.parametrize("angle", [-40, 0, 40])
def test_native_switch_grid_uses_requested_tilt(angle):
    result = APP.call(
        "mesh.create_switch_grid",
        scene_id=SCENE,
        columns=1,
        rows=1,
        spacing=0.1,
        base_width=0.1,
        base_height=0.1,
        toggle_angle_degrees=angle,
    )
    obj = APP.engine.object(result["object_id"])[1]
    vertices = np.asarray(obj.vertices)
    axis = np.array([0, -math.sin(math.radians(angle)), math.cos(math.radians(angle))])
    np.testing.assert_allclose(vertices[8], [0, 0, 0])
    np.testing.assert_allclose(vertices[-1], axis * 0.016)
