"""Deterministic planar polygon validation, hole bridges and ear clipping."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]) / 2)


def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    values = [cross(b - a, c - a), cross(b - a, d - a), cross(d - c, a - c), cross(d - c, b - c)]
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    for value, point, start, end in (
        (values[0], c, a, b),
        (values[1], d, a, b),
        (values[2], a, c, d),
        (values[3], b, c, d),
    ):
        if (
            abs(value) <= 1e-12
            and np.all(point >= np.minimum(start, end) - 1e-12)
            and np.all(point <= np.maximum(start, end) + 1e-12)
        ):
            return True
    return False


def validate_curve_path(path: np.ndarray, closed: bool, settings: Settings) -> None:
    count = len(path) if closed else len(path) - 1
    if count * count > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Curve crossing checks exceed pair budget")
    deadline = time.monotonic() + settings.operation_timeout
    for i in range(count):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Curve crossing checks timed out")
        for j in range(i + 2, count):
            if closed and i == 0 and j == count - 1:
                continue
            if intersects(path[i], path[(i + 1) % len(path)], path[j], path[(j + 1) % len(path)]):
                raise ScissorsError("GEOMETRY_ERROR", "Curve edges intersect", {"edges": [i, j]})


def inside(point: np.ndarray, ring: np.ndarray) -> bool:
    hit = False
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
            if point[0] < x:
                hit = not hit
    return hit


def normalize(points: list[list[float]], clockwise: bool, label: str) -> tuple[np.ndarray, int]:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ScissorsError(
            "GEOMETRY_ERROR", "Contour points must be finite pairs", {"contour": label}
        )
    result: list[np.ndarray] = []
    for point in value:
        if not result or not np.array_equal(result[-1], point):
            result.append(point)
    if len(result) > 1 and np.array_equal(result[0], result[-1]):
        result.pop()
    if len(result) < 3:
        raise ScissorsError(
            "GEOMETRY_ERROR", "Contour requires three distinct points", {"contour": label}
        )
    ring = np.asarray(result)
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        for j in range(i + 1, len(ring)):
            if j == i + 1 or (i == 0 and j == len(ring) - 1):
                continue
            if intersects(a, b, ring[j], ring[(j + 1) % len(ring)]):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Contour edges intersect", {"contour": label, "edges": [i, j]}
                )
    if abs(area(ring)) < 1e-12:
        raise ScissorsError("GEOMETRY_ERROR", "Contour has zero area", {"contour": label})
    if (area(ring) < 0) != clockwise:
        ring = ring[::-1].copy()
    return ring, len(value) - len(ring)


def triangulate(
    points: list[list[float]], holes: list[list[list[float]]], settings: Settings
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[str]]:
    if len(points) > 512 or len(holes) > 32 or any(len(hole) > 512 for hole in holes):
        raise ScissorsError("RESOURCE_LIMIT", "Profile contour limits exceeded")
    total = len(points) + sum(len(hole) for hole in holes)
    if total > settings.max_vertices or total * total > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Profile validation exceeds pair budget")
    outer, removed = normalize(points, False, "outer")
    rings = [outer]
    for index, points_hole in enumerate(holes):
        ring, duplicates = normalize(points_hole, True, f"hole {index}")
        removed += duplicates
        if not inside(ring[0], outer):
            raise ScissorsError(
                "GEOMETRY_ERROR", "Hole lies outside the outer contour", {"hole": index}
            )
        for existing in rings:
            if existing is not outer and (inside(ring[0], existing) or inside(existing[0], ring)):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Holes overlap or are nested", {"hole": index}
                )
            for i, a in enumerate(ring):
                for j, c in enumerate(existing):
                    if intersects(
                        a, ring[(i + 1) % len(ring)], c, existing[(j + 1) % len(existing)]
                    ):
                        raise ScissorsError(
                            "GEOMETRY_ERROR",
                            "Hole touches or crosses another contour",
                            {"hole": index, "edges": [i, j]},
                        )
        rings.append(ring)
    vertices = np.concatenate(rings)
    indices: list[list[int]] = []
    offset = 0
    for ring in rings:
        indices.append(list(range(offset, offset + len(ring))))
        offset += len(ring)
    polygon = list(indices[0])
    for hole_index, hole in enumerate(indices[1:], start=1):
        candidates = []
        for position, oid in enumerate(polygon):
            for hi, hid in enumerate(hole):
                a, b = vertices[oid], vertices[hid]
                midpoint = (a + b) / 2
                if not inside(midpoint, outer) or any(inside(midpoint, ring) for ring in rings[1:]):
                    continue
                blocked = False
                for boundary in [polygon, *indices[1:]]:
                    for j, cid in enumerate(boundary):
                        did = boundary[(j + 1) % len(boundary)]
                        if oid in {cid, did} or hid in {cid, did}:
                            continue
                        if intersects(a, b, vertices[cid], vertices[did]):
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    candidates.append((float(np.sum((a - b) ** 2)), position, hi))
        if not candidates:
            raise ScissorsError(
                "GEOMETRY_ERROR", "No visible bridge for hole", {"hole": hole_index - 1}
            )
        _, position, hi = min(candidates)
        cycle = hole[hi:] + hole[:hi]
        polygon = (
            polygon[: position + 1]
            + cycle
            + [cycle[0], polygon[position]]
            + polygon[position + 1 :]
        )
    faces: list[list[int]] = []
    while len(polygon) > 3:
        found = False
        for i, current in enumerate(polygon):
            previous, following = polygon[i - 1], polygon[(i + 1) % len(polygon)]
            a, b, c = vertices[[previous, current, following]]
            if cross(b - a, c - b) <= 1e-12:
                continue
            blocked = False
            for other in polygon:
                point = vertices[other]
                if other in {previous, current, following} or any(
                    np.array_equal(point, vertex) for vertex in (a, b, c)
                ):
                    continue
                if (
                    cross(b - a, point - a) >= -1e-12
                    and cross(c - b, point - b) >= -1e-12
                    and cross(a - c, point - c) >= -1e-12
                ):
                    blocked = True
                    break
            if not blocked:
                faces.append([previous, current, following])
                polygon.pop(i)
                found = True
                break
        if not found:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Polygon triangulation stalled; check collinear points"
            )
    if len(set(polygon)) != 3 or abs(area(vertices[polygon])) <= 1e-12:
        raise ScissorsError("GEOMETRY_ERROR", "Polygon triangulation produced a degenerate cap")
    faces.append(polygon)
    warnings = [f"Removed {removed} consecutive duplicate contour points."] if removed else []
    return vertices, np.asarray(faces), indices, warnings


def embed(points: np.ndarray, plane: str) -> np.ndarray:
    vertices = np.zeros((len(points), 3))
    axes = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}[plane]
    vertices[:, axes] = points
    return vertices


def profile_geometry(
    points: list[list[float]], holes: list[list[list[float]]], plane: str, settings: Settings
) -> tuple[trimesh.Trimesh, list[list[float]], dict[str, Any], list[str]]:
    flat, faces, rings, warnings = triangulate(points, holes, settings)
    if plane == "xz":
        faces = faces[:, ::-1]
        rings = [ring[::-1] for ring in rings]
    uv = (flat - flat.min(axis=0)) / np.maximum(np.ptp(flat, axis=0), 1e-12)
    mesh = trimesh.Trimesh(vertices=embed(flat, plane), faces=faces, process=False)
    return (
        mesh,
        uv.tolist(),
        {"plane": plane, "boundary_rings": rings, "hole_count": len(holes)},
        warnings,
    )


def resample(points: np.ndarray, count: int, closed: bool = True) -> np.ndarray:
    path = np.concatenate([points, points[:1]]) if closed else points
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 1e-12):
        raise ScissorsError("GEOMETRY_ERROR", "Resampling requires distinct consecutive points")
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    samples = np.linspace(0, cumulative[-1], count, endpoint=not closed)
    return np.column_stack([np.interp(samples, cumulative, path[:, axis]) for axis in range(3)])


def catmull(points: np.ndarray, samples: int, closed: bool) -> np.ndarray:
    extended = (
        np.concatenate([points[-1:], points, points[:2]])
        if closed
        else np.concatenate([points[:1], points, points[-1:]])
    )
    result = []
    for i in range(len(points) if closed else len(points) - 1):
        a, b, c, d = extended[i : i + 4]
        for t in np.linspace(0, 1, samples, endpoint=False):
            result.append(
                0.5
                * (
                    (2 * b)
                    + (-a + c) * t
                    + (2 * a - 5 * b + 4 * c - d) * t**2
                    + (-a + 3 * b - 3 * c + d) * t**3
                )
            )
    if not closed:
        result.append(points[-1])
    return np.asarray(result)
