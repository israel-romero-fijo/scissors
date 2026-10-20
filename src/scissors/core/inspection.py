"""Bounded triangle queries with exact feature distances and solid overlap."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_boolean
from scissors.core.mesh import MeshObject, topology_mesh


def point_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    t = np.sum((points - start) * vector, axis=-1) / np.maximum(
        np.sum(vector * vector, axis=-1), 1e-30
    )
    return np.linalg.norm(points - start - np.clip(t, 0, 1)[..., None] * vector, axis=-1)


def segment_triangle(start: np.ndarray, end: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return whether each finite segment crosses its paired triangle."""
    direction = end - start
    edge1, edge2 = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    cross = np.cross(direction, edge2)
    determinant = np.sum(edge1 * cross, axis=1)
    valid = np.abs(determinant) > 1e-15
    inverse = np.divide(1, determinant, out=np.zeros_like(determinant), where=valid)
    offset = start - triangle[:, 0]
    u = np.sum(offset * cross, axis=1) * inverse
    q = np.cross(offset, edge1)
    v = np.sum(direction * q, axis=1) * inverse
    t = np.sum(edge2 * q, axis=1) * inverse
    return valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)


def triangle_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Exact minimum from vertex-face, edge-edge and edge-face candidates."""
    distance = np.full(len(first), np.inf)
    for corner in range(3):
        for points, triangles in ((first[:, corner], second), (second[:, corner], first)):
            closest = trimesh.triangles.closest_point(triangles, points)
            distance = np.minimum(distance, np.linalg.norm(points - closest, axis=1))
        a, b = first[:, corner], first[:, (corner + 1) % 3]
        distance[segment_triangle(a, b, second)] = 0
        distance[segment_triangle(second[:, corner], second[:, (corner + 1) % 3], first)] = 0
        for other in range(3):
            c, d = second[:, other], second[:, (other + 1) % 3]
            for points, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
                distance = np.minimum(distance, point_segment(points, start, end))
            u, v, w = b - a, d - c, a - c
            aa, bb, cc = np.sum(u * u, axis=1), np.sum(u * v, axis=1), np.sum(v * v, axis=1)
            dd, ee = np.sum(u * w, axis=1), np.sum(v * w, axis=1)
            denominator = aa * cc - bb * bb
            valid = np.abs(denominator) > 1e-30
            s = np.divide(
                bb * ee - cc * dd, denominator, out=np.zeros_like(denominator), where=valid
            )
            t = np.divide(
                aa * ee - bb * dd, denominator, out=np.zeros_like(denominator), where=valid
            )
            valid &= (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
            value = np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1)
            distance = np.minimum(distance, np.where(valid, value, np.inf))
    return distance


def surface_distance(first: MeshObject, second: MeshObject, settings: Settings) -> float:
    a, b = first.geometry().triangles, second.geometry().triangles
    count = len(a) * len(b)
    if count > settings.max_geometry_pairs:
        raise ScissorsError(
            "RESOURCE_LIMIT",
            "Triangle query exceeds pair budget",
            {"pairs": count, "limit": settings.max_geometry_pairs},
        )
    chunk = min(4096, max(1, settings.max_memory_bytes // 8192))
    result = float("inf")
    for start in range(0, count, chunk):
        indices = np.arange(start, min(start + chunk, count))
        values = triangle_distances(a[indices // len(b)], b[indices % len(b)])
        result = min(result, float(values.min()))
        if result == 0:
            break
    return result


def intersection(
    first: MeshObject, second: MeshObject, settings: Settings, tolerance: float
) -> dict[str, Any]:
    if first.kind != "mesh" or second.kind != "mesh":
        raise ValueError("Intersection requires two meshes")
    a, b = topology_mesh(first.geometry()), topology_mesh(second.geometry())
    distance = surface_distance(first, second, settings)
    volume: float | None = None
    bounds_overlap = np.all(a.bounds[1] >= b.bounds[0]) and np.all(b.bounds[1] >= a.bounds[0])
    if a.is_volume and b.is_volume:
        volume = (
            abs(float(run_boolean([a, b], "intersection", settings).volume))
            if bounds_overlap
            else 0.0
        )
    penetration = volume is not None and volume > tolerance**3
    return {
        "object_ids": [first.id, second.id],
        "surface_distance": distance,
        "surface_contact": distance <= tolerance,
        "penetration": penetration,
        "intersects": penetration or distance <= tolerance,
        "overlap_volume": volume,
        "solid_classification": "closed oriented volumes" if volume is not None else "unavailable",
        "warnings": [] if volume is not None else ["Open meshes support surface contact only."],
    }


def contact_components(
    object_ids: list[str], collisions: list[dict[str, Any]], anchors: list[str] | None
) -> dict[str, Any]:
    """Classify spatial connectivity, with floating defined relative to explicit anchors."""
    neighbors: dict[str, set[str]] = {object_id: set() for object_id in object_ids}
    for contact in collisions:
        first, second = contact["object_ids"]
        neighbors[first].add(second)
        neighbors[second].add(first)
    remaining = set(object_ids)
    components: list[list[str]] = []
    anchor_set = set(anchors or [])
    attached: set[str] = set()
    for start in object_ids:
        if start not in remaining:
            continue
        component, pending = set(), [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        remaining -= component
        components.append([object_id for object_id in object_ids if object_id in component])
        if component & anchor_set:
            attached |= component
    return {
        "connected_components": components,
        "isolated_object_ids": [object_id for object_id in object_ids if not neighbors[object_id]],
        "anchor_object_ids": anchors or [],
        "floating_object_ids": (
            [object_id for object_id in object_ids if object_id not in attached]
            if anchors
            else None
        ),
        "connectivity_method": "surface_contact_or_solid_overlap",
        "connectivity_warnings": [
            "Spatial contact does not prove a mechanical joint or stable physical support.",
            *([] if anchors else ["Provide anchor_object_ids to classify floating parts."]),
        ],
    }


def self_intersections(obj: MeshObject, settings: Settings, tolerance: float) -> dict[str, Any]:
    if obj.kind != "mesh":
        raise ValueError("Self-intersection requires a mesh")
    triangles = obj.geometry().triangles
    count = len(triangles) * (len(triangles) - 1) // 2
    if count > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Self-intersection exceeds pair budget")
    hits: list[list[int]] = []
    checked = 0
    deadline = time.monotonic() + settings.operation_timeout
    for i, triangle in enumerate(triangles[:-1]):
        for start in range(i + 1, len(triangles), 1024):
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Self-intersection query timed out")
            indices = np.arange(start, min(start + 1024, len(triangles)))
            others = triangles[indices]
            # Shared positions identify topology adjacency even at render seams.
            adjacent = np.any(
                np.all(triangle[None, :, None, :] == others[:, None, :, :], axis=-1), axis=(1, 2)
            )
            overlap = np.all(others.max(axis=1) + tolerance >= triangle.min(axis=0), axis=1)
            overlap &= np.all(others.min(axis=1) - tolerance <= triangle.max(axis=0), axis=1)
            indices = indices[~adjacent & overlap]
            if not len(indices):
                continue
            checked += len(indices)
            distances = triangle_distances(
                np.repeat(triangle[None], len(indices), axis=0), triangles[indices]
            )
            hits.extend([i, int(j)] for j in indices[distances <= tolerance])
            if len(hits) > 4096:
                raise ScissorsError("RESOURCE_LIMIT", "Self-intersection result exceeds pair limit")
    return {
        "object_id": obj.id,
        "self_intersects": bool(hits),
        "face_pairs": hits,
        "checked_pairs": checked,
        "adjacent_faces_excluded": True,
    }
