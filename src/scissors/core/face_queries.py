"""Revision-bound selections and bounded material-interior thickness queries."""

from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject


def selection_indices(obj: MeshObject, selection: dict[str, Any]) -> np.ndarray:
    if selection["object_id"] != obj.id or selection["geometry_revision"] != obj.geometry_revision:
        raise ScissorsError("INVALID_PARAMETER", "Selection belongs to another geometry revision")
    indices = np.asarray(selection["face_indices"], dtype=np.int64)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(obj.faces)):
        raise ScissorsError("INVALID_PARAMETER", "Selection must contain valid faces")
    if len(np.unique(indices)) != len(indices):
        raise ScissorsError("INVALID_PARAMETER", "Selection contains duplicate face indices")
    return indices


def triangle_box(triangles: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Separating-axis test, including degenerate box dimensions."""
    points = triangles - (low + high) / 2
    half = (high - low) / 2
    edges = np.roll(points, -1, axis=1) - points
    axes = [np.broadcast_to(axis, (len(points), 3)) for axis in np.eye(3)]
    axes.append(np.cross(edges[:, 0], edges[:, 1]))
    axes.extend(np.cross(edges[:, edge], axis) for edge in range(3) for axis in np.eye(3))
    result = np.ones(len(points), dtype=bool)
    for axis in axes:
        projection = np.einsum("nvc,nc->nv", points, axis)
        radius = np.abs(axis) @ half
        result &= (projection.min(axis=1) <= radius) & (projection.max(axis=1) >= -radius)
    return result


def select_faces(
    obj: MeshObject,
    normal: dict[str, Any] | None,
    region: dict[str, Any] | None,
    tolerance: float,
    settings: Settings,
) -> dict[str, Any]:
    if obj.kind != "mesh":
        raise ValueError("Face selection requires a mesh")
    if len(obj.faces) * 1024 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Face selection exceeds memory budget")
    mesh = obj.geometry()
    mask = np.ones(len(obj.faces), dtype=bool)
    if normal is not None:
        direction = np.asarray(normal["direction"], dtype=float)
        if not np.isclose(np.linalg.norm(direction), 1, atol=1e-6, rtol=0):
            raise ValueError("Selection direction must be a unit vector")
        mask &= (
            mesh.face_normals @ direction
            >= np.cos(np.radians(normal["angular_tolerance_degrees"])) - 1e-12
        )
    if region is not None:
        low, high = np.asarray(region["bounds"], dtype=float)
        if np.any(low > high):
            raise ValueError("Region bounds are inverted")
        low, high = low - tolerance, high + tolerance
        points = mesh.triangles
        rule = region["rule"]
        if rule == "centroid_inside":
            mask &= np.all((points.mean(axis=1) >= low) & (points.mean(axis=1) <= high), axis=1)
        elif rule == "fully_inside":
            mask &= np.all((points >= low) & (points <= high), axis=(1, 2))
        else:
            if len(points) * 13 > settings.max_geometry_pairs:
                raise ScissorsError("RESOURCE_LIMIT", "Triangle-box query exceeds pair budget")
            mask &= triangle_box(points, low, high)
    indices = np.flatnonzero(mask)
    points = mesh.triangles[indices].reshape(-1, 3)
    return {
        "object_id": obj.id,
        "geometry_revision": obj.geometry_revision,
        "face_indices": indices.tolist(),
        "count": len(indices),
        "empty": not len(indices),
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
        if len(points)
        else None,
        "predicates": {
            "normal": normal,
            "region": region,
            "tolerance": tolerance,
            "combine": "and",
        },
        "preview": {
            "operation": "scene.preview",
            "highlights": [
                {
                    "object_id": obj.id,
                    "geometry_revision": obj.geometry_revision,
                    "face_indices": indices.tolist(),
                }
            ],
        },
    }


def check_thickness(
    obj: MeshObject,
    minimum: float,
    tolerance: float,
    settings: Settings,
    selection: dict[str, Any] | None = None,
    rim_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from scissors.core.thickness import check_thickness as inspect

    return inspect(obj, minimum, tolerance, settings, selection, rim_selection)
