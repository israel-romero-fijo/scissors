"""Measured world-axis resizing around one common subtree anchor."""

import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject

MAX_RESIZE_OBJECTS = 4096
DIMENSION_RTOL = 1e-9


def mesh_bounds(objects: list[MeshObject]) -> np.ndarray:
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    found = False
    for obj in objects:
        if obj.kind == "mesh" and obj.vertices:
            points = np.asarray(obj.vertices, dtype=float)
            low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
            found = True
    if not found:
        raise ScissorsError("GEOMETRY_ERROR", "Resize requires mesh geometry in the target subtree")
    result = np.asarray([low, high])
    if not np.isfinite(result).all():
        raise ScissorsError("GEOMETRY_ERROR", "Resize bounds are not finite")
    return result


def resize(
    engine: Engine,
    object_id: str,
    dimensions: dict[str, float],
    anchor: str | dict[str, Any],
    aspect_policy: str,
) -> dict[str, Any]:
    if not dimensions:
        raise ValueError("Specify at least one non-null dimension")
    scene, root = engine.object(object_id)
    if root.kind not in {"mesh", "group"}:
        raise ValueError("Resize target must be a mesh or group")
    deadline = time.monotonic() + engine.settings.operation_timeout
    objects = [root, *engine.descendants(scene, object_id)]
    if len(objects) > MAX_RESIZE_OBJECTS:
        raise ScissorsError("RESOURCE_LIMIT", "Resize exceeds 4096 objects")
    vertices = sum(len(obj.vertices) for obj in objects)
    faces = sum(len(obj.faces) for obj in objects)
    if (
        vertices > engine.settings.max_vertices
        or faces > engine.settings.max_faces
        or vertices * 768 + faces * 384 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError(
            "RESOURCE_LIMIT", "Resize exceeds aggregate geometry or working memory budget"
        )
    old = mesh_bounds(objects)
    extents = old[1] - old[0]
    center = old[0] / 2 + old[1] / 2
    if isinstance(anchor, str):
        point = center.copy()
        if anchor != "center":
            side, axis = anchor.split("_")
            index = "xyz".index(axis)
            point[index] = old[0 if side == "min" else 1, index]
            resolved_anchor: dict[str, Any] = {
                "type": "plane",
                "name": anchor,
                "axis": axis,
                "coordinate": float(point[index]),
            }
        else:
            resolved_anchor = {"type": "center"}
    else:
        point = np.asarray(anchor["position"], dtype=float)
        resolved_anchor = {"type": "point"}
    resolved_anchor["position"] = point.tolist()
    ratios = []
    factors = np.ones(3)
    for axis, value in sorted(dimensions.items()):
        index = "xyz".index(axis)
        if extents[index] <= 0 or not np.isfinite(extents[index]):
            raise ScissorsError(
                "GEOMETRY_ERROR", "Cannot resize a zero or nonfinite source extent", {"axis": axis}
            )
        factor = value / extents[index]
        if not np.isfinite(factor) or factor <= 0 or not np.isfinite(1 / factor):
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Resize scale is singular or outside numerical range",
                {"axis": axis},
            )
        factors[index] = factor
        ratios.append(factor)
    if aspect_policy == "uniform":
        if not np.allclose(ratios, ratios[0], rtol=DIMENSION_RTOL, atol=0):
            raise ValueError("Uniform resize dimensions require the same scale ratio")
        factors[:] = ratios[0]
    # Work with a delta on baked world vertices. The recorded transform is diagnostic.
    matrix = np.diag([*factors, 1.0])
    matrix[:3, 3] = (1 - factors) * point
    expected = old * factors + matrix[:3, 3]
    if not np.isfinite(matrix).all() or not np.isfinite(expected).all():
        raise ScissorsError(
            "GEOMETRY_ERROR", "Resize would produce nonfinite bounds or translation"
        )
    numeric_tolerance = float(
        max(1.0, np.abs(expected).max(), np.abs(matrix[:3, 3]).max()) * np.finfo(float).eps * 32
    )
    if any(value <= numeric_tolerance * 4 for value in dimensions.values()):
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Requested dimensions are below world-coordinate precision",
            {"numeric_tolerance": numeric_tolerance},
        )
    changed = not np.array_equal(factors, np.ones(3))
    if time.monotonic() > deadline:
        raise ScissorsError("OPERATION_TIMEOUT", "Resize preparation timed out")
    if changed:
        engine.transform(object_id, matrix)
    actual = mesh_bounds(objects)
    new_extents = actual[1] - actual[0]
    if np.any((extents > 0) & (new_extents <= 0)) or not np.allclose(
        actual, expected, rtol=0, atol=numeric_tolerance
    ):
        raise ScissorsError("GEOMETRY_ERROR", "Resize bounds could not be represented accurately")
    for axis, value in dimensions.items():
        if not np.isclose(
            new_extents["xyz".index(axis)], value, rtol=DIMENSION_RTOL, atol=numeric_tolerance
        ):
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Resize did not reach the requested dimension",
                {"axis": axis, "requested": value, "actual": float(new_extents["xyz".index(axis)])},
            )
    for obj in objects:
        if obj.kind == "mesh":
            # Geometry validation rejects degenerate faces. Also reject numerical
            # overflow rather than persisting nonfinite areas or normals.
            if (
                not np.isfinite(obj.geometry().area_faces).all()
                or not np.isfinite(obj.normals).all()
            ):
                raise ScissorsError(
                    "GEOMETRY_ERROR",
                    "Resize produced nonfinite geometry diagnostics",
                    {"object_id": obj.id},
                )
    if time.monotonic() > deadline:
        raise ScissorsError("OPERATION_TIMEOUT", "Resize exceeded its time budget")
    ids = sorted(obj.id for obj in objects)
    return {
        "object_id": object_id,
        "axes": "world",
        "aspect_policy": aspect_policy,
        "requested_dimensions": dimensions,
        "old_bounds": old.tolist(),
        "new_bounds": actual.tolist(),
        "old_dimensions": dict(zip("xyz", extents.tolist(), strict=True)),
        "new_dimensions": dict(zip("xyz", new_extents.tolist(), strict=True)),
        "anchor": resolved_anchor,
        "scale_factors": factors.tolist(),
        "delta_matrix": matrix.tolist(),
        "changed_ids": ids if changed else [],
        "unchanged_ids": [] if changed else ids,
        "bounds_mesh_ids": sorted(obj.id for obj in objects if obj.kind == "mesh"),
        "numeric_tolerance": numeric_tolerance,
        "dimension_relative_tolerance": DIMENSION_RTOL,
        "warnings": [],
    }
