"""World-space texture density and bounded, shared UV island placement."""

import math
import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.exchange import unit_scale
from scissors.core.mesh import MeshObject
from scissors.core.uv import generate_uv


def prepare_density_uv(
    obj: MeshObject, method: str, resolution: list[int], settings: Settings
) -> None:
    """Generate chart shapes in pixel space, retaining metric planar proportions."""
    effective = method
    if method == "auto":
        effective = {
            "sphere": "spherical",
            "cylinder": "cylindrical",
            "cone": "cylindrical",
            "plane": "planar",
        }.get(str(obj.metadata.get("primitive", "custom")), "box")
    span = np.ptp(np.asarray(obj.vertices), axis=0)
    generate_uv(obj, method, settings)
    coords = np.asarray(obj.uv) / resolution
    if effective == "planar":
        coords *= span[:2]
    elif effective == "box":
        for face, normal in zip(obj.faces, obj.geometry().face_normals, strict=True):
            axis = int(np.argmax(np.abs(normal)))
            axes = [i for i in range(3) if i != axis]
            coords[face] *= span[axes]
    obj.uv = coords.tolist()


def density_report(obj: MeshObject, units: str, resolution: list[int]) -> dict[str, Any]:
    """Measure pixel area per physical area; overlapping triangles count separately."""
    if obj.uv is None:
        raise ValueError("Texture density requires UV")
    points = np.asarray(obj.vertices)[obj.faces] * unit_scale(units)
    pixels = np.asarray(obj.uv)[obj.faces] * resolution
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    u, v = pixels[:, 1] - pixels[:, 0], pixels[:, 2] - pixels[:, 0]
    area = np.linalg.norm(np.cross(a, b), axis=1) / 2
    pixel_area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    if not np.isfinite(pixel_area).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Texture density needs finite, nondegenerate geometry and UV")
    density = np.sqrt(pixel_area / area)
    length = np.linalg.norm(a, axis=1)
    tangent = a / length[:, None]
    bx = np.sum(b * tangent, axis=1)
    by = np.linalg.norm(b - bx[:, None] * tangent, axis=1)
    local = np.zeros((len(points), 2, 2))
    local[:, 0, 0], local[:, 0, 1], local[:, 1, 1] = length, bx, by
    singular = np.linalg.svd(np.stack((u, v), axis=2) @ np.linalg.inv(local), compute_uv=False)
    covered = pixel_area > 1e-20
    anisotropy = singular[covered, 0] / singular[covered, 1]
    return {
        "object_id": obj.id,
        "resolution": resolution,
        "surface_area_m2": float(area.sum()),
        "pixel_area_with_overlap": float(pixel_area.sum()),
        "pixels_per_meter": float(np.sqrt(pixel_area.sum() / area.sum())),
        "min_pixels_per_meter": float(density.min()),
        "max_pixels_per_meter": float(density.max()),
        "covered_area_fraction": float(area[covered].sum() / area.sum()),
        "degenerate_uv_triangles": int(np.count_nonzero(~covered)),
        "max_anisotropy": float(anisotropy.max()) if len(anisotropy) else None,
        "measurement": "area-based; overlaps count per surface; image detail is not measured",
    }


def islands(obj: MeshObject, deadline: float) -> list[list[int]]:
    """Connect continuous UV edges; keep folded/overlapping neighbors separate."""
    faces, vertices, uv = np.asarray(obj.faces), np.asarray(obj.vertices), np.asarray(obj.uv)
    parent = list(range(len(faces)))

    def root(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: dict[tuple, list[tuple[int, float]]] = {}
    for index, face in enumerate(faces):
        if index % 1024 == 0 and time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "UV density island query timed out")
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            ka = tuple([*vertices[face[a]], *uv[face[a]]])
            kb = tuple([*vertices[face[b]], *uv[face[b]]])
            key = tuple(sorted((ka, kb)))
            ua, ub = np.asarray(key[0][-2:]), np.asarray(key[1][-2:])
            delta, other = ub - ua, uv[face[c]] - ua
            side = float(delta[0] * other[1] - delta[1] * other[0])
            edges.setdefault(key, []).append((index, side))
    for members in edges.values():
        if len(members) == 2 and members[0][1] * members[1][1] < 0:
            parent[root(members[0][0])] = root(members[1][0])
    groups: dict[int, list[int]] = {}
    for i in range(len(faces)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def fit_density(
    objects: list[MeshObject],
    units: str,
    resolution: list[int],
    target: float,
    mode: str,
    region: list[float],
    padding: int,
    settings: Settings,
) -> dict[str, Any]:
    """Plan every island before applying any UV or seam changes to detached objects."""
    deadline = time.monotonic() + settings.operation_timeout
    if not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
        raise ValueError("Atlas region must be [u0,v0,u1,v1] inside [0,1]")
    if mode == "repeat" and (region != [0, 0, 1, 1] or padding):
        raise ValueError("Repeat mode uses the whole texture and requires padding=0")
    faces_total = sum(len(obj.faces) for obj in objects)
    if (
        faces_total * 4608 + sum(len(obj.vertices) * 384 for obj in objects)
        > settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "UV density working set exceeds memory budget")
    charts = []
    pair_budget = settings.max_geometry_pairs
    for oi, obj in enumerate(objects):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "UV density planning timed out")
        if obj.uv is None:
            raise ValueError("Generate UV before fitting texture density")
        if len(obj.faces) * 3 > settings.max_vertices:
            raise ScissorsError("RESOURCE_LIMIT", "UV density seams exceed vertex budget")
        all_vertices, all_faces, all_uv = (
            np.asarray(obj.vertices),
            np.asarray(obj.faces),
            np.asarray(obj.uv),
        )
        for group in islands(obj, deadline):
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "UV density planning timed out")
            face_indices = np.asarray(group)
            indices, inverse = np.unique(all_faces[face_indices], return_inverse=True)
            # Measure this chart only, without using a bounding-box area estimate.
            part = obj.model_copy(
                update={
                    "vertices": all_vertices[indices].tolist(),
                    "faces": inverse.reshape(-1, 3).tolist(),
                    "uv": all_uv[indices].tolist(),
                }
            )
            report = density_report(part, units, resolution)
            if report["degenerate_uv_triangles"]:
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Density target needs nondegenerate UV", {"object_id": obj.id}
                )
            scale = target / report["pixels_per_meter"]
            coords = all_uv[indices] * resolution
            center = (coords.min(axis=0) + coords.max(axis=0)) / 2
            coords = (coords - center) * scale + center
            if not np.isfinite(coords).all():
                raise ValueError("Texture density scale is not finite")
            span = np.ptp(coords, axis=0)
            if mode == "atlas":
                pair_budget -= check_chart_overlap(
                    coords[inverse.reshape(-1, 3)], pair_budget, deadline
                )
            charts.append(
                {"object": oi, "faces": group, "indices": indices, "coords": coords, "span": span}
            )
    if mode == "atlas":
        low = np.ceil(np.asarray(region[:2]) * resolution).astype(int)
        high = np.floor(np.asarray(region[2:]) * resolution).astype(int)
        x, y, row_height = int(low[0]), int(low[1]), 0
        for chart in sorted(charts, key=lambda c: (-c["span"][1], -c["span"][0])):
            w, h = [math.ceil(float(value)) + padding * 2 for value in chart["span"]]
            if x + w > high[0]:
                x, y, row_height = int(low[0]), y + row_height, 0
            if w > high[0] - low[0] or y + h > high[1]:
                raise ScissorsError(
                    "RESOURCE_LIMIT",
                    "UV islands do not fit at the requested density",
                    {
                        "reason": "ATLAS_CAPACITY",
                        "pixels_per_meter": target,
                        "resolution": resolution,
                        "atlas_region": region,
                        "packing": "height-sorted shelves; no rotation or density reduction",
                    },
                )
            chart["coords"] += [x + padding, y + padding] - chart["coords"].min(axis=0)
            x, row_height = x + w, max(row_height, h)
    results = []
    for oi, obj in enumerate(objects):
        vertices: list[list[float]] = []
        normals: list[list[float]] = []
        uv_out: list[list[float]] = []
        colors: list[list[int]] = []
        output_faces = np.zeros((len(obj.faces), 3), dtype=int)
        for chart in [c for c in charts if c["object"] == oi]:
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "UV density placement timed out")
            mapping = {}
            for original, coords in zip(chart["indices"], chart["coords"], strict=True):
                mapping[original] = len(vertices)
                vertices.append(obj.vertices[original])
                normals.append(obj.normals[original])
                uv_out.append((coords / resolution).tolist())
                if obj.vertex_colors is not None:
                    colors.append(obj.vertex_colors[original])
            for fi in chart["faces"]:
                output_faces[fi] = [mapping[i] for i in obj.faces[fi]]
        if len(vertices) > settings.max_vertices:
            raise ScissorsError("RESOURCE_LIMIT", "UV seam output exceeds vertex budget")
        obj.vertices, obj.normals, obj.uv = vertices, normals, uv_out
        obj.faces = output_faces.tolist()
        obj.vertex_colors = colors if obj.vertex_colors is not None else None
        results.append(
            {
                **density_report(obj, units, resolution),
                "island_count": sum(c["object"] == oi for c in charts),
            }
        )
    return {
        "requested_pixels_per_meter": target,
        "mode": mode,
        "atlas_region": region,
        "padding": padding,
        "objects": results,
        "overlap_policy": "separate island rectangles"
        if mode == "atlas"
        else "overlap allowed; islands scale about their original centers",
        "warnings": [
            "Density measures pixel area, not source image detail. "
            "Per-face density and anisotropy can differ within a curved island."
        ],
    }


def check_chart_overlap(triangles: np.ndarray, pair_budget: int, deadline: float) -> int:
    """Reject positive-area UV overlap; touching chart triangles remain valid."""
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    active: list[int] = []
    pairs = 0

    def side(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
        u, v = b - a, p - a
        return float(u[0] * v[1] - u[1] * v[0])

    for current in np.argsort(low[:, 0], kind="stable"):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "UV overlap check timed out")
        active = [i for i in active if high[i, 0] > low[current, 0]]
        for other in active:
            pairs += 1
            if pairs > pair_budget:
                raise ScissorsError("RESOURCE_LIMIT", "UV overlap check exceeds pair budget")
            if high[other, 1] <= low[current, 1] or high[current, 1] <= low[other, 1]:
                continue
            polygon = list(triangles[current])
            clip = triangles[other]
            orientation = 1 if side(clip[0], clip[1], clip[2]) > 0 else -1
            for a, b in zip(clip, np.roll(clip, -1, axis=0), strict=True):
                if not polygon:
                    break
                output = []
                previous = polygon[-1]
                dp = orientation * side(a, b, previous)
                for p in polygon:
                    d = orientation * side(a, b, p)
                    if (d >= 0) != (dp >= 0):
                        output.append(previous + (p - previous) * (dp / (dp - d)))
                    if d >= 0:
                        output.append(p)
                    previous, dp = p, d
                polygon = output
            if len(polygon) >= 3:
                p = np.asarray(polygon)
                area = (
                    abs(
                        float(
                            np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
                        )
                    )
                    / 2
                )
                if area > 1e-10:
                    raise ScissorsError(
                        "GEOMETRY_ERROR",
                        "A UV island overlaps itself",
                        {"reason": "UV_OVERLAP", "overlap_area_pixels": area},
                    )
        active.append(int(current))
    return pairs
