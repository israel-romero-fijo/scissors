"""Targeted hole patches with fixed boundaries and measured tangent joins."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.bridging import boundary_loops
from scissors.core.engine import Engine
from scissors.core.persistence import json_bytes
from scissors.core.profiles import area, triangulate
from scissors.core.selections import indices


def polynomial(xy: np.ndarray, derivative: int | None = None) -> np.ndarray:
    """Cubic height chart and its first derivatives in normalized coordinates."""
    x, y = xy[:, 0], xy[:, 1]
    powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    columns = []
    for a, b in powers:
        coefficient = 1
        if derivative == 0:
            coefficient, a = a, max(a - 1, 0)
        elif derivative == 1:
            coefficient, b = b, max(b - 1, 0)
        columns.append(coefficient * x**a * y**b)
    return np.column_stack(columns)


@dataclass
class HeightPatch:
    points: list[list[float]]
    flat: list[list[float]]
    residuals: list[float]
    coefficients: np.ndarray
    origin: np.ndarray
    scale: float
    basis: np.ndarray
    uv: list[list[float]] | None
    colors: list[list[float]] | None

    def sample(self, parents: list[int]) -> int:
        xy = np.mean([self.flat[i] for i in parents], axis=0)
        residual = float(np.mean([self.residuals[i] for i in parents]))
        height = float((polynomial(xy[None]) @ self.coefficients)[0]) + residual
        point = self.origin + self.scale * (np.r_[xy, height] @ self.basis)
        if not np.isfinite(point).all():
            raise ScissorsError("GEOMETRY_ERROR", "Smooth patch exceeds numerical range")
        index = len(self.points)
        self.points.append(point.tolist())
        self.flat.append(xy.tolist())
        self.residuals.append(residual)
        if self.uv is not None:
            self.uv.append(np.mean([self.uv[i] for i in parents], axis=0).tolist())
        if self.colors is not None:
            self.colors.append(np.mean([self.colors[i] for i in parents], axis=0).tolist())
        return index


def fill_holes(
    engine: Engine,
    object_id: str,
    max_edges: int,
    selection: dict[str, Any] | None,
    method: str,
    refinement: int,
    tangent_tolerance_degrees: float,
) -> dict[str, Any]:
    scene, obj = engine.object(object_id)
    if obj.kind != "mesh":
        raise ValueError("Hole filling requires a mesh")
    settings = engine.settings
    deadline = time.monotonic() + settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Hole filling timed out")

    def check_size(vertex_count: int, face_count: int) -> None:
        if vertex_count > settings.max_vertices or face_count > settings.max_faces:
            raise ScissorsError("RESOURCE_LIMIT", "Hole filling exceeds mesh size limits")
        if vertex_count * 1024 + face_count * 4096 > settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Hole filling exceeds working memory budget")

    check_size(len(obj.vertices), len(obj.faces))
    if selection is not None and (
        selection["object_id"] != obj.id or selection["geometry_revision"] != obj.geometry_revision
    ):
        raise ValueError("Boundary selection belongs to another geometry revision")
    loops = boundary_loops(obj)
    chosen = (
        indices(selection["loop_indices"], len(loops))
        if selection is not None
        else np.asarray(
            [i for i, loop in enumerate(loops) if len(loop) <= max_edges], dtype=np.int64
        )
    )
    if selection is not None and not len(chosen):
        raise ValueError("Hole filling requires at least one selected boundary loop")
    if any(len(loops[i]) > max_edges for i in chosen):
        raise ValueError("Selected boundary exceeds max_edges; increase the limit explicitly")
    vertices = [list(vertex) for vertex in obj.vertices]
    uv = [list(value) for value in obj.uv] if obj.uv is not None else None
    colors = [list(value) for value in obj.vertex_colors] if obj.vertex_colors is not None else None
    original = np.asarray(obj.vertices)
    faces = np.asarray(obj.faces)
    source_mesh = obj.geometry()
    # Owners and collar lookup use exact-position topology, including render seams.
    _, welded = np.unique(original, axis=0, return_inverse=True)
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edge_keys = np.sort(welded[directed], axis=1)
    unique_edges, inverse, counts = np.unique(
        edge_keys, axis=0, return_inverse=True, return_counts=True
    )
    owners = {tuple(unique_edges[inverse[i]]): i // 3 for i in np.flatnonzero(counts[inverse] == 1)}
    caps: list[list[int]] = []
    reports = []
    pairs = 0
    for loop_index in chosen:
        check_time()
        # New patch winding opposes the directed source boundary.
        loop = loops[loop_index][::-1]
        points = original[loop]
        origin = points.mean(axis=0)
        _, singular, basis = np.linalg.svd(points - origin, full_matrices=False)
        scale = float(singular[0])
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ScissorsError("GEOMETRY_ERROR", "Boundary has no stable projection")
        chart = (points - origin) @ basis.T / scale
        flat = chart[:, :2]
        pairs += len(loop) ** 2
        if pairs > settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Hole projection exceeds pair budget")
        projected, triangles, _, _ = triangulate(flat.tolist(), [], settings)
        mapping = {tuple(point): i for i, point in enumerate(flat)}
        if len(mapping) != len(loop):
            raise ScissorsError("GEOMETRY_ERROR", "Boundary projection collapses vertices")
        local_faces = [
            [mapping[tuple(projected[i])] for i in (face if area(flat) > 0 else face[::-1])]
            for face in triangles
        ]
        boundary = {tuple(sorted((i, (i + 1) % len(loop)))) for i in range(len(loop))}
        adjacent = [
            owners[tuple(sorted((int(welded[loop[i]]), int(welded[loop[(i + 1) % len(loop)]]))))]
            for i in range(len(loop))
        ]
        local_points = points.tolist()
        local_uv = np.asarray(obj.uv)[loop].tolist() if uv is not None else None
        local_colors = (
            np.asarray(obj.vertex_colors)[loop].astype(float).tolist()
            if colors is not None
            else None
        )
        if method == "smooth":
            # Fit heights and boundary tangents using the adjacent surface collar.
            collar = np.unique(faces[np.any(np.isin(welded[faces], welded[loop]), axis=1)])
            collar_chart = (original[collar] - origin) @ basis.T / scale
            normals = np.asarray(source_mesh.face_normals)[adjacent] @ basis.T
            if np.any(np.abs(normals[:, 2]) < 0.1):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Smooth patch needs a single-valued height chart"
                )
            midpoints = (flat + np.roll(flat, -1, axis=0)) / 2
            system = np.concatenate(
                [
                    polynomial(collar_chart[:, :2]),
                    polynomial(midpoints, 0),
                    polynomial(midpoints, 1),
                ]
            )
            targets = np.r_[
                collar_chart[:, 2], -normals[:, 0] / normals[:, 2], -normals[:, 1] / normals[:, 2]
            ]
            coefficients, _, _, _ = np.linalg.lstsq(system, targets, rcond=None)
            residuals = (chart[:, 2] - polynomial(flat) @ coefficients).tolist()
            local_flat = flat.tolist()

            sample = HeightPatch(
                local_points,
                local_flat,
                residuals,
                coefficients,
                origin,
                scale,
                basis,
                local_uv,
                local_colors,
            ).sample

            for _ in range(refinement):
                check_time()
                # Count shared interior midpoints once. Source boundary edges stay fixed.
                interior_edges = {
                    tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                    for triangle in local_faces
                    for i in range(3)
                } - boundary
                check_size(
                    len(vertices)
                    + len(local_points)
                    - len(loop)
                    + len(local_faces)
                    + len(interior_edges),
                    len(faces) + len(caps) + 3 * len(local_faces) + 2 * len(interior_edges),
                )
                mid_cache: dict[tuple[int, int], int] = {}
                refined = []
                for triangle in local_faces:
                    check_time()
                    perimeter = []
                    for i, a in enumerate(triangle):
                        b = triangle[(i + 1) % 3]
                        perimeter.append(a)
                        edge = (min(a, b), max(a, b))
                        if edge not in boundary:
                            if edge not in mid_cache:
                                mid_cache[edge] = sample([a, b])
                            perimeter.append(mid_cache[edge])
                    center = sample(triangle)
                    refined.extend(
                        [
                            [center, a, perimeter[(i + 1) % len(perimeter)]]
                            for i, a in enumerate(perimeter)
                        ]
                    )
                local_faces = refined
        check_size(
            len(vertices) + len(local_points) - len(loop), len(faces) + len(caps) + len(local_faces)
        )
        patch = trimesh.Trimesh(vertices=local_points, faces=local_faces, process=False)
        patch_edges = np.asarray(local_faces)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        patch_owners = {
            tuple(sorted(edge)): i // 3
            for i, edge in enumerate(patch_edges)
            if tuple(sorted(edge)) in boundary
        }
        patch_normals = np.asarray(patch.face_normals)[
            [patch_owners[tuple(sorted((i, (i + 1) % len(loop))))] for i in range(len(loop))]
        ]
        dots = np.einsum("ij,ij->i", patch_normals, np.asarray(source_mesh.face_normals)[adjacent])
        max_angle = float(np.degrees(np.arccos(np.clip(dots, -1, 1))).max())
        if method == "smooth" and max_angle > tangent_tolerance_degrees + 1e-8:
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Smooth patch cannot meet tangent tolerance",
                {
                    "loop_index": int(loop_index),
                    "max_join_angle_degrees": max_angle,
                    "tangent_tolerance_degrees": tangent_tolerance_degrees,
                },
            )
        local_to_global = [
            *loop,
            *range(len(vertices), len(vertices) + len(local_points) - len(loop)),
        ]
        vertices.extend(local_points[len(loop) :])
        if uv is not None:
            assert local_uv is not None
            uv.extend(local_uv[len(loop) :])
        if colors is not None:
            assert local_colors is not None
            colors.extend(np.rint(local_colors[len(loop) :]).astype(int).tolist())
        first_face = len(faces) + len(caps)
        caps.extend([[local_to_global[i] for i in face] for face in local_faces])
        reports.append(
            {
                "loop_index": int(loop_index),
                "max_join_angle_degrees": max_angle,
                "face_indices": list(range(first_face, len(faces) + len(caps))),
            }
        )
    target = obj
    if len(chosen):
        target = obj.model_copy(deep=True)
        target.replace(
            trimesh.Trimesh(vertices=vertices, faces=[*obj.faces, *caps], process=False), uv
        )
        target.vertex_colors = colors
        for key in ("primitive", "boundary_rings", "center_pairs", "instance_source"):
            target.metadata.pop(key, None)
        engine.checked(target)
    patch_indices = list(range(len(faces), len(faces) + len(caps)))
    result = {
        **target.summary(),
        "filled_holes": len(chosen),
        "method": method,
        "filled_loop_indices": chosen.tolist(),
        "patches": reports,
        "patch_face_indices": patch_indices,
        "selection": {
            "object_id": obj.id,
            "geometry_revision": target.geometry_revision,
            "face_indices": patch_indices,
        },
        "warnings": ["Patch UV interpolates boundary values; inspect stretching and intersections."]
        if len(chosen)
        else [],
    }
    if len(json_bytes(result)) > settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Hole filling response exceeds byte budget")
    check_time()
    if target is not obj:
        scene.objects[obj.id] = target
    return result
