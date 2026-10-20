"""Local triangulation refinement and smoothing with pinned borders and creases."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.face_queries import selection_indices
from scissors.core.persistence import json_bytes


def reconstruct_patch(
    engine: Engine,
    object_id: str,
    selection: dict[str, Any],
    refinement: int,
    iterations: int,
    strength: float,
    sharp_angle_degrees: float,
) -> dict[str, Any]:
    scene, obj = engine.object(object_id)
    if obj.kind != "mesh":
        raise ValueError("Patch reconstruction requires a mesh")
    settings = engine.settings
    deadline = time.monotonic() + settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Patch reconstruction timed out")

    def budget(vertices: int, faces: int) -> None:
        if vertices > settings.max_vertices or faces > settings.max_faces:
            raise ScissorsError("RESOURCE_LIMIT", "Patch reconstruction exceeds geometry limits")
        if vertices * 1024 + faces * 4096 > settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Patch reconstruction exceeds memory budget")

    budget(len(obj.vertices), len(obj.faces))
    chosen = np.sort(selection_indices(obj, selection))
    original = np.asarray(obj.vertices)
    source_uv = np.asarray(obj.uv) if obj.uv is not None else None
    source_colors = np.asarray(obj.vertex_colors) if obj.vertex_colors is not None else None
    points, representative, welded = np.unique(
        original, axis=0, return_index=True, return_inverse=True
    )
    faces = np.asarray(obj.faces)
    topology = welded[faces]
    patch_faces = topology[chosen]
    edges = np.sort(topology[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ScissorsError("GEOMETRY_ERROR", "Patch reconstruction requires manifold edges")
    if len(unique) > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Patch topology exceeds comparison budget")
    ordered = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = starts[counts == 2]
    normals = np.asarray(obj.geometry().face_normals)
    dots = np.einsum("ij,ij->i", normals[ordered[paired] // 3], normals[ordered[paired + 1] // 3])
    sharp = unique[counts == 2][dots <= np.cos(np.radians(sharp_angle_degrees))]
    sharp = sharp[np.any(np.isin(sharp, np.unique(patch_faces)), axis=1)]
    patch_edges = np.sort(patch_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    boundary_edges, boundary_counts = np.unique(patch_edges, axis=0, return_counts=True)
    border = boundary_edges[boundary_counts == 1]
    pinned_edges = {tuple(edge) for edge in np.concatenate([border, sharp])}
    pinned = set(np.unique(np.concatenate([border, sharp])).tolist())
    used = set(np.unique(patch_faces).tolist())
    pinned |= set(range(len(points))) - used
    outside = np.ones(len(faces), dtype=bool)
    outside[chosen] = False
    # Point contact alone can share a vertex without creating a patch border edge.
    pinned |= set(np.unique(topology[outside]).tolist())
    coordinates = points.tolist()
    # Each refined corner retains barycentric provenance in its original render triangle.
    pieces = [
        (face.tolist(), np.eye(3), int(owner))
        for face, owner in zip(patch_faces, chosen, strict=True)
    ]
    for _ in range(refinement):
        check_time()
        interior_edges = {
            tuple(sorted((tri[i], tri[(i + 1) % 3]))) for tri, _, _ in pieces for i in range(3)
        } - pinned_edges
        next_count = 3 * len(pieces) + 2 * len(interior_edges)
        budget(len(original) + 3 * next_count, len(faces) - len(chosen) + next_count)
        midpoints: dict[tuple[int, int], int] = {}
        next_pieces = []
        for triangle, bary, owner in pieces:
            check_time()
            perimeter = []
            weights = []
            for i, a in enumerate(triangle):
                j = (i + 1) % 3
                b = triangle[j]
                perimeter.append(a)
                weights.append(bary[i])
                edge = (min(a, b), max(a, b))
                if edge not in pinned_edges:
                    if edge not in midpoints:
                        midpoints[edge] = len(coordinates)
                        coordinates.append(
                            ((np.asarray(coordinates[a]) + coordinates[b]) / 2).tolist()
                        )
                    perimeter.append(midpoints[edge])
                    weights.append((bary[i] + bary[j]) / 2)
            center = len(coordinates)
            coordinates.append(np.mean([coordinates[i] for i in triangle], axis=0).tolist())
            center_weights = bary.mean(axis=0)
            for i, a in enumerate(perimeter):
                j = (i + 1) % len(perimeter)
                next_pieces.append(
                    (
                        [center, a, perimeter[j]],
                        np.asarray([center_weights, weights[i], weights[j]]),
                        owner,
                    )
                )
        pieces = next_pieces
    positions = np.asarray(coordinates)
    refined_faces = np.asarray([triangle for triangle, _, _ in pieces])
    adjacency = np.unique(
        np.sort(refined_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0
    )
    if len(adjacency) > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Refined patch exceeds comparison budget")
    degree = np.bincount(adjacency.ravel(), minlength=len(positions))
    free = degree > 0
    free[list(pinned)] = False
    for _ in range(iterations):
        check_time()
        sums = np.zeros_like(positions)
        np.add.at(sums, adjacency[:, 0], positions[adjacency[:, 1]])
        np.add.at(sums, adjacency[:, 1], positions[adjacency[:, 0]])
        positions[free] += strength * (sums[free] / degree[free, None] - positions[free])
    # Original render indices survive. All unselected face positions remain exact.
    output_vertices = positions[welded].tolist()
    uv = [list(v) for v in obj.uv] if obj.uv is not None else None
    colors = [list(v) for v in obj.vertex_colors] if obj.vertex_colors is not None else None
    output_faces = faces.tolist()
    original_ids = list(range(len(faces)))
    selected_faces = []
    filled_slots: set[int] = set()
    for triangle, bary, owner in pieces:
        check_time()
        start = len(output_vertices)
        output_vertices.extend(positions[triangle].tolist())
        if uv is not None:
            assert source_uv is not None
            uv.extend((bary @ source_uv[faces[owner]]).tolist())
        if colors is not None:
            assert source_colors is not None
            colors.extend(np.rint(bary @ source_colors[faces[owner]]).astype(int).tolist())
        new_face = [start, start + 1, start + 2]
        if owner not in filled_slots:
            filled_slots.add(owner)
            index = owner
            output_faces[index] = new_face
        else:
            index = len(output_faces)
            output_faces.append(new_face)
            original_ids.append(owner)
        selected_faces.append(index)
    budget(len(output_vertices), len(output_faces))
    mesh = trimesh.Trimesh(vertices=output_vertices, faces=output_faces, process=False)
    # Do not allow smoothing to reverse a child relative to its source triangle.
    if np.any(
        np.einsum(
            "ij,ij->i",
            np.asarray(mesh.face_normals)[selected_faces],
            normals[np.asarray(original_ids)[selected_faces]],
        )
        <= 0
    ):
        raise ScissorsError(
            "GEOMETRY_ERROR", "Patch smoothing reversed triangles; reduce strength or iterations"
        )
    target = obj.model_copy(deep=True)
    target.replace(mesh, uv)
    target.vertex_colors = colors
    for key in ("primitive", "boundary_rings", "center_pairs", "instance_source", "tangents"):
        target.metadata.pop(key, None)
    engine.checked(target)
    result = {
        **target.summary(),
        "method": "constrained_laplacian_refinement",
        "source_geometry_revision": obj.geometry_revision,
        "protected_vertex_indices": np.flatnonzero(np.isin(welded, list(pinned))).tolist(),
        "protected_edges": representative[sharp].tolist(),
        "new_to_old_faces": original_ids,
        "selection": {
            "object_id": obj.id,
            "geometry_revision": target.geometry_revision,
            "face_indices": sorted(selected_faces),
        },
        "warnings": [
            "Laplacian reconstruction can shrink curved regions; "
            "inspect patch shape, UV and intersections."
        ],
    }
    if len(json_bytes(result)) > settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Patch reconstruction response exceeds byte budget")
    check_time()
    scene.objects[obj.id] = target
    return result
