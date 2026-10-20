"""Boundary discovery and oriented bridges on exact-position topology."""

from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject


def boundary_loops(obj: MeshObject) -> list[list[int]]:
    """Return directed closed loops as representative render vertex indices."""
    if obj.kind != "mesh":
        raise ValueError("Boundary discovery requires a mesh")
    _, representatives, inverse = np.unique(
        obj.vertices, axis=0, return_index=True, return_inverse=True
    )
    faces = inverse[np.asarray(obj.faces)]
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, indices, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ScissorsError("GEOMETRY_ERROR", "Boundary discovery requires manifold edges")
    boundary = edges[counts[indices] == 1]
    following: dict[int, int] = {}
    incoming: set[int] = set()
    for a, b in boundary:
        a, b = int(a), int(b)
        if a in following or b in incoming:
            raise ScissorsError("GEOMETRY_ERROR", "Boundary is branched or inconsistently wound")
        following[a] = b
        incoming.add(b)
    loops = []
    while following:
        first = min(following)
        loop = [first]
        current = following.pop(first)
        while current != first:
            if current not in following:
                raise ScissorsError("GEOMETRY_ERROR", "Boundary does not form a closed loop")
            loop.append(current)
            current = following.pop(current)
        if len(loop) < 3:
            raise ScissorsError("GEOMETRY_ERROR", "Boundary needs at least three vertices")
        loops.append([int(representatives[index]) for index in loop])
    return loops


def describe_boundaries(obj: MeshObject) -> dict[str, Any]:
    loops = boundary_loops(obj)
    return {
        "object_id": obj.id,
        "geometry_revision": obj.geometry_revision,
        "loops": [
            {"loop_index": index, "vertex_indices": loop, "edge_count": len(loop)}
            for index, loop in enumerate(loops)
        ],
    }


def bridge_geometry(
    first: MeshObject,
    second: MeshObject,
    first_loop: list[int],
    second_loop: list[int],
    settings: Settings,
    offset: int | None,
) -> tuple[trimesh.Trimesh, int]:
    """Zip loops by normalized arc length without resampling source surfaces."""
    vertex_count = len(first.vertices) + len(second.vertices)
    face_count = len(first.faces) + len(second.faces) + len(first_loop) + len(second_loop)
    if vertex_count > settings.max_vertices or face_count > settings.max_faces:
        raise ScissorsError("RESOURCE_LIMIT", "Bridge exceeds mesh size limits")
    if vertex_count * 384 + face_count * 192 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Bridge exceeds working memory limit")
    vertices = np.asarray([*first.vertices, *second.vertices])
    a = np.asarray(first_loop)
    # Each new boundary edge must oppose its source face's winding.
    b = np.asarray(second_loop[::-1]) + len(first.vertices)
    if offset is None:
        offset = int(np.argmin(np.linalg.norm(vertices[b] - vertices[a[0]], axis=1)))
    if offset >= len(b):
        raise ValueError("second_offset must be smaller than the second boundary edge count")
    b = np.roll(b, -offset)

    def lengths(loop: np.ndarray) -> np.ndarray:
        steps = np.linalg.norm(vertices[np.roll(loop, -1)] - vertices[loop], axis=1)
        if np.any(steps <= 1e-14):
            raise ScissorsError("GEOMETRY_ERROR", "Boundary contains zero-length edges")
        return np.r_[0.0, np.cumsum(steps) / steps.sum()]

    sa, sb = lengths(a), lengths(b)
    i = j = 0
    added = []
    while i < len(a) or j < len(b):
        ai, bj = int(a[i % len(a)]), int(b[j % len(b)])
        if i < len(a) and (j == len(b) or sa[i + 1] <= sb[j + 1]):
            added.append([ai, bj, int(a[(i + 1) % len(a)])])
            i += 1
        else:
            added.append([ai, bj, int(b[(j + 1) % len(b)])])
            j += 1
    triangles = vertices[np.asarray(added)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-14):
        raise ScissorsError("GEOMETRY_ERROR", "Bridge would create degenerate triangles")
    faces = [*first.faces, *(np.asarray(second.faces) + len(first.vertices)).tolist(), *added]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), offset
