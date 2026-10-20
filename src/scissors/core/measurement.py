"""Read-only measurements from current world geometry, including planar hole rims."""

from typing import Any

import numpy as np

from scissors.core.mesh import MeshObject, topology_mesh


def hole_rim(obj: MeshObject, face: int, hole_index: int, tolerance: float) -> np.ndarray:
    mesh = topology_mesh(obj.geometry())
    if face >= len(mesh.faces):
        raise ValueError("Face index is outside the mesh")
    origin, normal = mesh.triangles[face, 0], mesh.face_normals[face]
    planar = np.max(np.abs((mesh.triangles - origin) @ normal), axis=1) <= tolerance
    planar &= (mesh.face_normals @ normal) > 1 - 1e-8
    neighbors: dict[int, list[int]] = {}
    for a, b in mesh.face_adjacency:
        if planar[a] and planar[b]:
            neighbors.setdefault(int(a), []).append(int(b))
            neighbors.setdefault(int(b), []).append(int(a))
    selected, pending = {face}, [face]
    while pending:
        for other in neighbors.get(pending.pop(), []):
            if other not in selected:
                selected.add(other)
                pending.append(other)
    faces = mesh.faces[sorted(selected)]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    links: dict[int, list[int]] = {}
    for a, b in unique[count == 1]:
        links.setdefault(int(a), []).append(int(b))
        links.setdefault(int(b), []).append(int(a))
    if any(len(adjacent) != 2 for adjacent in links.values()):
        raise ValueError("Selected planar face region has a branched boundary")
    loops = []
    unseen = set(links)
    while unseen:
        start = min(unseen)
        ring, visited, previous, current = [], set(), -1, start
        while current not in visited:
            ring.append(current)
            visited.add(current)
            unseen.discard(current)
            following = [value for value in links[current] if value != previous]
            previous, current = current, following[0]
        if current != start:
            raise ValueError("Selected face region has an invalid boundary")
        loops.append(mesh.vertices[ring])
    # The largest enclosed area is the exterior rim. Remaining rims are holes.
    loops.sort(
        key=lambda ring: abs(float(np.sum(np.cross(ring, np.roll(ring, -1, axis=0)) @ normal))),
        reverse=True,
    )
    if hole_index >= len(loops) - 1:
        raise ValueError("Selected face region does not contain the requested hole")
    return loops[hole_index + 1]


def circle_measure(points: np.ndarray, tolerance: float) -> dict[str, Any]:
    points = np.unique(points, axis=0)
    if len(points) < 3:
        raise ValueError("Diameter measurement needs at least three distinct rim points")
    origin = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - origin, full_matrices=False)
    if singular[1] <= tolerance:
        raise ValueError("Rim points are collinear")
    coordinates = (points - origin) @ axes[:2].T
    if np.max(np.abs((points - origin) @ axes[2])) > tolerance:
        raise ValueError("Rim is not planar within tolerance")
    matrix = np.column_stack([2 * coordinates, np.ones(len(points))])
    solution = np.linalg.lstsq(matrix, np.sum(coordinates**2, axis=1), rcond=None)[0]
    radius = float(np.sqrt(max(0, solution[2] + np.dot(solution[:2], solution[:2]))))
    residual = float(np.max(np.abs(np.linalg.norm(coordinates - solution[:2], axis=1) - radius)))
    if radius <= tolerance or residual > tolerance:
        raise ValueError("Rim is not circular within tolerance")
    return {
        "diameter": 2 * radius,
        "radius": radius,
        "center": (origin + solution[:2] @ axes[:2]).tolist(),
        "normal": axes[2].tolist(),
        "max_radial_error": residual,
        "sample_count": len(points),
    }
