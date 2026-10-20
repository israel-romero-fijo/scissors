"""Split hard edges by angle while smoothing normals across UV seams."""

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject


def split_normals(obj: MeshObject, angle: float, settings: Settings) -> None:
    if obj.kind != "mesh":
        raise ValueError("Normal splitting requires a mesh")
    count = 3 * len(obj.faces)
    if count * 768 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Normal splitting exceeds temporary memory budget")
    mesh = obj.geometry()
    faces = np.asarray(obj.faces)
    points = np.asarray(obj.vertices)
    _, welded = np.unique(points, axis=0, return_inverse=True)
    topology = welded[faces]
    triangles = mesh.triangles
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths == 0):
        raise ScissorsError("GEOMETRY_ERROR", "Cannot split normals of a zero-area triangle")
    # Avoid the geometry library's absolute unit-vector cutoff on small faces.
    face_normals = cross / lengths[:, None]
    parent = np.arange(count)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    edges: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(topology):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges.setdefault(key, []).append(face_id)
    cosine = np.cos(np.radians(angle))
    for edge, adjacent in edges.items():
        if len(adjacent) != 2:
            continue
        first, second = adjacent
        if np.dot(face_normals[first], face_normals[second]) < cosine - 1e-12:
            continue
        for vertex in edge:
            a = 3 * first + int(np.flatnonzero(topology[first] == vertex)[0])
            b = 3 * second + int(np.flatnonzero(topology[second] == vertex)[0])
            one, two = root(a), root(b)
            parent[max(one, two)] = min(one, two)
    groups = np.asarray([root(i) for i in range(count)], dtype=np.int64)
    # atan2 retains the angle of skinny, valid backend triangles where arccos
    # rounds a corner to zero. Do not mistake a zero weight for opposing normals.
    angles = np.empty((len(triangles), 3))
    for corner in range(3):
        a = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        b = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        angles[:, corner] = np.arctan2(
            np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)
        )
    corner_normals = np.repeat(face_normals, 3, axis=0) * angles.reshape(-1, 1)
    accumulated = np.zeros((count, 3))
    np.add.at(accumulated, groups, corner_normals)
    lengths = np.linalg.norm(accumulated, axis=1)
    accumulated = np.divide(
        accumulated,
        lengths[:, None],
        out=np.zeros_like(accumulated),
        where=lengths[:, None] > 0,
    )
    uv = np.asarray(obj.uv)[faces].reshape(-1, 2) if obj.uv is not None else None
    colors = (
        np.asarray(obj.vertex_colors)[faces].reshape(-1, 4)
        if obj.vertex_colors is not None
        else None
    )
    key_parts = [groups[:, None].astype(float)]
    if uv is not None:
        key_parts.append(uv)
    if colors is not None:
        key_parts.append(colors)
    keys = np.column_stack(key_parts)
    _, representatives, mapping = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if len(representatives) > settings.max_vertices:
        raise ScissorsError("RESOURCE_LIMIT", "Split render vertices exceed configured limit")
    vertices = points[faces.reshape(-1)][representatives]
    center_pairs = obj.metadata.get("center_pairs")
    remapped_pairs = None
    if center_pairs is not None:
        point_indices = {tuple(point): i for i, point in enumerate(vertices)}
        remapped_pairs = [
            [point_indices[tuple(points[index])] for index in pair] for pair in center_pairs
        ]
    normals = accumulated[groups[representatives]]
    if np.any(np.linalg.norm(normals, axis=1) < 1e-12):
        raise ScissorsError("GEOMETRY_ERROR", "Opposing normals cancel at a smooth vertex")
    obj.replace(
        trimesh.Trimesh(vertices=vertices, faces=mapping.reshape(-1, 3), process=False),
        uv[representatives].tolist() if uv is not None else None,
    )
    obj.normals = normals.tolist()
    obj.vertex_colors = colors[representatives].tolist() if colors is not None else None
    obj.metadata["normal_method"] = "split_angle"
    obj.metadata["normal_angle_degrees"] = angle
    obj.metadata.pop("boundary_rings", None)
    obj.metadata.pop("center_pairs", None)
    if remapped_pairs is not None:
        obj.metadata["center_pairs"] = remapped_pairs
