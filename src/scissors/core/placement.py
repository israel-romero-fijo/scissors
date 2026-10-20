"""Affine placement and bounded global deformation geometry."""

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.mesh import MeshObject


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12 or not np.isfinite(length):
        raise ValueError("Direction must be finite and nonzero")
    return vector / length


def reflection(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    normal = unit(normal)
    matrix = np.eye(4)
    matrix[:3, :3] -= 2 * np.outer(normal, normal)
    matrix[:3, 3] = 2 * normal * np.dot(normal, point)
    return matrix


def warp(obj: MeshObject, bend: float, twist: float, taper: float, axis: str) -> None:
    if obj.kind != "mesh":
        raise ValueError("Global deformation requires a mesh")
    index = "xyz".index(axis)
    a, b = [(1, 2), (2, 0), (0, 1)][index]
    vertices = np.asarray(obj.vertices).copy()
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    length = high[index] - low[index]
    if length <= 1e-12:
        raise ScissorsError("GEOMETRY_ERROR", "Deformation axis has zero extent")
    center = (low + high) / 2
    t = (vertices[:, index] - low[index]) / length
    scale = 1 + (taper - 1) * t
    x = (vertices[:, a] - center[a]) * scale
    y = (vertices[:, b] - center[b]) * scale
    angles = np.radians(twist) * t
    x, y = x * np.cos(angles) - y * np.sin(angles), x * np.sin(angles) + y * np.cos(angles)
    vertices[:, a], vertices[:, b] = x + center[a], y + center[b]
    if bend:
        radius = length / np.radians(bend)
        if np.any(1 - x / radius <= 1e-10):
            raise ScissorsError("GEOMETRY_ERROR", "Bend radius would fold the cross-section")
        theta = np.radians(bend) * t
        vertices[:, a] = center[a] + radius * (1 - np.cos(theta)) + x * np.cos(theta)
        vertices[:, index] = low[index] + radius * np.sin(theta) - x * np.sin(theta)
    obj.replace(trimesh.Trimesh(vertices=vertices, faces=obj.faces, process=False), obj.uv)
    for key in ("primitive", "boundary_rings", "center_pairs", "instance_source"):
        obj.metadata.pop(key, None)
    obj.metadata["last_warp"] = {
        "bend_degrees": bend,
        "twist_degrees": twist,
        "end_scale": taper,
        "axis": axis,
    }


def face_frame(obj: MeshObject, index: int) -> tuple[np.ndarray, np.ndarray]:
    if obj.kind != "mesh" or index < 0 or index >= len(obj.faces):
        raise ValueError("Face index must refer to a triangle in a mesh")
    mesh = obj.geometry()
    return mesh.triangles[index].mean(axis=0), mesh.face_normals[index]


def nearest_face(obj: MeshObject, point: np.ndarray) -> tuple[int, np.ndarray]:
    if obj.kind != "mesh":
        raise ValueError("Surface snapping requires a target mesh")
    triangles = obj.geometry().triangles
    best, closest, distance = -1, np.zeros(3), float("inf")
    for start in range(0, len(triangles), 4096):
        chunk = triangles[start : start + 4096]
        points = trimesh.triangles.closest_point(chunk, np.repeat(point[None], len(chunk), axis=0))
        distances = np.linalg.norm(points - point, axis=1)
        index = int(np.argmin(distances))
        if distances[index] < distance:
            best, closest, distance = start + index, points[index], float(distances[index])
    return best, closest


def alignment(
    points: np.ndarray,
    anchor: np.ndarray,
    source_normal: np.ndarray,
    target_point: np.ndarray,
    target_normal: np.ndarray,
    gap: float,
    rotate: bool,
    face_anchor: bool,
) -> np.ndarray:
    normal = unit(target_normal)
    matrix = np.eye(4)
    if rotate:
        rotation = trimesh.geometry.align_vectors(unit(source_normal), -normal)
        matrix[:3, :3] = rotation[:3, :3]
        matrix[:3, 3] = anchor - matrix[:3, :3] @ anchor
    if face_anchor:
        contact = anchor
    else:
        transformed = trimesh.transformations.transform_points(points, matrix)
        lowest = float(np.min(transformed @ normal))
        contact = anchor + normal * (lowest - np.dot(anchor, normal))
    matrix[:3, 3] += target_point + gap * normal - contact
    return matrix
