"""Angle-weighted tangent frames for normal-map exports."""

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError


def tangent_frames(mesh: trimesh.Trimesh, uv: np.ndarray) -> np.ndarray:
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    if uv.shape != (len(vertices), 2):
        raise ScissorsError("GEOMETRY_ERROR", "Tangents require UV for every render vertex")
    triangles, coordinates = vertices[faces], uv[faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    a, b = coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if np.any(np.abs(determinant) <= 1e-14):
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Tangents require nondegenerate UV triangles",
            {"degenerate_uv_faces": int(np.count_nonzero(np.abs(determinant) <= 1e-14))},
        )
    tangent = (e1 * b[:, 1, None] - e2 * a[:, 1, None]) / determinant[:, None]
    bitangent = (e2 * a[:, 0, None] - e1 * b[:, 0, None]) / determinant[:, None]
    summed_t, summed_b = np.zeros_like(vertices), np.zeros_like(vertices)
    normals = np.asarray(mesh.vertex_normals)
    signs_min, signs_max = np.ones(len(vertices)), -np.ones(len(vertices))
    for corner in range(3):
        indices = faces[:, corner]
        weight = mesh.face_angles[:, corner, None]
        np.add.at(summed_t, indices, tangent * weight)
        np.add.at(summed_b, indices, bitangent * weight)
        signs = np.where(np.sum(np.cross(normals[indices], tangent) * bitangent, axis=1) < 0, -1, 1)
        np.minimum.at(signs_min, indices, signs)
        np.maximum.at(signs_max, indices, signs)
    if np.any(signs_min != signs_max):
        raise ScissorsError("GEOMETRY_ERROR", "Split render vertices at mirrored UV seams first")
    summed_t -= normals * np.sum(normals * summed_t, axis=1)[:, None]
    lengths = np.linalg.norm(summed_t, axis=1)
    if np.any(lengths <= 1e-14):
        raise ScissorsError(
            "GEOMETRY_ERROR", "UV tangent directions cancel or contain unused vertices"
        )
    summed_t /= lengths[:, None]
    signs = np.where(np.sum(np.cross(normals, summed_t) * summed_b, axis=1) < 0, -1, 1)
    return np.column_stack([summed_t, signs])


def gltf_tangents(tree: dict) -> None:
    """Promote a typed VEC4 custom accessor to the standard glTF semantic."""
    for mesh in tree.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "_SCISSORS_TANGENT" in attributes:
                attributes["TANGENT"] = attributes.pop("_SCISSORS_TANGENT")
