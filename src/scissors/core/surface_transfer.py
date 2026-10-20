"""Bounded nearest-surface correspondence and seam-preserving UV transfer."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.engine import Engine, identifier
from scissors.core.face_queries import selection_indices
from scissors.core.mesh import MeshObject
from scissors.core.tangents import tangent_frames


def chosen_faces(obj: MeshObject, selection: dict[str, Any] | None) -> np.ndarray:
    if obj.kind != "mesh":
        raise ValueError("Surface transfer requires mesh objects")
    return (
        np.sort(selection_indices(obj, selection))
        if selection is not None
        else np.arange(len(obj.faces))
    )


class Correspondence:
    def __init__(
        self,
        source: MeshObject,
        selected: np.ndarray,
        settings: Settings,
        max_distance: float,
        min_normal_dot: float,
    ):
        self.source, self.selected, self.settings = source, selected, settings
        if not len(selected) or source.uv is None:
            raise ValueError("Source transfer faces require UV")
        if len(selected) * 768 + len(source.vertices) * 512 > settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Source correspondence exceeds memory budget")
        self.triangles = source.geometry().triangles[selected]
        self.normals = source.geometry().face_normals[selected]
        self.max_distance, self.min_normal_dot = max_distance, min_normal_dot
        self.deadline = time.monotonic() + settings.operation_timeout
        self.pairs = 0

    def project(
        self, points: np.ndarray, normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.pairs += len(points) * len(self.selected)
        if self.pairs > self.settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Surface transfer exceeds comparison budget")
        batch = max(1, min(256, self.settings.max_memory_bytes // max(1, len(self.selected) * 768)))
        faces, weights, distances = [], [], []
        for start in range(0, len(points), batch):
            if time.monotonic() > self.deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Surface transfer timed out")
            query = points[start : start + batch]
            nearest = trimesh.triangles.closest_point(
                np.tile(self.triangles, (len(query), 1, 1)),
                np.repeat(query, len(self.selected), axis=0),
            ).reshape(len(query), len(self.selected), 3)
            distance = np.linalg.norm(nearest - query[:, None], axis=2)
            alignment = normals[start : start + batch] @ self.normals.T
            distance[alignment < self.min_normal_dot] = np.inf
            best = distance.min(axis=1)
            if np.any(~np.isfinite(best)) or np.any(best > self.max_distance + 1e-10):
                raise ScissorsError(
                    "GEOMETRY_ERROR",
                    "Surface transfer has unmatched samples",
                    {
                        "max_distance": self.max_distance,
                        "unmatched_samples": int(
                            np.count_nonzero(
                                ~np.isfinite(best) | (best > self.max_distance + 1e-10)
                            )
                        ),
                    },
                )
            candidates = distance <= best[:, None] + 1e-12
            pick = np.argmax(np.where(candidates, alignment, -np.inf), axis=1)
            closest = nearest[np.arange(len(query)), pick]
            bary = trimesh.triangles.points_to_barycentric(self.triangles[pick], closest)
            bary = np.clip(bary, 0, 1)
            bary /= bary.sum(axis=1)[:, None]
            faces.extend(self.selected[pick].tolist())
            weights.extend(bary.tolist())
            distances.extend(best.tolist())
        return np.asarray(faces, dtype=int), np.asarray(weights), np.asarray(distances)


def transfer_uv(
    engine: Engine,
    object_id: str,
    source_object_id: str,
    max_distance: float,
    selection: dict[str, Any] | None,
    source_selection: dict[str, Any] | None,
    material_policy: str,
    transfer_colors: bool,
    min_normal_dot: float,
) -> dict[str, Any]:
    scene, target = engine.object(object_id)
    source_scene, source = engine.object(source_object_id)
    if target.id == source.id:
        raise ValueError("Surface transfer requires distinct source and target objects")
    chosen = chosen_faces(target, selection)
    source_faces = chosen_faces(source, source_selection)
    whole = len(chosen) == len(target.faces)
    outside = np.ones(len(target.faces), dtype=bool)
    outside[chosen] = False
    planned_vertices = len(np.unique(np.asarray(target.faces)[outside])) + len(chosen) * 3
    if (
        planned_vertices > engine.settings.max_vertices
        or len(target.faces) > engine.settings.max_faces
        or planned_vertices * 1024 + len(target.faces) * 4096 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "UV transfer exceeds geometry or memory budget")
    if not whole and target.uv is None:
        raise ValueError("Partial UV transfer requires existing target UV")
    if material_policy == "source" and not whole and source.material_id != target.material_id:
        raise ValueError(
            "Partial transfer with different materials needs texture.transfer "
            "or material_policy='target'"
        )
    matcher = Correspondence(source, source_faces, engine.settings, max_distance, min_normal_dot)
    faces = np.asarray(target.faces).copy()
    points = np.asarray(target.vertices)[faces[chosen]].reshape(-1, 3)
    normal = np.repeat(target.geometry().face_normals[chosen], 3, axis=0)
    matched, bary, distances = matcher.project(points, normal)
    origins = np.r_[np.arange(len(target.vertices)), faces[chosen].ravel()]
    vertices = np.concatenate([np.asarray(target.vertices), points])
    uv = np.asarray(target.uv if target.uv is not None else [[0, 0]] * len(target.vertices))
    mapped_uv = np.einsum(
        "ni,nij->nj", bary, np.asarray(source.uv)[np.asarray(source.faces)[matched]]
    )
    triangle_uv = mapped_uv.reshape(-1, 3, 2)
    a, b = triangle_uv[:, 1] - triangle_uv[:, 0], triangle_uv[:, 2] - triangle_uv[:, 0]
    if np.any(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) <= 1e-14):
        raise ScissorsError("GEOMETRY_ERROR", "Transferred UV collapses target triangles")
    uv = np.concatenate([uv, mapped_uv])
    colors = None
    if target.vertex_colors is not None or (transfer_colors and source.vertex_colors is not None):
        old = target.vertex_colors or [[255, 255, 255, 255]] * len(target.vertices)
        new = (
            np.rint(
                np.einsum(
                    "ni,nij->nj",
                    bary,
                    np.asarray(source.vertex_colors)[np.asarray(source.faces)[matched]],
                )
            ).astype(int)
            if transfer_colors and source.vertex_colors is not None
            else np.asarray(old)[origins[len(target.vertices) :]]
        )
        colors = np.concatenate([old, new])
    faces[chosen] = np.arange(len(target.vertices), len(vertices)).reshape(-1, 3)
    used, inverse = np.unique(faces, return_inverse=True)
    if len(used) > engine.settings.max_vertices or len(faces) > engine.settings.max_faces:
        raise ScissorsError("RESOURCE_LIMIT", "UV seam splitting exceeds geometry limits")
    result_obj = target.model_copy(deep=True)
    result_obj.replace(
        trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False),
        uv[used].tolist(),
    )
    result_obj.normals = np.asarray(target.normals)[origins[used]].tolist()
    result_obj.metadata["normal_method"] = "split_angle"
    result_obj.vertex_colors = colors[used].tolist() if colors is not None else None
    if material_policy == "source":
        if source.material_id is None:
            result_obj.material_id = None
        elif source_scene.id == scene.id:
            result_obj.material_id = source.material_id
        else:
            material = source_scene.materials[source.material_id].model_copy(deep=True)
            material.id = identifier("mat")
            scene.materials[material.id] = material
            result_obj.material_id = material.id
    for key in ("primitive", "boundary_rings", "center_pairs", "instance_source"):
        result_obj.metadata.pop(key, None)
    if result_obj.material_id and scene.materials[result_obj.material_id].normal_texture:
        tangent_frames(result_obj.geometry(), np.asarray(result_obj.uv))
    engine.checked(result_obj)
    scene.objects[target.id] = result_obj
    return {
        **result_obj.summary(),
        "source_object_id": source.id,
        "max_measured_distance": float(distances.max()),
        "transferred_face_count": len(chosen),
        "new_to_old_vertices": origins[used].tolist(),
        "selection": {
            "object_id": target.id,
            "geometry_revision": result_obj.geometry_revision,
            "face_indices": chosen.tolist(),
        },
    }
