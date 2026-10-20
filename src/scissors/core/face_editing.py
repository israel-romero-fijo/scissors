"""Attribute-preserving local face edits with registry-owned rollback."""

import copy
import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.bridging import boundary_loops
from scissors.core.engine import Engine, identifier
from scissors.core.face_queries import selection_indices
from scissors.core.mesh import MeshObject, face_components, topology_mesh
from scissors.core.persistence import json_bytes

TOPOLOGY_KEYS = ("primitive", "boundary_rings", "center_pairs", "instance_source")


def flipped_faces(source: MeshObject, chosen: np.ndarray) -> MeshObject:
    """Reverse face order without moving or compacting render vertices."""
    faces = np.asarray(source.faces).copy()
    faces[chosen] = faces[chosen, ::-1]
    target = source.model_copy(deep=True)
    target.replace(
        trimesh.Trimesh(vertices=source.vertices, faces=faces, process=False),
        copy.deepcopy(source.uv),
    )
    target.vertex_colors = copy.deepcopy(source.vertex_colors)
    for key in TOPOLOGY_KEYS:
        target.metadata.pop(key, None)
    return target


def edit_faces(
    engine: Engine,
    object_id: str,
    selection: dict[str, Any],
    action: str,
    offset: list[float] | None = None,
    detach: bool = True,
    empty_policy: str = "error",
    name: str | None = None,
    copy_faces: bool = False,
) -> dict[str, Any]:
    scene, source = engine.object(object_id)
    if source.kind != "mesh":
        raise ValueError("Face editing requires a mesh")
    if len(source.vertices) * 1024 + len(source.faces) * 4096 > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Face editing exceeds working memory budget")
    deadline = time.monotonic() + engine.settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Face editing timed out")

    chosen = np.sort(selection_indices(source, selection))
    before_revision = source.geometry_revision
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    mask = np.zeros(len(faces), dtype=bool)
    mask[chosen] = True
    selected_vertices = np.unique(faces[chosen])
    warnings = []
    moved = np.array([], dtype=np.int64)
    side_count = 0

    def stage(
        target: MeshObject,
        points: np.ndarray,
        triangles: np.ndarray,
        origins: np.ndarray,
        uv: list[list[float]] | None = None,
    ) -> tuple[MeshObject, np.ndarray]:
        if len(points) > engine.settings.max_vertices or len(triangles) > engine.settings.max_faces:
            raise ScissorsError("RESOURCE_LIMIT", "Edited mesh exceeds geometry limits")
        if len(points) * 1024 + len(triangles) * 4096 > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Edited mesh exceeds working memory budget")
        if not np.isfinite(points).all():
            raise ScissorsError("GEOMETRY_ERROR", "Edited coordinates must remain finite")
        used, inverse = np.unique(triangles, return_inverse=True)
        result = target.model_copy(deep=True)
        mapped_uv = (
            np.asarray(uv)[used].tolist()
            if uv is not None
            else np.asarray(source.uv)[origins[used]].tolist()
            if source.uv is not None
            else None
        )
        mesh = trimesh.Trimesh(vertices=points[used], faces=inverse.reshape(-1, 3), process=False)
        if not np.isfinite(mesh.area_faces).all():
            raise ScissorsError("GEOMETRY_ERROR", "Edited face areas exceed numerical range")
        result.replace(mesh, mapped_uv)
        result.vertex_colors = (
            np.asarray(source.vertex_colors)[origins[used]].tolist()
            if source.vertex_colors is not None
            else None
        )
        for key in TOPOLOGY_KEYS:
            result.metadata.pop(key, None)
        # Leave revision-bound certificates and named selections unchanged. Their old
        # signatures make stale provenance explicit to their consumers.
        engine.checked(result)
        check_time()
        return result, origins[used]

    origins = np.arange(len(vertices), dtype=np.int64)
    face_origins = np.arange(len(faces), dtype=np.int64)
    selected_output = chosen.copy()
    source_deleted = False
    source_result = None
    if action in {"delete", "separate"}:
        if len(chosen) == len(faces) and action == "delete" and empty_policy != "delete_object":
            raise ValueError("Deleting every face requires empty_policy='delete_object'")
        remaining = None
        remaining_origins = np.array([], dtype=np.int64)
        if not (action == "separate" and copy_faces) and np.any(~mask):
            remaining, remaining_origins = stage(source, vertices, faces[~mask], origins)
        if action == "delete":
            result = remaining
            output_origins = remaining_origins
            face_origins = np.flatnonzero(~mask)
            selected_output = np.array([], dtype=np.int64)
        else:
            target = source.model_copy(deep=True)
            target.id = identifier("obj")
            target.name = name or f"{source.name} selected faces"
            target.transform = np.eye(4).tolist()
            # Names belong to the original object, not the extracted copy.
            target.metadata.pop("_named_selections", None)
            target.metadata.pop("_thickness_certificate", None)
            result, output_origins = stage(target, vertices, faces[chosen], origins)
            face_origins = chosen
            selected_output = np.arange(len(chosen), dtype=np.int64)
        if not (action == "separate" and copy_faces):
            if remaining is None:
                source_deleted = True
            else:
                source_result = {
                    **remaining.summary(),
                    "new_to_old_vertices": remaining_origins.tolist(),
                    "new_to_old_faces": np.flatnonzero(~mask).tolist(),
                }
        elif copy_faces:
            source_result = {
                **source.summary(),
                "new_to_old_vertices": list(range(len(vertices))),
                "new_to_old_faces": list(range(len(faces))),
            }
    elif action == "flip":
        result = flipped_faces(source, chosen)
        engine.checked(result)
        output_origins = origins
        if len(chosen) < len(faces):
            warnings.append(
                "Reversing part of a surface can leave inconsistent winding at its border."
            )
    elif action == "move":
        assert offset is not None
        delta = np.asarray(offset, dtype=float)
        if not np.any(delta):
            raise ValueError("Move offset must be nonzero")
        updated_vertices, updated_faces = vertices.copy(), faces.copy()
        if detach:
            shared = np.intersect1d(selected_vertices, np.unique(faces[~mask]))
            mapping = np.arange(len(vertices))
            mapping[shared] = np.arange(len(vertices), len(vertices) + len(shared))
            updated_vertices = np.concatenate([vertices, vertices[shared]])
            updated_faces[chosen] = mapping[faces[chosen]]
            origins = np.r_[origins, shared]
            moved = mapping[selected_vertices]
            warnings.append(
                "Detached movement can leave open boundaries between neighboring faces."
            )
        else:
            # Move every render copy at a selected position to avoid tears at UV seams.
            _, inverse = np.unique(vertices, axis=0, return_inverse=True)
            moved = np.flatnonzero(np.isin(inverse, np.unique(inverse[selected_vertices])))
            warnings.append("Moving shared vertices also deforms neighboring unselected faces.")
        updated_vertices[moved] += delta
        result, output_origins = stage(source, updated_vertices, updated_faces, origins)
    elif action == "extrude":
        assert offset is not None
        delta = np.asarray(offset, dtype=float)
        if not np.any(delta):
            raise ValueError("Extrusion offset must be nonzero")
        selected = source.model_copy(update={"faces": faces[chosen].tolist()})
        boundary_loops(selected)  # Require unbranched, directed manifold contours.
        selected_topology = topology_mesh(selected.geometry())
        if not selected_topology.is_winding_consistent:
            raise ScissorsError("GEOMETRY_ERROR", "Extrusion requires consistent selected winding")
        render_edges = faces[chosen][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        _, welded = np.unique(vertices, axis=0, return_inverse=True)
        _, inverse, counts = np.unique(
            np.sort(welded[render_edges], axis=1), axis=0, return_inverse=True, return_counts=True
        )
        border_mask = counts[inverse] == 1
        border = render_edges[border_mask]
        owners = np.repeat(np.arange(len(chosen)), 3)[border_mask]
        for component in face_components(len(chosen), selected_topology.face_adjacency):
            check_time()
            if not np.any(np.isin(owners, component)):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Every extruded component needs an open boundary"
                )
        side_count = 2 * len(border)
        planned_vertices = len(vertices) + len(selected_vertices) + 4 * len(border)
        planned_faces = len(faces) + side_count
        if (
            planned_vertices > engine.settings.max_vertices
            or planned_faces > engine.settings.max_faces
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Extrusion exceeds geometry limits")
        if planned_vertices * 1024 + planned_faces * 4096 > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Extrusion exceeds working memory budget")
        points = np.concatenate([vertices, vertices[selected_vertices] + delta]).tolist()
        origins_list = [*origins.tolist(), *selected_vertices.tolist()]
        top = np.full(len(vertices), -1, dtype=np.int64)
        top[selected_vertices] = np.arange(len(vertices), len(points))
        triangles = faces.copy()
        triangles[chosen] = top[faces[chosen]]
        side_faces = []
        uv = (
            [*source.uv, *np.asarray(source.uv)[selected_vertices].tolist()]
            if source.uv is not None
            else None
        )
        for a, b in border:
            check_time()
            start = len(points)
            points.extend(
                [
                    vertices[a].tolist(),
                    vertices[b].tolist(),
                    (vertices[b] + delta).tolist(),
                    (vertices[a] + delta).tolist(),
                ]
            )
            origins_list.extend([int(a), int(b), int(b), int(a)])
            side_faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
            if uv is not None:
                uv.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangles = np.concatenate([triangles, np.asarray(side_faces).reshape(-1, 3)])
        result, output_origins = stage(
            source, np.asarray(points), triangles, np.asarray(origins_list), uv
        )
        face_origins = np.r_[face_origins, np.full(side_count, -1, dtype=np.int64)]
        warnings.append(
            "Extruded walls use one unit UV rectangle per edge; "
            "inspect stretching and intersections."
        )
    else:
        raise ValueError("Unknown face edit action")
    check_time()
    response = {
        "object_id": result.id if result is not None else source.id,
        "scene_id": scene.id,
        "source_object_id": source.id,
        "source_geometry_revision": before_revision,
        "source_deleted": source_deleted,
        "edited_face_count": len(chosen),
        "side_face_count": side_count,
        "new_to_old_vertices": output_origins.tolist(),
        "new_to_old_faces": face_origins.tolist(),
        "warnings": warnings,
    }
    if result is not None:
        response.update(result.summary())
        response["selection"] = {
            "object_id": result.id,
            "geometry_revision": result.geometry_revision,
            "face_indices": selected_output.tolist(),
        }
    if action == "separate":
        response["source"] = source_result
        response["copied"] = copy_faces
    if len(json_bytes(response)) > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Face edit response exceeds byte budget")
    check_time()
    # Publish only validated meshes; the registry restores the entire scene if commit fails.
    if source_deleted:
        engine.remove_objects(scene.id, [source.id])
    elif action == "separate" and not copy_faces:
        assert remaining is not None
        scene.objects[source.id] = remaining
    if result is not None:
        scene.objects[result.id] = result
    return response
