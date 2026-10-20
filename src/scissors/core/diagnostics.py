"""Revision-bound repair selections derived from validation defects."""

import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject, compute_normals, face_components, topology_mesh
from scissors.core.persistence import json_bytes


def repair_selections(
    obj: MeshObject, report: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    vertices, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
    if (
        obj.kind != "mesh"
        or vertices.ndim != 2
        or vertices.shape[-1] != 3
        or faces.ndim != 2
        or faces.shape[-1] != 3
        or not len(vertices)
        or not len(faces)
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        return {
            "selection_status": "unavailable",
            "selection_reason": "invalid_geometry_arrays",
            "selections": {},
        }
    if (
        len(vertices) > settings.max_vertices
        or len(faces) > settings.max_faces
        or len(vertices) * 512 + len(faces) * 4096 > settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Repair selections exceed memory budget")
    deadline = time.monotonic() + settings.operation_timeout
    revision = obj.geometry_revision
    _, representatives, welded = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    topology_faces = welded[faces]
    directed = topology_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if len(edges) > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Repair selection topology exceeds comparison budget")
    result: dict[str, Any] = {}
    used_bytes = 0

    def make(kind: str, selected: np.ndarray) -> dict[str, Any]:
        nonlocal used_bytes
        selected = np.unique(selected).astype(np.int64)
        field = {"faces": "face_indices", "edges": "edge_indices", "vertices": "vertex_indices"}[
            kind
        ]
        base = {
            "object_id": obj.id,
            "geometry_revision": revision,
            "kind": kind,
            field: selected.tolist(),
        }
        data = {**base, "count": len(selected), "empty": not len(selected)}
        if kind == "edges":
            data["edges"] = representatives[edges[selected]].tolist()
            data["face_indices"] = np.unique(
                np.flatnonzero(np.isin(inverse, selected)) // 3
            ).tolist()
            used = np.unique(representatives[edges[selected]])
        elif kind == "faces":
            used = np.unique(faces[selected])
        else:
            used = selected
        if kind != "vertices":
            data["vertex_indices"] = used.tolist()
        points = vertices[used]
        data["bounds"] = (
            [points.min(axis=0).tolist(), points.max(axis=0).tolist()] if len(points) else None
        )
        data["preview"] = {"operation": "scene.preview", "highlights": [base]}
        used_bytes += len(json_bytes(data))
        if used_bytes > min(settings.max_file_bytes, settings.max_memory_bytes // 2):
            raise ScissorsError("RESOURCE_LIMIT", "Repair selections exceed response byte budget")
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Repair selections timed out")
        return data

    result["boundary_edges"] = make("edges", np.flatnonzero(counts == 1))
    result["non_manifold_edges"] = make("edges", np.flatnonzero(counts > 2))
    _, first, duplicate_inverse = np.unique(
        np.sort(topology_faces, axis=1), axis=0, return_index=True, return_inverse=True
    )
    duplicate = np.flatnonzero(np.arange(len(faces)) != first[duplicate_inverse])
    result["duplicate_faces"] = make("faces", duplicate)
    result["degenerate_faces"] = make("faces", np.flatnonzero(obj.geometry().area_faces <= 1e-14))
    ordered = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pairs = starts[counts == 2]
    same = directed[ordered[pairs], 0] == directed[ordered[pairs + 1], 0]
    bad_edges = np.flatnonzero(counts == 2)[same]
    result["winding_edges"] = make("edges", bad_edges)
    result["inconsistent_winding"] = make(
        "faces", np.unique(np.r_[ordered[pairs[same]] // 3, ordered[pairs[same] + 1] // 3])
    )
    mesh = obj.geometry()
    topology = topology_mesh(mesh)
    groups = face_components(len(faces), topology.face_adjacency)
    components = [
        {"component_index": i, "selection": make("faces", group)} for i, group in enumerate(groups)
    ]
    result["disconnected_components"] = make(
        "faces", np.arange(len(faces)) if len(groups) > 1 else np.array([], dtype=int)
    )
    result["nonpositive_volume"] = make(
        "faces",
        np.arange(len(faces))
        if topology.is_watertight and topology.volume <= 0
        else np.array([], dtype=int),
    )
    normals = np.asarray(obj.normals)
    if normals.shape != vertices.shape:
        incorrect = np.arange(len(vertices))
    else:
        bad = ~np.isfinite(normals).all(axis=1) | ~np.isclose(
            np.linalg.norm(normals, axis=1), 1, atol=1e-4
        )
        if obj.metadata.get("normal_method") != "split_angle":
            bad |= ~np.isclose(normals, compute_normals(mesh), atol=1e-4).all(axis=1)
        incorrect = np.flatnonzero(bad)
    result["incorrect_normals"] = make("vertices", incorrect)
    result["unused_vertices"] = make(
        "vertices", np.setdiff1d(np.arange(len(vertices)), faces.ravel())
    )
    for attr, key, maximum in [
        ("uv", "invalid_uv", 1),
        ("vertex_colors", "invalid_vertex_colors", 255),
    ]:
        raw = getattr(obj, attr)
        if raw is None:
            continue
        values = np.asarray(raw)
        expected = (len(vertices), 2 if attr == "uv" else 4)
        if values.shape != expected:
            invalid = np.arange(len(vertices))
        else:
            invalid = np.flatnonzero(~np.isfinite(values).all(axis=1))
            if attr == "vertex_colors":
                invalid = np.union1d(
                    invalid, np.flatnonzero(np.any((values < 0) | (values > maximum), axis=1))
                )
                if not np.issubdtype(values.dtype, np.integer):
                    invalid = np.arange(len(vertices))
            else:
                result["uv_outside_unit_square"] = make(
                    "vertices", np.flatnonzero(np.any((values < 0) | (values > 1), axis=1))
                )
        result[key] = make("vertices", invalid)
    intersections = report["properties"].get("self_intersections", {})
    statuses = {"self_intersections": intersections.get("status", "not_requested")}
    if intersections.get("status") in {"passed", "failed"}:
        pairs_found = np.asarray(intersections.get("face_pairs", []), dtype=int)
        result["self_intersections"] = make("faces", pairs_found.ravel())
    output = {
        "selection_status": "partial"
        if statuses["self_intersections"] == "uncompleted"
        else "complete",
        "selection_statuses": statuses,
        "selections": result,
        "components": components,
    }
    if len(json_bytes(output)) > min(settings.max_file_bytes, settings.max_memory_bytes // 2):
        raise ScissorsError("RESOURCE_LIMIT", "Repair selections exceed response byte budget")
    return output
