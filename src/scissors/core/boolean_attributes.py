"""Manifold UV/color/normal interpolation and original-face provenance."""

from typing import Any

import numpy as np
import trimesh


def planar_uv(triangles: np.ndarray, scale: float) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    dominant = np.argmax(np.abs(normals), axis=1)
    result = np.empty((len(triangles), 3, 2))
    for axis in range(3):
        selected = dominant == axis
        result[selected, :, 0] = triangles[selected, :, (axis + 1) % 3]
        result[selected, :, 1] = triangles[selected, :, (axis + 2) % 3]
        result[selected, :, 0] *= np.where(normals[selected, axis] < 0, -1, 1)[:, None]
    return result / scale


def attributed_boolean(value: dict[str, Any]) -> dict[str, Any]:
    """Run in the existing isolated geometry worker, never in a transport adapter."""
    import manifold3d as m

    solids = []
    first_id = m.Manifold.reserve_ids(len(value["objects"]))
    for index, obj in enumerate(value["objects"]):
        vertices, faces = np.asarray(obj["vertices"]), np.asarray(obj["faces"])
        corners = vertices[faces]
        uv = (
            np.asarray(obj["uv"])[faces]
            if obj["uv"] is not None
            else planar_uv(corners, value["uv_scale"])
        )
        colors = (
            np.asarray(obj["vertex_colors"])[faces]
            if obj["vertex_colors"] is not None
            else np.ones((*faces.shape, 4)) * 255
        )
        normals = np.asarray(obj["normals"])[faces]
        properties = np.column_stack(
            [
                corners.reshape(-1, 3),
                uv.reshape(-1, 2),
                colors.reshape(-1, 4),
                normals.reshape(-1, 3),
            ]
        )
        _, representatives, mapping = np.unique(
            properties[:, :3], axis=0, return_index=True, return_inverse=True
        )
        target = representatives[mapping]
        source = np.flatnonzero(target != np.arange(len(properties)))
        mesh = m.Mesh64(
            np.ascontiguousarray(properties, dtype=np.float64),
            np.arange(len(properties), dtype=np.uint64).reshape(-1, 3),
            merge_from_vert=source.astype(np.uint64),
            merge_to_vert=target[source].astype(np.uint64),
            run_index=np.asarray([0, len(properties)], dtype=np.uint64),
            run_original_id=np.asarray([first_id + index], dtype=np.uint32),
            face_id=np.arange(len(faces), dtype=np.uint64),
        )
        solid = m.Manifold(mesh)
        if solid.status() != m.Error.NoError:
            raise ValueError(f"Attributed input is not manifold: {solid.status()}")
        solids.append(solid)
    operation = {
        "union": m.OpType.Add,
        "difference": m.OpType.Subtract,
        "intersection": m.OpType.Intersect,
    }[value["mode"]]
    result = m.Manifold.batch_boolean(solids, operation)
    if result.status() == m.Error.NoError and result.is_empty() and value.get("allow_empty"):
        return {"empty": True, "properties": [], "faces": [], "face_sources": []}
    if result.status() != m.Error.NoError or result.is_empty():
        raise ValueError("Boolean failed or produced empty geometry")
    mesh = result.to_mesh64(normal_idx=6)
    properties, faces = np.asarray(mesh.vert_properties), np.asarray(mesh.tri_verts)
    provenance = np.full(len(faces), -1, dtype=int)
    runs, original_ids = mesh.run_index, mesh.run_original_id
    for index, original_id in enumerate(original_ids):
        source_id = int(original_id) - first_id
        if source_id < 0 or source_id >= len(solids):
            raise ValueError("Boolean returned unknown face provenance")
        provenance[int(runs[index]) // 3 : int(runs[index + 1]) // 3] = source_id
    if np.any(provenance < 0):
        raise ValueError("Boolean output has missing face provenance")
    expanded = properties[faces].copy()
    if value["mode"] == "difference" and value["cut_uv"] == "planar":
        cuts = provenance != 0
        expanded[cuts, :, 3:5] = planar_uv(expanded[cuts, :, :3], value["uv_scale"])
    normals = expanded[:, :, 9:12]
    lengths = np.linalg.norm(normals, axis=2)
    geometric = np.cross(
        expanded[:, 1, :3] - expanded[:, 0, :3], expanded[:, 2, :3] - expanded[:, 0, :3]
    )
    geometric /= np.maximum(np.linalg.norm(geometric, axis=1)[:, None], 1e-15)
    normals = np.where((lengths <= 1e-12)[:, :, None], geometric[:, None, :], normals)
    expanded[:, :, 9:12] = normals / np.maximum(np.linalg.norm(normals, axis=2)[:, :, None], 1e-15)
    unique, inverse = np.unique(expanded.reshape(-1, 12), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    combined = trimesh.Trimesh(vertices=unique[:, :3], faces=faces, process=False)
    return {
        "empty": False,
        "changed": bool(result.volume() != solids[0].volume() or np.any(provenance != 0)),
        "properties": unique.tolist(),
        "faces": faces.tolist(),
        "face_sources": provenance.tolist(),
        "face_original_ids": np.asarray(mesh.face_id).tolist(),
        "combined_volume": float(combined.volume),
        "source_ids": value["source_ids"],
    }
