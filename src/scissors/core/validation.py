from typing import Any

import numpy as np

from scissors.config import Settings
from scissors.core.mesh import MeshObject, compute_normals, face_components, topology_mesh


def validate_mesh(
    obj: MeshObject,
    settings: Settings,
    require_closed: bool = False,
    require_connected: bool = False,
    check_self_intersections: bool = False,
    include_selections: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if obj.kind in {"group", "marker"}:
        transform = np.asarray(obj.transform)
        if (
            obj.vertices
            or obj.faces
            or obj.uv is not None
            or obj.material_id is not None
            or obj.vertex_colors is not None
        ):
            errors.append("EMPTY_NODE_HAS_GEOMETRY")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            errors.append("INVALID_TRANSFORM")
        if not np.isfinite(obj.pivot).all():
            errors.append("INVALID_PIVOT")
        if require_closed:
            errors.append("NOT_CLOSED")
        if require_connected or check_self_intersections:
            errors.append("NOT_A_MESH")
        node_result = {
            "valid": not errors,
            "status": "failed" if errors else "passed",
            "errors": errors,
            "warnings": warnings,
            "properties": {"kind": obj.kind},
            "object_id": obj.id,
        }
        if include_selections:
            node_result.update({"selection_status": "not_applicable", "selections": {}})
        return node_result
    if obj.kind != "mesh":
        errors.append("INVALID_OBJECT_KIND")
    vertices, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
    if vertices.size == 0 or faces.size == 0:
        errors.append("EMPTY_GEOMETRY")
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        errors.append("INVALID_VERTICES")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        errors.append("INVALID_FACES")
    if not np.isfinite(vertices).all():
        errors.append("NONFINITE_VERTICES")
    transform = np.asarray(obj.transform)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        errors.append("INVALID_TRANSFORM")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        errors.append("INDEX_OUT_OF_RANGE")
    if len(vertices) > settings.max_vertices or len(faces) > settings.max_faces:
        errors.append("RESOURCE_LIMIT")
    if vertices.nbytes + faces.nbytes > settings.max_memory_bytes:
        errors.append("RESOURCE_LIMIT")
    properties: dict[str, Any] = {
        "self_intersections": {
            "status": "uncompleted" if check_self_intersections else "not_requested"
        }
    }
    if not errors:
        mesh = obj.geometry()
        degenerate = int(np.count_nonzero(mesh.area_faces <= 1e-14))
        if degenerate:
            errors.append("DEGENERATE_FACES")
        topology = topology_mesh(mesh)
        duplicate = len(topology.faces) - len(np.unique(np.sort(topology.faces, axis=1), axis=0))
        if duplicate:
            errors.append("DUPLICATE_FACES")
        _, counts = np.unique(topology.edges_sorted, axis=0, return_counts=True)
        if np.any(counts > 2):
            errors.append("NON_MANIFOLD_EDGES")
        if require_closed and not topology.is_watertight:
            errors.append("NOT_CLOSED")
        elif not topology.is_watertight:
            warnings.append("OPEN_MESH")
        if not topology.is_winding_consistent:
            (errors if require_closed else warnings).append("INCONSISTENT_WINDING")
        components = face_components(len(faces), topology.face_adjacency)
        if require_connected and len(components) != 1:
            errors.append("NOT_CONNECTED")
        if topology.is_watertight and topology.volume <= 0:
            (errors if require_closed else warnings).append("NONPOSITIVE_VOLUME")
        triangles = topology.triangles - topology.vertices.mean(axis=0)
        face_volumes = (
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
        )
        properties = {
            "watertight": bool(topology.is_watertight),
            "degenerate_faces": degenerate,
            "boundary_edges": int(np.count_nonzero(counts == 1)),
            "winding_consistent": bool(topology.is_winding_consistent),
            "duplicate_faces": duplicate,
            "non_manifold_edges": int(np.count_nonzero(counts > 2)),
            "surface_components": len(components),
            "component_face_counts": [len(component) for component in components],
            "component_signed_volumes": [
                float(face_volumes[component].sum()) for component in components
            ],
            "component_semantics": "shared_exact_position_edges; not solid material connectivity",
            "signed_volume": float(topology.volume),
            "orientation": "outward" if topology.is_volume else "open_or_invalid",
            "self_intersections": {"status": "not_requested"},
        }
        if check_self_intersections:
            if degenerate or duplicate or np.any(counts > 2):
                properties["self_intersections"] = {
                    "status": "uncompleted",
                    "reason": "invalid_topology",
                }
                errors.append("SELF_INTERSECTIONS_UNCOMPLETED")
            else:
                from scissors.core.inspection import self_intersections

                result = self_intersections(obj, settings, 0)
                properties["self_intersections"] = {
                    **result,
                    "status": "failed" if result["self_intersects"] else "passed",
                }
                if result["self_intersects"]:
                    errors.append("SELF_INTERSECTIONS")
        normals = np.asarray(obj.normals)
        if (
            normals.shape != vertices.shape
            or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-4)
            or (
                obj.metadata.get("normal_method") != "split_angle"
                and not np.allclose(normals, compute_normals(mesh), atol=1e-4)
            )
        ):
            warnings.append("INCORRECT_NORMALS")
    if obj.vertex_colors is not None:
        colors = np.asarray(obj.vertex_colors)
        if (
            colors.shape != (len(obj.vertices), 4)
            or not np.issubdtype(colors.dtype, np.integer)
            or np.any(colors < 0)
            or np.any(colors > 255)
        ):
            errors.append("INVALID_VERTEX_COLORS")
    if obj.uv is not None:
        uv = np.asarray(obj.uv)
        if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
            errors.append("INVALID_UV")
        elif np.any((uv < 0) | (uv > 1)):
            warnings.append("UV_OUTSIDE_UNIT_SQUARE")
    result = {
        "valid": not errors,
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "properties": properties,
        "object_id": obj.id,
    }
    if include_selections:
        from scissors.core.diagnostics import repair_selections

        result.update(repair_selections(obj, result, settings))
    return result
