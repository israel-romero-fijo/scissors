"""Bounded component removal with exact topology and lossless render remapping."""

import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.cut_volume import PARTS_KEY
from scissors.core.engine import Engine
from scissors.core.face_queries import triangle_box
from scissors.core.mesh import compute_normals, face_components, topology_mesh
from scissors.core.persistence import json_bytes

MAX_COMPONENTS = 8192
TOPOLOGY_METADATA = ("boundary_rings", "center_pairs")


def delete_components(
    engine: Engine,
    object_id: str,
    bounds: list[list[float]],
    selection_rule: str,
    dry_run: bool,
    empty_policy: str,
    tolerance: float,
) -> dict[str, Any]:
    scene, source = engine.object(object_id)
    if source.kind != "mesh":
        raise ValueError("Component deletion requires one mesh object")
    low, high = np.asarray(bounds, dtype=float)
    if np.any(low > high):
        raise ValueError("Component region bounds are inverted")
    with np.errstate(over="ignore"):
        low, high = low - tolerance, high + tolerance
        midpoint, span = (low + high) / 2, high - low
    if not np.isfinite([low, high, midpoint, span]).all():
        raise ValueError("Expanded component region bounds and span must remain finite")
    deadline = time.monotonic() + engine.settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Component deletion timed out")

    vertex_count, face_count = len(source.vertices), len(source.faces)
    if (
        vertex_count > engine.settings.max_vertices
        or face_count > engine.settings.max_faces
        or vertex_count * 256 + face_count * 2048 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError(
            "RESOURCE_LIMIT", "Component analysis exceeds geometry or working memory budget"
        )
    revision = source.geometry_revision
    mesh = source.geometry()
    topology = topology_mesh(mesh)
    groups = face_components(face_count, topology.face_adjacency)
    check_time()
    if len(groups) > MAX_COMPONENTS:
        raise ScissorsError("RESOURCE_LIMIT", "Component count exceeds 8192")
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    triangles = vertices[faces]
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ScissorsError("GEOMETRY_ERROR", "Component areas must be finite and positive")
    selected = []
    keep = np.ones(face_count, dtype=bool)
    axis_tests = 0
    for component_index, indices in enumerate(groups):
        check_time()
        used = np.unique(faces[indices])
        positions = vertices[used]
        component_low, component_high = positions.min(axis=0), positions.max(axis=0)
        origin = triangles[indices[0], 0]
        area = float(areas[indices].sum())
        centers = (triangles[indices] - origin).mean(axis=1)
        centroid = origin + np.sum(centers * areas[indices, None], axis=0) / area
        if not np.isfinite(centroid).all() or not np.isfinite(area):
            raise ScissorsError("GEOMETRY_ERROR", "Component centroid is outside numerical range")
        contained = bool(np.all(component_low >= low) and np.all(component_high <= high))
        if selection_rule == "fully_inside":
            chosen = contained
        elif selection_rule == "centroid_inside":
            chosen = bool(np.all(centroid >= low) and np.all(centroid <= high))
        else:
            chosen = contained
            if not chosen and np.all(component_high >= low) and np.all(component_low <= high):
                # Bounding boxes only reject candidates. Selection requires an actual
                # triangle/box separating-axis test, including zero-width regions.
                for start in range(0, len(indices), 4096):
                    check_time()
                    chunk = triangles[indices[start : start + 4096]]
                    overlapping = np.all(chunk.max(axis=1) >= low, axis=1)
                    overlapping &= np.all(chunk.min(axis=1) <= high, axis=1)
                    candidates = chunk[overlapping]
                    axis_tests += len(candidates) * 13
                    if axis_tests > engine.settings.max_geometry_pairs:
                        raise ScissorsError(
                            "RESOURCE_LIMIT", "Component triangle-box checks exceed pair budget"
                        )
                    if len(candidates) and np.any(triangle_box(candidates, low, high)):
                        chosen = True
                        break
        if chosen:
            keep[indices] = False
            selected.append(
                {
                    "index": component_index,
                    "first_face_index": int(indices[0]),
                    "face_indices": indices.tolist(),
                    "face_count": len(indices),
                    "render_vertex_count": len(used),
                    "topology_vertex_count": len(np.unique(topology.faces[indices])),
                    "bounds": [component_low.tolist(), component_high.tolist()],
                    "area": area,
                    "centroid": centroid.tolist(),
                }
            )
    check_time()
    changed = bool(selected)
    empty = changed and not bool(np.any(keep))
    allowed = not empty or empty_policy == "delete_object"
    if not dry_run and not allowed:
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "All components selected; choose empty_policy=delete_object",
            {"object_id": object_id, "selected_component_count": len(selected)},
        )
    kept_faces = np.flatnonzero(keep)
    used_vertices = np.unique(faces[keep]) if changed else np.arange(vertex_count)
    vertex_map = np.full(vertex_count, -1, dtype=np.int64)
    vertex_map[used_vertices] = np.arange(len(used_vertices))
    face_map = np.full(face_count, -1, dtype=np.int64)
    face_map[kept_faces] = np.arange(len(kept_faces))
    removed_metadata = (
        [key for key in TOPOLOGY_METADATA if key in source.metadata] if changed else []
    )
    warnings = []
    if changed and not empty:
        removed_vertices = np.unique(faces[~keep])
        if np.intersect1d(used_vertices, removed_vertices).size:
            warnings.append(
                "Normals at shared render vertices are preserved; "
                "review shading at point contacts after removal."
            )
    staged = None
    if changed and not empty:
        staged = source.model_copy(deep=True)
        staged.vertices = vertices[used_vertices].tolist()
        staged.faces = vertex_map[faces[keep]].tolist()
        if len(source.normals) == vertex_count:
            staged.normals = np.asarray(source.normals)[used_vertices].tolist()
        else:
            staged.normals = compute_normals(staged.geometry()).tolist()
            staged.metadata.pop("normal_method", None)
            staged.metadata.pop("normal_angle_degrees", None)
            warnings.append("Missing source normals were generated for the surviving geometry.")
        staged.uv = np.asarray(source.uv)[used_vertices].tolist() if source.uv is not None else None
        staged.vertex_colors = (
            np.asarray(source.vertex_colors)[used_vertices].tolist()
            if source.vertex_colors is not None
            else None
        )
        for key in removed_metadata:
            staged.metadata.pop(key, None)
        engine.checked(staged)
    old_bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    new_bounds = (
        [vertices[used_vertices].min(axis=0).tolist(), vertices[used_vertices].max(axis=0).tolist()]
        if len(used_vertices)
        else None
    )
    before = {
        "component_count": len(groups),
        "face_count": face_count,
        "vertex_count": vertex_count,
        "bounds": old_bounds,
    }
    planned = {
        "component_count": len(groups) - len(selected),
        "face_count": len(kept_faces),
        "vertex_count": len(used_vertices),
        "bounds": new_bounds,
    }
    action = (
        "unchanged"
        if not changed
        else "reject_empty"
        if not allowed
        else "delete_object"
        if empty
        else "remove_components"
    )
    reparenting = [
        {"object_id": obj.id, "parent_id": source.parent_id}
        for obj in scene.objects.values()
        if empty and obj.parent_id == source.id
    ]
    parent = scene.objects.get(source.parent_id or "")
    update_parent = bool(
        empty and parent is not None and source.id in parent.metadata.get(PARTS_KEY, [])
    )
    result: dict[str, Any] = {
        "object_id": object_id,
        "scene_id": scene.id,
        "dry_run": dry_run,
        "selection_rule": selection_rule,
        "region": {"type": "box", "bounds": bounds},
        "tolerance": tolerance,
        "empty_policy": empty_policy,
        "allowed": allowed,
        "planned_action": action,
        "performed_action": "none" if dry_run else action,
        "source_geometry_revision": revision,
        "geometry_revision": revision
        if dry_run or not changed
        else staged.geometry_revision
        if staged is not None
        else None,
        "selected_components": selected,
        "selected_component_count": len(selected),
        "removed_component_count": 0 if dry_run else len(selected),
        "before": before,
        "planned_after": planned,
        "after": before if dry_run else planned,
        "object_deleted": bool(empty and not dry_run),
        "index_maps": {
            "old_to_new_vertices": vertex_map.tolist(),
            "new_to_old_vertices": used_vertices.tolist(),
            "old_to_new_faces": face_map.tolist(),
            "new_to_old_faces": kept_faces.tolist(),
        },
        "invalidated_metadata_keys": removed_metadata,
        "planned_reparenting": reparenting,
        "reparented_ids": [entry["object_id"] for entry in reparenting] if not dry_run else [],
        "parent_volume_parts_updated": parent.id
        if update_parent and parent is not None and not dry_run
        else None,
        "work": {
            "triangle_box_axis_tests": axis_tests,
            "pair_limit": engine.settings.max_geometry_pairs,
        },
        "warnings": warnings,
    }
    if not allowed:
        result["warnings"].append(
            "This plan would empty the mesh and cannot be applied with empty_policy=error."
        )
    if len(json_bytes(result)) > engine.settings.max_file_bytes:
        raise ScissorsError(
            "RESOURCE_LIMIT", "Component result and index maps exceed output size limit"
        )
    check_time()
    if not dry_run and changed:
        if empty:
            engine.remove_objects(scene.id, [source.id])
            if update_parent and parent is not None:
                remaining = [oid for oid in parent.metadata[PARTS_KEY] if oid != source.id]
                if remaining:
                    parent.metadata[PARTS_KEY] = remaining
                else:
                    parent.metadata.pop(PARTS_KEY, None)
                    parent.metadata.pop("_cut_volume_default_material", None)
                parent.metadata.pop("_last_cut_opening", None)
        else:
            assert staged is not None
            scene.objects[source.id] = staged
    return result
