"""Atomic multi-object differences with inline world-space box cutters."""

import copy
import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.boolean_attributes import planar_uv
from scissors.core.engine import Engine, identifier
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, compute_normals, topology_mesh

MAX_CUT_TARGETS = 128
PARTS_KEY = "_cut_volume_parts"


def source_parts(engine: Engine, scene_id: str, obj: MeshObject) -> list[MeshObject]:
    """Resolve a material-split volume without treating open parts as solids."""
    if obj.kind == "mesh":
        return [obj]
    ids = obj.metadata.get(PARTS_KEY)
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ScissorsError("GEOMETRY_ERROR", "Cut volume group has invalid part references")
    scene = engine.scene(scene_id)
    parts = [scene.objects.get(oid) for oid in ids]
    if any(part is None or part.kind != "mesh" or part.parent_id != obj.id for part in parts):
        raise ScissorsError("GEOMETRY_ERROR", "Cut volume group parts changed; repair the group")
    return [part for part in parts if part is not None]


def combine_parts(parts: list[MeshObject], uv_scale: float) -> tuple[MeshObject, list[str | None]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    uv: list[list[float]] = []
    normals: list[list[float]] = []
    colors: list[list[int]] = []
    materials: list[str | None] = []
    colored = any(part.vertex_colors is not None for part in parts)
    for part in parts:
        # Expand corners to retain every UV and normal seam, including missing-UV parts.
        mesh = part.geometry()
        indices = np.asarray(part.faces)
        corners = np.asarray(part.vertices)[indices]
        offset = len(vertices)
        vertices.extend(corners.reshape(-1, 3).tolist())
        faces.extend((np.arange(len(indices) * 3).reshape(-1, 3) + offset).tolist())
        uv.extend(
            (np.asarray(part.uv)[indices] if part.uv is not None else planar_uv(corners, uv_scale))
            .reshape(-1, 2)
            .tolist()
        )
        ns = (
            np.asarray(part.normals)
            if len(part.normals) == len(part.vertices)
            else compute_normals(mesh)
        )
        normals.extend(ns[indices].reshape(-1, 3).tolist())
        cs = (
            np.asarray(part.vertex_colors)
            if part.vertex_colors is not None
            else np.full((len(part.vertices), 4), 255)
        )
        colors.extend(cs[indices].reshape(-1, 4).tolist())
        materials.extend([part.material_id] * len(indices))
    return MeshObject(
        id=parts[0].id,
        name=parts[0].name,
        vertices=vertices,
        faces=faces,
        uv=uv,
        normals=normals,
        vertex_colors=colors if colored else None,
    ), materials


def cut_contours(properties: np.ndarray, faces: np.ndarray, sources: np.ndarray) -> dict[str, Any]:
    """Report only unbranched boundaries of newly exposed cut surfaces."""
    selected = faces[sources != 0]
    if not len(selected):
        return {"status": "unavailable", "reason": "No new cut surface", "contours": []}
    positions, inverse = np.unique(properties[:, :3], axis=0, return_inverse=True)
    triangles = inverse[selected]
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency: dict[int, list[int]] = {}
    for a, b in edges[counts == 1]:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return {
            "status": "ambiguous",
            "reason": "Cut boundary is missing or branched",
            "contours": [],
        }
    unseen = set(adjacency)
    loops = []
    while unseen:
        start = min(unseen)
        path, previous, current = [], -1, start
        while current in unseen:
            unseen.remove(current)
            path.append(current)
            neighbors = sorted(adjacency[current])
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            previous, current = current, following
        if current != start or len(path) < 3:
            return {
                "status": "ambiguous",
                "reason": "Cut boundary is not a closed loop",
                "contours": [],
            }
        loops.append({"points": positions[path].tolist(), "closed": True})
    return {
        "status": "contours",
        "reason": "Choose a contour explicitly; rim side is not inferred",
        "contours": loops,
    }


def cut_volume(
    engine: Engine,
    scene_id: str,
    target_ids: list[str],
    minimum: list[float],
    maximum: list[float],
    cut_material_id: str | None,
    uv_scale: float,
    dry_run: bool,
    empty_policy: str,
) -> dict[str, Any]:
    scene = engine.scene(scene_id)
    lo, hi = np.asarray(minimum), np.asarray(maximum)
    if not np.all(np.isfinite([lo, hi])) or np.any(hi <= lo):
        raise ValueError("Box max must be finite and strictly greater than min on every axis")
    if cut_material_id is not None and engine.material(cut_material_id)[0].id != scene_id:
        raise ValueError("Cut material must belong to the selected scene")

    chosen: dict[str, MeshObject] = {}
    for oid in target_ids:
        owner, obj = engine.object(oid)
        if owner.id != scene_id:
            raise ValueError("Cut targets must belong to the selected scene")
        for candidate in [obj, *engine.descendants(scene, oid)]:
            if candidate.kind == "mesh" or PARTS_KEY in candidate.metadata:
                chosen[candidate.id] = candidate
    # Selecting a generated material part selects its whole logical volume.
    for candidate in list(chosen.values()):
        parent = scene.objects.get(candidate.parent_id or "")
        if parent is not None and candidate.id in parent.metadata.get(PARTS_KEY, []):
            chosen[parent.id] = parent
    part_ids = {
        part.id
        for obj in chosen.values()
        if PARTS_KEY in obj.metadata
        for part in source_parts(engine, scene_id, obj)
    }
    targets = [obj for oid, obj in sorted(chosen.items()) if oid not in part_ids]
    if not targets:
        raise ValueError("Cut selection contains no mesh volumes")
    if len(targets) > MAX_CUT_TARGETS:
        raise ScissorsError(
            "RESOURCE_LIMIT", "Cut target count exceeds limit", {"limit": MAX_CUT_TARGETS}
        )
    sources = [(obj, source_parts(engine, scene_id, obj)) for obj in targets]
    face_count = sum(len(part.faces) for _, parts in sources for part in parts)
    if face_count > engine.settings.max_faces or face_count * 3 > engine.settings.max_vertices:
        raise ScissorsError("RESOURCE_LIMIT", "Cut input exceeds aggregate geometry budget")
    if face_count * 3 * 12 * 8 * 6 > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Cut working memory exceeds budget")

    cutter_mesh = trimesh.creation.box(extents=hi - lo)
    cutter_mesh.apply_translation((hi + lo) / 2)
    cutter = MeshObject(
        id="inline_box",
        name="Inline box",
        vertices=cutter_mesh.vertices.tolist(),
        faces=cutter_mesh.faces.tolist(),
        normals=compute_normals(cutter_mesh).tolist(),
    )
    plans: list[Any] = []
    output_vertices = 0
    output_faces = 0
    deadline = time.monotonic() + engine.settings.operation_timeout
    for obj, parts in sources:
        combined, materials = combine_parts(parts, uv_scale)
        bounds = combined.geometry().bounds
        if bounds is None:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Cut target has no geometry", {"object_id": obj.id}
            )
        if not topology_mesh(combined.geometry()).is_volume:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Cut requires a closed oriented volume", {"object_id": obj.id}
            )
        if np.any(bounds[1] <= lo) or np.any(bounds[0] >= hi):
            plans.append((obj, parts, None, None, None))
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ScissorsError("RESOURCE_LIMIT", "Cut exceeded its total time budget")
        try:
            data = run_backend(
                {
                    "backend": "boolean_attributes",
                    "mode": "difference",
                    "cut_uv": "planar",
                    "uv_scale": uv_scale,
                    "allow_empty": True,
                    "source_ids": [obj.id, "inline_box"],
                    "objects": [
                        {
                            key: getattr(operand, key)
                            for key in ["vertices", "faces", "uv", "vertex_colors", "normals"]
                        }
                        for operand in [combined, cutter]
                    ],
                },
                engine.settings.model_copy(update={"operation_timeout": remaining}),
            )
        except ScissorsError as exc:
            raise ScissorsError(exc.code, str(exc), {**exc.details, "object_id": obj.id}) from exc
        if data["empty"]:
            if empty_policy == "error":
                raise ScissorsError(
                    "GEOMETRY_ERROR",
                    "Cutter removes the entire target; choose empty_policy=delete_object",
                    {"object_id": obj.id},
                )
            if any(
                child.id not in {p.id for p in parts} for child in engine.descendants(scene, obj.id)
            ):
                raise ValueError("Cannot delete an empty target with unrelated descendants")
            plans.append((obj, parts, data, None, None))
            continue
        if not data["changed"]:
            plans.append((obj, parts, None, None, None))
            continue
        properties, faces, provenance = (
            np.asarray(data["properties"]),
            np.asarray(data["faces"]),
            np.asarray(data["face_sources"]),
        )
        if len(properties) > engine.settings.max_vertices or len(faces) > engine.settings.max_faces:
            raise ScissorsError("RESOURCE_LIMIT", "Cut output exceeds geometry budget")
        output_vertices += len(properties)
        output_faces += len(faces)
        if (
            output_vertices > engine.settings.max_vertices
            or output_faces > engine.settings.max_faces
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Cut outputs exceed aggregate geometry budget")
        if not topology_mesh(
            trimesh.Trimesh(vertices=properties[:, :3], faces=faces, process=False)
        ).is_volume:
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Cut result is not a closed oriented volume",
                {"object_id": obj.id},
            )
        groups: dict[str | None, list[int]] = {}
        default_material = obj.metadata.get(
            "_cut_volume_default_material",
            obj.material_id if obj.kind == "mesh" else parts[0].material_id,
        )
        for index, source in enumerate(provenance):
            material = (
                (cut_material_id if cut_material_id is not None else default_material)
                if source
                else materials[data["face_original_ids"][index]]
            )
            groups.setdefault(material, []).append(index)
        outputs = []
        for material, selected in groups.items():
            used, mapping = np.unique(faces[selected], return_inverse=True)
            values = properties[used]
            part = MeshObject(
                id=identifier("obj"),
                name=obj.name,
                vertices=values[:, :3].tolist(),
                faces=mapping.reshape(-1, 3).tolist(),
                uv=values[:, 3:5].tolist(),
                normals=values[:, 9:12].tolist(),
                vertex_colors=np.clip(np.rint(values[:, 5:9]), 0, 255).astype(int).tolist()
                if combined.vertex_colors is not None
                else None,
                material_id=material,
                hidden=obj.hidden,
                transform=obj.transform,
                pivot=obj.pivot,
                metadata={"normal_method": "split_angle", "operation": "cut_volume"},
            )
            engine.checked(part)
            outputs.append(part)
        contours = cut_contours(properties, faces, provenance)
        plans.append((obj, parts, data, outputs, contours))

    if time.monotonic() > deadline:
        raise ScissorsError("RESOURCE_LIMIT", "Cut exceeded its total time budget")
    reports: list[dict[str, Any]] = []
    changed: list[str] = []
    unchanged: list[str] = []
    deleted: list[str] = []
    for obj, parts, data, outputs, contours in plans:
        report: dict[str, Any] = {
            "object_id": obj.id,
            "before_face_count": sum(len(p.faces) for p in parts),
        }
        if data is None:
            unchanged.append(obj.id)
            report.update(status="unchanged", after_face_count=report["before_face_count"])
        elif data["empty"]:
            deleted.append(obj.id)
            report.update(status="deleted", after_face_count=0, empty=True)
            if not dry_run:
                engine.remove_objects(scene_id, [p.id for p in parts if p.id != obj.id] + [obj.id])
        else:
            changed.append(obj.id)
            assert outputs is not None and contours is not None
            report.update(
                status="changed",
                after_face_count=len(data["faces"]),
                material_ids=[p.material_id for p in outputs],
                attributes={
                    "uv": "source interpolation; world-space planar cut UV",
                    "normals": "interpolated and normalized",
                    "vertex_colors": "interpolated" if outputs[0].vertex_colors else "absent",
                },
                opening=contours,
                prior_part_ids=[p.id for p in parts if p.id != obj.id],
            )
            if not dry_run:
                for part in parts:
                    if part.id != obj.id:
                        # Keep unrelated children attached to the stable logical root.
                        for child in scene.objects.values():
                            if child.parent_id == part.id:
                                child.parent_id = obj.id
                        del scene.objects[part.id]
                if len(outputs) == 1:
                    result = outputs[0]
                    result.id, result.name, result.parent_id = obj.id, obj.name, obj.parent_id
                    result.metadata = {**obj.metadata, **result.metadata}
                    result.metadata.pop(PARTS_KEY, None)
                    engine.checked(result)
                    scene.objects[obj.id] = result
                    report["part_ids"] = [obj.id]
                    report["geometry_revision"] = result.geometry_revision
                else:
                    root = obj.model_copy(deep=True)
                    root.kind, root.vertices, root.faces, root.normals = "group", [], [], []
                    root.uv, root.vertex_colors, root.material_id = None, None, None
                    root.metadata[PARTS_KEY] = [p.id for p in outputs]
                    root.metadata.setdefault("_cut_volume_default_material", obj.material_id)
                    root.metadata["operation"] = "cut_volume"
                    engine.checked(root)
                    scene.objects[root.id] = root
                    for part in outputs:
                        part.parent_id = root.id
                        scene.objects[part.id] = part
                    report["part_ids"] = root.metadata[PARTS_KEY]
                    report["geometry_revision"] = combine_parts(outputs, uv_scale)[
                        0
                    ].geometry_revision
                contours["object_id"] = obj.id
                contours["geometry_revision"] = report["geometry_revision"]
                scene.objects[obj.id].metadata["_last_cut_opening"] = copy.deepcopy(contours)
            else:
                contours["status"] = "preview_" + contours["status"]
        reports.append(report)
    return {
        "scene_id": scene_id,
        "dry_run": dry_run,
        "atomic": True,
        "changed_ids": changed,
        "unchanged_ids": unchanged,
        "deleted_ids": deleted,
        "targets": reports,
        "warnings": [
            "Different cut materials retain the target ID as a group with material parts; "
            "part IDs may change on later cuts.",
            "Opening contours require explicit selection; no rim side is inferred.",
        ],
    }
