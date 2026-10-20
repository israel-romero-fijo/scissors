"""Read-only export views, material consolidation, LODs and split artifacts."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.export import export_scene, node_names
from scissors.core.mesh import MeshObject
from scissors.core.persistence import atomic_write, json_bytes
from scissors.core.scene import Scene
from scissors.core.simplification import simplify_mesh


def selection_view(
    engine: Engine,
    scene_id: str,
    only: list[str] | None,
    exclude: list[str] | None,
    expand_subtrees: bool = True,
) -> Scene:
    scene = engine.scene(scene_id)
    for values in [only, exclude]:
        if values is not None and any(oid not in scene.objects for oid in values):
            raise ScissorsError(
                "OBJECT_NOT_FOUND", "Export selection references unknown scene objects"
            )

    def expand(values: list[str]) -> set[str]:
        result = set(values)
        for oid in values:
            result.update(obj.id for obj in engine.descendants(scene, oid))
        return result

    selected = (
        (expand(only) if expand_subtrees else set(only)) if only is not None else set(scene.objects)
    )
    selected -= expand(exclude or [])
    selected = {oid for oid in selected if engine.visible(scene, scene.objects[oid])}
    included = set(selected)
    for oid in selected:
        parent = scene.objects[oid].parent_id
        while parent:
            included.add(parent)
            parent = scene.objects[parent].parent_id
    objects = {}
    for oid, obj in scene.objects.items():
        if oid not in included:
            continue
        if oid not in selected and obj.kind == "mesh":
            obj = obj.model_copy(deep=False)
            obj.kind, obj.vertices, obj.faces, obj.normals = "group", [], [], []
            obj.uv, obj.vertex_colors, obj.material_id = None, None, None
            obj.pivot = np.asarray(obj.transform)[:3, 3].tolist()
        objects[oid] = obj
    return scene.model_copy(update={"objects": objects}, deep=False)


def consolidate_materials(engine: Engine, view: Scene) -> Scene:
    groups: dict[str | None, list[MeshObject]] = {}
    objects = {}
    for obj in view.objects.values():
        if obj.kind != "mesh":
            node = obj.model_copy(deep=False)
            node.parent_id = None
            objects[node.id] = node
        else:
            groups.setdefault(obj.material_id, []).append(obj)
    for material, meshes in groups.items():
        vertex_count = sum(len(obj.vertices) for obj in meshes)
        face_count = sum(len(obj.faces) for obj in meshes)
        if (
            vertex_count > engine.settings.max_vertices
            or face_count > engine.settings.max_faces
            or vertex_count * 768 + face_count * 384 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Export material mesh exceeds geometry budget")
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        normals: list[list[float]] = []
        uv: list[list[float]] = []
        colors: list[list[int]] = []
        textured = any(obj.uv is not None for obj in meshes)
        colored = any(obj.vertex_colors is not None for obj in meshes)
        parts = []
        for obj in meshes:
            # Validate within each source part. Coincident positions in different
            # parts are render contacts, not a request to weld their topology.
            engine.checked(obj, track_revision=False)
            start = len(vertices)
            parts.append(
                {
                    "object_id": obj.id,
                    "vertex_start": start,
                    "vertex_count": len(obj.vertices),
                    "face_start": len(faces),
                    "face_count": len(obj.faces),
                }
            )
            vertices.extend(obj.vertices)
            faces.extend((np.asarray(obj.faces) + start).tolist())
            normals.extend(obj.normals)
            if textured:
                if (
                    obj.uv is None
                    and material
                    and any(
                        value
                        for key, value in view.materials[material].model_dump().items()
                        if key.endswith("_texture")
                    )
                ):
                    raise ScissorsError(
                        "GEOMETRY_ERROR", "Textured material split requires UV on every part"
                    )
                uv.extend(obj.uv if obj.uv is not None else [[0, 0]] * len(obj.vertices))
            if colored:
                colors.extend(
                    obj.vertex_colors
                    if obj.vertex_colors is not None
                    else [[255, 255, 255, 255]] * len(obj.vertices)
                )
        oid = f"export_material_{material or 'unassigned'}"
        while oid in objects:
            oid += "_mesh"
        merged = MeshObject(
            id=oid,
            name=view.materials[material].name if material else "Unassigned",
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv=uv if textured else None,
            vertex_colors=colors if colored else None,
            material_id=material,
            metadata={
                "normal_method": "split_angle",
                "export_material_split": True,
                "export_material_parts": parts,
                "export_tangents": any(obj.metadata.get("export_tangents") for obj in meshes),
            },
        )
        # Concatenation offsets each validated part's indices into disjoint
        # ranges. A whole-solid check would incorrectly weld touching parts.
        # MeshObject validates finite arrays; source checks above validate every
        # face and attribute. This transient view never enters engine state.
        objects[oid] = merged
    return view.model_copy(update={"objects": objects}, deep=False)


def lod_view(
    engine: Engine, view: Scene, reduction: float, max_error: float
) -> tuple[Scene, list[dict[str, Any]]]:
    objects, reports = dict(view.objects), []
    for oid, original in view.objects.items():
        if original.kind != "mesh":
            continue
        obj = original.model_copy(deep=True)
        color_map: dict[tuple, list[int]] = {}
        if original.vertex_colors is not None:
            for index, vertex in enumerate(original.vertices):
                key = tuple([*vertex, *(original.uv[index] if original.uv is not None else [])])
                color = original.vertex_colors[index]
                if key in color_map and color_map[key] != color:
                    raise ScissorsError(
                        "GEOMETRY_ERROR", "Bake vertex-color seams before LOD simplification"
                    )
                color_map[key] = color
        report = simplify_mesh(obj, engine.settings, reduction, "preserve", max_error, False)
        if color_map:
            obj.vertex_colors = [
                color_map[tuple([*vertex, *(obj.uv[index] if obj.uv is not None else [])])]
                for index, vertex in enumerate(obj.vertices)
            ]
        engine.checked(obj)
        reports.append(
            {
                "object_id": oid,
                "face_count": len(obj.faces),
                "warnings": report["warnings"],
                "target_reached": report["target_reached"],
            }
        )
        objects[oid] = obj
    return view.model_copy(update={"objects": objects}, deep=False), reports


def export_delivery(
    engine: Engine,
    scene_id: str,
    output: str,
    format: str,
    only: list[str] | None,
    exclude: list[str] | None,
    transform: list[list[float]] | None,
    root_name: str | None,
    material_split: bool,
    lods: list[float] | None,
    max_error: float,
    require_target: bool,
) -> dict[str, Any]:
    view = selection_view(engine, scene_id, only, exclude)
    geometry_bytes = sum(
        len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in view.objects.values()
    )
    if geometry_bytes * (3 if lods else 2) > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Export working view exceeds geometry memory budget")
    source_view = view
    if material_split:
        view = consolidate_materials(engine, source_view)
    if lods is not None and (lods != sorted(set(lods))):
        raise ValueError("LOD reductions must be unique and increasing")
    before = sum(len(obj.faces) for obj in view.objects.values())
    result = export_scene(engine, scene_id, output, format, view, transform, root_name)
    levels = []
    previous = before
    base = engine.artifact_path(output)
    for index, reduction in enumerate(lods or [], 1):
        reduced, reports = lod_view(engine, source_view, reduction, max_error)
        if material_split:
            reduced = consolidate_materials(engine, reduced)
        count = sum(len(obj.faces) for obj in reduced.objects.values())
        target = int(np.ceil(before * (1 - reduction / 100)))
        if count >= previous or (require_target and count > target):
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "LOD did not reduce geometry to the requested budget",
                {
                    "reason": "TARGET_NOT_REACHED",
                    "level": index,
                    "faces": count,
                    "previous_faces": previous,
                    "target": target,
                },
            )
        stem = f"{base.stem}_LOD{index}"
        path = base.parent / stem / f"{stem}.{format}"
        level = export_scene(
            engine,
            scene_id,
            str(path),
            format,
            reduced,
            transform,
            f"{root_name or view.name}_LOD{index}",
        )
        levels.append(
            {
                **level,
                "level": index,
                "requested_reduction_percent": reduction,
                "face_count": count,
                "target_face_count": target,
                "target_reached": count <= target,
                "objects": reports,
            }
        )
        previous = count
    if levels:
        manifest = engine.artifact_path(base.with_suffix(".lods.json"))
        payload = {
            "format_version": 1,
            "base": base.name,
            "base_face_count": before,
            "levels": [
                {
                    "level": level["level"],
                    "path": Path(level["path"]).relative_to(base.parent).as_posix(),
                    "face_count": level["face_count"],
                }
                for level in levels
            ],
        }
        atomic_write(manifest, json_bytes(payload), engine.settings.max_file_bytes)
        result["lod_manifest"] = str(manifest)
    return {
        **result,
        "lods": levels,
        "source_face_count": before,
        "material_split": material_split,
        "exported_mesh_count": sum(obj.kind == "mesh" for obj in view.objects.values()),
        "warnings": ["Material splitting flattens mesh hierarchy into one mesh per material."]
        if material_split
        else [],
    }


def split_export(
    engine: Engine,
    scene_id: str,
    output_dir: str,
    by: str,
    format: str,
    only: list[str] | None,
    transform: list[list[float]] | None,
) -> dict[str, Any]:
    view = selection_view(engine, scene_id, only, None)
    directory = engine.artifact_path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError("Split export output_dir must be a directory")
    groups: dict[str, list[str]] = {}
    labels = {}
    if by == "material":
        for obj in view.objects.values():
            if obj.kind != "mesh":
                continue
            key = obj.material_id or "unassigned"
            groups.setdefault(key, []).append(obj.id)
            labels[key] = view.materials[key].name if obj.material_id else "Unassigned"
    else:
        for obj in view.objects.values():
            if obj.kind != "mesh":
                continue
            key = obj.id
            if by == "group":
                current = obj
                while current.parent_id:
                    current = view.objects[current.parent_id]
                    key = current.id
            groups.setdefault(key, []).append(obj.id)
            labels[key] = view.objects[key].name
    deadline = time.monotonic() + engine.settings.operation_timeout
    names = node_names([SimpleNamespace(id=key, name=labels[key]) for key in groups])
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }
    names = {
        key: (f"Object_{name}" if name.upper() in reserved else name)[:80]
        for key, name in names.items()
    }
    names = node_names([SimpleNamespace(id=key, name=name) for key, name in names.items()])
    results = []
    for key, selected in groups.items():
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Split export timed out")
        stem = names[key]
        path = directory / stem / f"{stem}.{format}"
        part = selection_view(engine, scene_id, selected, None, expand_subtrees=False)
        results.append(
            {
                "key": key,
                "object_ids": selected,
                **export_scene(engine, scene_id, str(path), format, part, transform, stem),
            }
        )
    if not results:
        raise ScissorsError("GEOMETRY_ERROR", "Split export contains no visible meshes")
    manifest = engine.artifact_path(directory / "manifest.json")
    atomic_write(
        manifest,
        json_bytes(
            {
                "format_version": 1,
                "by": by,
                "parts": [
                    {
                        "key": result["key"],
                        "path": Path(result["path"]).relative_to(directory).as_posix(),
                        "object_ids": result["object_ids"],
                    }
                    for result in results
                ],
            }
        ),
        engine.settings.max_file_bytes,
    )
    return {
        "scene_id": scene_id,
        "by": by,
        "files": results,
        "manifest": str(manifest),
        "part_count": len(results),
    }
