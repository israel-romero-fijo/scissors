"""Attributed boolean orchestration with first/source/cut material policies."""

from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine, identifier
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, topology_mesh


def boolean_with_attributes(
    engine: Engine,
    scene_id: str,
    object_ids: list[str],
    mode: str,
    name: str,
    material_policy: str,
    cut_material_id: str | None,
    cut_uv: str,
    uv_scale: float,
) -> dict[str, Any]:
    scene = engine.scene(scene_id)
    objects = [engine.object(oid)[1] for oid in object_ids]
    if cut_material_id and engine.material(cut_material_id)[0].id != scene_id:
        raise ValueError("Cut material must belong to the boolean scene")
    estimate = sum(len(obj.faces) * 3 * 12 * 8 * 6 for obj in objects)
    if estimate > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Attributed boolean working memory exceeds budget")
    data = run_backend(
        {
            "backend": "boolean_attributes",
            "mode": mode,
            "cut_uv": cut_uv,
            "uv_scale": uv_scale,
            "source_ids": object_ids,
            "objects": [
                {
                    key: getattr(obj, key)
                    for key in ["vertices", "faces", "uv", "vertex_colors", "normals"]
                }
                for obj in objects
            ],
        },
        engine.settings,
    )
    properties, faces, sources = (
        np.asarray(data["properties"]),
        np.asarray(data["faces"]),
        np.asarray(data["face_sources"]),
    )
    whole = trimesh.Trimesh(vertices=properties[:, :3], faces=faces, process=False)
    welded = topology_mesh(whole)
    if not welded.is_volume:
        raise ScissorsError(
            "GEOMETRY_ERROR", "Attributed boolean output is not a closed oriented volume"
        )
    materials: list[str | None] = []
    for source in sources:
        cut = mode == "difference" and source != 0
        material = (
            objects[source].material_id if material_policy == "source" else objects[0].material_id
        )
        if cut:
            material = cut_material_id or material or objects[0].material_id
        materials.append(material)
    groups: dict[str | None, list[int]] = {}
    for face_index, material in enumerate(materials):
        groups.setdefault(material, []).append(face_index)
    root = None
    if len(groups) > 1:
        root = MeshObject(
            id=identifier("obj"),
            name=name,
            kind="group",
            vertices=[],
            faces=[],
            metadata={"operation": "boolean", "mode": mode},
        )
        scene.objects[root.id] = root
    ids: list[str] = []
    colored = any(obj.vertex_colors is not None for obj in objects)
    for material, selected in groups.items():
        used, mapping = np.unique(faces[selected], return_inverse=True)
        part = properties[used]
        obj = MeshObject(
            id=identifier("obj"),
            name=name if root is None else f"{name} {len(ids) + 1}",
            vertices=part[:, :3].tolist(),
            faces=mapping.reshape(-1, 3).tolist(),
            uv=part[:, 3:5].tolist(),
            normals=part[:, 9:12].tolist(),
            vertex_colors=np.clip(np.rint(part[:, 5:9]), 0, 255).astype(int).tolist()
            if colored
            else None,
            material_id=material,
            parent_id=root.id if root is not None else None,
            metadata={"primitive": "boolean", "normal_method": "split_angle", "mode": mode},
        )
        engine.checked(obj)
        scene.objects[obj.id] = obj
        ids.append(obj.id)
    result = root if root is not None else scene.objects[ids[0]]
    return {
        **result.summary(),
        "object_ids": ids,
        "material_policy": material_policy,
        "combined": {
            "volume": float(whole.volume),
            "watertight": True,
            "face_count": len(faces),
            "vertex_count": len(properties),
        },
        "cut_uv": cut_uv,
        "uv_scale": uv_scale,
        "faces_by_source": {
            oid: int(np.count_nonzero(sources == index)) for index, oid in enumerate(object_ids)
        },
        "warnings": [
            "New planar UV is measured in world units; review texture placement on cut faces.",
            "Source policy can split the closed result into open material parts under a group.",
        ],
    }
