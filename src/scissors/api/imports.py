"""Native import and scaled reference-plane registry operations."""

import base64
import io
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from PIL import Image
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine, identifier
from scissors.core.geometry import run_backend
from scissors.core.image_editing import check_pixels
from scissors.core.importing import reference_geometry
from scissors.core.materials import Material
from scissors.core.mesh import MeshObject
from scissors.core.textures import load_image, store_image

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


def register_imports(registry: OperationRegistry, engine: Engine) -> None:
    def import_mesh(
        path: str,
        scene_id: str | None = None,
        name: str | None = None,
        source_up: Literal["auto", "x", "y", "z"] = "auto",
        scale: Positive = 1,
        position: Vector | None = None,
    ) -> dict[str, Any]:
        source = engine.path(path)
        if source.suffix.lower() not in {".obj", ".stl", ".glb", ".ply"}:
            raise ValueError("Import supports OBJ, STL, GLB and PLY")
        if source.stat().st_size > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Import file exceeds byte limit")
        if scene_id is not None:
            engine.scene(scene_id)
        settings = engine.settings
        data = run_backend(
            {
                "backend": "import",
                "root": str(engine.root),
                "path": str(source),
                **{
                    key: getattr(settings, key)
                    for key in [
                        "max_file_bytes",
                        "max_texture_pixels",
                        "max_memory_bytes",
                        "max_vertices",
                        "max_faces",
                    ]
                },
            },
            settings,
        )
        nodes, geometries = data["nodes"], data["geometries"]
        if len(nodes) > settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Imported hierarchy exceeds node budget")
        extra = sum(
            len(part["vertices"]) * 384 + len(part["faces"]) * 192
            for node in nodes
            for part in geometries.get(node["geometry"], [])
        )
        used = sum(
            len(obj.vertices) * 384 + len(obj.faces) * 192
            for scene in engine.scenes.values()
            for obj in scene.objects.values()
        )
        if used + extra > settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Import instances exceed workspace memory budget")
        axis = (
            ("y" if source.suffix.lower() == ".glb" else "z") if source_up == "auto" else source_up
        )
        conversion = trimesh.geometry.align_vectors(np.eye(3)["xyz".index(axis)], [0, 0, 1])
        conversion[:3, :3] *= scale
        conversion[:3, 3] = position or [0, 0, 0]
        for node in nodes:
            matrix = np.asarray(node["matrix"])
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
                raise ValueError("Imported node has a singular or nonfinite transform")
        if scene_id is None:
            scene_id = engine.create_scene(name or source.stem)["scene_id"]
        scene = engine.scene(scene_id)
        texture_ids = {}
        for key, encoded in data["textures"].items():
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                check_pixels(engine, image.width, image.height)
                texture_ids[key] = store_image(engine, image)["texture_id"]
        material_ids = []
        for properties in data["materials"]:
            values = {
                key: texture_ids.get(value) if key.endswith("_texture") else value
                for key, value in properties.items()
            }
            material = Material.model_validate({"id": identifier("mat"), **values})
            scene.materials[material.id] = material
            material_ids.append(material.id)
        root = MeshObject(
            id=identifier("obj"),
            name=name or source.stem,
            kind="group",
            vertices=[],
            faces=[],
            pivot=position or [0, 0, 0],
        )
        scene.objects[root.id] = root
        node_ids = {node["name"]: identifier("obj") for node in nodes}
        for node in nodes:
            matrix = conversion @ np.asarray(node["matrix"])
            group = MeshObject(
                id=node_ids[node["name"]],
                name=node["name"],
                kind="group",
                vertices=[],
                faces=[],
                parent_id=node_ids.get(node["parent"], root.id),
                pivot=matrix[:3, 3].tolist(),
                transform=matrix.tolist(),
            )
            scene.objects[group.id] = group
        ids = []
        sources: dict[tuple[str, int], str] = {}
        for node in nodes:
            for index, part in enumerate(geometries.get(node["geometry"], [])):
                obj = MeshObject(
                    id=identifier("obj"),
                    name=part["name"],
                    vertices=part["vertices"],
                    faces=part["faces"],
                    normals=part["normals"],
                    uv=part["uv"],
                    vertex_colors=part["vertex_colors"],
                    parent_id=node_ids[node["name"]],
                    material_id=material_ids[part["material"]]
                    if part["material"] is not None
                    else None,
                    metadata={
                        "primitive": "imported",
                        "normal_method": "split_angle",
                        "source_format": source.suffix[1:].lower(),
                    },
                )
                key = (node["geometry"], index)
                if key in sources:
                    obj.metadata["instance_source"] = sources[key]
                else:
                    sources[key] = obj.id
                scene.objects[obj.id] = obj
                engine.transform(obj.id, conversion @ np.asarray(node["matrix"]))
                ids.append(obj.id)
        return {
            "scene_id": scene_id,
            "root_object_id": root.id,
            "object_ids": ids,
            "node_ids": node_ids,
            "material_ids": material_ids,
            "texture_ids": list(texture_ids.values()),
            "source_up": axis,
            "warnings": data["diagnostics"]
            + ["Cameras, lights, skins and animation are not imported."],
        }

    def reference_plane(
        scene_id: str,
        path: str,
        axis: Literal["x", "y", "z"] = "y",
        size: Annotated[list[Positive], Field(min_length=2, max_length=2)] | None = None,
        position: Vector | None = None,
        name: str = "Reference plane",
    ) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        texture = load_image(engine, path)
        width, height = size or [1, texture["height"] / texture["width"]]
        material = Material(
            id=identifier("mat"),
            name=name,
            base_color=[1, 1, 1, 1],
            roughness=1,
            double_sided=True,
            base_color_texture=texture["texture_id"],
        )
        scene.materials[material.id] = material
        mesh, matrix = reference_geometry(width, height, axis, position or [0, 0, 0])
        result = engine.add_mesh(scene_id, mesh, name, "reference")
        obj = engine.object(result["object_id"])[1]
        obj.material_id, obj.uv = material.id, [[0, 0], [1, 0], [1, 1], [0, 1]]
        obj.metadata["reference"] = True
        engine.transform(obj.id, matrix)
        return {
            **obj.summary(),
            "texture_id": texture["texture_id"],
            "axis": axis,
            "size": [width, height],
            "warnings": ["Reference planes are ordinary geometry; hide before game export."],
        }

    registry.add_function(
        "scene.import_mesh",
        import_mesh,
        "Import local OBJ/STL/GLB/PLY into a new or existing scene; GLB Y-up becomes Z-up.",
        permissions=["scene.write", "filesystem.read", "filesystem.write"],
    )
    registry.add_function(
        "reference.plane",
        reference_plane,
        "Create a measured, double-sided image plane with world placement and UV.",
        permissions=["scene.write", "filesystem.read", "filesystem.write"],
    )
