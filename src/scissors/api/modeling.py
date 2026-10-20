"""Native anatomical modeling operations, shared by every transport."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import Field

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.export import compare_reference as render_reference
from scissors.core.organic import brush_weights, edit_surface, reconstruct
from scissors.core.simplification import simplify_mesh
from scissors.core.surface_textures import project_texture
from scissors.core.uv import unwrap_atlas

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Scale = Annotated[list[Positive], Field(min_length=3, max_length=3)]
ImageSize = Annotated[int, Field(ge=64, le=4096)]


def register_modeling(registry: OperationRegistry, engine: Engine) -> None:
    def organic_union(
        scene_id: str,
        object_ids: Annotated[list[str], Field(min_length=2, max_length=32)],
        voxel_size: Positive,
        blend_radius: Positive,
        name: str = "Organic union",
    ) -> dict[str, Any]:
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Object IDs must be unique")
        objects = [engine.object(oid) for oid in object_ids]
        if any(scene.id != scene_id for scene, _ in objects):
            raise ValueError("All source objects must belong to the destination scene")
        mesh = reconstruct(
            [obj.geometry() for _, obj in objects], voxel_size, blend_radius, engine.settings
        )
        return {
            **engine.add_mesh(scene_id, mesh, name),
            "voxel_size": voxel_size,
            "blend_radius": blend_radius,
            "warnings": [
                "Sources retained. Hide or delete them for export. UV and materials are not "
                "transferred.",
                "Features near voxel size may be lost; smooth union can change volume.",
            ],
        }

    def local_deform(
        object_id: str,
        center: Vector,
        radius: Positive,
        displacement: Vector | None = None,
        scale: Scale | None = None,
        axis: Literal["x", "y", "z"] = "z",
        angle_degrees: Finite = 0,
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        vertices = np.asarray(obj.vertices)
        weights = brush_weights(vertices, center, radius)
        if not np.any(weights):
            raise ValueError("Region does not contain any vertices")
        rotation = trimesh.transformations.rotation_matrix(
            np.radians(angle_degrees), np.eye(3)["xyz".index(axis)]
        )[:3, :3]
        target = (
            ((vertices - center) * (scale if scale is not None else [1, 1, 1])) @ rotation.T
            + center
            + np.asarray(displacement if displacement is not None else [0, 0, 0])
        )
        vertices = vertices + weights[:, None] * (target - vertices)
        obj.replace(trimesh.Trimesh(vertices=vertices, faces=obj.faces, process=False), obj.uv)
        obj.metadata.pop("primitive", None)
        engine.checked(obj)
        return {
            **obj.summary(),
            "affected_vertices": int(np.count_nonzero(weights)),
            "warnings": [
                "Local affine edit preserves UV and faces; inspect for self-intersections."
            ],
        }

    def sculpt_brush(
        object_id: str,
        center: Vector,
        radius: Positive,
        mode: Literal["inflate", "deflate", "move", "smooth", "sharpen"] = "inflate",
        strength: Finite = 0.05,
        displacement: Vector | None = None,
        iterations: Annotated[int, Field(ge=1, le=50)] = 1,
    ) -> dict[str, Any]:
        if mode in {"smooth", "sharpen"} and not 0 < strength <= 1:
            raise ValueError("Smoothing and sharpening strength must be in (0, 1]")
        if mode in {"inflate", "deflate"} and strength <= 0:
            raise ValueError("Inflate and deflate strength must be positive")
        obj = engine.object(object_id)[1]
        signed_strength = -strength if mode in {"deflate", "sharpen"} else strength
        result = edit_surface(
            obj,
            center,
            radius,
            "inflate" if mode == "deflate" else mode,
            displacement if displacement is not None else [0, 0, 0],
            signed_strength,
            iterations,
        )
        engine.checked(obj)
        return result

    def remesh(
        object_id: str,
        voxel_size: Positive,
        reduction_percent: Annotated[float, Field(ge=0, lt=100, allow_inf_nan=False)] = 0,
        max_error_relative: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.001,
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        mesh = reconstruct([obj.geometry()], voxel_size, 0, engine.settings)
        obj.replace(mesh)
        obj.metadata.pop("primitive", None)
        result = (
            simplify_mesh(
                obj, engine.settings, reduction_percent, "discard", max_error_relative, False
            )
            if reduction_percent
            else obj.summary()
        )
        engine.checked(obj)
        return {
            **result,
            "voxel_size": voxel_size,
            "warnings": result.get("warnings", [])
            + [
                "UV cleared; regenerate before export. Material retained.",
                "Uniform reconstruction followed by optional conservative reduction; "
                "sub-voxel details can disappear.",
                "Requires closed oriented input; does not guarantee repair of arbitrary "
                "self-intersections.",
            ],
        }

    def uv_unwrap(
        object_id: str,
        face_groups: list[Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]]
        | None = None,
        density_weights: list[Positive] | None = None,
        resolution: ImageSize = 1024,
        padding: Annotated[int, Field(ge=0, le=64)] = 4,
        max_anisotropy: Annotated[float, Field(ge=1, allow_inf_nan=False)] | None = None,
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        result = unwrap_atlas(
            obj, engine.settings, face_groups, density_weights, resolution, padding
        )
        distortion = result["distortion"]
        if distortion["degenerate_triangles"] or (
            max_anisotropy is not None and distortion["max_anisotropy"] > max_anisotropy
        ):
            from scissors.api.errors import ScissorsError

            raise ScissorsError(
                "GEOMETRY_ERROR", "Atlas exceeds requested distortion limit", distortion
            )
        engine.checked(obj)
        return result

    def texture_project(
        object_id: str,
        texture_id: str,
        center: Vector,
        radius: Positive,
        size: Annotated[list[Positive], Field(min_length=2, max_length=2)],
        u_axis: Vector | None = None,
        v_axis: Vector | None = None,
        opacity: Unit = 1,
        resolution: ImageSize = 1024,
        front_only: bool = True,
    ) -> dict[str, Any]:
        return project_texture(
            engine,
            object_id,
            texture_id,
            center,
            radius,
            u_axis if u_axis is not None else [1, 0, 0],
            v_axis if v_axis is not None else [0, 1, 0],
            size,
            opacity,
            resolution,
            front_only,
        )

    def compare_reference(
        scene_id: str,
        reference_path: str,
        output: str = "previews/reference.png",
        mode: Literal["side_by_side", "overlay"] = "side_by_side",
        opacity: Unit = 0.5,
        width: ImageSize = 512,
        height: ImageSize = 512,
        azimuth_degrees: Finite = 35,
        elevation_degrees: Finite = 65,
        zoom: Annotated[float, Field(gt=0, le=20, allow_inf_nan=False)] = 1,
        offset_x: Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)] = 0,
        offset_y: Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)] = 0,
    ) -> dict[str, Any]:
        return render_reference(
            engine,
            scene_id,
            reference_path,
            output,
            mode,
            opacity,
            width,
            height,
            azimuth_degrees,
            elevation_degrees,
            zoom,
            offset_x,
            offset_y,
        )

    operations: list[tuple[str, Callable[..., Any], str]] = [
        (
            "mesh.organic_union",
            organic_union,
            "Blend closed volumes using a sampled signed-distance field. Uses the included "
            "VTK backend. Retains sources; creates geometry without UV or material.",
        ),
        (
            "deform.local",
            local_deform,
            "Translate, rotate and scale a spherical region with smooth falloff; retains UV "
            "and topology.",
        ),
        (
            "sculpt.brush",
            sculpt_brush,
            "Inflate, deflate, move, smooth or sharpen vertices with a spherical brush. "
            "Strength is world distance for volume brushes, a fraction for smoothing.",
        ),
        (
            "mesh.remesh",
            remesh,
            "Reconstruct a closed volume at voxel_size, optionally reduce faces with "
            "bounded geometric error. Uses the included VTK backend. Clears UV; fine detail can be "
            "lost.",
        ),
        (
            "uv.unwrap",
            uv_unwrap,
            "Pack a non-repeating xatlas UV atlas. Face groups force region cuts; density "
            "weights allocate relative texel area. Uses the included xatlas backend.",
        ),
    ]
    for name, function, description in operations:
        registry.add_function(name, function, description)
    registry.add_function(
        "texture.project",
        texture_project,
        "Bake a feathered orthographic image into UV and assign a new base-color texture. "
        "Requires non-overlapping UV and a material; affects all objects sharing that "
        "material.",
        permissions=["scene.write", "filesystem.write"],
    )
    registry.add_function(
        "scene.compare_reference",
        compare_reference,
        "Write a CPU PNG beside a workspace photo or overlay the solid-color model. Camera "
        "keeps world +Z as screen up: azimuth 0 looks along -Y, 90 along +X, 180 along +Y and "
        "270 along -X, while elevation 90 looks straight down. No perspective matching or "
        "texture sampling.",
        permissions=["filesystem.write"],
    )
    examples: dict[str, dict[str, Any]] = {
        "mesh.organic_union": {
            "scene_id": "SCENE_ID",
            "object_ids": ["BODY_ID", "NECK_ID"],
            "voxel_size": 0.03,
            "blend_radius": 0.2,
        },
        "deform.local": {
            "object_id": "OBJECT_ID",
            "center": [0, 0, 1],
            "radius": 0.5,
            "displacement": [0.1, 0, 0],
        },
        "sculpt.brush": {
            "object_id": "OBJECT_ID",
            "center": [0, 0, 1],
            "radius": 0.2,
            "mode": "deflate",
            "strength": 0.03,
        },
        "mesh.remesh": {"object_id": "OBJECT_ID", "voxel_size": 0.02},
        "uv.unwrap": {"object_id": "OBJECT_ID"},
        "texture.project": {
            "object_id": "OBJECT_ID",
            "texture_id": "TEXTURE_ID",
            "center": [0, 0, 1],
            "radius": 0.5,
            "size": [1, 1],
        },
        "scene.compare_reference": {
            "scene_id": "SCENE_ID",
            "reference_path": "references/animal.png",
        },
    }
    for name, example in examples.items():
        registry.operations[name].examples = [example]
    # Keep discovery useful for clients that validate compact output envelopes.
    mesh_properties = {
        "object_id": {"type": "string"},
        "id": {"type": "string"},
        "vertex_count": {"type": "integer", "minimum": 1},
        "face_count": {"type": "integer", "minimum": 1},
        "has_uv": {"type": "boolean"},
        "watertight": {"type": "boolean"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    }
    for name, _, _ in operations:
        registry.operations[name].result_schema = {
            "type": "object",
            "required": list(mesh_properties),
            "properties": mesh_properties,
        }
    registry.operations["texture.project"].result_schema = {
        "type": "object",
        "required": ["texture_id", "object_id", "path", "painted_pixels", "warnings"],
        "properties": {
            "texture_id": {"type": "string"},
            "object_id": {"type": "string"},
            "path": {"type": "string"},
            "painted_pixels": {"type": "integer", "minimum": 1},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }
    registry.operations["scene.compare_reference"].result_schema = {
        "type": "object",
        "required": ["path", "format", "size_bytes", "diagnostics"],
        "properties": {
            "path": {"type": "string"},
            "format": {"const": "png"},
            "size_bytes": {"type": "integer", "minimum": 1},
            "diagnostics": {"type": "array", "items": {"type": "string"}},
        },
    }
