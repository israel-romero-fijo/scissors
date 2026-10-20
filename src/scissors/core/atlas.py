"""Deterministic aligned PBR atlases and compatible leaf-mesh consolidation."""

from typing import Any, cast

import numpy as np
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine, identifier
from scissors.core.image_editing import check_pixels, read_texture
from scissors.core.materials import Material
from scissors.core.mesh import MeshObject
from scissors.core.persistence import json_bytes
from scissors.core.tangents import tangent_frames
from scissors.core.textures import store_image

SLOTS = {
    "base_color_texture": (255, 255, 255, 255),
    "normal_texture": (128, 128, 255, 255),
    "metallic_roughness_texture": (255, 255, 255, 255),
    "emissive_texture": (255, 255, 255, 255),
    "occlusion_texture": (255, 255, 255, 255),
}


def padded(image: Image.Image, padding: int) -> Image.Image:
    if padding == 0:
        return image
    width, height = image.size
    result = Image.new("RGBA", (width + 2 * padding, height + 2 * padding))
    result.paste(image, (padding, padding))
    for rectangle, destination, size in [
        ((0, 0, 1, height), (0, padding), (padding, height)),
        ((width - 1, 0, width, height), (width + padding, padding), (padding, height)),
        ((0, 0, width, 1), (padding, 0), (width, padding)),
        ((0, height - 1, width, height), (padding, height + padding), (width, padding)),
    ]:
        result.paste(image.crop(rectangle).resize(size, Image.Resampling.NEAREST), destination)
    for source, destination in [
        ((0, 0), (0, 0)),
        ((width - 1, 0), (width + padding, 0)),
        ((0, height - 1), (0, height + padding)),
        ((width - 1, height - 1), (width + padding, height + padding)),
    ]:
        result.paste(Image.new("RGBA", (padding, padding), image.getpixel(source)), destination)
    return result


def pack(
    engine: Engine,
    scene_id: str,
    object_ids: list[str] | None,
    size: list[int],
    padding: int,
    merge: bool,
) -> dict[str, Any]:
    scene = engine.scene(scene_id)
    selected = (
        object_ids
        if object_ids is not None
        else [
            obj.id
            for obj in scene.objects.values()
            if obj.kind == "mesh" and engine.visible(scene, obj)
        ]
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Atlas selection must be nonempty and contain unique objects")
    objects = []
    for oid in selected:
        owner, obj = engine.object(oid)
        if owner.id != scene_id or obj.kind != "mesh" or obj.uv is None or not obj.material_id:
            raise ValueError("Atlas objects require UV and a material in the selected scene")
        if np.any(np.asarray(obj.uv) < 0) or np.any(np.asarray(obj.uv) > 1):
            raise ValueError("Atlas UV must stay in [0,1]; tiled UV need baking before packing")
        objects.append(obj)
    materials = {
        obj.material_id: scene.materials[obj.material_id] for obj in objects if obj.material_id
    }
    channels = [
        slot
        for slot in SLOTS
        if slot == "base_color_texture" or any(getattr(mat, slot) for mat in materials.values())
    ]
    if "normal_texture" in channels:
        for obj in objects:
            tangent_frames(obj.geometry(), np.asarray(obj.uv))
    check_pixels(engine, *size)
    width, height = size
    dimensions = {}
    for mid, mat in materials.items():
        source_textures = [
            engine.textures[getattr(mat, slot)] for slot in channels if getattr(mat, slot)
        ]
        dimensions[mid] = (
            max([2, *[value["width"] for value in source_textures]]),
            max([2, *[value["height"] for value in source_textures]]),
        )
    positions = {}
    x = y = row = 0
    for mid in sorted(dimensions, key=lambda key: (-dimensions[key][1], -dimensions[key][0], key)):
        w, h = dimensions[mid]
        if w + 2 * padding > width or h + 2 * padding > height:
            raise ScissorsError("RESOURCE_LIMIT", "A texture tile is larger than the atlas")
        if x + w + 2 * padding > width:
            x, y, row = 0, y + row, 0
        if y + h + 2 * padding > height:
            raise ScissorsError("RESOURCE_LIMIT", "Textures do not fit; increase atlas size")
        positions[mid] = (x + padding, y + padding)
        x += w + 2 * padding
        row = max(row, h + 2 * padding)
    geometry_bytes = sum(
        len(obj.vertices) * 384 + len(obj.faces) * 192
        for other in engine.scenes.values()
        for obj in other.objects.values()
    )
    if geometry_bytes + width * height * 32 > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Atlas working memory exceeds workspace budget")
    grouped: dict[bytes, list[Material]] = {}
    for mat in materials.values():
        properties = {
            key: value
            for key, value in mat.model_dump().items()
            if key not in {"id", "name"} and not key.endswith("_texture")
        }
        grouped.setdefault(json_bytes(properties), []).append(mat)
    replacements = {}
    new_materials = []
    for members in grouped.values():
        mat = members[0].model_copy(deep=True)
        mat.id, mat.name = identifier("mat"), f"Atlas {members[0].name}"
        new_materials.append(mat)
        replacements.update({member.id: mat.id for member in members})
    # Validate merge budgets before writing any image. Only compatible leaf siblings merge.
    merge_groups: dict[tuple[str, str | None, bool], list[MeshObject]] = {}
    parents = {obj.parent_id for obj in scene.objects.values()}
    for obj in objects:
        if merge and obj.id not in parents:
            merge_groups.setdefault(
                (replacements[obj.material_id or ""], obj.parent_id, obj.hidden), []
            ).append(obj)
    for meshes in merge_groups.values():
        if (
            sum(len(obj.vertices) for obj in meshes) > engine.settings.max_vertices
            or sum(len(obj.faces) for obj in meshes) > engine.settings.max_faces
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Merged atlas mesh exceeds geometry limits")
    textures = {}
    for slot in channels:
        canvas = Image.new("RGBA", (width, height), SLOTS[slot])
        for mid, mat in materials.items():
            tile_size = dimensions[mid]
            tid = getattr(mat, slot)
            tile = read_texture(engine, tid) if tid else Image.new("RGBA", tile_size, SLOTS[slot])
            if tile.size != tile_size:
                tile = tile.resize(tile_size, Image.Resampling.BILINEAR)
                if slot == "normal_texture":
                    pixels = np.asarray(tile).copy()
                    vectors = pixels[..., :3].astype(float) / 127.5 - 1
                    vectors /= np.maximum(np.linalg.norm(vectors, axis=2)[..., None], 1e-15)
                    pixels[..., :3] = np.clip(np.rint((vectors + 1) * 127.5), 0, 255).astype(
                        np.uint8
                    )
                    tile = Image.fromarray(pixels)
            px, py = positions[mid]
            canvas.paste(padded(tile, padding), (px - padding, py - padding))
        textures[slot] = store_image(engine, canvas)
    for mat in new_materials:
        for slot in channels:
            setattr(mat, slot, textures[slot]["texture_id"])
        scene.materials[mat.id] = mat
    rectangles = {}
    for mid, (px, py) in positions.items():
        w, h = dimensions[mid]
        rectangles[mid] = [
            (px + 0.5) / width,
            1 - (py + h - 0.5) / height,
            (px + w - 0.5) / width,
            1 - (py + 0.5) / height,
        ]
    for obj in objects:
        old = obj.material_id or ""
        u0, v0, u1, v1 = rectangles[old]
        uv = np.asarray(obj.uv)
        obj.uv = (uv * [u1 - u0, v1 - v0] + [u0, v0]).tolist()
        obj.material_id = replacements[old]
    mapping = {obj.id: obj.id for obj in objects}
    visible_ids = [obj.id for obj in objects if engine.visible(scene, obj)]
    for meshes in merge_groups.values():
        if len(meshes) < 2:
            continue
        first = meshes[0]
        geometry = cast(
            trimesh.Trimesh, trimesh.util.concatenate([obj.geometry() for obj in meshes])
        )
        merged_uv = [coordinate for obj in meshes for coordinate in obj.uv or []]
        # Preserve shading arrays rather than smooth disconnected pieces together.
        normals = [normal for obj in meshes for normal in obj.normals]
        colors = (
            [
                color
                for obj in meshes
                for color in (
                    obj.vertex_colors
                    if obj.vertex_colors is not None
                    else [[255, 255, 255, 255]] * len(obj.vertices)
                )
            ]
            if any(obj.vertex_colors is not None for obj in meshes)
            else None
        )
        first.replace(geometry, merged_uv)
        first.normals = normals
        first.vertex_colors = colors
        first.transform = np.eye(4).tolist()
        first.metadata = {
            "primitive": "atlas_merged",
            "normal_method": "split_angle",
            "atlas_sources": [obj.id for obj in meshes],
            "pivot_explicit": first.metadata.get("pivot_explicit", False),
            "export_tangents": any(obj.metadata.get("export_tangents", False) for obj in meshes),
        }
        engine.checked(first)
        for obj in meshes[1:]:
            mapping[obj.id] = first.id
            del scene.objects[obj.id]
    return {
        "scene_id": scene_id,
        "textures": textures,
        "rectangles_by_material": rectangles,
        "material_mapping": replacements,
        "object_mapping": mapping,
        "draw_calls_before": len(visible_ids),
        "draw_calls_after": len({mapping[oid] for oid in visible_ids}),
        "warnings": [
            "Padding protects base-level filtering, not every mip level.",
            "Only leaf siblings with equal PBR factors and visibility merge.",
        ],
    }
