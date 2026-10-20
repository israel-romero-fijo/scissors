"""CPU PBR baking through bounded surface correspondence, with immutable output maps."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine, identifier
from scissors.core.image_editing import read_texture
from scissors.core.materials import Material
from scissors.core.persistence import json_bytes
from scissors.core.surface_transfer import Correspondence, chosen_faces
from scissors.core.tangents import tangent_frames
from scissors.core.textures import store_image

CHANNELS = {
    "base_color": "base_color_texture",
    "normal": "normal_texture",
    "metallic_roughness": "metallic_roughness_texture",
    "emissive": "emissive_texture",
    "occlusion": "occlusion_texture",
}


def linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def sample(image: np.ndarray, uv: np.ndarray, wrap: str) -> np.ndarray:
    height, width = image.shape[:2]
    uv = np.mod(uv, 1) if wrap == "repeat" else np.clip(uv, 0, 1)
    x, y = uv[:, 0] * width - 0.5, (1 - uv[:, 1]) * height - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]

    def pick(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        if wrap == "repeat":
            return image[yy % height, xx % width]
        return image[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]

    return (1 - fy) * ((1 - fx) * pick(x0, y0) + fx * pick(x0 + 1, y0)) + fy * (
        (1 - fx) * pick(x0, y0 + 1) + fx * pick(x0 + 1, y0 + 1)
    )


def material_sample(
    engine: Engine,
    material: Material,
    channel: str,
    uv: np.ndarray,
    wrap: str,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    tid = getattr(material, CHANNELS[channel])
    if tid:
        key = f"{channel}:{tid}"
        if key not in cache:
            image = read_texture(engine, tid)
            if (
                sum(value.nbytes for value in cache.values()) + image.width * image.height * 32
                > engine.settings.max_memory_bytes // 2
            ):
                raise ScissorsError(
                    "RESOURCE_LIMIT", "Transfer texture cache exceeds memory budget"
                )
            pixels = np.asarray(image, dtype=float) / 255
            if channel in {"base_color", "emissive"}:
                pixels[:, :, :3] = linear(pixels[:, :, :3])
            cache[key] = pixels
        values = sample(cache[key], uv, wrap)
    else:
        values = np.ones((len(uv), 4))
        if channel == "normal":
            values[:, :3] = [0.5, 0.5, 1]
    if channel == "base_color":
        values *= material.base_color
        values[:, 3] *= material.alpha
    elif channel == "emissive":
        values[:, :3] *= material.emissive
    elif channel == "metallic_roughness":
        values[:, 1] *= material.roughness
        values[:, 2] *= material.metallic
    return values


def compact(obj: Any) -> tuple[Any, np.ndarray]:
    faces = np.asarray(obj.faces)
    used, inverse = np.unique(faces, return_inverse=True)
    if len(used) == len(obj.vertices):
        return obj.model_copy(deep=True), used
    clone = obj.model_copy(deep=True)
    clone.replace(
        trimesh.Trimesh(
            vertices=np.asarray(obj.vertices)[used], faces=inverse.reshape(-1, 3), process=False
        ),
        np.asarray(obj.uv)[used].tolist(),
    )
    clone.normals = np.asarray(obj.normals)[used].tolist()
    clone.metadata["normal_method"] = "split_angle"
    clone.vertex_colors = (
        np.asarray(obj.vertex_colors)[used].tolist() if obj.vertex_colors is not None else None
    )
    for key in ("primitive", "boundary_rings", "center_pairs", "instance_source"):
        clone.metadata.pop(key, None)
    return clone, used


def frames(obj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(obj.normals)
    if (
        normals.shape != (len(obj.vertices), 3)
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-4)
    ):
        raise ScissorsError("GEOMETRY_ERROR", "Normal baking requires valid unit shading normals")
    tangents = tangent_frames(obj.geometry(), np.asarray(obj.uv))
    return tangents[:, :3], np.cross(normals, tangents[:, :3]) * tangents[:, 3, None], normals


def interpolate_frame(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray], corners: np.ndarray, bary: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, b, n = [np.einsum("ni,nij->nj", bary, component[corners]) for component in frame]
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-15)
    t -= n * np.einsum("ni,ni->n", t, n)[:, None]
    t /= np.maximum(np.linalg.norm(t, axis=1)[:, None], 1e-15)
    sign = np.where(np.einsum("ni,ni->n", np.cross(n, t), b) < 0, -1, 1)
    return t, np.cross(n, t) * sign[:, None], n


def bake_transfer(
    engine: Engine,
    object_id: str,
    source_object_id: str,
    max_distance: float,
    selection: dict[str, Any] | None,
    source_selection: dict[str, Any] | None,
    resolution: int,
    channels: list[str],
    padding: int,
    wrap: str,
    min_normal_dot: float,
    owned_paths: list[Path],
) -> dict[str, Any]:
    scene, original_target = engine.object(object_id)
    source_scene, original_source = engine.object(source_object_id)
    if object_id == source_object_id:
        raise ValueError("Texture transfer requires distinct source and target objects")
    chosen = chosen_faces(original_target, selection)
    available = chosen_faces(original_source, source_selection)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Texture channels must be nonempty and unique")
    if original_target.uv is None or original_source.uv is None:
        raise ValueError("Texture transfer requires UV on source and target")
    if original_source.material_id is None or original_target.material_id is None:
        raise ValueError("Texture transfer requires source and target materials")
    pixels = resolution**2
    geometry = (len(original_target.vertices) + len(original_source.vertices)) * 512 + (
        len(original_target.faces) + len(original_source.faces)
    ) * 2048
    if (
        pixels > engine.settings.max_texture_pixels
        or pixels * (len(channels) * 64 + 128) + geometry > engine.settings.max_memory_bytes // 2
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Texture transfer exceeds image/memory limits")
    deadline = time.monotonic() + engine.settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Texture transfer timed out")

    target, target_used = compact(original_target)
    source, _ = compact(original_source)
    source_mat, target_mat = (
        source_scene.materials[source.material_id],
        scene.materials[target.material_id],
    )
    uv, faces = np.asarray(target.uv), np.asarray(target.faces)
    if (
        uv.shape != (len(target.vertices), 2)
        or not np.isfinite(uv).all()
        or np.any((uv < 0) | (uv > 1))
    ):
        raise ValueError("Target baking UV must be finite and stay in the unit square")
    owner = np.full((resolution, resolution), -1, dtype=int)
    weights = np.zeros((resolution, resolution, 3))
    interior = np.zeros((resolution, resolution), dtype=bool)
    work = 0
    for index, face in enumerate(faces):
        check_time()
        coords = uv[face] * [resolution, -resolution] + [0, resolution]
        matrix = np.column_stack([coords[1] - coords[0], coords[2] - coords[0]])
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Baking requires nondegenerate target UV triangles"
            )
        low = np.maximum(np.floor(coords.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(coords.max(axis=0)).astype(int), resolution)
        work += int(np.prod(np.maximum(high - low, 0)))
        if work > engine.settings.max_texture_pixels * 16:
            raise ScissorsError("RESOURCE_LIMIT", "Texture transfer raster work exceeds limit")
        yy, xx = np.mgrid[low[1] : high[1], low[0] : high[0]]
        points = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
        pair = (points - coords[0]) @ np.linalg.inv(matrix).T
        bary = np.column_stack([1 - pair.sum(axis=1), pair])
        inside = np.all(bary >= -1e-9, axis=1)
        strict = np.all(bary > 1e-8, axis=1)
        if np.any(interior[yy.ravel()[strict], xx.ravel()[strict]]):
            raise ScissorsError("GEOMETRY_ERROR", "Target UV triangles overlap at bake samples")
        interior[yy.ravel()[strict], xx.ravel()[strict]] = True
        px, py = xx.ravel()[inside], yy.ravel()[inside]
        unset = owner[py, px] < 0
        owner[py[unset], px[unset]] = index
        weights[py[unset], px[unset]] = bary[inside][unset]
    mask = np.isin(owner, chosen)
    py, px = np.nonzero(mask)
    target_faces = owner[py, px]
    missing = np.setdiff1d(chosen, np.unique(target_faces))
    if len(missing):
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Selected faces have no bake samples; increase resolution",
            {"face_indices": missing.tolist()},
        )
    bary_target = weights[py, px]
    world = np.einsum("ni,nij->nj", bary_target, np.asarray(target.vertices)[faces[target_faces]])
    matcher = Correspondence(source, available, engine.settings, max_distance, min_normal_dot)
    source_faces, bary_source, distances = matcher.project(
        world, target.geometry().face_normals[target_faces]
    )
    source_corners = np.asarray(source.faces)[source_faces]
    source_uv = np.einsum("ni,nij->nj", bary_source, np.asarray(source.uv)[source_corners])
    yy, xx = np.mgrid[:resolution, :resolution]
    target_uv = np.column_stack(
        [(xx.ravel() + 0.5) / resolution, 1 - (yy.ravel() + 0.5) / resolution]
    )
    normal_frames = (frames(source), frames(target)) if "normal" in channels else None
    cache: dict[str, np.ndarray] = {}
    images = {}
    for channel in channels:
        check_time()
        canvas = material_sample(engine, target_mat, channel, target_uv, wrap, cache).reshape(
            resolution, resolution, 4
        )
        values = material_sample(engine, source_mat, channel, source_uv, wrap, cache)
        if channel == "normal":
            assert normal_frames is not None
            st, sb, sn = interpolate_frame(normal_frames[0], source_corners, bary_source)
            tt, tb, tn = interpolate_frame(normal_frames[1], faces[target_faces], bary_target)
            vectors = values[:, :3] * 2 - 1
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-15)
            world_normal = st * vectors[:, :1] + sb * vectors[:, 1:2] + sn * vectors[:, 2:3]
            encoded = np.column_stack(
                [np.einsum("ni,ni->n", world_normal, axis) for axis in (tt, tb, tn)]
            )
            values[:, :3] = (encoded + 1) / 2
        canvas[py, px] = values
        filled = mask.copy()
        for _ in range(padding):
            check_time()
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                rolled = np.roll(filled, (dy, dx), axis=(0, 1))
                if dy == 1:
                    rolled[0] = False
                if dy == -1:
                    rolled[-1] = False
                if dx == 1:
                    rolled[:, 0] = False
                if dx == -1:
                    rolled[:, -1] = False
                fill = rolled & ~filled & (owner < 0)
                canvas[fill] = np.roll(canvas, (dy, dx), axis=(0, 1))[fill]
                filled |= fill
        if channel in {"base_color", "emissive"}:
            canvas[:, :, :3] = srgb(canvas[:, :, :3])
        if channel == "normal":
            v = canvas[:, :, :3] * 2 - 1
            v /= np.maximum(np.linalg.norm(v, axis=2)[:, :, None], 1e-15)
            canvas[:, :, :3] = (v + 1) / 2
        images[channel] = Image.fromarray(np.clip(np.rint(canvas * 255), 0, 255).astype(np.uint8))
    material = target_mat.model_copy(deep=True)
    material.id, material.name = identifier("mat"), f"{target_mat.name} transferred"
    results = {}
    for channel, image in images.items():
        check_time()
        relative = f"textures/{identifier('transfer')}.png"
        path = engine.path(relative)
        if path.exists():
            raise ValueError("Transfer texture output already exists")
        owned_paths.append(path)
        result = store_image(engine, image, relative)
        results[channel] = result
        setattr(material, CHANNELS[channel], result["texture_id"])
    if "base_color" in channels:
        material.base_color, material.alpha = [1, 1, 1, 1], 1
    if "metallic_roughness" in channels:
        material.metallic, material.roughness = 1, 1
    if "emissive" in channels:
        material.emissive = [1, 1, 1]
    scene.materials[material.id] = material
    target.material_id = material.id
    target.metadata["export_tangents"] = "normal" in channels or bool(
        target.metadata.get("export_tangents")
    )
    engine.checked(target)
    output = {
        **target.summary(),
        "source_object_id": source.id,
        "textures": results,
        "painted_pixels": len(px),
        "max_measured_distance": float(distances.max()),
        "new_to_old_vertices": target_used.tolist(),
        "selection": {
            "object_id": target.id,
            "geometry_revision": target.geometry_revision,
            "face_indices": chosen.tolist(),
        },
        "warnings": [
            "Baking uses nearest compatible surfaces; "
            "inspect source seams, texel density and gutters."
        ],
    }
    if len(json_bytes(output)) > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Texture transfer response exceeds byte budget")
    check_time()
    scene.objects[target.id] = target
    return output
