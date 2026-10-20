"""Bake a feathered orthographic image projection into an object's UV atlas."""

from typing import Any

import numpy as np
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.organic import brush_weights
from scissors.core.textures import store_image


def project_texture(
    engine: Engine,
    object_id: str,
    texture_id: str,
    center: list[float],
    radius: float,
    u_axis: list[float],
    v_axis: list[float],
    size: list[float],
    opacity: float,
    resolution: int,
    front_only: bool,
) -> dict[str, Any]:
    scene, obj = engine.object(object_id)
    if obj.uv is None or obj.material_id is None:
        raise ScissorsError("GEOMETRY_ERROR", "Projection requires UV and an assigned material")
    if texture_id not in engine.textures:
        raise ScissorsError("TEXTURE_NOT_FOUND", "Projection image does not exist")
    uv = np.asarray(obj.uv)
    if np.any((uv < 0) | (uv > 1)):
        raise ScissorsError("GEOMETRY_ERROR", "Projection requires UV inside the unit square")
    u, v = np.asarray(u_axis, dtype=float), np.asarray(v_axis, dtype=float)
    if min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-12:
        raise ValueError("Projection axes must be nonzero")
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    if abs(float(u @ v)) > 1e-6:
        raise ValueError("Projection axes must be perpendicular")
    pixels = resolution**2
    if (
        pixels > engine.settings.max_texture_pixels
        or pixels * 256 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Projection exceeds pixel or memory budget")
    material = scene.materials[obj.material_id]

    def read(tid: str) -> np.ndarray:
        path = engine.path(engine.textures[tid]["path"])
        if path.stat().st_size > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Projection texture exceeds file limit")
        with Image.open(path) as image:
            if image.width * image.height > engine.settings.max_texture_pixels or (
                image.width * image.height * 64 + pixels * 256 > engine.settings.max_memory_bytes
            ):
                raise ScissorsError("RESOURCE_LIMIT", "Projection texture exceeds memory budget")
            return np.asarray(image.convert("RGBA"), dtype=float) / 255

    source = read(texture_id)
    had_texture = material.base_color_texture is not None
    if material.base_color_texture:
        base = read(material.base_color_texture)
        base_image = Image.fromarray(np.round(base * 255).astype(np.uint8))
        canvas = (
            np.asarray(
                base_image.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=float
            )
            / 255
        )
    else:
        canvas = np.broadcast_to(material.base_color, (resolution, resolution, 4)).copy()
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    covered = np.zeros_like(occupancy)
    vertices = np.asarray(obj.vertices)
    work = 0
    changed = 0
    for face in np.asarray(obj.faces):
        coords = uv[face].copy()
        coords[:, 0] *= resolution
        coords[:, 1] = (1 - coords[:, 1]) * resolution
        low = np.maximum(np.floor(coords.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(coords.max(axis=0)).astype(int), resolution)
        width, height = high - low
        work += int(width * height)
        if work > engine.settings.max_texture_pixels * 16:
            raise ScissorsError("RESOURCE_LIMIT", "Projection raster work exceeds limit")
        matrix = np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ScissorsError("GEOMETRY_ERROR", "Projection requires nondegenerate UV triangles")
        y, x = np.mgrid[low[1] : high[1], low[0] : high[0]]
        points = np.column_stack((x.reshape(-1) + 0.5, y.reshape(-1) + 0.5))
        pair = (points - coords[0]) @ np.linalg.inv(matrix).T
        bary = np.column_stack((1 - pair.sum(axis=1), pair))
        inside = np.all(bary >= -1e-9, axis=1)
        interior = np.all(bary > 1e-8, axis=1)
        px, py = x.reshape(-1)[inside], y.reshape(-1)[inside]
        ix, iy = x.reshape(-1)[interior], y.reshape(-1)[interior]
        if occupancy[iy, ix].any():
            raise ScissorsError("GEOMETRY_ERROR", "UV triangles overlap; unwrap an atlas first")
        occupancy[iy, ix] = True
        if not len(px):
            continue
        world = bary[inside] @ vertices[face]
        relative = world - center
        projected = np.column_stack((relative @ u / size[0] + 0.5, relative @ v / size[1] + 0.5))
        weights = brush_weights(world, center, radius) * opacity
        weights *= np.all((projected >= 0) & (projected <= 1), axis=1)
        normal = np.cross(
            vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]]
        )
        if front_only and normal @ np.cross(u, v) <= 0:
            weights[:] = 0
        # Pixel centers shared by two triangle edges are painted once.
        weights[covered[py, px]] = 0
        covered[py, px] = True
        sx = np.clip(projected[:, 0], 0, 1) * (source.shape[1] - 1)
        sy = (1 - np.clip(projected[:, 1], 0, 1)) * (source.shape[0] - 1)
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        x1, y1 = np.minimum(x0 + 1, source.shape[1] - 1), np.minimum(y0 + 1, source.shape[0] - 1)
        fx, fy = (sx - x0)[:, None], (sy - y0)[:, None]
        sample = (source[y0, x0] * (1 - fx) + source[y0, x1] * fx) * (1 - fy) + (
            source[y1, x0] * (1 - fx) + source[y1, x1] * fx
        ) * fy
        alpha = weights * sample[:, 3]
        old = canvas[py, px]
        out_alpha = alpha + old[:, 3] * (1 - alpha)
        rgb = (
            sample[:, :3] * alpha[:, None] + old[:, :3] * old[:, 3, None] * (1 - alpha[:, None])
        ) / np.maximum(out_alpha[:, None], 1e-12)
        canvas[py, px] = np.column_stack((rgb, out_alpha))
        changed += int(np.count_nonzero(alpha))
    if not changed:
        raise ScissorsError("GEOMETRY_ERROR", "Projection does not cover any atlas pixels")
    # Copy border colors into four gutter pixels to reduce seams under filtering.
    filled = covered.copy()
    for _ in range(4):
        next_filled = filled.copy()
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = np.roll(filled, (dy, dx), axis=(0, 1))
            if dy == 1:
                neighbor[0] = False
            elif dy == -1:
                neighbor[-1] = False
            if dx == 1:
                neighbor[:, 0] = False
            elif dx == -1:
                neighbor[:, -1] = False
            target = neighbor & ~next_filled
            canvas[target] = np.roll(canvas, (dy, dx), axis=(0, 1))[target]
            next_filled[target] = True
        filled = next_filled
    result = store_image(
        engine, Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    )
    material.base_color_texture = result["texture_id"]
    # Bake the factor into the image once, then avoid multiplying it again on export.
    if not had_texture:
        material.base_color = [1, 1, 1, 1]
    return {
        **result,
        "object_id": object_id,
        "painted_pixels": changed,
        "warnings": [
            "Orthographic projection has no occlusion test. Split shared materials before painting."
        ],
    }
