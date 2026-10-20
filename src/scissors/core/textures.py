import io
from typing import Any

import numpy as np
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine, identifier
from scissors.core.persistence import atomic_write


def store_image(engine: Engine, image: Image.Image, output: str | None = None) -> dict[str, Any]:
    if image.width * image.height > engine.settings.max_texture_pixels:
        raise ScissorsError("RESOURCE_LIMIT", "Texture pixel limit exceeded")
    tid = identifier("tex")
    path = engine.path(output or f"textures/{tid}.png")
    if path.exists():
        raise ValueError("Managed texture output must use a new path")
    if path.suffix.lower() != ".png":
        raise ValueError("Generated textures must use .png")
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    atomic_write(path, buffer.getvalue(), engine.settings.max_file_bytes)
    value = {
        "id": tid,
        "texture_id": tid,
        "path": str(path.relative_to(engine.root)),
        "width": image.width,
        "height": image.height,
        "format": "png",
        "size_bytes": path.stat().st_size,
        "diagnostics": [],
    }
    engine.textures[tid] = value
    return {**value, "path": str(path)}


def checker(
    engine: Engine,
    width: int,
    height: int,
    cells: int,
    color_a: list[int],
    color_b: list[int],
    output: str | None = None,
) -> dict[str, Any]:
    if width * height > engine.settings.max_texture_pixels:
        raise ScissorsError("RESOURCE_LIMIT", "Texture pixel limit exceeded")
    if width * height * 64 > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Estimated texture memory exceeds limit")
    y, x = np.indices((height, width))
    mask = ((x * cells // width + y * cells // height) % 2).astype(bool)
    pixels = np.where(mask[..., None], color_b, color_a).astype(np.uint8)
    return store_image(engine, Image.fromarray(pixels), output)


def load_image(engine: Engine, path: str) -> dict[str, Any]:
    source = engine.path(path)
    if source.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Image file is too large")
    with Image.open(source) as image:
        if image.width * image.height > engine.settings.max_texture_pixels:
            raise ScissorsError("RESOURCE_LIMIT", "Texture pixel limit exceeded")
        if image.width * image.height * 32 > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Texture decode exceeds memory budget")
        return store_image(engine, image)
