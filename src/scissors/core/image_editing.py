"""Bounded CPU image edits. Managed source images remain immutable."""

import io
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.export import artifact
from scissors.core.persistence import atomic_write
from scissors.core.textures import store_image


def check_pixels(engine: Engine, width: int, height: int, buffers: int = 8) -> None:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if width * height > engine.settings.max_texture_pixels:
        raise ScissorsError("RESOURCE_LIMIT", "Texture pixel limit exceeded")
    if width * height * 4 * buffers > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Texture working memory exceeds limit")


def read_texture(engine: Engine, texture_id: str) -> Image.Image:
    if texture_id not in engine.textures:
        raise ScissorsError("TEXTURE_NOT_FOUND", "Texture does not exist")
    path = engine.path(engine.textures[texture_id]["path"])
    if path.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Texture file exceeds size limit")
    with Image.open(path) as image:
        check_pixels(engine, image.width, image.height)
        return image.convert("RGBA")


def font(
    engine: Engine, size: int, path: str | None
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is None:
        return ImageFont.load_default(size=size)
    source = engine.path(path)
    if source.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError("Font must be a workspace TTF or OTF file")
    if source.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Font file exceeds size limit")
    return ImageFont.truetype(str(source), size=size)


def draw_text(
    engine: Engine,
    text: str,
    position: list[int],
    size: int,
    color: list[int],
    width: int,
    height: int,
    background: list[int],
    texture_id: str | None,
    font_path: str | None,
    output: str | None,
) -> dict[str, Any]:
    check_pixels(engine, width, height)
    base = (
        read_texture(engine, texture_id)
        if texture_id
        else Image.new(
            "RGBA", (width, height), (background[0], background[1], background[2], background[3])
        )
    )
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)
    selected = font(engine, size, font_path)
    bounds = draw.multiline_textbbox((position[0], position[1]), text, font=selected)
    check_pixels(
        engine, max(1, math.ceil(bounds[2] - bounds[0])), max(1, math.ceil(bounds[3] - bounds[1]))
    )
    draw.multiline_text(
        (position[0], position[1]),
        text,
        font=selected,
        fill=(color[0], color[1], color[2], color[3]),
    )
    result = store_image(engine, Image.alpha_composite(base, overlay), output)
    return {
        **result,
        "text_bounds": list(bounds),
        "clipped": bounds[0] < 0
        or bounds[1] < 0
        or bounds[2] > base.width
        or bounds[3] > base.height,
    }


def compose(
    engine: Engine,
    width: int,
    height: int,
    background: list[int],
    layers: list[dict[str, Any]],
    output: str | None,
) -> dict[str, Any]:
    check_pixels(engine, width, height)
    canvas = Image.new(
        "RGBA", (width, height), (background[0], background[1], background[2], background[3])
    )
    for layer in layers:
        image = read_texture(engine, layer["texture_id"])
        size = layer.get("size")
        if size is not None:
            check_pixels(engine, *size)
            image = image.resize(tuple(size), Image.Resampling.LANCZOS)
        opacity = layer.get("opacity", 1)
        if opacity != 1:
            image.putalpha(
                image.getchannel("A").point([round(value * opacity) for value in range(256)])
            )
        canvas.alpha_composite(image, dest=tuple(layer.get("position", [0, 0])))
    return store_image(engine, canvas, output)


def edit(
    engine: Engine,
    texture_id: str,
    mode: str,
    values: list[int] | float,
    expand: bool,
    output: str | None,
) -> dict[str, Any]:
    image = read_texture(engine, texture_id)
    if mode == "resize":
        assert isinstance(values, list)
        check_pixels(engine, *values)
        image = image.resize((values[0], values[1]), Image.Resampling.LANCZOS)
    elif mode == "crop":
        assert isinstance(values, list)
        x0, y0, x1, y1 = values
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError("Crop rectangle must lie inside the source image")
        image = image.crop((x0, y0, x1, y1))
    else:
        assert isinstance(values, float)
        if expand:
            angle = values % 360
            if angle in {0, 180}:
                check_pixels(engine, image.width, image.height)
            elif angle in {90, 270}:
                check_pixels(engine, image.height, image.width)
            else:
                c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
                check_pixels(
                    engine,
                    math.ceil(image.width * c + image.height * s) + 2,
                    math.ceil(image.width * s + image.height * c) + 2,
                )
        image = image.rotate(values, Image.Resampling.BICUBIC, expand=expand)
    check_pixels(engine, image.width, image.height)
    return store_image(engine, image, output)


def export_texture(engine: Engine, texture_id: str, output: str) -> dict[str, Any]:
    path = engine.path(output)
    if path.exists():
        raise ValueError("Texture export must use a new path")
    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
    if path.suffix.lower() not in formats:
        raise ValueError("Texture export supports PNG, JPEG and WebP")
    image = read_texture(engine, texture_id)
    target_format = formats[path.suffix.lower()]
    diagnostics = []
    if target_format == "JPEG":
        flattened = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(flattened, image).convert("RGB")
        diagnostics.append("JPEG alpha flattened over white")
    buffer = io.BytesIO()
    image.save(buffer, format=target_format)
    atomic_write(path, buffer.getvalue(), engine.settings.max_file_bytes)
    return artifact(path, target_format.lower(), diagnostics)
