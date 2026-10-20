"""Typed image-editing registry contracts."""

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core import image_editing
from scissors.core.atlas import pack
from scissors.core.engine import Engine

Dimension = Annotated[int, Field(ge=1, le=4096)]
Pair = Annotated[
    list[Annotated[int, Field(ge=-1000000, le=1000000)]], Field(min_length=2, max_length=2)
]
Size = Annotated[list[Dimension], Field(min_length=2, max_length=2)]
Color = Annotated[list[Annotated[int, Field(ge=0, le=255)]], Field(min_length=4, max_length=4)]


class ImageLayer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    texture_id: str
    position: Pair = Field(default_factory=lambda: [0, 0])
    size: Size | None = None
    opacity: float = Field(default=1, ge=0, le=1)


def register_images(registry: OperationRegistry, engine: Engine) -> None:
    def atlas_pack(
        scene_id: str,
        object_ids: Annotated[list[str], Field(min_length=1, max_length=4096)] | None = None,
        size: Size | None = None,
        padding: Annotated[int, Field(ge=0, le=64)] = 4,
        merge: bool = True,
    ) -> dict[str, Any]:
        return pack(engine, scene_id, object_ids, size or [2048, 2048], padding, merge)

    registry.add_function(
        "texture.atlas_pack",
        atlas_pack,
        "Pack PBR images, remap UV/materials and optionally merge compatible leaf siblings.",
        permissions=["scene.write", "filesystem.write"],
    )

    def draw_text(
        text: Annotated[str, Field(max_length=4096)],
        position: Pair | None = None,
        font_size: Annotated[int, Field(ge=1, le=512)] = 24,
        color: Color | None = None,
        width: Dimension = 256,
        height: Dimension = 256,
        background: Color | None = None,
        texture_id: str | None = None,
        font_path: str | None = None,
        output: str | None = None,
    ) -> dict[str, Any]:
        return image_editing.draw_text(
            engine,
            text,
            position or [0, 0],
            font_size,
            color if color is not None else [255, 255, 255, 255],
            width,
            height,
            background if background is not None else [0, 0, 0, 0],
            texture_id,
            font_path,
            output,
        )

    def compose(
        layers: Annotated[list[ImageLayer], Field(max_length=128)],
        width: Dimension = 256,
        height: Dimension = 256,
        background: Color | None = None,
        output: str | None = None,
    ) -> dict[str, Any]:
        return image_editing.compose(
            engine,
            width,
            height,
            background if background is not None else [0, 0, 0, 0],
            [ImageLayer.model_validate(layer).model_dump(exclude_none=True) for layer in layers],
            output,
        )

    def resize(texture_id: str, size: Size, output: str | None = None) -> dict[str, Any]:
        return image_editing.edit(engine, texture_id, "resize", size, False, output)

    def crop(
        texture_id: str,
        rectangle: Annotated[list[int], Field(min_length=4, max_length=4)],
        output: str | None = None,
    ) -> dict[str, Any]:
        return image_editing.edit(engine, texture_id, "crop", rectangle, False, output)

    def rotate(
        texture_id: str,
        angle: Annotated[float, Field(ge=-360, le=360, allow_inf_nan=False)],
        expand: bool = True,
        output: str | None = None,
    ) -> dict[str, Any]:
        return image_editing.edit(engine, texture_id, "rotate", float(angle), expand, output)

    def export(texture_id: str, output: str) -> dict[str, Any]:
        return image_editing.export_texture(engine, texture_id, output)

    functions: list[tuple[str, Callable[..., Any], str]] = [
        ("draw_text", draw_text, "Draw bounded text with Pillow's default or a workspace font."),
        ("compose", compose, "Alpha-compose ordered texture layers on a new RGBA canvas."),
        ("resize", resize, "Resize a managed texture into a new texture using Lanczos filtering."),
        ("crop", crop, "Crop an internal pixel rectangle into a new managed texture."),
        ("rotate", rotate, "Rotate counterclockwise in degrees into a new managed texture."),
        ("export", export, "Export a managed texture as PNG, JPEG or WebP."),
    ]
    for name, function, description in functions:
        registry.add_function(
            f"texture.{name}",
            function,
            description,
            side_effects=name != "export",
            permissions=["filesystem.write"]
            if name == "export"
            else ["scene.write", "filesystem.write"],
        )
