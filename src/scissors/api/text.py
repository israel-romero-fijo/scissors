"""Native extruded typography with bounded font files and figure discovery."""

import unicodedata
from typing import Annotated, Any, Literal

import trimesh
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.geometry import run_backend
from scissors.core.normals import split_normals
from scissors.core.solids import orient

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def register_text(registry: OperationRegistry, engine: Engine) -> None:
    def text(
        scene_id: str,
        text: Annotated[str, Field(min_length=1, max_length=256)],
        size: Positive = 1,
        depth: Positive = 0.1,
        font: str = "Arial",
        segments: Annotated[int, Field(ge=2, le=64)] = 8,
        capped: bool = True,
        axis: Literal["x", "y", "z"] = "z",
        pivot: Literal["base", "center"] = "base",
        name: str = "Text",
        material_id: str | None = None,
        bold: bool = False,
        italic: bool = False,
        line_spacing: Annotated[float, Field(ge=0.5, le=4, allow_inf_nan=False)] = 1,
    ) -> dict[str, Any]:
        engine.scene(scene_id)
        if material_id and engine.material(material_id)[0].id != scene_id:
            raise ValueError("Text material must belong to its scene")
        normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").expandtabs(4))
        if not normalized.strip() or len(normalized) > 1024:
            raise ValueError("Text needs visible glyphs within its expanded size limit")
        if any(
            (unicodedata.category(char).startswith("C") and char != "\n")
            or unicodedata.combining(char)
            or unicodedata.bidirectional(char) in {"R", "AL", "AN"}
            for char in normalized
        ):
            raise ValueError(
                "Text needs left-to-right glyphs without unsupported shaping/control codes"
            )
        families = {"arial": "Arial", "courier": "Courier", "times": "Times"}
        family = families.get(font.lower())
        font_path = None
        if family is None:
            source = engine.path(font)
            if source.suffix.lower() not in {".ttf", ".otf"}:
                raise ValueError("Font must be Arial/Courier/Times or a workspace TTF/OTF path")
            if source.stat().st_size > engine.settings.max_file_bytes:
                raise ScissorsError("RESOURCE_LIMIT", "Font file exceeds byte limit")
            if source.stat().st_size * 8 > engine.settings.max_memory_bytes:
                raise ScissorsError("RESOURCE_LIMIT", "Font working memory exceeds budget")
            if bold or italic:
                raise ValueError(
                    "Choose a styled font file instead of bold/italic flags for custom fonts"
                )
            font_path = str(source)
        elif any(ord(char) > 126 or (ord(char) < 32 and char != "\n") for char in normalized):
            raise ValueError("Built-in text fonts support ASCII; provide a Unicode font file")
        result = run_backend(
            {
                "backend": "text",
                "text": normalized,
                "font": family,
                "font_path": font_path,
                "root": str(engine.root),
                "size": size,
                "depth": depth,
                "segments": segments,
                "capped": capped,
                "bold": bold,
                "italic": italic,
                "line_spacing": line_spacing,
                **{
                    key: getattr(engine.settings, key)
                    for key in ["max_file_bytes", "max_memory_bytes", "max_vertices", "max_faces"]
                },
            },
            engine.settings,
        )
        mesh = orient(
            trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False),
            axis,
            pivot,
        )
        created = engine.add_mesh(scene_id, mesh, name, "text")
        obj = engine.object(created["object_id"])[1]
        obj.uv, obj.material_id = result["uv"], material_id
        obj.metadata.update(
            {
                "text": normalized,
                "font": family or source.relative_to(engine.root).as_posix(),
                "font_sha256": result["font_sha256"],
                "size": size,
                "depth": depth,
                "segments": segments,
                "axis": axis,
                "pivot": pivot,
                "capped": capped,
            }
        )
        split_normals(obj, 30, engine.settings)
        engine.checked(obj)
        if capped and not obj.summary()["watertight"]:
            raise ScissorsError("GEOMETRY_ERROR", "Text extrusion did not produce closed contours")
        return {
            **obj.summary(),
            "contour_count": result["contours"],
            "warnings": [
                "Size is nominal font em-height; glyph ink height depends on font.",
                "Complex shaping, right-to-left layout and font collections are unsupported.",
            ],
        }

    registry.add_function(
        "mesh.create_text",
        text,
        "Extrude vector font outlines, including letter holes, with material and UV.",
        permissions=["scene.write", "filesystem.read"],
    )
