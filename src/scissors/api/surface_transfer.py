"""Surface UV transfer and managed PBR baking through the shared registry."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from scissors.api.inspection import FaceSelection
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.surface_transfer import transfer_uv
from scissors.core.texture_transfer import CHANNELS, bake_transfer

Distance = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Dot = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
Channel = Literal["base_color", "normal", "metallic_roughness", "emissive", "occlusion"]


def parsed(value: FaceSelection | None) -> dict[str, Any] | None:
    return FaceSelection.model_validate(value).model_dump() if value is not None else None


def register_surface_transfer(registry: OperationRegistry, engine: Engine) -> None:
    def uv_transfer(
        object_id: str,
        source_object_id: str,
        max_distance: Distance,
        selection: FaceSelection | None = None,
        source_selection: FaceSelection | None = None,
        material_policy: Literal["source", "target"] = "source",
        transfer_colors: bool = True,
        min_normal_dot: Dot = 0,
    ) -> dict[str, Any]:
        return transfer_uv(
            engine,
            object_id,
            source_object_id,
            max_distance,
            parsed(selection),
            parsed(source_selection),
            material_policy,
            transfer_colors,
            min_normal_dot,
        )

    def texture_transfer(
        object_id: str,
        source_object_id: str,
        max_distance: Distance,
        selection: FaceSelection | None = None,
        source_selection: FaceSelection | None = None,
        resolution: Annotated[int, Field(ge=16, le=4096)] = 256,
        channels: Annotated[list[Channel], Field(min_length=1, max_length=5)] | None = None,
        padding: Annotated[int, Field(ge=0, le=16)] = 2,
        wrap: Literal["repeat", "clamp"] = "repeat",
        min_normal_dot: Dot = 0,
    ) -> dict[str, Any]:
        owned: list[Path] = []

        def cleanup() -> None:
            for path in reversed(owned):
                if not path.resolve().is_relative_to(
                    engine.root.resolve()
                ) or not path.resolve().is_relative_to((engine.root / "textures").resolve()):
                    raise ValueError("Transfer cleanup path escaped managed textures")
                path.unlink(missing_ok=True)

        registry._on_rollback(cleanup)
        return bake_transfer(
            engine,
            object_id,
            source_object_id,
            max_distance,
            parsed(selection),
            parsed(source_selection),
            resolution,
            list(CHANNELS) if channels is None else [str(channel) for channel in channels],
            padding,
            wrap,
            min_normal_dot,
            owned,
        )

    registry.add_function(
        "mesh.transfer_uv",
        uv_transfer,
        "Project source UV and optional colors onto selected target face corners. "
        "Bound distance and normal alignment; split render seams. Source material policy "
        "shares/clones its PBR references and requires matching material for partial edits.",
    )
    registry.add_function(
        "texture.transfer",
        texture_transfer,
        "Bake selected target UV texels from nearest compatible source surfaces. "
        "Transfer five PBR channels and factors; reorient tangent normals. Clone target "
        "material, preserve unselected texels, and remove owned maps on rollback. "
        "Requires nonoverlapping target UV and rejects unmatched/subpixel selected faces.",
        permissions=["scene.write", "filesystem.write"],
    )
