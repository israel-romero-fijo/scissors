"""Portable asset bundle parameters for the shared operation registry."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from scissors.api.registry import OperationRegistry
from scissors.api.render_sheet import EyeViewSet, SheetView
from scissors.core.bundles import export_bundle
from scissors.core.engine import Engine


class BundleVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
    preset: Literal["hierarchy", "material_batches"] = "hierarchy"
    format: Literal["glb"] = "glb"


def register_bundles(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        scene_id: str,
        output_path: str,
        variants: Annotated[list[BundleVariant], Field(min_length=1, max_length=8)] | None = None,
        include_scene: bool = True,
        include_textures: bool = True,
        preview_views: Annotated[list[SheetView], Field(min_length=1, max_length=24)] | None = None,
        preview_view_sets: Annotated[list[EyeViewSet], Field(min_length=1, max_length=16)]
        | None = None,
        preview_size: Annotated[int, Field(ge=64, le=2048)] = 256,
        preview_renderer: Literal["auto", "vtk", "pillow"] = "auto",
        report: bool = True,
        package_format: Literal["directory", "zip"] = "directory",
        overwrite: bool = False,
        verification: Literal["reopen", "structure"] = "reopen",
    ) -> dict[str, Any]:
        parsed_variants = (
            [BundleVariant.model_validate(value).model_dump() for value in variants]
            if variants is not None
            else [
                {"name": "hierarchy", "preset": "hierarchy", "format": "glb"},
                {"name": "material_batches", "preset": "material_batches", "format": "glb"},
            ]
        )
        views: list[dict[str, Any]] = [
            TypeAdapter(SheetView).validate_python(value).model_dump(exclude_none=True)
            for value in preview_views or []
        ]
        sets = [
            EyeViewSet.model_validate(value).model_dump(exclude_none=True)
            for value in preview_view_sets or []
        ]
        return export_bundle(
            engine,
            scene_id,
            output_path,
            parsed_variants,
            include_scene,
            include_textures,
            views,
            sets,
            preview_size,
            preview_renderer,
            report,
            package_format,
            overwrite,
            verification,
        )

    registry.add_function(
        "scene.export_bundle",
        execute,
        "Publish a staged directory or ZIP from one scene snapshot. Include hierarchy or "
        "material-batched GLBs, an editable scene, required textures, optional review views, "
        "and a hash manifest. Reuse export and camera services. Verify embedded GLB structure "
        "and optionally reopen artifacts. Keep source state unchanged; reject atomic batches.",
        side_effects=True,
        scene_mutation=False,
        permissions=["scene.read", "filesystem.read", "filesystem.write"],
        examples=[
            {
                "scene_id": "SCENE_ID",
                "output_path": "exports/asset.zip",
                "package_format": "zip",
                "verification": "reopen",
            }
        ],
    )
