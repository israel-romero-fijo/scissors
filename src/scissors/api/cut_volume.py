"""Typed inline cut contract shared by Python, REST, and MCP."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core.cut_volume import cut_volume
from scissors.core.engine import Engine

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class BoxVolume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["box"] = "box"
    min: Vector
    max: Vector


def register_cut_volume(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        scene_id: str,
        target_ids: Annotated[list[str], Field(min_length=1, max_length=128)],
        volume: BoxVolume,
        operation: Literal["difference"] = "difference",
        cut_material_id: str | None = None,
        cut_uv: Literal["planar"] = "planar",
        uv_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1,
        dry_run: bool = False,
        empty_policy: Literal["error", "delete_object"] = "error",
    ) -> dict[str, Any]:
        box = BoxVolume.model_validate(volume)
        return cut_volume(
            engine,
            scene_id,
            target_ids,
            box.min,
            box.max,
            cut_material_id,
            uv_scale,
            dry_run,
            empty_policy,
        )

    registry.add_function(
        "scene.cut_volume",
        execute,
        "Atomically subtract an inline world-space box from up to 128 independent closed "
        "volumes. Groups expand and deduplicate. Preserve target IDs and source attributes; "
        "a distinct cut material makes the stable target a group of material parts. "
        "dry_run computes without scene changes. Geometry and total backend time are bounded.",
        examples=[
            {
                "scene_id": "SCENE_ID",
                "target_ids": ["OBJECT_ID"],
                "volume": {"type": "box", "min": [-1, -1, 0], "max": [0, 1, 1]},
            }
        ],
    )
