"""Dimension-based resizing through the shared operation registry."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.resizing import resize

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class Dimensions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: Positive | None = None
    y: Positive | None = None
    z: Positive | None = None


class PointAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["point"]
    position: Vector


NamedAnchor = Literal["center", "min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]


def register_resize(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        object_id: str,
        dimensions: Dimensions,
        axes: Literal["world"] = "world",
        anchor: NamedAnchor | PointAnchor = "center",
        aspect_policy: Literal["independent", "uniform"] = "independent",
    ) -> dict[str, Any]:
        requested = Dimensions.model_validate(dimensions).model_dump(exclude_none=True)
        selected_anchor = (
            anchor if isinstance(anchor, str) else PointAnchor.model_validate(anchor).model_dump()
        )
        return resize(engine, object_id, requested, selected_anchor, aspect_policy)

    registry.add_function(
        "mesh.resize",
        execute,
        "Set positive final world-axis dimensions of a mesh or group subtree around a fixed "
        "bounds center, named bounds plane, or explicit world point. Bounds include all mesh "
        "descendants, including hidden meshes, but exclude markers. Transform every descendant "
        "once through the engine. Independent mode retains omitted extents; uniform mode "
        "scales all axes and requires consistent requested ratios. Preserve IDs and attributes.",
        examples=[
            {
                "object_id": "PEDESTAL_ID",
                "dimensions": {"z": 0.29},
                "anchor": "min_z",
                "aspect_policy": "independent",
            }
        ],
    )
