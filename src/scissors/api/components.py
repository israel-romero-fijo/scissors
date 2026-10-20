"""Whole-component selection and deletion contract shared by public transports."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core.components import delete_components
from scissors.core.engine import Engine

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class ComponentRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    type: Literal["box"] = "box"
    bounds: Annotated[list[Vector], Field(min_length=2, max_length=2)]


def register_components(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        object_id: str,
        region: ComponentRegion,
        selection_rule: Literal["fully_inside", "intersects", "centroid_inside"] = "fully_inside",
        dry_run: bool = False,
        empty_policy: Literal["error", "delete_object"] = "error",
        tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1e-9,
    ) -> dict[str, Any]:
        box = ComponentRegion.model_validate(region)
        return delete_components(
            engine, object_id, box.bounds, selection_rule, dry_run, empty_policy, tolerance
        )

    registry.add_function(
        "mesh.delete_components",
        execute,
        "Remove whole edge-connected face components selected by a world box. Topology "
        "welds exact positions only; render seams remain separate and point contact does "
        "not connect pieces. Rules test all vertices, actual triangle surface intersection, "
        "or area-weighted surface centroid. Preserve surviving attributes and return planned "
        "index maps. Dry runs report forbidden empty plans without changing scene state.",
        examples=[
            {
                "object_id": "MERGED_FASTENERS_ID",
                "region": {"type": "box", "bounds": [[-1, -1, 0], [0, 1, 2]]},
                "selection_rule": "fully_inside",
                "dry_run": True,
            }
        ],
    )
