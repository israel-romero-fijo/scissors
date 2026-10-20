"""Shared read-only mesh query contracts."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core import face_queries
from scissors.core.engine import Engine

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class FaceNormal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Vector
    angular_tolerance_degrees: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)]


class FaceRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bounds: Annotated[list[Vector], Field(min_length=2, max_length=2)]
    rule: Literal["centroid_inside", "fully_inside", "intersects"]


class FaceSelection(BaseModel):
    # Accept a complete select_faces response, while validating consumed fields.
    model_config = ConfigDict(extra="ignore")
    object_id: str
    geometry_revision: str
    face_indices: list[Annotated[int, Field(ge=0)]]


def register_inspection(registry: OperationRegistry, engine: Engine) -> None:
    from scissors.api.selections import register_selections

    register_selections(registry, engine)

    def select_faces(
        object_id: str,
        normal: FaceNormal | None = None,
        region: FaceRegion | None = None,
        tolerance: Positive = 1e-6,
    ) -> dict[str, Any]:
        parsed_normal = (
            FaceNormal.model_validate(normal).model_dump() if normal is not None else None
        )
        parsed_region = (
            FaceRegion.model_validate(region).model_dump() if region is not None else None
        )
        return face_queries.select_faces(
            engine.object(object_id)[1], parsed_normal, parsed_region, tolerance, engine.settings
        )

    def check_thickness(
        object_id: str,
        minimum: Positive,
        tolerance: Positive,
        selection: FaceSelection | None = None,
        rim_selection: FaceSelection | None = None,
    ) -> dict[str, Any]:
        return face_queries.check_thickness(
            engine.object(object_id)[1],
            minimum,
            tolerance,
            engine.settings,
            FaceSelection.model_validate(selection).model_dump() if selection is not None else None,
            FaceSelection.model_validate(rim_selection).model_dump()
            if rim_selection is not None
            else None,
        )

    registry.add_function(
        "mesh.select_faces",
        select_faces,
        "Select revision-bound world triangles by geometric normal AND explicit box rule.",
        False,
    )
    registry.add_function(
        "mesh.check_thickness",
        check_thickness,
        "Bound material-interior face-normal chords with exact whole-face partitions. "
        "Inspect all faces or a revision-bound face/component selection. Explicit rim_selection "
        "excludes rim faces and domains whose first exit hits a rim; reports excluded areas. "
        "Require closed oriented nonintersecting nested boundaries. Report pass, fail or "
        "inconclusive with coverage and global bounds. Limits never yield an unchecked pass.",
        False,
    )
