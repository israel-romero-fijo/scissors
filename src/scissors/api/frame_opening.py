"""Revision-bound opening frame contracts for the shared registry."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.opening_frames import frame_opening

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CutOpening(StrictModel):
    type: Literal["cut"]
    geometry_revision: str
    contour_index: Annotated[int, Field(ge=0)] | None = None


class RimOpening(StrictModel):
    type: Literal["rim_faces"]
    geometry_revision: str
    face_indices: Annotated[list[int], Field(min_length=1, max_length=400_000)]
    contour_index: Annotated[int, Field(ge=0)] | None = None


class ExplicitOpening(StrictModel):
    type: Literal["points"]
    geometry_revision: str
    points: Annotated[list[Vector], Field(min_length=2, max_length=512)]
    closed: bool = True


class CircleSection(StrictModel):
    type: Literal["circle"]
    radius: Positive
    sides: Annotated[int, Field(ge=4, le=64, multiple_of=4)] = 12


class RectangleSection(StrictModel):
    type: Literal["rectangle"]
    width: Positive
    depth: Positive


Opening = Annotated[CutOpening | RimOpening | ExplicitOpening, Field(discriminator="type")]
Section = Annotated[CircleSection | RectangleSection, Field(discriminator="type")]


def register_frame_opening(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        object_id: str,
        opening: Opening,
        section: Section,
        contour_side: Literal["selected", "inner", "outer"] = "selected",
        normal: Vector | None = None,
        placement: Literal["centered", "inside", "outside"] = "centered",
        offset: Finite = 0,
        clearance: Nonnegative = 0,
        joint_policy: Literal["miter", "transport"] = "miter",
        end_policy: Literal["auto", "cap", "open"] = "auto",
        miter_limit: Annotated[float, Field(ge=1, le=100, allow_inf_nan=False)] = 4,
        material_id: str | None = None,
        parent_id: str | None = None,
        name: Annotated[str, Field(min_length=1)] = "Opening frame",
        allow_incomplete_checks: bool = False,
    ) -> dict[str, Any]:
        from pydantic import TypeAdapter

        resolved: dict[str, Any] = TypeAdapter(Opening).validate_python(opening).model_dump()
        cross_section: dict[str, Any] = TypeAdapter(Section).validate_python(section).model_dump()
        return frame_opening(
            engine,
            object_id,
            resolved,
            cross_section,
            contour_side,
            normal,
            placement,
            offset,
            clearance,
            joint_policy,
            end_policy,
            miter_limit,
            material_id,
            parent_id,
            name,
            allow_incomplete_checks,
        )

    registry.add_function(
        "mesh.frame_opening",
        execute,
        "Create a separate circle or rectangle frame from revision-bound cut contours, "
        "selected rim faces, or explicit ordered points. Choose ambiguous contours explicitly. "
        "Planar miter frames support inside/outside clearance and signed normal offset. "
        "Transport frames support spatial paths. Preserve the source body and bound checks.",
        examples=[
            {
                "object_id": "BODY_ID",
                "opening": {
                    "type": "cut",
                    "geometry_revision": "REVISION_FROM_CUT",
                    "contour_index": 0,
                },
                "section": {"type": "circle", "radius": 0.019, "sides": 12},
            }
        ],
    )
