"""Read-only sampled visibility contract shared by every public transport."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.visibility import visibility_report

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
AngleRange = Annotated[list[Finite], Field(min_length=2, max_length=2)]
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
ObjectIDs = Annotated[list[str], Field(max_length=4096)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Eye(StrictModel):
    name: Name
    position: Vector


class HeadOffset(StrictModel):
    name: Name
    offset: Vector


class StepSampling(StrictModel):
    type: Literal["steps"]
    horizontal: Positive = 5
    vertical: Positive = 5


class CountSampling(StrictModel):
    type: Literal["counts"]
    horizontal: Annotated[int, Field(ge=1, le=1024)]
    vertical: Annotated[int, Field(ge=1, le=1024)]


Sampling = Annotated[StepSampling | CountSampling, Field(discriminator="type")]


class BaselineEye(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    clear: Annotated[int, Field(ge=0)]
    blocked: Annotated[int, Field(ge=0)]
    unresolved: Annotated[int, Field(ge=0)]


class Baseline(BaseModel):
    """Accept a previous full report, validating the fields used for comparison."""

    model_config = ConfigDict(extra="ignore")
    scene_id: str
    scene_revision: str
    configuration_fingerprint: str
    configuration: dict[str, Any]
    eyes: Annotated[list[BaselineEye], Field(min_length=1, max_length=32)]


def register_visibility(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        scene_id: str,
        eyes: Annotated[list[Eye], Field(min_length=1, max_length=16)],
        forward: Vector | None = None,
        up: Vector | None = None,
        horizontal_range: AngleRange | None = None,
        vertical_range: AngleRange | None = None,
        sampling: Sampling | None = None,
        near: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.001,
        max_distance: Positive = 100,
        head_offsets: Annotated[list[HeadOffset], Field(min_length=1, max_length=16)] | None = None,
        include_ids: ObjectIDs | None = None,
        exclude_ids: ObjectIDs | None = None,
        include_hidden: bool = False,
        obstruction_policy: Literal["geometric_surfaces"] = "geometric_surfaces",
        detail: Literal["summary", "samples"] = "summary",
        tolerance: Positive = 1e-9,
        baseline: Baseline | None = None,
    ) -> dict[str, Any]:
        parsed_eyes = [Eye.model_validate(eye).model_dump() for eye in eyes]
        offsets = (
            [HeadOffset.model_validate(value).model_dump() for value in head_offsets]
            if head_offsets is not None
            else [{"name": "neutral", "offset": [0.0, 0.0, 0.0]}]
        )
        parsed_sampling: dict[str, Any] = (
            TypeAdapter(Sampling)
            .validate_python(sampling if sampling is not None else {"type": "steps"})
            .model_dump()
        )
        if parsed_sampling["type"] == "steps":
            parsed_sampling["horizontal"] = float(parsed_sampling["horizontal"])
            parsed_sampling["vertical"] = float(parsed_sampling["vertical"])
        previous = Baseline.model_validate(baseline).model_dump() if baseline is not None else None
        return visibility_report(
            engine,
            scene_id,
            parsed_eyes,
            forward if forward is not None else [-1, 0, 0],
            up if up is not None else [0, 0, 1],
            horizontal_range if horizontal_range is not None else [-45, 45],
            vertical_range if vertical_range is not None else [-30, 30],
            parsed_sampling,
            near,
            max_distance,
            offsets,
            include_ids,
            exclude_ids,
            include_hidden,
            detail,
            tolerance,
            previous,
        )

    registry.add_function(
        "scene.visibility_report",
        execute,
        "Measure bounded, double-sided geometric surface visibility on a yaw/elevation grid "
        "from named world-space eyes and head offsets. Uses a CPU BVH. Reports nearest hits "
        "and clear/blocked/unresolved samples without scene or file changes. Material alpha "
        "and textures do not make surfaces transparent. Compare only matching configurations.",
        side_effects=False,
        permissions=["scene.read"],
        examples=[
            {
                "scene_id": "SCENE_ID",
                "eyes": [{"name": "player", "position": [0.5, 0, 1.72]}],
                "forward": [-1, 0, 0],
                "up": [0, 0, 1],
                "horizontal_range": [-40, 40],
                "vertical_range": [-25, 20],
                "sampling": {"type": "steps", "horizontal": 5, "vertical": 5},
                "max_distance": 5,
                "detail": "samples",
            }
        ],
    )
