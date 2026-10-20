"""Typed orbit, explicit-camera, and eye-set sheet views."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.render_sheets import render_sheet

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
RGB = Annotated[list[Unit], Field(min_length=3, max_length=3)]
Dimension = Annotated[int, Field(ge=32, le=2048)]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]


class ViewOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: Annotated[str, Field(min_length=1, max_length=192)] | None = None
    width: Dimension | None = None
    height: Dimension | None = None
    projection: Literal["orthographic", "perspective"] | None = None
    fov: Annotated[float, Field(gt=1, lt=179, allow_inf_nan=False)] | None = None
    orthographic_scale: Positive | None = None
    near: Positive = 0.001
    far: Positive | None = None
    zoom: Annotated[float, Field(gt=0, le=20, allow_inf_nan=False)] = 1
    offset_x: Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)] = 0
    offset_y: Annotated[float, Field(ge=-2, le=2, allow_inf_nan=False)] = 0
    wireframe: bool = False
    renderer: Literal["auto", "vtk", "pillow"] | None = None
    sample_textures: bool | None = None
    light_direction: Vector | None = None
    ambient: Unit = 0.35
    light_strength: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)] = 0.65
    background: RGB | None = None


class OrbitView(ViewOptions):
    type: Literal["orbit"] = "orbit"
    azimuth_degrees: Finite = 35
    elevation_degrees: Finite = 35


class CameraView(ViewOptions):
    type: Literal["camera"] = "camera"
    position: Vector
    target: Vector
    up: Vector = Field(default_factory=lambda: [0.0, 0.0, 1.0])


class HeadOffset(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: Identifier
    offset: Vector


class EyeViewSet(CameraView):
    name: Identifier
    stereo: bool = False
    eye_separation: Positive | None = None
    include_center: bool = True
    head_offsets: Annotated[list[HeadOffset], Field(min_length=1, max_length=16)] = Field(
        default_factory=lambda: [HeadOffset(name="neutral", offset=[0, 0, 0])]
    )


SheetView = OrbitView | CameraView


def register_render_sheet(registry: OperationRegistry, engine: Engine) -> None:
    def execute(
        scene_id: str,
        output: str = "previews/sheet.png",
        views: Annotated[list[SheetView], Field(min_length=1, max_length=24)] | None = None,
        columns: Annotated[int, Field(ge=1, le=8)] = 3,
        size: Annotated[int, Field(ge=64, le=2048)] = 256,
        only_object_ids: list[str] | None = None,
        sample_textures: bool = True,
        projection: Literal["orthographic", "perspective"] | None = None,
        transparent: bool = False,
        renderer: Literal["auto", "vtk", "pillow"] = "auto",
        view_sets: Annotated[list[EyeViewSet], Field(min_length=1, max_length=16)] | None = None,
        exclude_object_ids: list[str] | None = None,
        background: RGB | None = None,
    ) -> dict[str, Any]:
        parsed: list[dict[str, Any]] = (
            [
                TypeAdapter(SheetView).validate_python(view).model_dump(exclude_none=True)
                for view in views
            ]
            if views is not None
            else []
        )
        sets = (
            [EyeViewSet.model_validate(value).model_dump(exclude_none=True) for value in view_sets]
            if view_sets is not None
            else []
        )
        return render_sheet(
            engine,
            scene_id,
            output,
            parsed,
            sets,
            columns,
            size,
            only_object_ids,
            exclude_object_ids,
            sample_textures,
            projection,
            transparent,
            renderer,
            background,
        )

    registry.add_function(
        "scene.render_sheet",
        execute,
        "Render a labeled PNG sheet and retain each view PNG. Accept legacy orbit views, "
        "typed world position/target/up cameras, and named head/stereo sets. Stereo uses "
        "parallel directions with meter eye separation. Reuse preview rendering and camera "
        "validation. Bound expanded views, pixels, geometry work, files, and time. Stage all "
        "images before publication; scene geometry is unchanged.",
        permissions=["scene.read", "filesystem.write"],
        examples=[
            {
                "scene_id": "SCENE_ID",
                "views": [
                    {
                        "type": "camera",
                        "name": "Player",
                        "position": [0.5, 0, 1.72],
                        "target": [-2, 0, 1.72],
                        "up": [0, 0, 1],
                        "fov": 90,
                    }
                ],
            }
        ],
    )
