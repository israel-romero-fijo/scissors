"""Animated review operations for the fast preview renderer."""

from typing import Annotated, Any, Literal

from pydantic import Field

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.rendering import turntable as render_turntable


def register_rendering(registry: OperationRegistry, engine: Engine) -> None:
    registry.operations["scene.preview"].description = (
        "PNG preview drawn by the VTK offscreen renderer, or by the Pillow painter when no "
        "OpenGL context exists. Orbit cameras use world +Z as screen up: azimuth 0 looks "
        "along -Y, 90 along +X, 180 along +Y, 270 along -X, and elevation 90 looks down. "
        "Explicit position/target/up cameras can roll. orthographic_scale sets vertical "
        "world span before zoom and requires orthographic projection. Sections hide whole "
        "triangles. renderer forces a backend and the result reports the backend used. "
        "highlights accepts revision-bound geometry selections as orange x-ray overlays "
        "on the color pass; it does not change scene materials or geometry."
    )

    def turntable(
        scene_id: str,
        output: str = "previews/turntable.gif",
        frames: Annotated[int, Field(ge=2, le=180)] = 24,
        size: Annotated[int, Field(ge=32, le=2048)] = 256,
        elevation_degrees: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)] = 25,
        fps: Annotated[int, Field(ge=1, le=60)] = 12,
        projection: Literal["orthographic", "perspective"] = "perspective",
        sample_textures: bool = True,
        only_object_ids: list[str] | None = None,
        renderer: Literal["auto", "vtk", "pillow"] = "auto",
    ) -> dict[str, Any]:
        return render_turntable(
            engine,
            scene_id,
            output,
            frames,
            size,
            elevation_degrees,
            fps,
            projection,
            sample_textures,
            only_object_ids,
            renderer,
        )

    registry.add_function(
        "scene.turntable",
        turntable,
        "Render a full camera orbit as a looping GIF without changing scene geometry. "
        "renderer selects the backend and the result reports it.",
        permissions=["filesystem.write"],
    )
