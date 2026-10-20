"""Native mechanical figures backed by reusable core geometry."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core import mechanical
from scissors.core.engine import Engine
from scissors.core.solids import open_flat_caps, orient

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Size = Annotated[list[Positive], Field(min_length=3, max_length=3)]
Name = Annotated[str, Field(min_length=1)]


def register_mechanical(registry: OperationRegistry, engine: Engine) -> None:
    def budget(vertices: int, faces: int) -> None:
        if (
            vertices > engine.settings.max_vertices
            or faces > engine.settings.max_faces
            or vertices * 384 + faces * 192 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Mechanical figure exceeds geometry budget")

    def rounded_box(
        scene_id: str,
        size: Size,
        corner_radius: Nonnegative = 0,
        segments: Annotated[int, Field(ge=1, le=16)] = 4,
        edge_bevel: Nonnegative = 0,
        axis: Literal["x", "y", "z"] = "z",
        name: Name = "Rounded box",
        pivot: Literal["base", "center"] = "base",
        capped: bool = True,
    ) -> dict[str, Any]:
        count = len(mechanical._rounded_outline(size[0], size[1], corner_radius, segments))
        if corner_radius == 0 and edge_bevel == 0:
            budget(8, 12)
        else:
            budget(
                4 * count + 2 if edge_bevel else 2 * count + 2,
                8 * count if edge_bevel else 4 * count,
            )
        mesh = mechanical.rounded_box(size[0], size[1], size[2], corner_radius, segments, name)
        if edge_bevel:
            mesh = mechanical.beveled_box(size, corner_radius, segments, edge_bevel)
        if not capped:
            mesh.update_faces(np.abs(mesh.face_normals[:, 2]) < 1 - 1e-10)
            mesh.remove_unreferenced_vertices()
        return engine.add_mesh(scene_id, orient(mesh, axis, pivot), name, "rounded_box")

    def tube(
        scene_id: str,
        points: Annotated[list[Vector], Field(min_length=2, max_length=256)],
        radius: Positive,
        sides: Annotated[int, Field(ge=3, le=32)] = 8,
        caps: bool = True,
        closed: bool = False,
        name: Name = "Tube",
        capped: bool | None = None,
        axis: Literal["x", "y", "z"] = "z",
        pivot: Literal["origin", "base", "center"] = "origin",
    ) -> dict[str, Any]:
        if capped is not None:
            if caps is not True and caps != capped:
                raise ValueError("caps and capped must agree")
            caps = capped
        budget(len(points) * sides + 2, 2 * len(points) * sides)
        mesh = mechanical.tube(points, radius, sides, caps, closed, name)
        return engine.add_mesh(scene_id, orient(mesh, axis, pivot), name, "tube")

    def louver(
        scene_id: str,
        width: Positive,
        height: Positive,
        blade_count: Annotated[int, Field(ge=1, le=40)],
        blade_angle_degrees: Annotated[float, Field(ge=-80, le=80, allow_inf_nan=False)] = 30,
        blade_thickness: Positive = 0.004,
        frame_thickness: Nonnegative = 0.012,
        frame_depth: Positive = 0.02,
        depth: Positive = 0.01,
        name: Name = "Louver",
        axis: Literal["x", "y", "z"] = "z",
        pivot: Literal["origin", "base", "center"] = "origin",
        capped: bool = True,
    ) -> dict[str, Any]:
        budget(8 * blade_count + 16, 12 * blade_count + 32)
        mesh = mechanical.louver(
            width,
            height,
            blade_count,
            blade_angle_degrees,
            blade_thickness,
            frame_thickness,
            frame_depth,
            depth,
            name,
        )
        if not capped:
            angle = np.radians(blade_angle_degrees)
            mesh = open_flat_caps(
                mesh, [[0, 0, 1], [0, -float(np.sin(angle)), float(np.cos(angle))]]
            )
        return engine.add_mesh(scene_id, orient(mesh, axis, pivot), name, "louver")

    def switch_grid(
        scene_id: str,
        columns: Annotated[int, Field(ge=1, le=12)],
        rows: Annotated[int, Field(ge=1, le=12)],
        spacing: Positive,
        base_width: Positive,
        base_height: Positive,
        base_thickness: Positive = 0.004,
        toggle_length: Positive = 0.016,
        toggle_radius: Positive = 0.0035,
        toggle_angle_degrees: Annotated[float, Field(ge=-80, le=80, allow_inf_nan=False)] = 20,
        name: Name = "Switch grid",
        axis: Literal["x", "y", "z"] = "z",
        pivot: Literal["origin", "base", "center"] = "origin",
        capped: bool = True,
    ) -> dict[str, Any]:
        budget(8 + columns * rows * 26, 12 + columns * rows * 48)
        mesh = mechanical.switch_grid(
            columns,
            rows,
            spacing,
            base_width,
            base_height,
            base_thickness,
            toggle_length,
            toggle_radius,
            toggle_angle_degrees,
            name,
        )
        if not capped:
            angle = np.radians(toggle_angle_degrees)
            mesh = open_flat_caps(
                mesh, [[0, 0, 1], [0, -float(np.sin(angle)), float(np.cos(angle))]]
            )
        return engine.add_mesh(scene_id, orient(mesh, axis, pivot), name, "switch_grid")

    registry.add_function(
        "mesh.create_rounded_box",
        rounded_box,
        "Create a rounded XY box with base at zero and optional top/bottom chamfers.",
    )
    registry.add_function(
        "mesh.create_tube",
        tube,
        "Sweep a tube along a polyline with parallel transport and closed-loop frame correction.",
    )
    registry.add_function(
        "mesh.create_louver",
        louver,
        "Create a closed ventilation frame and tilted blades with clearance.",
    )
    registry.add_function(
        "mesh.create_switch_grid",
        switch_grid,
        "Create a base plate and grid of closed tilted twelve-sided toggles.",
    )
