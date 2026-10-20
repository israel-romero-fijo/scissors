"""Native rounded bodies, sheet figures and regular polyhedra."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core import solids
from scissors.core.engine import Engine

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Segments = Annotated[int, Field(ge=4, le=256)]
Axis = Literal["x", "y", "z"]
Pivot = Literal["base", "center"]
Name = Annotated[str, Field(min_length=1)]
Point = Annotated[list[Finite], Field(min_length=2, max_length=2)]


class PlateHole(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    center: Point
    radius: Positive
    sides: Annotated[int, Field(ge=3, le=128)] = 24


def register_solids(registry: OperationRegistry, engine: Engine) -> None:
    def budget(vertices: int, faces: int) -> None:
        if (
            vertices > engine.settings.max_vertices
            or faces > engine.settings.max_faces
            or vertices * 384 + faces * 192 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Solid figure exceeds geometry budget")

    def store(
        scene_id: str,
        mesh: Any,
        name: str,
        primitive: str,
        axis: str,
        pivot: str,
        material_id: str | None = None,
    ) -> dict[str, Any]:
        result = engine.add_mesh(scene_id, solids.orient(mesh, axis, pivot), name, primitive)
        obj = engine.object(result["object_id"])[1]
        if material_id:
            if engine.material(material_id)[0].id != scene_id:
                raise ValueError("Material must belong to the figure scene")
            obj.material_id = material_id
        return obj.summary()

    def capsule(
        scene_id: str,
        radius: Positive = 1,
        height: Positive = 4,
        segments: Segments = 32,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Capsule",
        capped: bool = True,
    ) -> dict[str, Any]:
        rings = 2 * (max(2, segments // 4) + 1)
        budget(rings * segments + 2, 2 * rings * segments)
        mesh = solids.capsule(radius, height, segments)
        if not capped:
            mesh = solids.open_poles(mesh)
        return store(scene_id, mesh, name, "capsule", axis, pivot)

    def dome(
        scene_id: str,
        radius: Positive = 1,
        height: Positive = 0.5,
        segments: Segments = 32,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Dome",
    ) -> dict[str, Any]:
        rings = max(2, segments // 4) + 1
        budget(rings * segments + 2, 2 * rings * segments)
        return store(
            scene_id, solids.dome(radius, height, segments, capped), name, "dome", axis, pivot
        )

    def hemisphere(
        scene_id: str,
        radius: Positive = 1,
        segments: Segments = 32,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Hemisphere",
    ) -> dict[str, Any]:
        result = dome(scene_id, radius, radius, segments, capped, axis, pivot, name)
        engine.object(result["object_id"])[1].metadata["primitive"] = "hemisphere"
        return result

    def polyhedron(
        scene_id: str,
        kind: Literal["tetrahedron", "octahedron", "dodecahedron", "icosahedron"] = "icosahedron",
        radius: Positive = 1,
        axis: Axis = "z",
        pivot: Pivot = "center",
        name: Name = "Polyhedron",
        capped: bool = True,
    ) -> dict[str, Any]:
        budget(108, 36)
        mesh = solids.polyhedron(kind, radius, engine.settings)
        if not capped:
            z = mesh.face_normals[:, 2]
            mesh.update_faces(z > z.min() + 1e-10)
            mesh.remove_unreferenced_vertices()
        return store(
            scene_id,
            mesh,
            name,
            "polyhedron",
            axis,
            pivot,
        )

    def plate(
        scene_id: str,
        size: Annotated[list[Positive], Field(min_length=2, max_length=2)],
        thickness: Positive = 0.01,
        holes: Annotated[list[PlateHole], Field(max_length=32)] | None = None,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        material_id: str | None = None,
        name: Name = "Plate",
    ) -> dict[str, Any]:
        parsed = [PlateHole.model_validate(value) for value in holes or []]
        n = 4 + sum(value.sides for value in parsed)
        budget(3 * (4 * n + 4 * len(parsed) - 4), 4 * n + 4 * len(parsed) - 4)
        contours = []
        for hole in parsed:
            theta = np.arange(hole.sides) * 2 * np.pi / hole.sides
            contours.append(
                (
                    np.column_stack([hole.radius * np.cos(theta), hole.radius * np.sin(theta)])
                    + hole.center
                ).tolist()
            )
        w, d = size[0] / 2, size[1] / 2
        mesh = solids.polygon_solid(
            [[-w, -d], [w, -d], [w, d], [-w, d]], contours, thickness, engine.settings, capped
        )
        result = store(scene_id, mesh, name, "plate", axis, pivot, material_id)
        from scissors.core.uv import generate_uv

        obj = engine.object(result["object_id"])[1]
        generate_uv(obj, "box", engine.settings)
        engine.checked(obj)
        return obj.summary()

    def wedge(
        scene_id: str,
        size: Annotated[list[Positive], Field(min_length=3, max_length=3)],
        taper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Wedge",
    ) -> dict[str, Any]:
        budget(36, 12)
        return store(scene_id, solids.wedge(size, taper, capped), name, "wedge", axis, pivot)

    def gear(
        scene_id: str,
        teeth: Annotated[int, Field(ge=3, le=128)] = 12,
        radius: Positive = 1,
        thickness: Positive = 0.1,
        tooth_depth: Positive | None = None,
        bore_radius: Nonnegative = 0,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Gear",
    ) -> dict[str, Any]:
        n = 4 * teeth + (min(128, 4 * teeth) if bore_radius else 0)
        budget(12 * n, 4 * n)
        mesh = solids.gear(
            teeth,
            radius,
            thickness,
            tooth_depth or 0.15 * radius,
            bore_radius,
            engine.settings,
            capped,
        )
        return store(scene_id, mesh, name, "gear", axis, pivot)

    def spring(
        scene_id: str,
        radius: Positive = 1,
        wire_radius: Positive = 0.05,
        coils: Annotated[float, Field(gt=0, le=128, allow_inf_nan=False)] = 5,
        height: Positive = 2,
        segments: Annotated[int, Field(ge=8, le=128)] = 32,
        sides: Annotated[int, Field(ge=3, le=32)] = 8,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Pivot = "base",
        name: Name = "Spring",
    ) -> dict[str, Any]:
        count = int(np.ceil(coils * segments)) + 1
        budget(count * sides + 2, 2 * count * sides)
        mesh = solids.spring(radius, wire_radius, coils, height, segments, sides, capped)
        return store(scene_id, mesh, name, "spring", axis, pivot)

    functions: dict[str, Callable[..., Any]] = {
        "capsule": capsule,
        "dome": dome,
        "hemisphere": hemisphere,
        "polyhedron": polyhedron,
        "plate": plate,
        "wedge": wedge,
        "gear": gear,
        "spring": spring,
    }
    descriptions = {
        "capsule": "Create a hemispherical-ended capsule; height is its total axial length.",
        "dome": "Create an elliptical dome with optional base disk; radius is the base radius.",
        "hemisphere": "Create half a sphere with an optional base disk.",
        "polyhedron": "Create a regular Platonic solid with flat faces and circumscribed radius.",
        "plate": "Create a plate with validated through holes, UV and optional material.",
        "wedge": "Create a ramp; taper is the lower-end height divided by the upper-end height.",
        "gear": "Create a flat straight-tooth gear with optional circular bore.",
        "spring": "Create a capped or open helical wire; height is the centerline axial travel.",
    }
    for name, fn in functions.items():
        registry.add_function(f"mesh.create_{name}", fn, descriptions[name])
