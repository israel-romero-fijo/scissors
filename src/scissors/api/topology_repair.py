"""Public repair workflows for scene meshes and damaged triangle soups."""

from typing import Annotated, Any, Literal

from pydantic import Field

from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.topology_repair import repair_surface

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Point = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=3, max_length=3)
]
Triangle = Annotated[
    list[Annotated[int, Field(ge=0, strict=True)]], Field(min_length=3, max_length=3)
]


def register_topology_repair(registry: OperationRegistry, engine: Engine) -> None:
    def repair_topology(
        object_id: str,
        voxel_size: Positive,
        mode: Literal["solid", "shell"] = "solid",
        shell_thickness: Positive | None = None,
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh":
            raise ValueError("Topology repair requires a mesh")
        mesh, details = repair_surface(
            obj.vertices, obj.faces, voxel_size, mode, shell_thickness, engine.settings
        )
        obj.replace(mesh)
        for key in ("primitive", "boundary_rings", "center_pairs"):
            obj.metadata.pop(key, None)
        engine.checked(obj)
        details["validation"]["object_id"] = obj.id
        return {**obj.summary(), **details}

    def repair_topology_data(
        scene_id: str,
        vertices: Annotated[list[Point], Field(min_length=3)],
        faces: Annotated[list[Triangle], Field(min_length=1)],
        voxel_size: Positive,
        mode: Literal["solid", "shell"] = "solid",
        shell_thickness: Positive | None = None,
        name: str = "Repaired mesh",
    ) -> dict[str, Any]:
        engine.scene(scene_id)
        mesh, details = repair_surface(
            vertices, faces, voxel_size, mode, shell_thickness, engine.settings
        )
        result = engine.add_mesh(scene_id, mesh, name, "repaired")
        details["validation"]["object_id"] = result["object_id"]
        return {**result, **details}

    description = (
        "Reconstruct a closed manifold surface and verify topology, vertex links and "
        "nonadjacent self-intersections before publication. Solid mode samples absolute "
        "generalized winding above 0.5; oriented surfaces define the intended interior. "
        "Shell mode creates a two-sided envelope with explicit shell_thickness >= 3*voxel_size. "
        "Resolution changes geometry and can close small gaps. UV/colors are discarded. "
        "Resource limits or unverifiable results fail atomically. "
    )
    registry.add_function(
        "mesh.repair_topology",
        repair_topology,
        description + "Replace an existing mesh; retain its material and object identity.",
    )
    registry.add_function(
        "mesh.repair_topology_data",
        repair_topology_data,
        description + "Accept raw damaged triangles, including non-manifold topology; "
        "create only the validated result, without adding invalid input to the scene.",
    )
