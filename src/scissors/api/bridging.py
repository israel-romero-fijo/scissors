"""Public boundary queries and bridges through the shared registry."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.bridging import boundary_loops, bridge_geometry, describe_boundaries
from scissors.core.engine import Engine
from scissors.core.mesh import topology_mesh

Index = Annotated[int, Field(ge=0)]


class BoundarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    geometry_revision: str
    loop_index: Index


def register_bridging(registry: OperationRegistry, engine: Engine) -> None:
    def boundaries(object_id: str) -> dict[str, Any]:
        return describe_boundaries(engine.object(object_id)[1])

    def bridge(
        first: BoundarySelection,
        second: BoundarySelection,
        name: str = "Bridged mesh",
        replace: bool = False,
        second_offset: Index | None = None,
        material_id: str | None = None,
        uv_policy: Literal["preserve", "discard"] = "preserve",
    ) -> dict[str, Any]:
        selections = [BoundarySelection.model_validate(value) for value in (first, second)]
        objects = []
        loops = []
        scenes = []
        for selection in selections:
            scene, obj = engine.object(selection.object_id)
            if obj.geometry_revision != selection.geometry_revision:
                raise ScissorsError("INVALID_PARAMETER", "Boundary selection is stale")
            available = boundary_loops(obj)
            if selection.loop_index >= len(available):
                raise ValueError("Boundary loop_index is out of range")
            scenes.append(scene)
            objects.append(obj)
            loops.append(available[selection.loop_index])
        a, b = objects
        if a.id == b.id or scenes[0].id != scenes[1].id:
            raise ValueError("Bridge requires two different meshes in the same scene")
        if material_id is None:
            if a.material_id != b.material_id:
                raise ValueError("Different source materials require an explicit material_id")
            material_id = a.material_id
        elif material_id not in scenes[0].materials:
            raise ScissorsError("MATERIAL_NOT_FOUND", "Material does not belong to the scene")
        uv = None
        if uv_policy == "preserve":
            if (a.uv is None) != (b.uv is None):
                raise ValueError(
                    "Mixed UV availability requires uv_policy='discard' or UV generation"
                )
            if a.uv is not None and b.uv is not None:
                uv = [*a.uv, *b.uv]
        mesh, offset = bridge_geometry(a, b, loops[0], loops[1], engine.settings, second_offset)
        topology = topology_mesh(mesh)
        # Coincident source surfaces can otherwise introduce hidden non-manifold joins.
        _, counts = np.unique(topology.edges_sorted, axis=0, return_counts=True)
        if np.any(counts > 2) or not topology.is_winding_consistent:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Bridge requires consistently wound manifold sources"
            )
        result = engine.add_mesh(scenes[0].id, mesh, name, "bridge")
        obj = engine.object(result["object_id"])[1]
        obj.uv = uv
        obj.material_id = material_id
        if a.vertex_colors is not None or b.vertex_colors is not None:
            obj.vertex_colors = [
                *(a.vertex_colors or [[255, 255, 255, 255]] * len(a.vertices)),
                *(b.vertex_colors or [[255, 255, 255, 255]] * len(b.vertices)),
            ]
        engine.checked(obj)
        if replace:
            engine.remove_objects(scenes[0].id, [a.id, b.id])
        return {
            **obj.summary(),
            "bridge_face_indices": list(range(len(a.faces) + len(b.faces), len(obj.faces))),
            "source_object_ids": [a.id, b.id],
            "source_vertex_offsets": [0, len(a.vertices)],
            "second_offset": offset,
            "sources_replaced": replace,
            "warnings": [
                "Bridge faces reuse boundary UV; inspect texture stretching and intersections."
            ],
        }

    registry.add_function(
        "mesh.boundaries",
        boundaries,
        "List directed closed boundary loops on exact-position topology "
        "with revision and render indices.",
        side_effects=False,
        permissions=["scene.read"],
    )
    registry.add_function(
        "mesh.bridge",
        bridge,
        "Join revision-bound boundary loops of two meshes in one scene "
        "with an oriented triangle strip. "
        "Supports unequal edge counts. Create a combined mesh; optionally replace sources. "
        "Preserve source UV and colors; differing materials require an explicit material_id. "
        "No guarantee against self-intersections; inspect the result.",
    )
