"""Discoverable geometry selection operations and named selection management."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.inspection import FaceRegion
from scissors.api.registry import OperationRegistry
from scissors.core import selections
from scissors.core.engine import Engine

Index = Annotated[int, Field(ge=0)]
Tolerance = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Name = Annotated[str, Field(min_length=1, max_length=128, pattern=r"\S")]


class GeometrySelection(BaseModel):
    # Accept complete query responses. Only the kind's primary indices are consumed.
    model_config = ConfigDict(extra="ignore")
    object_id: str
    geometry_revision: str
    kind: Literal["faces", "vertices", "edges", "boundaries", "components"] = "faces"
    face_indices: list[Index] | None = None
    vertex_indices: list[Index] | None = None
    edge_indices: list[Index] | None = None
    loop_indices: list[Index] | None = None
    component_indices: list[Index] | None = None


def register_selections(registry: OperationRegistry, engine: Engine) -> None:
    def query(
        object_id: str,
        kind: str,
        requested: list[int] | None,
        region: FaceRegion | None,
        tolerance: float,
        boundary_only: bool = False,
        seed_faces: list[int] | None = None,
    ) -> dict[str, Any]:
        box = FaceRegion.model_validate(region).model_dump() if region is not None else None
        return selections.select_geometry(
            engine.object(object_id)[1],
            kind,
            requested,
            box,
            tolerance,
            engine.settings,
            boundary_only,
            seed_faces,
        )

    def select_vertices(
        object_id: str,
        vertex_indices: list[Index] | None = None,
        region: FaceRegion | None = None,
        tolerance: Tolerance = 1e-6,
    ) -> dict[str, Any]:
        return query(object_id, "vertices", vertex_indices, region, tolerance)

    def select_edges(
        object_id: str,
        edge_indices: list[Index] | None = None,
        region: FaceRegion | None = None,
        boundary_only: bool = False,
        tolerance: Tolerance = 1e-6,
    ) -> dict[str, Any]:
        return query(object_id, "edges", edge_indices, region, tolerance, boundary_only)

    def select_boundaries(
        object_id: str,
        loop_indices: list[Index] | None = None,
        region: FaceRegion | None = None,
        tolerance: Tolerance = 1e-6,
    ) -> dict[str, Any]:
        return query(object_id, "boundaries", loop_indices, region, tolerance)

    def select_components(
        object_id: str,
        component_indices: list[Index] | None = None,
        seed_face_indices: list[Index] | None = None,
        region: FaceRegion | None = None,
        tolerance: Tolerance = 1e-6,
    ) -> dict[str, Any]:
        return query(
            object_id,
            "components",
            component_indices,
            region,
            tolerance,
            seed_faces=seed_face_indices,
        )

    def selection_save(
        name: Name,
        selection: GeometrySelection,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        parsed = GeometrySelection.model_validate(selection)
        return selections.save_selection(
            engine.object(parsed.object_id)[1],
            name,
            parsed.model_dump(),
            overwrite,
            engine.settings,
        )

    def selection_get(object_id: str, name: Name) -> dict[str, Any]:
        return selections.get_selection(engine.object(object_id)[1], name)

    def selection_list(object_id: str) -> dict[str, Any]:
        return selections.list_selections(engine.object(object_id)[1])

    def selection_delete(object_id: str, name: Name) -> dict[str, Any]:
        return selections.delete_selection(engine.object(object_id)[1], name)

    queries: list[tuple[str, Callable[..., Any], str]] = [
        (
            "select_vertices",
            select_vertices,
            "Select render vertices by explicit indices AND a world box; "
            "coincident seam copies stay distinct.",
        ),
        (
            "select_edges",
            select_edges,
            "Select deterministic exact-position topology edges by indices AND box "
            "AND optional boundary filter. "
            "Return representative render endpoint pairs; intersects tests actual segments.",
        ),
        (
            "select_boundaries",
            select_boundaries,
            "Select whole closed boundary loops by indices AND box. "
            "Loop indices match mesh.boundaries.",
        ),
        (
            "select_components",
            select_components,
            "Select whole edge-connected face components by indices AND seed faces AND world box. "
            "Point contact does not connect components. Return revision-bound face indices.",
        ),
        (
            "selection_get",
            selection_get,
            "Retrieve a named selection; reject stale geometry revisions.",
        ),
        (
            "selection_list",
            selection_list,
            "List named selections with kind, count and stale status.",
        ),
    ]
    for name, fn, description in queries:
        registry.add_function(f"mesh.{name}", fn, description, side_effects=False)
    registry.add_function(
        "mesh.selection_save",
        selection_save,
        "Validate and persist a named revision-bound selection on its mesh. "
        "Require overwrite=true to replace an existing name; preserve geometry.",
    )
    registry.add_function(
        "mesh.selection_delete",
        selection_delete,
        "Delete a named selection, including stale selections; preserve geometry.",
    )
