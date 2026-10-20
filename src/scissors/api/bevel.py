"""Solid edge treatments and constrained local surface reconstruction."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scissors.api.inspection import FaceSelection
from scissors.api.registry import OperationRegistry
from scissors.core.bevel import edge_treatment
from scissors.core.engine import Engine, identifier

Width = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Segments = Annotated[int, Field(ge=1, le=32)]
RoundSegments = Annotated[int, Field(ge=2, le=32)]
Mode = Literal["convex", "concave", "both"]


class EdgeSelection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    object_id: str
    geometry_revision: str
    edge_indices: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]


def register_bevel(registry: OperationRegistry, engine: Engine) -> None:
    def reconstruct_patch(
        object_id: str,
        selection: FaceSelection,
        refinement: Annotated[int, Field(ge=1, le=3)] = 1,
        iterations: Annotated[int, Field(ge=0, le=100)] = 5,
        strength: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.5,
        sharp_angle_degrees: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)] = 45,
    ) -> dict[str, Any]:
        from scissors.core.patches import reconstruct_patch as reconstruct

        return reconstruct(
            engine,
            object_id,
            FaceSelection.model_validate(selection).model_dump(),
            refinement,
            iterations,
            strength,
            sharp_angle_degrees,
        )

    registry.add_function(
        "mesh.reconstruct_patch",
        reconstruct_patch,
        "Refine and smooth selected face regions while pinning their borders and "
        "sharp edges. Preserve unselected face positions and interpolate UV/colors. "
        "Reject reversed or degenerate children; inspect shrinkage and intersections.",
    )

    def apply(
        object_id: str,
        width: float,
        segments: int,
        mode: str,
        preserve_uv: bool,
        preserve_topology: bool,
        replace: bool,
        name: str | None,
        selection: EdgeSelection | None = None,
        radius: bool = False,
    ) -> dict[str, Any]:
        scene, obj = engine.object(object_id)
        selected = None
        if selection is None:
            target = edge_treatment(
                obj, width, segments, mode, preserve_uv, preserve_topology, engine.settings
            )
        else:
            from scissors.core.local_bevel import selected_edge_treatment

            target, selected = selected_edge_treatment(
                obj,
                EdgeSelection.model_validate(selection).model_dump(),
                width,
                segments,
                mode,
                preserve_uv,
                preserve_topology,
                engine.settings,
                radius,
            )
        if not replace:
            target.id = identifier("obj")
            target.name = name or f"{obj.name} beveled"
        elif name is not None:
            target.name = name
        engine.checked(target)
        scene.objects[target.id] = target
        return {
            **target.summary(),
            "source_object_id": object_id,
            "method": "selected_edge_prisms"
            if selected is not None
            else "rolling_polyhedron"
            if segments == 1
            else "rolling_sphere",
            "selected_edge_indices": selected,
            "width": width,
            "segments": segments,
            "mode": mode,
            "cleanup_tolerance": 0
            if selected is not None
            else target.metadata["bevel_cleanup_tolerance"],
            "uv_method": "source_interpolation_and_planar_cuts"
            if selected is not None and preserve_uv and obj.uv is not None
            else "nearest_source"
            if preserve_uv and obj.uv is not None
            else "planar",
            "warnings": [
                "Finite selected-edge prisms stop at edge endpoints; inspect widths and junctions."
                if selected is not None
                else "Global solid treatment: small features may disappear.",
                "Projected UV can stretch on new surfaces; unwrap and rebake for final textures.",
            ],
        }

    def bevel(
        object_id: str,
        width: Width = 0.01,
        segments: Segments = 1,
        mode: Mode = "convex",
        preserve_uv: bool = True,
        preserve_topology: bool = True,
        replace: bool = True,
        name: str | None = None,
        selection: EdgeSelection | None = None,
    ) -> dict[str, Any]:
        return apply(
            object_id,
            width,
            segments,
            mode,
            preserve_uv,
            preserve_topology,
            replace,
            name,
            selection,
        )

    def chamfer(
        object_id: str,
        width: Width = 0.01,
        mode: Mode = "convex",
        preserve_uv: bool = True,
        preserve_topology: bool = True,
        replace: bool = True,
        name: str | None = None,
        selection: EdgeSelection | None = None,
    ) -> dict[str, Any]:
        return apply(
            object_id, width, 1, mode, preserve_uv, preserve_topology, replace, name, selection
        )

    def fillet(
        object_id: str,
        radius: Width = 0.01,
        segments: RoundSegments = 8,
        mode: Mode = "convex",
        preserve_uv: bool = True,
        preserve_topology: bool = True,
        replace: bool = True,
        name: str | None = None,
        selection: EdgeSelection | None = None,
    ) -> dict[str, Any]:
        return apply(
            object_id,
            radius,
            segments,
            mode,
            preserve_uv,
            preserve_topology,
            replace,
            name,
            selection,
            True,
        )

    operations: list[tuple[str, Callable[..., Any], str]] = [
        (
            "mesh.bevel",
            bevel,
            "Bevel selected revision-bound solid creases with finite prisms, or apply a global "
            "closed-solid rolling kernel when selection is omitted. "
            "Selected width is face setback.",
        ),
        (
            "mesh.chamfer",
            chamfer,
            "Chamfer selected solid creases, or all solid edges when selection is omitted.",
        ),
        (
            "mesh.fillet",
            fillet,
            "Fillet selected solid creases with sampled circular arcs, "
            "or use a global rolling sphere.",
        ),
    ]
    for operation, function, description in operations:
        registry.add_function(operation, function, description)
