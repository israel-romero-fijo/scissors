"""Surface construction/editing contracts with shared rollback and material ownership."""

import copy
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.inspection import FaceSelection
from scissors.api.registry import OperationRegistry
from scissors.core import surfaces
from scissors.core.engine import Engine, identifier
from scissors.core.face_queries import selection_indices
from scissors.core.mesh import MeshObject, topology_mesh

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Method = Literal["miter", "normal"]
MiterLimit = Annotated[float, Field(ge=1, le=100, allow_inf_nan=False)]


def _boundary_report(mesh: trimesh.Trimesh, details: dict[str, Any]) -> dict[str, Any]:
    """Summarise why a clipped result is not watertight, for error details."""
    edges = np.asarray(topology_mesh(mesh).edges_sorted)
    if len(edges):
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        non_manifold = int(np.count_nonzero(counts > 2))
    else:
        boundary = np.zeros((0, 2), dtype=int)
        non_manifold = 0
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    seen: set[int] = set()
    closed = open_loops = 0
    for start in adjacency:
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node])
        if all(len(adjacency[node]) == 2 for node in component):
            closed += 1
        else:
            open_loops += 1
    return {
        "boundary_edges": int(len(boundary)),
        "non_manifold_edges": non_manifold,
        "loops_found": closed + open_loops,
        "closed_loops": closed,
        "open_loops": open_loops,
        "cap_faces": int(details.get("cap_faces", 0)),
        "suggestion": (
            "Use capped=false for a diagnostic section, move the plane away from vertices it "
            "touches exactly, or check that the source has consistent winding; a cap that "
            "duplicates edges is rejected instead of returned."
        ),
    }


def register_surfaces(registry: OperationRegistry, engine: Engine) -> None:
    def source(object_id: str) -> MeshObject:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh":
            raise ValueError("Surface operation requires a mesh")
        estimate = len(obj.faces) * 3 * 12 * 8 * 4 + len(obj.vertices) * 384
        if estimate > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Surface working geometry exceeds memory budget")
        return obj

    def apply(
        obj: MeshObject,
        result: surfaces.SurfaceMesh,
        replace: bool,
        name: str | None,
        operation: str,
    ) -> dict[str, Any]:
        scene = engine.object(obj.id)[0]
        target = obj if replace else obj.model_copy(deep=True)
        if not replace:
            target.id = identifier("obj")
            target.name = name or f"{obj.name} {operation}"
            scene.objects[target.id] = target
        elif name is not None:
            target.name = name
        target.replace(result.mesh, copy.deepcopy(result.uv))
        target.vertex_colors = copy.deepcopy(result.colors)
        for key in ["primitive", "boundary_rings", "center_pairs", "instance_source"]:
            target.metadata.pop(key, None)
        if result.normals is not None:
            target.normals = copy.deepcopy(result.normals)
            target.metadata["normal_method"] = "split_angle"
        target.metadata["last_surface_operation"] = operation
        engine.checked(target)
        return {
            **target.summary(),
            **result.details,
            "warnings": [
                "Offset/thickness can self-intersect in tight concave regions; inspect the result."
            ]
            if operation in {"offset", "shell"}
            else ["New caps use planar UV and the source material."],
        }

    def offset(
        object_id: str,
        distance: Finite = 0.01,
        method: Method = "miter",
        miter_limit: MiterLimit = 10,
        replace: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        obj = source(object_id)
        result = surfaces.offset(obj, distance, method, miter_limit)
        return apply(obj, result, replace, name, "offset")

    def shell(
        object_id: str,
        thickness: Positive = 0.01,
        offset: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] = 0,
        method: Method = "miter",
        miter_limit: MiterLimit = 10,
        replace: bool = True,
        name: str | None = None,
        opening: FaceSelection | None = None,
    ) -> dict[str, Any]:
        obj = source(object_id)
        indices = (
            selection_indices(obj, FaceSelection.model_validate(opening).model_dump())
            if opening is not None
            else None
        )
        source_mesh = topology_mesh(obj.geometry())
        if opening is not None and not source_mesh.is_volume:
            raise ValueError("A selected shell opening requires a closed outward source volume")
        result = surfaces.shell(
            obj, thickness, offset, method, miter_limit, engine.settings, indices
        )
        if not topology_mesh(result.mesh).is_watertight:
            raise ScissorsError("GEOMETRY_ERROR", "Shell did not produce a closed surface")
        # Certify only an exact rectangular source and either no opening or one
        # complete axial side. Other shells keep rim provenance, without a bound.
        certificate: dict[str, Any] = {
            "method": "sampled",
            "rim_faces": result.details["rim_face_indices"],
        }
        rectangular = (
            method == "miter"
            and len(source_mesh.vertices) == 8
            and len(source_mesh.faces) == 12
            and np.all(np.abs(source_mesh.face_normals).max(axis=1) == 1)
            and np.isclose(source_mesh.volume, np.prod(source_mesh.extents), rtol=1e-12)
        )
        complete_side = indices is None
        if rectangular and indices is not None:
            directions = obj.geometry().face_normals
            chosen = directions[indices]
            complete_side = bool(
                np.all(chosen == chosen[0])
                and np.array_equal(
                    np.sort(indices), np.flatnonzero(np.all(directions == chosen[0], axis=1))
                )
            )
        if rectangular and complete_side:
            if np.any(source_mesh.extents + thickness * (offset - 1) <= 0):
                raise ValueError("Shell thickness consumes the rectangular cavity")
            n = len(obj.vertices)
            deltas = np.asarray(result.mesh.vertices[:n]) - np.asarray(
                result.mesh.vertices[n : 2 * n]
            )
            distances = np.abs(
                np.einsum(
                    "ij,ij->i", obj.geometry().face_normals, deltas[np.asarray(obj.faces)[:, 0]]
                )
            )
            certificate.update(method="rectangular_miter_shell", minimum=float(distances.min()))
        response = apply(obj, result, replace, name, "shell")
        target = engine.object(response["object_id"])[1]
        certificate["signature"] = target.geometry_signature()
        target.metadata["_thickness_certificate"] = certificate
        return response

    def clip(
        object_id: str,
        normal: Vector,
        point: Vector | None = None,
        keep: Literal["positive", "negative"] = "positive",
        capped: bool = True,
        tolerance: Positive = 1e-8,
        replace: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        obj = source(object_id)
        direction = np.asarray(normal, dtype=float) * (1 if keep == "positive" else -1)
        result = surfaces.plane_clip(
            obj, direction, np.asarray(point or [0, 0, 0]), capped, tolerance, engine.settings
        )
        if (
            capped
            and topology_mesh(obj.geometry()).is_watertight
            and not topology_mesh(result.mesh).is_watertight
        ):
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Capped cut did not remain watertight",
                _boundary_report(result.mesh, result.details),
            )
        return apply(obj, result, replace, name, "clip")

    def slice(
        object_id: str,
        normal: Vector,
        point: Vector | None = None,
        capped: bool = True,
        tolerance: Positive = 1e-8,
        replace: bool = False,
        name: str | None = None,
    ) -> dict[str, Any]:
        obj = source(object_id)
        direction, origin = np.asarray(normal, dtype=float), np.asarray(point or [0, 0, 0])
        a = surfaces.plane_clip(obj, direction, origin, capped, tolerance, engine.settings)
        b = surfaces.plane_clip(obj, -direction, origin, capped, tolerance, engine.settings)
        if capped and topology_mesh(obj.geometry()).is_watertight:
            broken = {
                side: _boundary_report(result.mesh, result.details)
                for side, result in [("positive", a), ("negative", b)]
                if not topology_mesh(result.mesh).is_watertight
            }
            if broken:
                raise ScissorsError(
                    "GEOMETRY_ERROR",
                    "Capped slice did not remain watertight",
                    {"sides": broken},
                )
        # Apply the new negative part first so replacing the source cannot affect its template.
        negative = apply(obj, b, False, f"{name or obj.name} negative", "slice")
        positive = apply(obj, a, replace, f"{name or obj.name} positive", "slice")
        return {
            "source_object_id": object_id,
            "positive": positive,
            "negative": negative,
            "source_replaced": replace,
        }

    registry.add_function(
        "mesh.offset",
        offset,
        "Offset welded surface vertices by signed world distance; keep UV/material.",
    )
    registry.add_function(
        "mesh.shell",
        shell,
        "Build outer/inner surfaces and boundary walls with UV and source material.",
    )
    registry.add_function(
        "mesh.clip",
        clip,
        "Keep one side of a world plane, interpolate attributes and optionally cap holes.",
    )
    registry.add_function(
        "mesh.slice",
        slice,
        "Create both sides of a world plane with optional caps and preserved material.",
    )
