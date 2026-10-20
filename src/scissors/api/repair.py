"""Native mesh repair with explicit seam and geometry policies."""

from collections.abc import Callable
from itertools import product
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.inspection import FaceSelection
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject, topology_mesh
from scissors.core.simplification import simplify_mesh

Finite = Annotated[float, Field(allow_inf_nan=False)]


class BoundaryLoopSelection(BaseModel):
    # Consume complete select_boundaries and saved boundary selection responses.
    model_config = ConfigDict(extra="ignore")
    object_id: str
    geometry_revision: str
    loop_indices: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]


def register_repair(registry: OperationRegistry, engine: Engine) -> None:
    from scissors.api.bridging import register_bridging
    from scissors.api.face_editing import register_face_editing
    from scissors.api.surface_transfer import register_surface_transfer
    from scissors.api.topology_repair import register_topology_repair

    register_bridging(registry, engine)
    register_face_editing(registry, engine)
    register_surface_transfer(registry, engine)
    register_topology_repair(registry, engine)

    def mesh_object(object_id: str) -> MeshObject:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh":
            raise ValueError("Operation requires a mesh")
        return obj

    def split_normals(
        object_id: str, angle: Annotated[float, Field(ge=0, le=180, allow_inf_nan=False)] = 30
    ) -> dict[str, Any]:
        from scissors.core.normals import split_normals as split

        obj = mesh_object(object_id)
        split(obj, angle, engine.settings)
        engine.checked(obj)
        return obj.summary()

    def update(
        obj: MeshObject, mesh: trimesh.Trimesh, uv: list[list[float]] | None
    ) -> dict[str, Any]:
        obj.replace(mesh, uv)
        obj.metadata.pop("boundary_rings", None)
        obj.metadata.pop("center_pairs", None)
        obj.metadata.pop("primitive", None)
        engine.checked(obj)
        return obj.summary()

    def weld(
        object_id: str,
        tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0,
        preserve_uv: bool = True,
    ) -> dict[str, Any]:
        obj = mesh_object(object_id)
        vertices = np.asarray(obj.vertices)
        uv = np.asarray(obj.uv) if obj.uv is not None else None
        representatives: list[int] = []
        cells: dict[tuple[Any, ...], list[int]] = {}
        mapping = np.empty(len(vertices), dtype=np.int64)
        neighbor_offsets = list(product((-1, 0, 1), repeat=3)) if tolerance else [(0, 0, 0)]
        comparisons = 0
        for i, vertex in enumerate(vertices):
            position = np.floor(vertex / tolerance).astype(np.int64) if tolerance else vertex
            cell = tuple(position.tolist())
            chosen: int | None = None
            for offset in neighbor_offsets:
                key = tuple(position + offset) if tolerance else cell
                for candidate in cells.get(key, []):
                    comparisons += 1
                    if comparisons > engine.settings.max_geometry_pairs:
                        raise ScissorsError("RESOURCE_LIMIT", "Welding exceeds comparison budget")
                    if np.linalg.norm(vertex - vertices[candidate]) <= tolerance and (
                        not preserve_uv or uv is None or np.array_equal(uv[i], uv[candidate])
                    ):
                        chosen = candidate
                        break
                if chosen is not None:
                    break
            if chosen is None:
                mapping[i] = len(representatives)
                representatives.append(i)
                cells.setdefault(cell, []).append(i)
            else:
                mapping[i] = mapping[chosen]
        faces = mapping[np.asarray(obj.faces)]
        keep = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        mesh = trimesh.Trimesh(vertices=vertices[representatives], faces=faces[keep], process=False)
        mesh_uv = uv[representatives].tolist() if uv is not None else None
        return {
            **update(obj, mesh, mesh_uv),
            "merged_vertices": len(vertices) - len(representatives),
            "removed_faces": int(np.count_nonzero(~keep)),
            "preserve_uv": preserve_uv,
        }

    def clean(object_id: str) -> dict[str, Any]:
        obj = mesh_object(object_id)
        vertices, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
        areas = obj.geometry().area_faces
        keep = areas > 1e-14
        # Duplicate faces are duplicates regardless of cyclic order or winding.
        _, unique = np.unique(np.sort(faces[keep], axis=1), axis=0, return_index=True)
        selected = faces[keep][np.sort(unique)]
        used, inverse = np.unique(selected, return_inverse=True)
        uv = np.asarray(obj.uv)[used].tolist() if obj.uv is not None else None
        mesh = trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)
        return {
            **update(obj, mesh, uv),
            "removed_faces": len(faces) - len(selected),
            "removed_vertices": len(vertices) - len(used),
        }

    def flip(object_id: str, selection: FaceSelection | None = None) -> dict[str, Any]:
        from scissors.core.face_editing import edit_faces

        obj = mesh_object(object_id)
        parsed = (
            FaceSelection.model_validate(selection).model_dump()
            if selection is not None
            else {
                "object_id": obj.id,
                "geometry_revision": obj.geometry_revision,
                "face_indices": list(range(len(obj.faces))),
            }
        )
        return edit_faces(engine, object_id, parsed, "flip")

    def smooth(
        object_id: str,
        iterations: Annotated[int, Field(ge=1, le=100)] = 1,
        factor: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.5,
        preserve_boundary: bool = True,
    ) -> dict[str, Any]:
        obj = mesh_object(object_id)
        mesh = topology_mesh(obj.geometry())
        vertices = np.asarray(mesh.vertices).copy()
        edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        fixed = np.unique(edges[counts == 1]) if preserve_boundary else np.array([], dtype=int)
        degree = np.zeros(len(vertices))
        for corner in range(2):
            np.add.at(degree, edges[:, corner], 1)
        for _ in range(iterations):
            neighbors = np.zeros_like(vertices)
            np.add.at(neighbors, edges[:, 0], vertices[edges[:, 1]])
            np.add.at(neighbors, edges[:, 1], vertices[edges[:, 0]])
            displacement = factor * (neighbors / np.maximum(degree[:, None], 1) - vertices)
            displacement[fixed] = 0
            vertices += displacement
        _, mapping = np.unique(np.asarray(obj.vertices), axis=0, return_inverse=True)
        updated = trimesh.Trimesh(vertices=vertices[mapping], faces=obj.faces, process=False)
        return {
            **update(obj, updated, obj.uv),
            "iterations": iterations,
            "warnings": ["Laplacian smoothing can shrink volume; inspect the result."],
        }

    def convex_hull(object_id: str, name: str | None = None) -> dict[str, Any]:
        from scissors.core.geometry import run_backend

        scene, obj = engine.object(object_id)
        if obj.kind != "mesh":
            raise ValueError("Convex hull requires a mesh")
        data = run_backend({"backend": "convex_hull", "vertices": obj.vertices}, engine.settings)
        return engine.add_mesh(
            scene.id,
            trimesh.Trimesh(**data, process=False),
            name or f"{obj.name} hull",
            "convex_hull",
        )

    def collider(
        object_id: str,
        max_faces: Annotated[int, Field(ge=4)] = 256,
        name: str | None = None,
    ) -> dict[str, Any]:
        result = convex_hull(object_id, name or "Collider")
        obj = engine.object(result["object_id"])[1]
        if len(obj.faces) > max_faces:
            simplify_mesh(
                obj, engine.settings, 100 * (1 - max_faces / len(obj.faces)), "discard", 1, True
            )
        obj.metadata["collider_for"] = object_id
        if len(obj.faces) > max_faces:
            raise ScissorsError("GEOMETRY_ERROR", "Collider cannot meet the requested face budget")
        engine.checked(obj)
        return {
            **obj.summary(),
            "max_faces": max_faces,
            "warnings": ["A convex collider covers concave cavities."],
        }

    def fill_holes(
        object_id: str,
        max_edges: Annotated[int, Field(ge=3, le=512)] = 64,
        selection: BoundaryLoopSelection | None = None,
        method: Literal["projected", "smooth"] = "projected",
        refinement: Annotated[int, Field(ge=1, le=4)] = 2,
        tangent_tolerance_degrees: Annotated[float, Field(gt=0, le=90, allow_inf_nan=False)] = 20,
    ) -> dict[str, Any]:
        from scissors.core.hole_filling import fill_holes as fill

        parsed = (
            BoundaryLoopSelection.model_validate(selection).model_dump()
            if selection is not None
            else None
        )
        return fill(
            engine, object_id, max_edges, parsed, method, refinement, tangent_tolerance_degrees
        )

    def repair_winding(
        object_id: str,
        selection: FaceSelection | None = None,
        orientation: Literal["preserve", "outward", "inward"] = "preserve",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        from scissors.core.winding import repair_winding as repair

        parsed = (
            FaceSelection.model_validate(selection).model_dump() if selection is not None else None
        )
        return repair(engine, object_id, parsed, orientation, dry_run)

    functions: dict[str, Callable[..., Any]] = {
        "split_normals": split_normals,
        "weld": weld,
        "clean": clean,
        "flip": flip,
        "smooth": smooth,
        "convex_hull": convex_hull,
        "collider": collider,
        "fill_holes": fill_holes,
        "repair_winding": repair_winding,
    }
    descriptions = {
        "split_normals": "Split hard edges by angle; smooth normals across UV seams.",
        "weld": "Weld coincident or nearby vertices; preserve UV seams by default.",
        "clean": "Remove degenerate/duplicate triangles and unused render vertices.",
        "flip": "Reverse selected triangles, or every triangle when selection is omitted; "
        "preserve attributes and recompute normals.",
        "smooth": "Smooth welded topology with Laplacian neighbors and boundary preservation.",
        "convex_hull": "Create a separate convex hull mesh; keep the source.",
        "collider": "Create a convex collider, requiring the requested triangle budget.",
        "fill_holes": "Fill selected revision-bound boundary loops, or all loops up to max_edges. "
        "Preserve source geometry and attributes. Smooth mode fits a refined collar patch and "
        "requires measured boundary face angles within tangent_tolerance_degrees; "
        "failure is atomic.",
        "repair_winding": "Repair edge winding within a face/component selection; keep outside "
        "faces fixed. Preserve majority orientation when unanchored. Outward/inward modes require "
        "complete closed components and use signed volume, without inferring cavity nesting. "
        "Dry runs report planned flips without changing geometry.",
    }
    for name, fn in functions.items():
        registry.add_function(f"mesh.{name}", fn, descriptions[name])
