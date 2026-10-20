"""Native arrays, surface placement, scene symmetry and global deformation."""

import hashlib
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.inspection import triangle_distances
from scissors.core.measurement import circle_measure, hole_rim
from scissors.core.mesh import MeshObject
from scissors.core.placement import alignment, face_frame, nearest_face, reflection, unit, warp
from scissors.core.tangents import tangent_frames

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Axis = Literal["x", "y", "z"]
Face = Annotated[int, Field(ge=0)]


def fibonacci_directions(count: int) -> np.ndarray:
    """Deterministic low-discrepancy directions on the unit sphere."""
    golden = np.pi * (1 + 5**0.5)
    index = np.arange(count, dtype=float)
    y = 1.0 - 2.0 * (index + 0.5) / count
    ring = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    angle = golden * index
    return np.column_stack([ring * np.cos(angle), ring * np.sin(angle), y])


def register_editing(registry: OperationRegistry, engine: Engine) -> None:
    def tangents(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh" or obj.uv is None:
            raise ValueError("Tangents require a mesh with UV")
        values = tangent_frames(obj.geometry(), np.asarray(obj.uv))
        obj.metadata["export_tangents"] = True
        return {
            "object_id": obj.id,
            "tangents": values.tolist(),
            "vertex_count": len(values),
            "method": "angle-weighted UV derivative frames",
            "warnings": ["Frames are not MikkTSpace; rebake for tools requiring exact MikkTSpace."],
        }

    registry.add_function(
        "mesh.tangents",
        tangents,
        "Compute world XYZ/handedness tangent frames and enable fresh glTF TANGENT export.",
    )

    def measure(
        object_id: str,
        mode: Literal["distance", "angle", "diameter"],
        face_a: Face | None = None,
        object_id_b: str | None = None,
        face_b: Face | None = None,
        vertex_indices: list[Face] | None = None,
        hole_index: Face = 0,
        tolerance: Positive = 1e-6,
    ) -> dict[str, Any]:
        scene, obj = engine.object(object_id)
        if obj.kind != "mesh":
            raise ValueError("Measurements require a mesh")
        if mode == "diameter":
            if vertex_indices is not None:
                if not vertex_indices or max(vertex_indices) >= len(obj.vertices):
                    raise ValueError("Rim vertex index is outside the mesh")
                points = np.asarray(obj.vertices)[vertex_indices]
            elif face_a is not None:
                points = hole_rim(obj, face_a, hole_index, tolerance)
            else:
                raise ValueError("Diameter needs rim vertex_indices or a planar face_a with holes")
            return {"object_id": object_id, "mode": mode, **circle_measure(points, tolerance)}
        if face_a is None or face_b is None:
            raise ValueError("Face distance and angle require face_a and face_b")
        other_scene, other = engine.object(object_id_b or object_id)
        if other_scene.id != scene.id:
            raise ValueError("Measurements require objects in the same scene")
        _, first_normal = face_frame(obj, face_a)
        _, second_normal = face_frame(other, face_b)
        if mode == "distance":
            distance = triangle_distances(
                obj.geometry().triangles[[face_a]], other.geometry().triangles[[face_b]]
            )[0]
            return {"object_id": object_id, "mode": mode, "distance": float(distance)}
        degrees = float(np.degrees(np.arccos(np.clip(np.dot(first_normal, second_normal), -1, 1))))
        return {
            "object_id": object_id,
            "mode": mode,
            "angle_degrees": degrees,
            "acute_plane_angle_degrees": min(degrees, 180 - degrees),
        }

    registry.add_function(
        "mesh.measure",
        measure,
        "Measure triangle distance, normal/plane angle or circular hole diameter in world units.",
        side_effects=False,
    )

    def array(
        object_id: str,
        count: Annotated[int, Field(ge=1, le=4096)] = 2,
        mode: Literal["linear", "polar"] = "linear",
        offset: Vector | None = None,
        axis: Axis = "z",
        center: Vector | None = None,
        angle_degrees: Annotated[float, Field(gt=0, le=360, allow_inf_nan=False)] = 360,
        rotate: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        scene, source = engine.object(object_id)
        subtree = [source, *engine.descendants(scene, object_id)]
        if len(subtree) * count > engine.settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Array object count exceeds workload budget")
        used = sum(
            len(obj.vertices) * 384 + len(obj.faces) * 192
            for other in engine.scenes.values()
            for obj in other.objects.values()
        )
        extra = sum(len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in subtree) * (count - 1)
        if used + extra > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Array exceeds workspace geometry budget")
        ids = [object_id]
        origin = np.asarray(center or [0, 0, 0], dtype=float)
        geometry = [np.asarray(obj.vertices) for obj in subtree if obj.kind == "mesh"]
        if geometry:
            vertices = np.concatenate(geometry)
            anchor = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        else:
            anchor = np.asarray(source.pivot)
        denominator = count if angle_degrees == 360 else max(count - 1, 1)
        for i in range(1, count):
            obj = engine.clone_object(object_id, f"{name or source.name} {i + 1}")
            if mode == "linear":
                matrix = trimesh.transformations.translation_matrix(
                    np.asarray(offset or [1, 0, 0]) * i
                )
            else:
                matrix = trimesh.transformations.rotation_matrix(
                    np.radians(angle_degrees * i / denominator),
                    np.eye(3)["xyz".index(axis)],
                    origin,
                )
                if not rotate:
                    destination = trimesh.transformations.transform_points([anchor], matrix)[0]
                    matrix = trimesh.transformations.translation_matrix(destination - anchor)
            engine.transform(obj.id, matrix)
            ids.append(obj.id)
        return {
            "source_object_id": object_id,
            "object_ids": ids,
            "count": count,
            "created_count": count - 1,
            "mode": mode,
        }

    def deform(
        object_id: str,
        bend: Annotated[float, Field(gt=-360, lt=360, allow_inf_nan=False)] = 0,
        twist: Annotated[float, Field(ge=-3600, le=3600, allow_inf_nan=False)] = 0,
        taper: Positive = 1,
        axis: Axis = "z",
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        warp(obj, bend, twist, taper, axis)
        engine.checked(obj)
        return {
            **obj.summary(),
            "warnings": [
                "Global deformation preserves UV and material; sparse rings produce coarse bends.",
                "Reapply split_normals after deformation and inspect self-intersections.",
            ],
        }

    def source_data(
        obj: MeshObject, source_face: int | None, axis: str, source_side: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if source_face is not None:
            anchor, normal = face_frame(obj, source_face)
            return np.asarray(obj.vertices), anchor, normal
        scene = engine.object(obj.id)[0]
        objects = [obj, *engine.descendants(scene, obj.id)]
        points = [np.asarray(child.vertices) for child in objects if child.kind == "mesh"]
        if not points:
            raise ValueError("Surface placement requires mesh geometry in the object subtree")
        vertices = np.concatenate(points)
        anchor = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        normal = np.linalg.inv(np.asarray(obj.transform)[:3, :3]).T @ np.eye(3)["xyz".index(axis)]
        if source_side == "negative":
            normal = -normal
        return vertices, anchor, unit(normal)

    def valid_target(object_id: str, target_object_id: str) -> tuple[MeshObject, MeshObject]:
        scene, source = engine.object(object_id)
        target_scene, target = engine.object(target_object_id)
        if scene.id != target_scene.id or target.kind != "mesh":
            raise ValueError("Target must be a mesh in the same scene")
        if object_id == target_object_id or target_object_id in {
            child.id for child in engine.descendants(scene, object_id)
        }:
            raise ValueError("Placement cannot target the moving object or its descendants")
        return source, target

    def align(
        object_id: str,
        target_object_id: str,
        target_face: Face,
        source_face: Face | None = None,
        gap: Finite = 0,
        rotate: bool = True,
        axis: Axis = "z",
        source_side: Literal["negative", "positive"] = "negative",
    ) -> dict[str, Any]:
        source, target = valid_target(object_id, target_object_id)
        target_point, target_normal = face_frame(target, target_face)
        points, anchor, normal = source_data(source, source_face, axis, source_side)
        matrix = alignment(
            points,
            anchor,
            normal,
            target_point,
            target_normal,
            gap,
            rotate,
            source_face is not None,
        )
        return {
            **engine.transform(object_id, matrix),
            "target_face": target_face,
            "target_point": target_point.tolist(),
            "gap": gap,
        }

    def snap(
        object_id: str,
        target_object_id: str,
        point: Vector | None = None,
        source_face: Face | None = None,
        gap: Finite = 0,
        rotate: bool = False,
        axis: Axis = "z",
        source_side: Literal["negative", "positive"] = "negative",
    ) -> dict[str, Any]:
        source, target = valid_target(object_id, target_object_id)
        points, anchor, normal = source_data(source, source_face, axis, source_side)
        target_face, target_point = nearest_face(
            target, np.asarray(point) if point is not None else anchor
        )
        _, target_normal = face_frame(target, target_face)
        matrix = alignment(
            points,
            anchor,
            normal,
            target_point,
            target_normal,
            gap,
            rotate,
            source_face is not None,
        )
        return {
            **engine.transform(object_id, matrix),
            "target_face": target_face,
            "target_point": target_point.tolist(),
            "gap": gap,
        }

    def symmetrize(
        scene_id: str,
        axis: Axis = "x",
        plane: Finite = 0,
        tolerance: Positive = 1e-8,
        object_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        if object_ids is not None:
            unknown = [oid for oid in object_ids if oid not in scene.objects]
            if unknown:
                raise ScissorsError(
                    "OBJECT_NOT_FOUND",
                    "Symmetry selection has unknown objects",
                    {"object_ids": unknown},
                )
        sources = (
            [scene.objects[oid] for oid in object_ids]
            if object_ids is not None
            else list(scene.objects.values())
        )
        if object_ids is not None and len(set(object_ids)) != len(object_ids):
            raise ValueError("Symmetry selection must contain unique IDs")
        matrix = reflection(np.eye(3)["xyz".index(axis)], np.eye(3)["xyz".index(axis)] * plane)

        def signature(obj: MeshObject, reflected: bool = False) -> str:
            points = np.asarray(obj.vertices) if obj.kind == "mesh" else np.asarray([obj.pivot])
            if reflected:
                points = trimesh.transformations.transform_points(points, matrix)
            quantized = np.round(points / tolerance)
            quantized[quantized == 0] = 0.0
            if not np.isfinite(quantized).all():
                raise ValueError("Symmetry tolerance is too small for the coordinate range")
            points = np.unique(quantized, axis=0)
            header = f"{obj.kind}:{len(obj.faces)}:".encode()
            return hashlib.sha256(header + points.tobytes()).hexdigest()

        signatures = {signature(obj) for obj in scene.objects.values() if obj.kind != "group"}
        ids = []
        skipped = []
        for source in sources:
            if source.kind == "group" or signature(source, True) in signatures:
                skipped.append(source.id)
                continue
            obj = engine.clone_object(source.id, f"{source.name} mirrored")
            engine.transform(obj.id, matrix)
            ids.append(obj.id)
            signatures.update(
                signature(child)
                for child in [obj, *engine.descendants(scene, obj.id)]
                if child.kind != "group"
            )
        return {
            "scene_id": scene_id,
            "created_object_ids": ids,
            "skipped_object_ids": skipped,
            "method": "quantized world point-set and triangle-count matching",
            "warnings": ["Center-crossing asymmetric objects can overlap their mirrored copies."],
        }

    def distribute(
        scene_id: str,
        object_id: str,
        count: Annotated[int, Field(ge=1, le=4096)],
        hole_radius: Positive,
        pattern: Literal["fibonacci"] = "fibonacci",
        orient: Literal["normal", "z"] = "normal",
        length: Positive | None = None,
        sections: Annotated[int, Field(ge=3, le=512)] = 32,
        name: str = "Cutter",
    ) -> dict[str, Any]:
        """Create cutter cylinders spread over the target's bounding sphere."""
        scene, target = engine.object(object_id)
        if scene.id != scene_id:
            raise ValueError("Distribution target belongs to another scene")
        if target.kind != "mesh" or not target.vertices:
            raise ValueError("Distribution requires a mesh target")
        vertices = np.asarray(target.vertices, dtype=float)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        center = (low + high) / 2
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        if radius <= 1e-9:
            raise ValueError("Distribution target is degenerate")
        axes = fibonacci_directions(count)
        cutter_length = float(length) if length is not None else 2.4 * radius
        rotation = (
            [trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis) for axis in axes]
            if orient == "normal"
            else [np.eye(4) for _ in axes]
        )
        matrices = [
            trimesh.transformations.translation_matrix(center) @ value for value in rotation
        ]
        rows = [
            {
                "figure": "cylinder",
                "name": name,
                "parameters": {
                    "radius": float(hole_radius),
                    "height": cutter_length,
                    "sections": int(sections),
                },
                "transform": {"matrix": matrix.tolist()},
            }
            for matrix in matrices
        ]
        table = registry.execute("scene.create_table", {"scene_id": scene_id, "rows": rows})
        if not table["success"]:
            raise ScissorsError(
                "BATCH_FAILED", "Cutter distribution rolled back", {"error": table["error"]}
            )
        created = table["result"]["objects"]
        return {
            "scene_id": scene_id,
            "target_object_id": object_id,
            "count": count,
            "pattern": pattern,
            "orient": orient,
            "center": center.tolist(),
            "radius": radius,
            "length": cutter_length,
            "object_ids": [entry["object_id"] for entry in created],
            "matrices": [matrix.tolist() for matrix in matrices],
            "warnings": [
                "Each cutter spans the target centre, so one axis drills two holes."
                if orient == "normal"
                else "Cutters keep the world Z axis; use orient=normal for radial holes."
            ],
        }

    functions: list[tuple[str, Callable[..., Any], str]] = [
        (
            "mesh.array",
            array,
            "Create a linear or polar array including the source and copied descendants.",
        ),
        (
            "mesh.distribute",
            distribute,
            "Create cutter cylinders over a Fibonacci sphere around a target, oriented along "
            "the radius, and return their matrices.",
        ),
        (
            "deform.warp",
            deform,
            "Apply native global bend/twist in degrees and end-scale taper about a world axis.",
        ),
        (
            "mesh.align",
            align,
            "Align a source face or support plane to an opposing target triangle with a gap.",
        ),
        (
            "mesh.snap",
            snap,
            "Snap to the closest target surface point; optional normal alignment and gap.",
        ),
        (
            "scene.symmetrize",
            symmetrize,
            "Mirror unmatched world mesh/marker point sets across an axis plane.",
        ),
    ]
    for name, fn, description in functions:
        registry.add_function(name, fn, description)
