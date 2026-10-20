"""Native profile-based modeling and discoverable figure families."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject
from scissors.core.profiles import catmull, embed, profile_geometry, validate_curve_path
from scissors.core.solids import orient_planar

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Point = Annotated[list[Finite], Field(min_length=2, max_length=2)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Plane = Literal["xy", "xz", "yz"]
Axis = Literal["x", "y", "z"]
Sides = Annotated[int, Field(ge=3, le=512)]


class LoftRingCorrespondence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ring: Annotated[int, Field(ge=0, le=32)] = 0
    start_index: Annotated[int, Field(ge=0, le=511)] | None = None
    reverse: bool = False


class LoftSectionCorrespondence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    geometry_revision: str
    rings: Annotated[list[LoftRingCorrespondence], Field(min_length=1, max_length=33)]


def register_figures(registry: OperationRegistry, engine: Engine) -> None:
    def budget(vertices: int, faces: int) -> None:
        if (
            vertices > engine.settings.max_vertices
            or faces > engine.settings.max_faces
            or vertices * 384 + faces * 192 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Figure exceeds geometry budget")

    def store(
        scene_id: str,
        mesh: trimesh.Trimesh,
        name: str,
        primitive: str,
        uv: list[list[float]] | None = None,
        metadata: dict[str, Any] | None = None,
        axis: str | None = None,
        pivot: str = "origin",
    ) -> dict[str, Any]:
        if axis is not None or pivot != "origin":
            mesh = orient_planar(mesh, (metadata or {}).get("plane", "xy"), axis, pivot)
        budget(len(mesh.vertices), len(mesh.faces))
        result = engine.add_mesh(scene_id, mesh, name, primitive)
        obj = engine.object(result["object_id"])[1]
        obj.uv = uv
        obj.metadata.update(metadata or {})
        engine.checked(obj)
        return obj.summary()

    def profile(
        scene_id: str,
        points: list[Point],
        holes: list[list[Point]] | None = None,
        plane: Plane = "xy",
        name: str = "Profile",
        axis: Axis | None = None,
        pivot: Literal["origin", "base", "center"] = "origin",
        capped: Literal[True] = True,
    ) -> dict[str, Any]:
        mesh, uv, metadata, warnings = profile_geometry(points, holes or [], plane, engine.settings)
        return {
            **store(scene_id, mesh, name, "profile", uv, metadata, axis, pivot),
            "warnings": warnings,
        }

    def curve(
        scene_id: str,
        points: Annotated[list[Point], Field(min_length=2, max_length=512)],
        closed: bool = False,
        mode: Literal["linear", "smooth"] = "linear",
        width: Positive = 0.01,
        samples: Annotated[int, Field(ge=1, le=256)] = 24,
        plane: Plane = "xy",
        name: str = "Curve",
        axis: Axis | None = None,
        pivot: Literal["origin", "base", "center"] = "origin",
        capped: Literal[True] = True,
    ) -> dict[str, Any]:
        path = np.asarray(points, dtype=float)
        if closed and len(path) < 3:
            raise ScissorsError("GEOMETRY_ERROR", "Closed curves require three points")
        if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) <= 1e-12):
            raise ScissorsError("GEOMETRY_ERROR", "Curve points must be distinct")
        count = len(path) * samples if closed else (len(path) - 1) * samples + 1
        budget(2 * (count if mode == "smooth" else len(path)), 2 * count)
        validate_curve_path(path, closed, engine.settings)
        if mode == "smooth":
            path = catmull(path, samples, closed)
            validate_curve_path(path, closed, engine.settings)
        before = np.roll(path, 1, axis=0)
        after = np.roll(path, -1, axis=0)
        if not closed:
            before[0], after[-1] = path[0], path[-1]
        tangent = after - before
        lengths = np.linalg.norm(tangent, axis=1)
        if np.any(lengths < 1e-12):
            raise ScissorsError("GEOMETRY_ERROR", "Curve reverses direction")
        perpendicular = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / lengths[:, None]
        left, right = path + perpendicular * width / 2, path - perpendicular * width / 2
        vertices = embed(np.stack([left, right], axis=1).reshape(-1, 2), plane)
        faces = []
        for i in range(len(path) if closed else len(path) - 1):
            j = (i + 1) % len(path)
            faces.extend([[2 * i, 2 * i + 1, 2 * j + 1], [2 * i, 2 * j + 1, 2 * j]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if plane == "xz":
            mesh.invert()
        rings = [list(range(0, len(vertices), 2)) + list(range(len(vertices) - 1, 0, -2))]
        if plane == "xz":
            rings = [rings[0][::-1]]
        if closed:
            rings = [list(range(0, len(vertices), 2))[::-1], list(range(1, len(vertices), 2))]
            if plane == "xz":
                rings = [ring[::-1] for ring in rings]
        flat = np.stack([left, right], axis=1).reshape(-1, 2)
        uv = (flat - flat.min(axis=0)) / np.maximum(np.ptp(flat, axis=0), 1e-12)
        return store(
            scene_id,
            mesh,
            name,
            "curve",
            uv.tolist(),
            {
                "plane": plane,
                "boundary_rings": rings,
                "closed": closed,
                "center_pairs": [[2 * i, 2 * i + 1] for i in range(len(path))],
            },
            axis,
            pivot,
        )

    def source(object_id: str, preserve_order: bool = False) -> MeshObject:
        from scissors.core.loft import profile_source

        return profile_source(engine.object(object_id)[1], preserve_order)

    def extrude(
        object_id: str,
        distance: Positive,
        direction: Literal["normal"] | Vector = "normal",
        taper: Positive = 1,
        segments: Annotated[int, Field(ge=1, le=256)] = 1,
        caps: bool = True,
        replace: bool = False,
        name: str | None = None,
    ) -> dict[str, Any]:
        obj = source(object_id)
        scene = engine.object(object_id)[0]
        points = np.asarray(obj.vertices)
        normal = np.asarray(obj.geometry().face_normals).mean(axis=0)
        vector = normal if isinstance(direction, str) else np.asarray(direction, dtype=float)
        length = np.linalg.norm(vector)
        if length < 1e-12 or abs(np.dot(vector / length, normal)) < 1e-10:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Extrusion direction must leave the profile plane"
            )
        vector = vector / length * distance
        n = len(points)
        edges = sum(len(ring) for ring in obj.metadata["boundary_rings"])
        budget(
            3 * (2 * edges * segments + (2 * len(obj.faces) if caps else 0)),
            2 * edges * segments + (2 * len(obj.faces) if caps else 0),
        )
        center = points.mean(axis=0)
        vertices = np.concatenate(
            [
                center + (points - center) * (1 + (taper - 1) * t) + vector * t
                for t in np.linspace(0, 1, segments + 1)
            ]
        )
        faces: list[list[int]] = []
        uv_faces: list[list[list[float]]] = []
        for ring in obj.metadata["boundary_rings"]:
            lengths = np.linalg.norm(np.roll(points[ring], -1, axis=0) - points[ring], axis=1)
            perimeter = np.concatenate([[0], np.cumsum(lengths)]) / lengths.sum()
            for layer in range(segments):
                v0, v1 = layer / segments, (layer + 1) / segments
                for i, a in enumerate(ring):
                    b = ring[(i + 1) % len(ring)]
                    faces.extend(
                        [
                            [a + layer * n, b + layer * n, b + (layer + 1) * n],
                            [a + layer * n, b + (layer + 1) * n, a + (layer + 1) * n],
                        ]
                    )
                    u0, u1 = float(perimeter[i]), float(perimeter[i + 1])
                    uv_faces.extend(
                        [[[u0, v0], [u1, v0], [u1, v1]], [[u0, v0], [u1, v1], [u0, v1]]]
                    )
        if caps:
            for face in obj.faces:
                faces.extend([face[::-1], [index + n * segments for index in face]])
                coords = np.asarray(obj.uv)[face] if obj.uv is not None else np.zeros((3, 2))
                uv_faces.extend([coords[::-1].tolist(), coords.tolist()])
        array = np.asarray(faces)
        if np.dot(vector, normal) < 0:
            array = array[:, ::-1]
            uv_faces = [values[::-1] for values in uv_faces]
        # Split at UV seams. Topology checks weld exact positions.
        render_vertices = vertices[array].reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=render_vertices,
            faces=np.arange(len(render_vertices)).reshape(-1, 3),
            process=False,
        )
        uv = np.asarray(uv_faces).reshape(-1, 2).tolist()
        if replace:
            obj.replace(mesh, uv)
            obj.metadata = {"primitive": "extrusion", "source_primitive": obj.metadata["primitive"]}
            obj.name = name or obj.name
            engine.checked(obj)
            return obj.summary()
        result = store(scene.id, mesh, name or f"{obj.name} extrusion", "extrusion", uv)
        engine.object(result["object_id"])[1].material_id = obj.material_id
        return engine.object(result["object_id"])[1].summary()

    def loft(
        object_ids: Annotated[list[str], Field(min_length=2, max_length=64)],
        resample: Annotated[int, Field(ge=0, le=512)] = 0,
        smooth: bool = False,
        caps: bool = True,
        closed: bool = False,
        name: str = "Loft",
        preserve_vertices: bool = False,
        correspondence: Annotated[list[LoftSectionCorrespondence], Field(max_length=64)]
        | None = None,
    ) -> dict[str, Any]:
        from scissors.core.profiles import resample as resample_ring

        if resample in {1, 2} or len(set(object_ids)) != len(object_ids):
            raise ValueError("Loft requires unique objects and resample=0 or at least 3")
        if preserve_vertices and resample:
            raise ScissorsError("INVALID_PARAMETER", "preserve_vertices requires resample=0")
        if correspondence is not None and not preserve_vertices:
            raise ScissorsError(
                "INVALID_PARAMETER", "Correspondence requires preserve_vertices=true"
            )
        if preserve_vertices and closed and len(object_ids) < 3:
            raise ScissorsError(
                "INVALID_PARAMETER", "Closed loft spine requires at least three sections"
            )
        objects = [source(oid, preserve_vertices) for oid in object_ids]
        scene_id = engine.object(object_ids[0])[0].id
        if any(engine.object(oid)[0].id != scene_id for oid in object_ids):
            raise ValueError("Loft inputs must belong to the same scene")
        curves = [obj.metadata["primitive"] == "curve" for obj in objects]
        if any(curves) and not all(curves):
            raise ValueError("Loft inputs must all be profiles or all be curves")
        details: dict[str, Any] = {}
        if preserve_vertices:
            from scissors.core.loft import prepare_preserved

            arrays, counts, sections_closed, details = prepare_preserved(
                objects,
                [
                    LoftSectionCorrespondence.model_validate(item).model_dump()
                    for item in correspondence or []
                ],
                engine.settings,
                spine_closed=closed,
            )
        else:
            if all(curves):
                paths = [
                    np.asarray(obj.vertices)[obj.metadata["center_pairs"]].mean(axis=1)
                    for obj in objects
                ]
                sections_closed = all(obj.metadata["closed"] for obj in objects)
                if any(obj.metadata["closed"] != sections_closed for obj in objects):
                    raise ValueError("Curve sections must agree on closed status")
                rings_per_section = [[path] for path in paths]
            else:
                count = len(objects[0].metadata["boundary_rings"])
                if any(len(obj.metadata["boundary_rings"]) != count for obj in objects):
                    raise ValueError("Loft profiles must have matching hole counts")
                rings_per_section = [
                    [np.asarray(obj.vertices)[ring] for ring in obj.metadata["boundary_rings"]]
                    for obj in objects
                ]
                sections_closed = True
            counts = [
                resample or max(len(section[i]) for section in rings_per_section)
                for i in range(len(rings_per_section[0]))
            ]
            prepared = []
            for section in rings_per_section:
                prepared.append(
                    [
                        resample_ring(ring, count, sections_closed)
                        for ring, count in zip(section, counts, strict=True)
                    ]
                )
            for ring_index in range(len(counts)):
                reference = prepared[0][ring_index]
                for section in prepared[1:]:
                    ring = section[ring_index]
                    if sections_closed:
                        # Match orientation by normal, then choose the nearest start.
                        normal = np.cross(reference[1] - reference[0], reference[2] - reference[0])
                        other_normal = np.cross(ring[1] - ring[0], ring[2] - ring[0])
                        if np.dot(normal, other_normal) < 0:
                            ring = ring[::-1]
                        start = int(np.argmin(np.linalg.norm(ring - reference[0], axis=1)))
                        section[ring_index] = np.roll(ring, -start, axis=0)
            arrays = np.asarray([np.concatenate(section) for section in prepared])
        from scissors.core.loft import build_mesh

        mesh = build_mesh(
            arrays,
            counts,
            objects,
            sections_closed,
            smooth,
            caps,
            closed,
            engine.settings,
            preserve_vertices=preserve_vertices,
        )
        if preserve_vertices:
            from scissors.core.loft import preserved_metadata

            preserved_metadata(
                mesh, counts, sections_closed, caps, closed, smooth, details, engine.settings
            )
        result = store(
            scene_id, mesh, name, "loft", metadata={"loft": details} if preserve_vertices else None
        )
        from scissors.core.uv import generate_uv

        obj = engine.object(result["object_id"])[1]
        generate_uv(obj, "box", engine.settings)
        engine.checked(obj)
        if preserve_vertices:
            details["geometry_revision"] = obj.geometry_revision
            obj.metadata["loft"] = details
            return {**obj.summary(), "loft": details}
        return obj.summary()

    def facet(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh":
            raise ValueError("Facet requires a mesh")
        budget(3 * len(obj.faces), len(obj.faces))
        indices = np.asarray(obj.faces).reshape(-1)
        uv = np.asarray(obj.uv)[indices].tolist() if obj.uv is not None else None
        obj.replace(
            trimesh.Trimesh(
                vertices=np.asarray(obj.vertices)[indices],
                faces=np.arange(len(indices)).reshape(-1, 3),
                process=False,
            ),
            uv,
        )
        obj.metadata.pop("boundary_rings", None)
        engine.checked(obj)
        return obj.summary()

    def orient(mesh: trimesh.Trimesh, axis: str, pivot: str) -> trimesh.Trimesh:
        if pivot == "center":
            mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
        else:
            mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
        if axis != "z":
            matrix = trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)["xyz".index(axis)])
            mesh.apply_transform(matrix)
        return mesh

    def prism(
        scene_id: str,
        sides: Sides = 6,
        radius: Positive = 1,
        height: Positive = 1,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Literal["base", "center"] = "base",
        name: str = "Prism",
    ) -> dict[str, Any]:
        budget(6 * sides + (6 * sides if capped else 0), 4 * sides)
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sides)
        if not capped:
            mesh.update_faces(np.abs(mesh.face_normals[:, 2]) < 0.9)
            mesh.remove_unreferenced_vertices()
        result = store(scene_id, orient(mesh, axis, pivot), name, "prism")
        return facet(result["object_id"])

    def pyramid(
        scene_id: str,
        sides: Sides = 4,
        radius: Positive = 1,
        height: Positive = 1,
        capped: bool = True,
        axis: Axis = "z",
        pivot: Literal["base", "center"] = "base",
        name: str = "Pyramid",
    ) -> dict[str, Any]:
        budget(6 * sides, 2 * sides)
        mesh = trimesh.creation.cone(radius=radius, height=height, sections=sides)
        if not capped:
            mesh.update_faces(mesh.face_normals[:, 2] > -0.9)
            mesh.remove_unreferenced_vertices()
        result = store(scene_id, orient(mesh, axis, pivot), name, "pyramid")
        return facet(result["object_id"])

    def polygon(
        scene_id: str,
        points: list[Point],
        thickness: Annotated[float, Field(ge=0)] = 0,
        holes: list[list[Point]] | None = None,
        plane: Plane = "xy",
        name: str = "Polygon",
        capped: bool = True,
        axis: Axis | None = None,
        pivot: Literal["origin", "base", "center"] = "origin",
    ) -> dict[str, Any]:
        if not capped and thickness == 0:
            raise ValueError("A planar polygon has no extrusion caps to remove")
        result = profile(scene_id, points, holes, plane, name)
        if thickness > 0:
            result = extrude(result["object_id"], thickness, caps=capped, replace=True, name=name)
            engine.object(result["object_id"])[1].metadata["primitive"] = "polygon"
        if axis is not None or pivot != "origin":
            obj = engine.object(result["object_id"])[1]
            obj.replace(orient_planar(obj.geometry(), plane, axis, pivot), obj.uv)
            engine.checked(obj)
            result = {**result, **obj.summary()}
        return result

    def star(
        scene_id: str,
        points: Annotated[int, Field(ge=3, le=128)] = 5,
        outer_radius: Positive = 1,
        inner_radius: Positive = 0.5,
        thickness: Annotated[float, Field(ge=0)] = 0,
        name: str = "Star",
        kind: Literal["star", "cross", "arrow"] = "star",
        capped: bool = True,
        axis: Axis | None = None,
        pivot: Literal["origin", "base", "center"] = "origin",
    ) -> dict[str, Any]:
        if inner_radius >= outer_radius:
            raise ValueError("Inner radius must be smaller than outer radius")
        if kind == "cross":
            r, s = outer_radius, inner_radius
            contour = [
                [r, s],
                [s, s],
                [s, r],
                [-s, r],
                [-s, s],
                [-r, s],
                [-r, -s],
                [-s, -s],
                [-s, -r],
                [s, -r],
                [s, -s],
                [r, -s],
            ]
            return polygon(
                scene_id, contour, thickness, name=name, capped=capped, axis=axis, pivot=pivot
            )
        if kind == "arrow":
            r, s = outer_radius, inner_radius
            contour = [[-r, -s], [0, -s], [0, -r], [r, 0], [0, r], [0, s], [-r, s]]
            return polygon(
                scene_id, contour, thickness, name=name, capped=capped, axis=axis, pivot=pivot
            )
        angles = np.arange(2 * points) * np.pi / points
        radius = np.where(np.arange(2 * points) % 2 == 0, outer_radius, inner_radius)
        return polygon(
            scene_id,
            np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]).tolist(),
            thickness,
            name=name,
            capped=capped,
            axis=axis,
            pivot=pivot,
        )

    functions: dict[str, Callable[..., Any]] = {
        "create_profile": profile,
        "create_curve": curve,
        "extrude": extrude,
        "loft": loft,
        "facet": facet,
        "create_prism": prism,
        "create_pyramid": pyramid,
        "create_polygon": polygon,
        "create_star": star,
    }
    descriptions = {
        "create_profile": "Create a validated planar polygon with holes and planar UV.",
        "create_curve": "Create a linear or Catmull-Rom planar ribbon with nonzero width.",
        "extrude": "Extrude a profile or ribbon with holes, taper, rings, caps and perimeter UV.",
        "loft": "Loft profiles or curves with resampling or preserved-vertex correspondence.",
        "facet": "Split render vertices per triangle for flat normals; retain UV and material.",
        "create_prism": "Create an n-sided prism with flat faces and optional caps.",
        "create_pyramid": "Create an n-sided pyramid with flat faces and optional base.",
        "create_polygon": "Create a filled planar polygon, optionally extruded with holes.",
        "create_star": "Create a star, cross or +X arrow with optional thickness.",
    }
    for name, fn in functions.items():
        registry.add_function(f"mesh.{name}", fn, descriptions[name])

    def figures() -> dict[str, Any]:
        families = {
            "cube": "solid",
            "sphere": "rounded",
            "cylinder": "revolution",
            "cone": "revolution",
            "plane": "planar",
            "torus": "revolution",
            "prism": "prism",
            "pyramid": "pyramid",
            "polygon": "profile",
            "star": "profile",
            "profile": "profile",
            "curve": "curve",
            "rounded_box": "rounded",
            "tube": "tube",
            "louver": "plate",
            "switch_grid": "plate",
            "capsule": "rounded",
            "hemisphere": "rounded",
            "dome": "rounded",
            "polyhedron": "polyhedron",
            "plate": "plate",
            "wedge": "prism",
            "gear": "profile",
            "spring": "tube",
            "text": "text",
        }
        variants = {
            "star": ["star", "cross", "arrow"],
            "polyhedron": ["tetrahedron", "octahedron", "dodecahedron", "icosahedron"],
            "torus": ["full", "arc", "knot"],
        }
        return {
            "figures": [
                {
                    "kind": kind,
                    "family": family,
                    "operation": f"mesh.create_{kind}",
                    "variants": variants.get(kind, [kind]),
                    "common_options": {
                        parameter: schema
                        for parameter, schema in registry.operations[f"mesh.create_{kind}"]
                        .parameters["properties"]
                        .items()
                        if parameter in {"sides", "segments", "capped", "axis", "pivot", "name"}
                    },
                }
                for kind, family in families.items()
                if f"mesh.create_{kind}" in registry.operations
            ]
        }

    registry.add_function(
        "mesh.figures", figures, "Discover available native figure families.", False
    )
