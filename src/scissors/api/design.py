"""Scene design commands registered once for Python, REST and MCP."""

import copy
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from pydantic import Field

from scissors.api.errors import ScissorsError
from scissors.api.registry import OperationRegistry
from scissors.core.engine import Engine, identifier
from scissors.core.export import as_trimesh_scene
from scissors.core.mesh import MeshObject
from scissors.core.validation import validate_mesh

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
IDs = Annotated[list[str], Field(min_length=1, max_length=4096)]


def register_design(registry: OperationRegistry, engine: Engine) -> None:
    def parent(object_id: str, parent_id: str | None = None) -> dict[str, Any]:
        scene, obj = engine.object(object_id)
        if parent_id is not None:
            parent_scene, candidate = engine.object(parent_id)
            if parent_scene.id != scene.id:
                raise ValueError("Parent must belong to the same scene")
            while True:
                if candidate.id == object_id:
                    raise ValueError("Parent would create a hierarchy cycle")
                if candidate.parent_id is None:
                    break
                candidate = scene.objects[candidate.parent_id]
        obj.parent_id = parent_id
        return obj.summary()

    def marker(
        scene_id: str,
        name: str = "Marker",
        position: Vector | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        obj = MeshObject(
            id=identifier("obj"),
            name=name,
            vertices=[],
            faces=[],
            kind="marker",
            pivot=position or [0, 0, 0],
        )
        scene.objects[obj.id] = obj
        return parent(obj.id, parent_id)

    def group(scene_id: str, object_ids: IDs, name: str = "Group") -> dict[str, Any]:
        scene = engine.scene(scene_id)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Object IDs must be unique")
        if any(engine.object(oid)[0].id != scene_id for oid in object_ids):
            raise ValueError("All objects must belong to the group scene")
        selected = set(object_ids)
        roots = [oid for oid in object_ids if scene.objects[oid].parent_id not in selected]
        obj = MeshObject(id=identifier("obj"), name=name, vertices=[], faces=[], kind="group")
        scene.objects[obj.id] = obj
        for oid in roots:
            parent(oid, obj.id)
        return {**obj.summary(), "children": roots}

    def instance(object_id: str, name: str | None = None) -> dict[str, Any]:
        scene, obj = engine.object(object_id)
        if obj.kind != "mesh":
            raise ValueError("Instances require a mesh")
        result = obj.model_copy(deep=True)
        result.id = identifier("obj")
        result.name = name or f"{obj.name} instance"
        result.metadata["instance_source"] = object_id
        scene.objects[result.id] = result
        return result.summary()

    def mesh_rename(object_id: str, name: Annotated[str, Field(min_length=1)]) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        obj.name = name
        return obj.summary()

    def scene_update(
        scene_id: str,
        name: str | None = None,
        units: Literal["meters", "centimeters", "millimeters"] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        if name is not None:
            if not name.strip():
                raise ValueError("Scene name cannot be empty")
            scene.name = name
        if units is not None:
            scene.units = units
        if metadata is not None:
            scene.metadata = copy.deepcopy(metadata)
        return scene.summary()

    def hide(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        obj.hidden = True
        return obj.summary()

    def show(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        obj.hidden = False
        return obj.summary()

    def hidden(scene_id: str) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        return {
            "object_ids": [
                obj.id for obj in scene.objects.values() if not engine.visible(scene, obj)
            ]
        }

    def pivot(object_id: str, position: Vector) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        obj.pivot = list(position)
        obj.metadata["pivot_explicit"] = True
        return obj.summary()

    def frame(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        basis = np.asarray(obj.transform)[:3, :3]
        lengths = np.linalg.norm(basis, axis=0)
        if np.any(lengths < 1e-12):
            raise ValueError("Object transform has a singular local frame")
        return {
            "object_id": object_id,
            "origin": obj.pivot,
            "axes": (basis / lengths).T.tolist(),
            "scale": lengths.tolist(),
        }

    def set_transform(
        object_id: str,
        matrix: Annotated[
            list[Annotated[list[Finite], Field(min_length=4, max_length=4)]],
            Field(min_length=4, max_length=4),
        ],
    ) -> dict[str, Any]:
        target = np.asarray(matrix, dtype=float)
        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not np.allclose(target[3], [0, 0, 0, 1])
            or abs(np.linalg.det(target[:3, :3])) < 1e-12
        ):
            raise ValueError("Transform must be an invertible affine 4x4 matrix")
        obj = engine.object(object_id)[1]
        delta = target @ np.linalg.inv(np.asarray(obj.transform))
        return engine.transform(object_id, delta)

    def place(
        object_id: str,
        position: Vector | None = None,
        rotation: Vector | None = None,
        scale: Annotated[list[Positive], Field(min_length=3, max_length=3)] | None = None,
    ) -> dict[str, Any]:
        angles = np.radians(rotation or [0, 0, 0])
        target = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2], axes="sxyz")
        target[:3, :3] = target[:3, :3] @ np.diag(scale or [1, 1, 1])
        target[:3, 3] = position or [0, 0, 0]
        return set_transform(object_id, target.tolist())

    def planar(
        object_id: str,
        u0: Unit = 0,
        v0: Unit = 0,
        u1: Unit = 1,
        v1: Unit = 1,
        axis: Literal["x", "y", "z"] = "z",
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        if obj.kind != "mesh" or u1 <= u0 or v1 <= v0:
            raise ValueError("Planar UV requires a mesh and a positive rectangle")
        axes = [index for index in range(3) if index != "xyz".index(axis)]
        points = np.asarray(obj.vertices)[:, axes]
        span = np.ptp(points, axis=0)
        if np.any(span <= 1e-12):
            raise ValueError("Projection has zero area; choose another axis")
        uv = (points - points.min(axis=0)) / span
        if flip_u:
            uv[:, 0] = 1 - uv[:, 0]
        if flip_v:
            uv[:, 1] = 1 - uv[:, 1]
        obj.uv = (np.array([u0, v0]) + uv * [u1 - u0, v1 - v0]).tolist()
        return obj.summary()

    def copy_uv(object_id: str, source_object_id: str) -> dict[str, Any]:
        obj, source = engine.object(object_id)[1], engine.object(source_object_id)[1]
        if (
            source.uv is None
            or len(obj.vertices) != len(source.vertices)
            or obj.faces != source.faces
        ):
            raise ValueError("UV copy requires matching indexed topology and source UV")
        obj.uv = copy.deepcopy(source.uv)
        return obj.summary()

    def flip_v(object_id: str) -> dict[str, Any]:
        obj = engine.object(object_id)[1]
        if obj.uv is None:
            raise ValueError("Object has no UV")
        obj.uv = [[u, 1 - v] for u, v in obj.uv]
        return obj.summary()

    def digest(scene_id: str) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        return {
            "scene_id": scene.id,
            "name": scene.name,
            "units": scene.units,
            "object_count": len(scene.objects),
            "material_count": len(scene.materials),
            "vertex_count": sum(len(obj.vertices) for obj in scene.objects.values()),
            "triangle_count": sum(len(obj.faces) for obj in scene.objects.values()),
            "objects": [
                {
                    "id": obj.id,
                    "name": obj.name,
                    "kind": obj.kind,
                    "parent_id": obj.parent_id,
                    "hidden": obj.hidden,
                }
                for obj in scene.objects.values()
            ],
        }

    def report(scene_id: str) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        return {
            **digest(scene_id),
            "validation": [validate_mesh(obj, engine.settings) for obj in scene.objects.values()],
        }

    def query(
        scene_id: str,
        point: Vector | None = None,
        radius: Annotated[float, Field(ge=0)] = 0,
        box: Annotated[list[Vector], Field(min_length=2, max_length=2)] | None = None,
    ) -> dict[str, Any]:
        if (point is None) == (box is None):
            raise ValueError("Provide either point/radius or box")
        if box is not None and np.any(np.asarray(box[0]) > box[1]):
            raise ValueError("Box minimum must not exceed maximum")
        result = []
        for obj in engine.scene(scene_id).objects.values():
            bounds = np.asarray(obj.summary()["bounds"])
            hit = (
                (np.all(bounds[1] >= box[0]) and np.all(bounds[0] <= box[1]))
                if box
                else (
                    np.linalg.norm(
                        np.maximum(bounds[0] - point, 0) + np.maximum(point - bounds[1], 0)
                    )
                    <= radius
                )
            )
            if hit:
                result.append(obj.id)
        return {"object_ids": result, "method": "world axis-aligned bounds"}

    def orphans(scene_id: str) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        used = {obj.material_id for obj in scene.objects.values()}
        texture_ids = {
            tid
            for other in engine.scenes.values()
            for mat in other.materials.values()
            for key, tid in mat.model_dump().items()
            if key.endswith("_texture") and tid
        }
        return {
            "material_ids": [mid for mid in scene.materials if mid not in used],
            "texture_ids": [tid for tid in engine.textures if tid not in texture_ids],
            "texture_scope": "workspace",
        }

    def material_delete(material_id: str) -> dict[str, Any]:
        scene, _ = engine.material(material_id)
        if any(obj.material_id == material_id for obj in scene.objects.values()):
            raise ValueError("Material is still assigned to an object")
        del scene.materials[material_id]
        return {"deleted": material_id}

    def texture_list() -> dict[str, Any]:
        return {"textures": [dict(value, id=tid) for tid, value in engine.textures.items()]}

    def texture_get(texture_id: str) -> dict[str, Any]:
        if texture_id not in engine.textures:
            raise ScissorsError("TEXTURE_NOT_FOUND", "Texture does not exist")
        return dict(engine.textures[texture_id], id=texture_id)

    def texture_delete(texture_id: str) -> dict[str, Any]:
        texture_get(texture_id)
        for scene in engine.scenes.values():
            for mat in scene.materials.values():
                if any(
                    value == texture_id
                    for key, value in mat.model_dump().items()
                    if key.endswith("_texture")
                ):
                    raise ValueError("Texture is still referenced by a material")
        # Remove the managed reference. Retain the file for saved scene snapshots.
        del engine.textures[texture_id]
        return {"deleted": texture_id, "file_retained": True}

    def material_copy(material_id: str, name: str | None = None) -> dict[str, Any]:
        scene, mat = engine.material(material_id)
        duplicate = mat.model_copy(deep=True)
        duplicate.id = identifier("mat")
        duplicate.name = name or f"{mat.name} copy"
        scene.materials[duplicate.id] = duplicate
        return duplicate.model_dump()

    def from_texture(
        scene_id: str, texture_id: str, name: str = "Textured material"
    ) -> dict[str, Any]:
        from scissors.core.materials import Material

        texture_get(texture_id)
        mat = Material(
            id=identifier("mat"), name=name, base_color=[1, 1, 1, 1], base_color_texture=texture_id
        )
        engine.scene(scene_id).materials[mat.id] = mat
        return mat.model_dump()

    def schemas(names: IDs) -> dict[str, Any]:
        return {"operations": [registry.describe_operation(name) for name in names]}

    def edit(operations: Annotated[list[dict[str, Any]], Field(max_length=100)]) -> dict[str, Any]:
        return registry.batch(operations, atomic=True)

    def budget(scene_id: str) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        triangles: dict[str, int] = {}
        tids: set[str] = set()
        draws = 0
        for obj in scene.objects.values():
            if obj.kind != "mesh" or not engine.visible(scene, obj):
                continue
            key = obj.material_id or "unassigned"
            triangles[key] = triangles.get(key, 0) + len(obj.faces)
            draws += 1
            mat = scene.materials.get(obj.material_id or "")
            if mat:
                tids.update(
                    value
                    for key, value in mat.model_dump().items()
                    if key.endswith("_texture") and value
                )
        texture_bytes = sum(
            engine.textures[tid]["width"] * engine.textures[tid]["height"] * 4 for tid in tids
        )
        return {
            "triangles_by_material": triangles,
            "draw_calls_estimate": draws,
            "unique_meshes": len(
                as_trimesh_scene(engine, scene_id, y_up=True, with_materials=False).geometry
            ),
            "texture_bytes_estimate": texture_bytes,
            "method": "one draw per visible mesh; glTF mesh sharing; RGBA8 without mipmaps",
        }

    def intersect_test(
        object_id: str,
        other_object_id: str,
        tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1e-8,
    ) -> dict[str, Any]:
        from scissors.core.inspection import intersection

        if object_id == other_object_id:
            raise ValueError("Provide distinct objects; use mesh.self_intersect for a single mesh")
        scene, first = engine.object(object_id)
        other_scene, second = engine.object(other_object_id)
        if scene.id != other_scene.id:
            raise ValueError("Objects must belong to the same scene")
        return intersection(first, second, engine.settings, tolerance)

    def self_intersect(
        object_id: str,
        tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1e-8,
    ) -> dict[str, Any]:
        from scissors.core.inspection import self_intersections

        return self_intersections(engine.object(object_id)[1], engine.settings, tolerance)

    def collisions(
        scene_id: str,
        object_ids: IDs | None = None,
        tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1e-8,
        anchor_object_ids: IDs | None = None,
    ) -> dict[str, Any]:
        scene = engine.scene(scene_id)
        ids = object_ids or [obj.id for obj in scene.objects.values() if obj.kind == "mesh"]
        if len(set(ids)) != len(ids) or any(oid not in scene.objects for oid in ids):
            raise ValueError("Provide unique objects from the selected scene")
        objects = [scene.objects[oid] for oid in ids]
        if any(obj.kind != "mesh" for obj in objects):
            raise ValueError("Collision selection must contain meshes")
        if anchor_object_ids is not None and (
            not anchor_object_ids
            or len(set(anchor_object_ids)) != len(anchor_object_ids)
            or not set(anchor_object_ids) <= set(ids)
        ):
            raise ValueError("Anchors must be a nonempty unique subset of the selected meshes")
        pairs = len(objects) * (len(objects) - 1) // 2
        if pairs > engine.settings.max_geometry_pairs:
            raise ScissorsError(
                "RESOURCE_LIMIT", "Scene collision query exceeds object pair budget"
            )
        results = []
        triangle_work = 0
        for i, first in enumerate(objects[:-1]):
            a = first.geometry().bounds
            for second in objects[i + 1 :]:
                b = second.geometry().bounds
                if np.any(a[1] + tolerance < b[0]) or np.any(b[1] + tolerance < a[0]):
                    continue
                triangle_work += len(first.faces) * len(second.faces)
                if triangle_work > engine.settings.max_geometry_pairs:
                    raise ScissorsError(
                        "RESOURCE_LIMIT", "Scene collision query exceeds triangle pair budget"
                    )
                result = intersect_test(first.id, second.id, tolerance)
                if result["intersects"]:
                    results.append(result)
        from scissors.core.inspection import contact_components

        return {
            "scene_id": scene_id,
            "collisions": results,
            "pair_count": len(results),
            **contact_components(ids, results, anchor_object_ids),
        }

    entries: list[tuple[str, Callable[..., Any], bool, str]] = [
        (
            "mesh.intersect_test",
            intersect_test,
            False,
            "Measure surface contact and closed-solid overlap.",
        ),
        (
            "scene.distance",
            intersect_test,
            False,
            "Measure minimum surface distance and solid penetration.",
        ),
        (
            "mesh.self_intersect",
            self_intersect,
            False,
            "Find nonadjacent intersecting triangle pairs.",
        ),
        (
            "scene.collisions",
            collisions,
            False,
            "Find contacts/penetrations, connected components and floating parts "
            "relative to anchors.",
        ),
        ("object.parent", parent, True, "Set a same-scene parent; preserve world geometry."),
        ("scene.group", group, True, "Group selected objects under an empty parent."),
        ("scene.marker", marker, True, "Create an empty attachment marker at a world position."),
        ("scene.instance", instance, True, "Copy a mesh; matching export geometry is shared."),
        ("mesh.rename", mesh_rename, True, "Rename an object and its exported node."),
        ("scene.rename", scene_update, True, "Rename a scene; optional units and metadata."),
        ("scene.update", scene_update, True, "Update scene units and metadata without rescaling."),
        ("mesh.hide", hide, True, "Hide an object and descendants in previews and exports."),
        ("mesh.show", show, True, "Clear an object's own hidden flag."),
        ("mesh.hidden", hidden, False, "List effectively hidden objects, including descendants."),
        ("mesh.pivot", pivot, True, "Set the world pivot without moving geometry."),
        ("mesh.set_origin", pivot, True, "Set the world pivot without moving geometry."),
        ("mesh.frame", frame, False, "Read diagnostic local axes, scale and world pivot."),
        (
            "mesh.set_transform",
            set_transform,
            True,
            "Set an absolute affine transform; bake its delta.",
        ),
        ("mesh.place", place, True, "Set absolute position, XYZ Euler degrees and positive scale."),
        (
            "uv.planar",
            planar,
            True,
            "Project UV onto a selected axis and explicit atlas rectangle.",
        ),
        ("uv.copy_from", copy_uv, True, "Copy UV between meshes with matching indexed topology."),
        ("uv.flip_v", flip_v, True, "Reflect UV V around 0.5 without changing geometry."),
        (
            "scene.digest",
            digest,
            False,
            "Read compact counts and object names without mesh arrays.",
        ),
        ("scene.report", report, False, "Read scene counts and per-object validation."),
        (
            "scene.query",
            query,
            False,
            "Find objects by world bounding-box overlap or point radius.",
        ),
        ("scene.orphans", orphans, False, "List scene material and workspace texture orphans."),
        ("material.delete", material_delete, True, "Delete an unassigned material."),
        ("material.copy", material_copy, True, "Copy all PBR properties into a new material."),
        (
            "material.from_texture",
            from_texture,
            True,
            "Create a white material with a base-color texture.",
        ),
        ("texture.list", texture_list, False, "List managed texture metadata."),
        ("texture.get", texture_get, False, "Read managed texture metadata."),
        (
            "texture.delete",
            texture_delete,
            True,
            "Delete an unused texture reference; retain snapshot files.",
        ),
        ("operations.schema", schemas, False, "Read several operation schemas in one call."),
        (
            "scene.edit",
            edit,
            True,
            "Execute up to 100 scene operations with one atomic checkpoint.",
        ),
        (
            "scene.budget",
            budget,
            False,
            "Estimate visible game triangles, draw calls and texture memory.",
        ),
    ]
    for name, fn, write, description in entries:
        registry.add_function(name, fn, description, write)
    from scissors.api.render_sheet import register_render_sheet

    register_render_sheet(registry, engine)
