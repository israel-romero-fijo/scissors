"""FBX bind-pose adaptation and retained source data in the geometry child."""

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh
from PIL import Image


def matrix4(value: Any) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3] = np.asarray(value, dtype=float).T
    return matrix


def triangle_corners(face_records: Any, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate polygons without losing their original attribute corner indices."""
    faces = np.asarray(face_records, dtype=np.int64).reshape(-1, 2)
    if len(faces) and np.all(faces[:, 1] == 3):
        return faces[:, :1] + np.arange(3), np.arange(len(faces))
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkCommonDataModel import vtkPolygon

    triangles, owners = [], []
    for owner, (start, count) in enumerate(faces):
        if count < 3:
            continue
        if count == 3:
            triangles.append([start, start + 1, start + 2])
            owners.append(owner)
            continue
        if count == 4:
            points = positions[start : start + count]
            choices = [np.array([[0, 1, 2], [0, 2, 3]]), np.array([[0, 1, 3], [1, 2, 3]])]
            scores = []
            for choice in choices:
                corners = points[choice]
                normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
                lengths = np.linalg.norm(normal, axis=1)
                aligned = float(normal[0] @ normal[1]) >= 0
                scores.append(float(lengths.min()) if aligned else -float(lengths.max()))
            choice = choices[int(np.argmax(scores))]
            triangles.extend((choice + start).tolist())
            owners.extend([owner, owner])
            continue
        polygon = vtkPolygon()
        polygon.GetPoints().SetDataTypeToDouble()
        polygon.GetPoints().SetNumberOfPoints(int(count))
        polygon.GetPointIds().SetNumberOfIds(int(count))
        # Authoring tools commonly store curved surfaces as nonplanar quads.
        # Triangulate a best-fit plane while keeping the original render corners.
        points = positions[start : start + count]
        centered = points - points.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        projected = np.column_stack((centered @ basis[:2].T, np.zeros(count)))
        for index, point in enumerate(projected):
            polygon.GetPoints().SetPoint(index, point)
            polygon.GetPointIds().SetId(index, index)
        indices = vtkIdList()
        if (
            not polygon.TriangulateLocalIds(0, indices)
            or indices.GetNumberOfIds() != (count - 2) * 3
        ):
            raise ValueError("FBX polygon could not be triangulated")
        local = np.array([indices.GetId(i) for i in range(indices.GetNumberOfIds())])
        triangles.extend((local.reshape(-1, 3) + start).tolist())
        owners.extend([owner] * (count - 2))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3), np.asarray(owners, dtype=np.int64)


class FbxMaterials:
    def __init__(self, resolver: Any, limits: dict[str, Any], diagnostics: list[str]):
        self.resolver, self.limits, self.diagnostics = resolver, limits, diagnostics
        self.materials: dict[int, Any] = {}
        self.images: dict[int, Image.Image] = {}
        self.image_bytes = 0
        self.references: list[Any] = []

    def native(self, value: Any, attribute: str) -> Any:
        result = getattr(value, attribute)
        self.references.append(result)
        return result

    def image(self, texture: Any, uv_name: str) -> Image.Image | None:
        if texture is None:
            return None
        import ufbx

        if texture.type != ufbx.TextureType.FILE:
            raise ValueError("FBX layered, procedural and shader textures must be baked to images")
        if texture.has_uv_transform or (texture.uv_set and texture.uv_set != uv_name):
            raise ValueError(
                "FBX texture transforms and additional UV sets must be baked before import"
            )
        key = int(texture.typed_id)
        if key in self.images:
            return self.images[key]
        # Embedded images take precedence over stale absolute paths in portable FBX files.
        content = texture.content
        video = self.native(texture, "video")
        if not content and video is not None:
            content = video.content
        if not content:
            filename = texture.relative_filename or texture.filename
            if not filename:
                raise ValueError("FBX texture has no embedded content or relative filename")
            content = self.resolver.get(filename)
        result = self.decode_image(content)
        self.images[key] = result
        return result

    def decode_image(self, content: bytes) -> Image.Image:
        if len(content) > self.limits["max_file_bytes"]:
            raise ValueError("FBX image exceeds encoded byte limit")
        with Image.open(io.BytesIO(content)) as image:
            self.reserve_image(image.width, image.height)
            return image.convert("RGBA")

    def reserve_image(self, width: int, height: int) -> None:
        if width * height > self.limits["max_texture_pixels"]:
            raise ValueError("FBX image exceeds texture pixel limit")
        self.image_bytes += width * height * 4
        if self.image_bytes > self.limits["max_memory_bytes"]:
            raise ValueError("FBX image memory exceeds budget")

    def material(self, source: Any, uv_name: str) -> Any:
        key = int(source.typed_id)
        pbr = self.native(source, "pbr")
        slots = {
            name: self.native(pbr, name)
            for name in (
                "base_color",
                "base_factor",
                "metalness",
                "roughness",
                "opacity",
                "emission_color",
                "emission_factor",
                "normal_map",
                "ambient_occlusion",
            )
        }
        props = self.native(source, "props")
        features = self.native(source, "features")
        if key in self.materials:
            # Validate UV set compatibility even for an already cached material.
            for slot in (
                slots["base_color"],
                slots["normal_map"],
                slots["roughness"],
                slots["metalness"],
                slots["emission_color"],
                slots["ambient_occlusion"],
            ):
                self.image(self.native(slot, "texture"), uv_name)
            return self.materials[key]

        def scalar(slot: Any, default: float) -> float:
            return float(slot.value_vec4[0]) if slot.has_value else default

        def color(slot: Any, default: list[float]) -> np.ndarray:
            return np.asarray(slot.value_vec4[:3] if slot.has_value else default, dtype=float)

        base = color(slots["base_color"], [1, 1, 1]) * scalar(slots["base_factor"], 1)
        opacity = scalar(slots["opacity"], 1)
        emissive = color(slots["emission_color"], [0, 0, 0]) * scalar(slots["emission_factor"], 1)
        metallic = props.find_real("ScissorsMetallic", scalar(slots["metalness"], 0))
        roughness = props.find_real("ScissorsRoughness", scalar(slots["roughness"], 0.7))
        images = {
            name: self.image(self.native(slot, "texture"), uv_name)
            for name, slot in slots.items()
            if name
            in {
                "base_color",
                "normal_map",
                "emission_color",
                "ambient_occlusion",
                "roughness",
                "metalness",
            }
        }
        base_image, normal_image, emissive_image = (
            images["base_color"],
            images["normal_map"],
            images["emission_color"],
        )
        occlusion, roughness_image, metallic_image = (
            images["ambient_occlusion"],
            images["roughness"],
            images["metalness"],
        )
        orm = None
        custom_orm = props.find_prop("ScissorsOrmImage")
        if custom_orm is not None and custom_orm.value_str:
            orm = self.decode_image(self.resolver.get(custom_orm.value_str))
        elif roughness_image is not None or metallic_image is not None:
            orm_images = [im for im in (roughness_image, metallic_image) if im is not None]
            width, height = max(im.width for im in orm_images), max(im.height for im in orm_images)
            self.reserve_image(width, height)

            def channel(image: Image.Image | None) -> Image.Image:
                if image is None:
                    return Image.new("L", (width, height), 255)
                return image.getchannel("R").resize((width, height), Image.Resampling.BILINEAR)

            orm = Image.merge(
                "RGB",
                (
                    Image.new("L", (width, height), 255),
                    channel(roughness_image),
                    channel(metallic_image),
                ),
            )
        custom_occlusion = props.find_prop("ScissorsOcclusionImage")
        if custom_occlusion is not None and custom_occlusion.value_str:
            occlusion = self.decode_image(self.resolver.get(custom_occlusion.value_str))
        if not self.native(features, "pbr").enabled:
            self.diagnostics.append("FBX Phong/Lambert materials are approximated as PBR.")
        if self.native(slots["opacity"], "texture") is not None:
            self.diagnostics.append(
                "FBX separate opacity images are omitted; base image alpha is retained."
            )
        bindings = list(self.native(source, "textures"))
        self.references.extend(bindings)
        if any(self.native(binding, "texture").typed_id not in self.images for binding in bindings):
            self.diagnostics.append("FBX additional material texture channels are omitted.")
        material = trimesh.visual.material.PBRMaterial(
            name=source.name or "FBX material",
            baseColorFactor=[*np.clip(base, 0, 1), np.clip(opacity, 0, 1)],
            metallicFactor=float(np.clip(metallic, 0, 1)),
            roughnessFactor=float(np.clip(roughness, 0, 1)),
            emissiveFactor=emissive,
            baseColorTexture=base_image,
            normalTexture=normal_image,
            emissiveTexture=emissive_image,
            metallicRoughnessTexture=orm,
            occlusionTexture=occlusion,
            doubleSided=props.find_bool(
                "ScissorsDoubleSided", self.native(features, "double_sided").enabled
            ),
            alphaMode="BLEND"
            if opacity < 1 or (base_image and base_image.getextrema()[3][0] < 255)
            else "OPAQUE",
        )
        self.materials[key] = material
        return material


def load_fbx(
    source: Path, resolver: Any, limits: dict[str, Any]
) -> tuple[trimesh.Scene, list[str], dict[str, Any]]:
    import ufbx

    override = limits.get("source_up", "auto")
    options: dict[str, Any] = {
        "load_external_files": False,
        "ignore_animation": limits.get("fbx_mode", "preserve") == "static",
        "retain_dom": True,
        "generate_missing_normals": True,
        "target_unit_meters": 1.0,
        "node_depth_limit": 256,
        "file_format": ufbx.FileFormat.FBX,
        "index_error_handling": ufbx.IndexErrorHandling.ABORT_LOADING,
    }
    if override == "auto":
        options["target_axes"] = ufbx.axes_right_handed_z_up
    scene = ufbx.load_memory(source.read_bytes(), **options)
    # Keep native element wrappers alive throughout conversion. The binding caches
    # wrappers shared by parent/instance/material lookups while traversing the scene.
    elements = (list(scene.nodes), list(scene.meshes), list(scene.materials), list(scene.textures))
    nodes = elements[0]
    preserve = limits.get("fbx_mode", "preserve") == "preserve" and bool(
        len(scene.skin_deformers) or len(scene.anim_stacks) or len(scene.blend_deformers)
    )
    settings = scene.settings
    references: list[Any] = []

    def retain(value: Any) -> Any:
        references.append(value)
        return value

    if len(nodes) > limits["max_geometry_pairs"]:
        raise ValueError("FBX hierarchy exceeds node budget")
    diagnostics = ["FBX preview shows the static bind pose; animation playback is not supported."]
    document = None
    if preserve:
        from scissors.core.fbx_document import retain_document

        document = retain_document(
            source, resolver, SimpleNamespace(**limits, operation_timeout=300)
        )
        diagnostics.append("Original rig and animation are retained for complete FBX export.")
    else:
        diagnostics.append("Static FBX import does not retain rig or animation data.")
    materials = FbxMaterials(resolver, limits, diagnostics)
    result = trimesh.Scene(base_frame="fbx")
    names: dict[int, str] = {}
    used_names = {"fbx"}
    for node in nodes:
        name = node.name or "FBX root"
        while name in used_names:
            name += f"_{node.typed_id}"
        used_names.add(name)
        names[node.typed_id] = name
    if override == "auto" or override == "z":
        correction = np.eye(4)
    elif override == "y":
        correction = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    else:
        correction = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
    cache: dict[tuple, str] = {}
    vertices_total = faces_total = 0
    for node in nodes:
        world = matrix4(node.node_to_world)
        parent_world = matrix4(node.parent.node_to_world) if node.parent else np.eye(4)
        local = np.linalg.inv(parent_world) @ world if node.parent else correction @ world
        result.graph.update(
            frame_to=names[node.typed_id],
            frame_from=names[node.parent.typed_id] if node.parent else "fbx",
            matrix=local,
        )
        mesh = node.mesh
        if mesh is None:
            continue
        node_materials = retain(list(retain(node.materials)))
        key = (mesh.typed_id, tuple(mat.typed_id for mat in node_materials))
        geometry = cache.get(key)
        if geometry is None:
            geometry = f"FBX mesh {mesh.typed_id} binding {key[1]}"
            if (
                mesh.num_vertices > limits["max_vertices"]
                or mesh.num_triangles > limits["max_faces"]
            ):
                raise ValueError("FBX geometry exceeds vertex/face limits")
            if mesh.num_indices * 128 + mesh.num_triangles * 192 > limits["max_memory_bytes"]:
                raise ValueError("FBX working geometry exceeds memory budget")
            if any(retain(mesh.face_hole)):
                raise ValueError("FBX polygon holes must be triangulated before import")
            position_attribute = retain(mesh.vertex_position)
            normal_attribute = retain(mesh.vertex_normal)
            uv_attribute, color_attribute = retain(mesh.vertex_uv), retain(mesh.vertex_color)
            attributes = [position_attribute, normal_attribute]
            has_uv, has_color = uv_attribute.exists, color_attribute.exists
            if has_uv:
                attributes.append(uv_attribute)
            if has_color:
                attributes.append(color_attribute)
            indices = np.column_stack(
                [np.asarray(retain(a.indices), dtype=np.int64) for a in attributes]
            )
            keys, inverse = np.unique(indices, axis=0, return_inverse=True)
            values = [retain(a.values) for a in attributes]
            native_arrays = [np.asarray(v, dtype=float) for v in values]
            if any(not np.isfinite(array).all() for array in native_arrays):
                raise ValueError("FBX mesh attributes must be finite")
            arrays = [v[keys[:, i]] for i, v in enumerate(native_arrays)]
            positions = native_arrays[0][indices[:, 0]]
            corners, owners = triangle_corners(retain(mesh.faces), positions)
            faces = inverse[corners]
            vertices_total += len(keys)
            faces_total += len(faces)
            if vertices_total > limits["max_vertices"] or faces_total > limits["max_faces"]:
                raise ValueError("FBX render geometry exceeds vertex/face limits")
            if (
                vertices_total * 384 + faces_total * 192 + materials.image_bytes
                > limits["max_memory_bytes"]
            ):
                raise ValueError("FBX geometry and images exceed memory budget")
            uv = arrays[2] if has_uv else None
            colors = (
                np.rint(np.clip(arrays[-1], 0, 1) * 255).astype(np.uint8) if has_color else None
            )
            uv_sets = retain(mesh.uv_sets)
            uv_name = retain(uv_sets[0]).name if len(uv_sets) else ""
            face_materials = np.asarray(retain(mesh.face_material), dtype=np.int64)
            if not len(face_materials):
                face_materials = np.zeros(mesh.num_faces, dtype=np.int64)
            face_material = face_materials[owners]
            for material_index in np.unique(face_material):
                chosen = np.flatnonzero(face_material == material_index)
                used, remap = np.unique(faces[chosen], return_inverse=True)
                part = trimesh.Trimesh(
                    vertices=arrays[0][used],
                    faces=remap.reshape(-1, 3),
                    vertex_normals=arrays[1][used],
                    process=False,
                )
                if preserve:
                    dom = retain(mesh.dom_node)
                    values_dom = retain(dom.values)
                    part.metadata["fbx_binding"] = {
                        "geometry_id": int(retain(values_dom[0]).value_int),
                        "control_points": keys[used, 0].tolist(),
                    }
                mat = (
                    materials.material(node_materials[int(material_index)], uv_name)
                    if len(node_materials)
                    else None
                )
                if uv is not None or mat is not None:
                    part.visual = trimesh.visual.texture.TextureVisuals(
                        uv=uv[used] if uv is not None else None, material=mat
                    )
                    if colors is not None:
                        part.visual.vertex_attributes["color"] = colors[used]
                elif colors is not None:
                    part.visual = trimesh.visual.color.ColorVisuals(
                        mesh=part, vertex_colors=colors[used]
                    )
                name = f"{geometry} material {material_index}"
                result.geometry[name] = part
            # Cache the material parts rather than merging away their UV/normal boundaries.
            cache[key] = geometry
        face_materials = np.asarray(retain(mesh.face_material), dtype=np.int64)
        for material_index in np.unique(face_materials) if len(face_materials) else [0]:
            part_name = f"{geometry} material {material_index}"
            helper = f"{names[node.typed_id]} geometry {material_index}"
            while helper in used_names:
                helper += "_"
            used_names.add(helper)
            result.graph.update(
                frame_to=helper,
                frame_from=names[node.typed_id],
                matrix=matrix4(node.geometry_to_node),
                geometry=part_name,
            )
    up = int(settings.axes.up) // 2
    return (
        result,
        diagnostics,
        {
            "coordinates_normalized": True,
            "source_up": "xyz"[up] if override == "auto" else override,
            "source_unit_meters": float(settings.unit_meters),
            **({"fbx_document": document} if document is not None else {}),
        },
    )
