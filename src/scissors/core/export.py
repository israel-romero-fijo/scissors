import io
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import trimesh
from PIL import Image, ImageOps

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.persistence import atomic_write
from scissors.core.scene import Scene
from scissors.core.tangents import gltf_tangents, tangent_frames

# glTF 2.0 defines +Y as up, so scene +Z must become glTF +Y before export.
# Rotating -90 degrees around X maps (x, y, z) to (x, z, -y) and keeps handedness.
Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0])


def artifact(path: Path, format: str, diagnostics: list[str] | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "format": format,
        "size_bytes": path.stat().st_size,
        "diagnostics": diagnostics or [],
    }


def node_names(objects: Any) -> dict[str, str]:
    """Give exported nodes readable, unique names so engine hierarchies stay usable."""
    # Trimesh uses this frame name for the scene root. Reusing it creates a cycle.
    taken: set[str] = {"world"}
    names: dict[str, str] = {}
    for obj in objects:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", obj.name).strip("_") or "Object"
        candidate, suffix = base, 2
        while candidate in taken:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        taken.add(candidate)
        names[obj.id] = candidate
    return names


def affine_mesh(mesh: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    """Apply positions once and preserve shading with the inverse normal matrix."""
    vertex_normals, face_normals = (
        np.asarray(mesh.vertex_normals).copy(),
        np.asarray(mesh.face_normals).copy(),
    )
    mesh.apply_transform(matrix)
    inverse = np.linalg.inv(matrix[:3, :3])
    vertex_normals, face_normals = vertex_normals @ inverse, face_normals @ inverse
    new_vertex_normals = vertex_normals / np.maximum(
        np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-15
    )
    mesh.vertex_normals = new_vertex_normals  # type: ignore[method-assign]
    mesh.face_normals = face_normals / np.maximum(
        np.linalg.norm(face_normals, axis=1)[:, None], 1e-15
    )
    return mesh


def baked_meshes(scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh]]:
    result = []
    for node in scene.graph.nodes_geometry:
        matrix, key = scene.graph.get(node)
        mesh = cast(trimesh.Trimesh, scene.geometry[key]).copy()
        result.append((node, affine_mesh(mesh, matrix)))
    return result


def ply_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    normals: list[list[float]] = []
    colors: list[list[int]] = []
    uv: list[list[float]] = []
    has_uv = False
    for _, mesh in baked_meshes(scene):
        start = len(vertices)
        vertices.extend(mesh.vertices.tolist())
        faces.extend((np.asarray(mesh.faces) + start).tolist())
        normals.extend(mesh.vertex_normals.tolist())
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            tint = getattr(mesh.visual.material, "baseColorFactor", None)
            factor = np.asarray(tint, dtype=float) / 255 if tint is not None else np.ones(4)
            color = mesh.visual.vertex_attributes.get("color")
            color = (
                np.asarray(color) if color is not None else np.full((len(mesh.vertices), 4), 255)
            )
            colors.extend(np.clip(np.rint(color * factor), 0, 255).astype(int).tolist())
            coordinates = mesh.visual.uv
            uv.extend(
                coordinates.tolist() if coordinates is not None else [[0, 0]] * len(mesh.vertices)
            )
            has_uv |= coordinates is not None
        else:
            colors.extend(
                mesh.visual.vertex_colors.tolist()
                if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals)
                else [[255, 255, 255, 255]] * len(mesh.vertices)
            )
            uv.extend([[0, 0]] * len(mesh.vertices))
    result = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=normals, vertex_colors=colors, process=False
    )
    if has_uv:
        result.vertex_attributes["s"] = np.asarray(uv)[:, 0]
        result.vertex_attributes["t"] = np.asarray(uv)[:, 1]
    return result


def as_trimesh_scene(
    engine: Engine,
    scene_id: str,
    y_up: bool = False,
    with_materials: bool = True,
    source_scene: Scene | None = None,
    export_transform: list[list[float]] | None = None,
    root_name: str | None = None,
) -> trimesh.Scene:
    scene = source_scene if source_scene is not None else engine.scene(scene_id)
    matrix = np.asarray(
        export_transform if export_transform is not None else np.eye(4), dtype=float
    )
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
    ):
        raise ValueError("Export transform must be an invertible affine 4x4 matrix")
    result = trimesh.Scene()
    names = node_names(scene.objects.values())
    root: str | None = None
    if y_up or root_name is not None or export_transform is not None:
        # One root node carries the up-axis conversion, so an engine receives a single
        # object to place instead of the same rotation repeated on every part.
        root = (
            re.sub(
                r"[^0-9A-Za-z_]+", "_", root_name if root_name is not None else scene.name
            ).strip("_")
            or "scene"
        )
        while root in names.values() or root == result.graph.base_frame:
            root = f"{root}_root"
    objects = [obj for obj in scene.objects.values() if engine.visible(scene, obj)]
    image_cache: dict[str, Image.Image] = {}
    image_bytes = 0
    geometry_bytes = sum(len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in objects)
    instance_sources = {obj.metadata.get("instance_source") for obj in objects}
    geometry_cache: dict[str, tuple[str, trimesh.Trimesh, Any, Any, Any]] = {}
    world_matrices: dict[str, np.ndarray] = {}
    for obj in objects:
        world_matrices[obj.id] = (
            trimesh.transformations.translation_matrix(obj.pivot)
            if obj.kind in {"group", "marker"}
            else np.asarray(obj.transform)
            if obj.id in instance_sources or obj.metadata.get("instance_source")
            else np.eye(4)
        )
        if obj.kind in {"group", "marker"}:
            world_matrices[obj.id][:3, :3] = np.asarray(obj.transform)[:3, :3]
    for obj in objects:
        parent = names[obj.parent_id] if obj.parent_id else root or result.graph.base_frame
        parent_world = world_matrices[obj.parent_id] if obj.parent_id else np.eye(4)
        local = np.linalg.inv(parent_world) @ world_matrices[obj.id]
        if obj.kind in {"group", "marker"}:
            result.graph.update(
                frame_to=names[obj.id],
                frame_from=parent,
                matrix=local,
                metadata={"object_id": obj.id, "kind": obj.kind},
            )
            continue
        mesh = obj.geometry()
        if obj.id in instance_sources or obj.metadata.get("instance_source"):
            # Explicitly unbake once so instances can share local geometry. The node
            # matrix reapplies the same transform exactly once at export.
            affine_mesh(mesh, np.linalg.inv(world_matrices[obj.id]))
        if obj.material_id and with_materials:
            mat = scene.materials[obj.material_id]
            if (
                any(
                    (
                        mat.base_color_texture,
                        mat.normal_texture,
                        mat.metallic_roughness_texture,
                        mat.emissive_texture,
                        mat.occlusion_texture,
                    )
                )
                and obj.uv is None
            ):
                raise ScissorsError(
                    "GEOMETRY_ERROR",
                    "Textured objects require UV before export",
                    {"object_id": obj.id},
                )

            def texture(tid: str | None) -> Image.Image | None:
                nonlocal image_bytes
                if not tid:
                    return None
                if tid in image_cache:
                    return image_cache[tid]
                source = engine.path(engine.textures[tid]["path"])
                if source.stat().st_size > engine.settings.max_file_bytes:
                    raise ScissorsError("RESOURCE_LIMIT", "Export texture exceeds file budget")
                with Image.open(source) as image:
                    pixels = image.width * image.height
                    if (
                        pixels > engine.settings.max_texture_pixels
                        or geometry_bytes + image_bytes + pixels * 16
                        > engine.settings.max_memory_bytes
                    ):
                        raise ScissorsError(
                            "RESOURCE_LIMIT", "Export texture cache exceeds memory/pixel budget"
                        )
                    image_cache[tid] = image.convert("RGBA")
                    image_bytes += pixels * 4
                    return image_cache[tid]

            color = list(mat.base_color)
            color[3] *= mat.alpha
            base_image = texture(mat.base_color_texture)
            transparent = color[3] < 1 or (
                base_image is not None and base_image.getchannel("A").getextrema()[0] < 255
            )
            pbr = trimesh.visual.material.PBRMaterial(
                name=mat.name,
                baseColorFactor=color,
                metallicFactor=mat.metallic,
                roughnessFactor=mat.roughness,
                doubleSided=mat.double_sided,
                alphaMode="BLEND" if transparent else "OPAQUE",
                baseColorTexture=base_image,
                normalTexture=texture(mat.normal_texture),
                metallicRoughnessTexture=texture(mat.metallic_roughness_texture),
                emissiveFactor=mat.emissive,
                emissiveTexture=texture(mat.emissive_texture),
                occlusionTexture=texture(mat.occlusion_texture),
            )
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=obj.uv, material=pbr)
        elif obj.uv is not None:
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=obj.uv)
        if obj.vertex_colors is not None:
            if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                mesh.visual.vertex_attributes["color"] = np.asarray(
                    obj.vertex_colors, dtype=np.uint8
                )
            else:
                mesh.visual = trimesh.visual.color.ColorVisuals(
                    mesh=mesh, vertex_colors=np.asarray(obj.vertex_colors, dtype=np.uint8)
                )
        normal_map = obj.material_id and scene.materials[obj.material_id].normal_texture
        if y_up and (obj.metadata.get("export_tangents") or normal_map):
            if obj.uv is None:
                raise ScissorsError("GEOMETRY_ERROR", "Tangent export requires UV")
            frames = tangent_frames(mesh, np.asarray(obj.uv))
            # Trimesh exports 1-V, so the exported bitangent handedness is reversed.
            frames[:, 3] *= -1
            mesh.vertex_attributes["_SCISSORS_TANGENT"] = frames.astype(np.float32)
        source = str(obj.metadata.get("instance_source", obj.id))
        cached = geometry_cache.get(source)
        if (
            cached is not None
            and cached[2] == obj.uv
            and cached[3] == obj.material_id
            and cached[4] == obj.vertex_colors
            and np.array_equal(cached[1].faces, mesh.faces)
            and cached[1].vertices.shape == mesh.vertices.shape
            and np.allclose(cached[1].vertices, mesh.vertices, rtol=1e-12, atol=1e-12)
            and np.allclose(cached[1].vertex_normals, mesh.vertex_normals, rtol=1e-12, atol=1e-12)
            and np.array_equal(
                cached[1].vertex_attributes.get("_SCISSORS_TANGENT", []),
                mesh.vertex_attributes.get("_SCISSORS_TANGENT", []),
            )
        ):
            result.graph.update(
                frame_to=names[obj.id],
                frame_from=parent,
                matrix=local,
                geometry=cached[0],
                metadata={"object_id": obj.id, "name": obj.name},
            )
            continue
        geometry_cache[source] = (names[obj.id], mesh, obj.uv, obj.material_id, obj.vertex_colors)
        result.add_geometry(
            mesh,
            node_name=names[obj.id],
            geom_name=names[obj.id],
            parent_node_name=parent,
            transform=local,
            metadata={"name": obj.name, "object_id": obj.id, "units": scene.units},
        )
    if root is not None:
        result.graph.update(
            frame_to=root,
            frame_from=result.graph.base_frame,
            matrix=(Z_UP_TO_Y_UP if y_up else np.eye(4)) @ matrix,
        )
    return result


def export_scene(
    engine: Engine,
    scene_id: str,
    output: str,
    format: str = "glb",
    source_scene: Scene | None = None,
    export_transform: list[list[float]] | None = None,
    root_name: str | None = None,
) -> dict[str, Any]:
    path = engine.artifact_path(output)
    if path.suffix.lower() != f".{format}":
        raise ValueError("Output extension must match format")
    retained = source_scene if source_scene is not None else engine.scene(scene_id)
    if engine.scene(scene_id).fbx_documents:
        from scissors.core.fbx_document import export_document

        if root_name is not None:
            raise ScissorsError("UNSUPPORTED_OPERATION", "Retained rigs keep the source FBX root")
        return export_document(engine, retained, path, format, export_transform)
    scene = as_trimesh_scene(
        engine,
        scene_id,
        y_up=format in {"glb", "gltf"},
        source_scene=source_scene,
        export_transform=export_transform,
        root_name=root_name,
    )
    if not scene.geometry:
        raise ScissorsError("GEOMETRY_ERROR", "Cannot export an empty scene")
    diagnostics: list[str] = []
    if format in {"glb", "gltf"}:
        diagnostics.append(
            "glTF is Y-up: scene +Z became glTF +Y, so engines load the model upright; "
            "OBJ and STL keep scene Z-up coordinates."
        )
    files: dict[str, bytes] = {}
    if format == "glb":
        files[path.name] = trimesh.exchange.gltf.export_glb(
            scene, include_normals=True, tree_postprocessor=gltf_tangents
        )
    elif format == "gltf":
        files = trimesh.exchange.gltf.export_gltf(
            scene, include_normals=True, tree_postprocessor=gltf_tangents
        )
        files[path.name] = files.pop("model.gltf")
    elif format == "obj":
        baked = trimesh.Scene()
        for node, mesh in baked_meshes(scene):
            baked.add_geometry(mesh, node_name=node, geom_name=node)
        obj, assets = trimesh.exchange.obj.export_obj(
            baked, return_texture=True, mtl_name=f"{path.stem}.mtl"
        )
        files[path.name] = obj.encode("utf-8")
        files.update(assets or {})
        diagnostics.append("OBJ approximates PBR materials using MTL")
    elif format == "stl":
        files[path.name] = cast(
            bytes,
            trimesh.util.concatenate([mesh for _, mesh in baked_meshes(scene)]).export(
                file_obj=None, file_type="stl"
            ),
        )
        diagnostics.append("STL contains geometry only; materials, UV and textures omitted")
    elif format == "ply":
        files[path.name] = cast(bytes, ply_mesh(scene).export(file_obj=None, file_type="ply"))
        diagnostics.append(
            "PLY flattens hierarchy and bakes base factors into vertex colors; images are omitted"
        )
    elif format in {"dae", "usdz", "fbx"}:
        from scissors.core.collada import export_dae
        from scissors.core.fbx import export_fbx
        from scissors.core.usdz import export_usdz

        units = (source_scene or engine.scene(scene_id)).units
        writer = {"dae": export_dae, "usdz": export_usdz, "fbx": export_fbx}[format]
        data, assets, notes = writer(scene, engine.settings, path.stem, units)
        files[path.name] = data
        files.update(assets)
        diagnostics.extend(notes)
    else:
        raise ValueError("Unsupported export format")
    if sum(map(len, files.values())) > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Export exceeds size limit")
    # Use a dedicated output directory for glTF to avoid collisions between sidecars.
    written = []
    for name, data in files.items():
        target = engine.artifact_path(path.parent / name)
        if target.parent != path.parent:
            raise ScissorsError("PERMISSION_DENIED", "Invalid export asset path")
        atomic_write(target, data, engine.settings.max_file_bytes)
        written.append(str(target))
    return {**artifact(path, format, diagnostics), "files": written}


def view_matrix(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    """Orthographic camera basis whose rows are screen right, screen up and depth.

    Azimuth 0 looks along -Y, 90 along +X, 180 along +Y and 270 along -X.
    Elevation 90 looks straight down and -90 straight up. World +Z stays screen up,
    so a horizontal view never rolls; when looking straight down or up, -Y (forward
    for a scene whose model faces -Y) becomes screen up.
    """
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    direction = np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            -math.cos(azimuth) * math.cos(elevation),
            -math.sin(elevation),
        ]
    )
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ up)) > 1.0 - 1e-9:
        up = np.array([0.0, -1.0, 0.0])
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, direction)
    return np.vstack([right, screen_up, -direction])


def preview(
    engine: Engine,
    scene_id: str,
    output: str = "previews/scene.png",
    width: int = 512,
    height: int = 512,
    wireframe: bool = False,
    azimuth_degrees: float = 35,
    elevation_degrees: float = 65,
    zoom: float = 1,
    offset_x: float = 0,
    offset_y: float = 0,
    clip_axis: Literal["x", "y", "z"] | None = None,
    clip_position: float | None = None,
    clip_keep: Literal["positive", "negative"] = "positive",
    only_object_ids: list[str] | None = None,
    exclude_object_ids: list[str] | None = None,
    sample_textures: bool = False,
    projection: str | None = None,
    camera_position: list[float] | None = None,
    camera_target: list[float] | None = None,
    camera_up: list[float] | None = None,
    fov: float = 45,
    near: float = 0.001,
    far: float | None = None,
    light_direction: list[float] | None = None,
    ambient: float = 0.35,
    light_strength: float = 0.65,
    transparent: bool = False,
    background: list[float] | None = None,
    renderer: str = "auto",
    render_pass: str = "color",
    orthographic_scale: float | None = None,
    highlights: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a preview PNG with the VTK backend or the Pillow painter fallback.

    The camera keeps world +Z as screen up and never rolls; see view_matrix.
    Object filters narrow the render to a sub-assembly and framing follows the
    selection. Sections hide whole triangles on the discarded side of a world
    axis plane. The result reports the backend that drew the image in renderer.
    render_pass="depth" writes the numeric z-buffer as a 16-bit PNG plus a
    `.depth.npz` sidecar and needs the VTK backend.
    """
    if renderer not in {"auto", "vtk", "pillow"}:
        raise ValueError("Renderer must be auto, vtk or pillow")
    if render_pass not in {"color", "depth", "normal"}:
        raise ValueError("Render pass must be color, depth or normal")
    from scissors.core.rendering import render

    return render(
        engine,
        scene_id,
        output,
        width,
        height,
        wireframe,
        azimuth_degrees,
        elevation_degrees,
        zoom,
        offset_x,
        offset_y,
        clip_axis,
        clip_position,
        clip_keep,
        only_object_ids,
        exclude_object_ids,
        sample_textures,
        projection or "orthographic",
        camera_position,
        camera_target,
        camera_up,
        fov,
        near,
        far,
        light_direction,
        ambient,
        light_strength,
        transparent,
        background if background is not None else [242 / 255, 244 / 255, 248 / 255],
        renderer,
        render_pass,
        orthographic_scale,
        highlights,
    )


def compare_reference(
    engine: Engine,
    scene_id: str,
    reference_path: str,
    output: str = "previews/reference.png",
    mode: Literal["side_by_side", "overlay"] = "side_by_side",
    opacity: float = 0.5,
    width: int = 512,
    height: int = 512,
    azimuth_degrees: float = 35,
    elevation_degrees: float = 65,
    zoom: float = 1,
    offset_x: float = 0,
    offset_y: float = 0,
) -> dict[str, Any]:
    """Place a reference photo beside or over a fast preview render."""
    path = engine.artifact_path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("Preview output must be PNG")
    pixels = width * height * (2 if mode == "side_by_side" else 1)
    if (
        pixels > engine.settings.max_texture_pixels
        or pixels * 32 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Preview pixel limit exceeded")
    source = engine.path(reference_path)
    if source.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Reference file exceeds size limit")
    with Image.open(source) as photo:
        if photo.width * photo.height > engine.settings.max_texture_pixels or (
            photo.width * photo.height * 32 + pixels * 32 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Reference image exceeds memory budget")
        reference = ImageOps.pad(
            ImageOps.exif_transpose(photo).convert("RGB"), (width, height), color=(242, 244, 248)
        )
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=engine.root) as folder:
        rendered = preview(
            engine,
            scene_id,
            f"{folder}/render.png",
            width,
            height,
            False,
            azimuth_degrees,
            elevation_degrees,
            zoom,
            offset_x,
            offset_y,
            transparent=True,
        )
        with Image.open(rendered["path"]) as image:
            shot = image.convert("RGBA")
    flattened = Image.new("RGB", (width, height), (242, 244, 248))
    flattened.paste(shot, (0, 0), shot)
    if mode == "overlay":
        mask = shot.getchannel("A").point(lambda value: round(value * opacity))
        canvas = Image.composite(flattened, reference, mask)
    else:
        canvas = Image.new("RGB", (width * 2, height), (242, 244, 248))
        canvas.paste(reference, (0, 0))
        canvas.paste(flattened, (width, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_write(path, buffer.getvalue(), engine.settings.max_file_bytes)
    diagnostics = [
        f"Camera: azimuth {azimuth_degrees} degrees (0 looks along -Y, 90 along +X, "
        f"180 along +Y, 270 along -X), elevation {elevation_degrees} degrees "
        f"(90 looks straight down); zoom {zoom}, offset ({offset_x}, {offset_y}).",
        f"Renderer: {rendered['renderer']}.",
        f"Reference comparison: {mode} with opacity {opacity}.",
    ]
    return {**artifact(path, "png", diagnostics), "renderer": rendered["renderer"]}
