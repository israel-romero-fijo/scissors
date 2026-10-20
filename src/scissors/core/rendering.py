"""Shared preview rendering and camera preparation for VTK and Pillow backends."""

import io
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.export import artifact
from scissors.core.mesh import MeshObject
from scissors.core.persistence import atomic_write
from scissors.core.rendering_fast import Style, collect_surfaces, render_fast
from scissors.core.view import Frame, build_frame, project


def normalized(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-15)


def prepare_frame(
    engine: Engine,
    scene_id: str,
    width: int,
    height: int,
    only_object_ids: list[str] | None,
    exclude_object_ids: list[str] | None,
    **camera: Any,
) -> tuple[list[MeshObject], Frame]:
    """Validate a preview selection, budget, and camera without rendering or writing."""
    scene = engine.scene(scene_id)
    for ids in [only_object_ids, exclude_object_ids]:
        if ids and any(oid not in scene.objects for oid in ids):
            raise ScissorsError("OBJECT_NOT_FOUND", "Preview selection contains unknown objects")
    objects = [
        obj
        for obj in scene.objects.values()
        if obj.kind == "mesh"
        and engine.visible(scene, obj)
        and (only_object_ids is None or obj.id in only_object_ids)
        and (exclude_object_ids is None or obj.id not in exclude_object_ids)
    ]
    geometry_bytes = sum(len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in objects)
    if (
        width * height > engine.settings.max_texture_pixels
        or width * height * 64 + geometry_bytes > engine.settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Preview working image exceeds memory/pixel budget")
    points = (
        np.concatenate([np.asarray(obj.vertices) for obj in objects])
        if objects
        else np.zeros((1, 3))
    )
    return objects, build_frame(points, width=width, height=height, **camera)


def render(
    engine: Engine,
    scene_id: str,
    output: str,
    width: int,
    height: int,
    wireframe: bool,
    azimuth_degrees: float,
    elevation_degrees: float,
    zoom: float,
    offset_x: float,
    offset_y: float,
    clip_axis: str | None,
    clip_position: float | None,
    clip_keep: str,
    only_object_ids: list[str] | None,
    exclude_object_ids: list[str] | None,
    sample_textures: bool,
    projection: str,
    camera_position: list[float] | None,
    camera_target: list[float] | None,
    camera_up: list[float] | None,
    fov: float,
    near: float,
    far: float | None,
    light_direction: list[float] | None,
    ambient: float,
    light_strength: float,
    transparent: bool,
    background: list[float],
    renderer: str = "auto",
    render_pass: str = "color",
    orthographic_scale: float | None = None,
    highlights: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one PNG preview and report which backend drew it.

    renderer=auto uses the VTK offscreen renderer when an OpenGL context exists
    and the Pillow painter otherwise. vtk fails with BACKEND_UNAVAILABLE when no
    context can be created; pillow always works. render_pass="depth" reads the
    OpenGL z-buffer and render_pass="normal" draws world normals as colors; both
    need VTK.
    """
    if renderer not in {"auto", "vtk", "pillow"}:
        raise ValueError("Renderer must be auto, vtk or pillow")
    if render_pass not in {"color", "depth", "normal"}:
        raise ValueError("Render pass must be color, depth or normal")
    path = engine.artifact_path(output)
    if path.suffix.lower() != ".png":
        raise ValueError("Preview output must be PNG")
    scene = engine.scene(scene_id)
    objects, frame = prepare_frame(
        engine,
        scene_id,
        width=width,
        height=height,
        only_object_ids=only_object_ids,
        exclude_object_ids=exclude_object_ids,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
        zoom=zoom,
        offset_x=offset_x,
        offset_y=offset_y,
        projection=projection,
        camera_position=camera_position,
        camera_target=camera_target,
        camera_up=camera_up,
        fov=fov,
        near=near,
        far=far,
        orthographic_scale=orthographic_scale,
    )
    axis = "xyz".index(clip_axis) if clip_axis is not None else None
    plane = (
        float(
            clip_position if clip_position is not None else (frame.low[axis] + frame.high[axis]) / 2
        )
        if axis is not None
        else None
    )
    surfaces = collect_surfaces(
        engine,
        scene,
        objects,
        sample_textures=sample_textures,
        clip_axis=clip_axis,
        clip_position=plane,
        clip_keep=clip_keep,
    )
    highlight_primitives: list[tuple[str, np.ndarray]] = []
    highlight_summary: list[dict[str, Any]] = []
    if highlights:
        from scissors.core.highlighting import prepare_highlights

        if render_pass != "color":
            raise ValueError("Highlights require the color render pass")
        highlight_primitives, highlight_summary = prepare_highlights(
            engine, objects, highlights, frame, clip_axis, plane, clip_keep
        )
    light = (
        normalized(np.asarray(light_direction, dtype=float))
        if light_direction is not None
        else frame.light
    )
    style = Style(
        background=np.asarray(background, dtype=float),
        light=light,
        ambient=ambient,
        light_strength=light_strength,
        transparent=transparent,
        sample_textures=sample_textures,
        wireframe=wireframe,
    )
    rendered = render_fast(
        surfaces=surfaces,
        frame=frame,
        style=style,
        requested=renderer,
        pass_name=render_pass,
    )
    data = rendered.data
    if highlights:
        from scissors.core.highlighting import draw_highlights

        data = draw_highlights(data, highlight_primitives, engine.settings.operation_timeout)
    atomic_write(path, data, engine.settings.max_file_bytes)
    sidecar: str | None = None
    if rendered.depth is not None and rendered.coverage is not None:
        sidecar_path = engine.artifact_path(path.with_suffix(".depth.npz"))
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            depth=rendered.depth.astype(np.float32),
            coverage=rendered.coverage,
            near=np.asarray(frame.near),
            far=np.asarray(frame.far),
        )
        atomic_write(sidecar_path, buffer.getvalue(), engine.settings.max_file_bytes)
        sidecar = str(sidecar_path)
    work = 0
    for surface in surfaces:
        screen, _ = project(frame, surface.vertices)
        corners = screen[surface.faces]
        spans = corners.max(axis=1) - corners.min(axis=1)
        work += int(np.sum(np.maximum(spans[:, 0] + 1, 0) * np.maximum(spans[:, 1] + 1, 0)))
    camera_description = (
        f"Camera: explicit world position {frame.eye.tolist()}, target {frame.target.tolist()}, "
        f"up {frame.up.tolist()}; zoom {zoom}, offset ({offset_x}, {offset_y})."
        if camera_position is not None
        else f"Camera: azimuth {azimuth_degrees} degrees (0 looks along -Y, 90 along +X, "
        f"180 along +Y, 270 along -X), elevation {elevation_degrees} degrees "
        f"(90 looks straight down); zoom {zoom}, offset ({offset_x}, {offset_y})."
    )
    diagnostics = [
        camera_description,
        f"Renderer: {rendered.renderer}"
        + (f" ({rendered.detail['device']})" if rendered.detail.get("device") else "")
        + ".",
    ]
    if render_pass == "depth":
        diagnostics.append(
            "Depth pass: positive camera-forward distance in scene units; background is "
            "infinity. Load the NPZ sidecar with allow_pickle=False."
        )
    if render_pass == "normal":
        diagnostics.append(
            "Normal pass: world-space normals encoded as (normal+1)/2 in RGB, lighting off."
        )
    if axis is not None and plane is not None:
        diagnostics.append(
            f"Section view: keeping {clip_axis} "
            f"{'>=' if clip_keep == 'positive' else '<='} {round(plane, 6)}; hidden "
            "triangles do not change framing."
        )
    if only_object_ids is not None:
        diagnostics.append(f"Render limited to {len(only_object_ids)} selected object(s).")
    if exclude_object_ids is not None:
        diagnostics.append(f"Render excluded {len(exclude_object_ids)} object(s).")
    if render_pass == "color":
        diagnostics.append(
            "Base-colour textures sampled at preview resolution; alpha is not blended."
            if sample_textures and not wireframe
            else "Solid colours; no texture sampling."
        )
        diagnostics.append(
            "VTK offscreen renderer: smooth shading, no shadows or environment lighting."
            if rendered.renderer == "vtk"
            else "Pillow painter fallback: flat per-face shading, no smooth normals."
        )
    result = {
        **artifact(path, "png", diagnostics),
        "projection": projection,
        "render_pass": render_pass,
        "renderer": rendered.renderer,
        "renderer_detail": rendered.detail,
        "camera_position": frame.eye.tolist(),
        "camera_target": frame.target.tolist(),
        "camera_up": frame.up.tolist(),
        "fov": frame.fov,
        "effective_fov": float(np.degrees(2 * np.arctan(height / (2 * frame.focal))))
        if projection == "perspective"
        else None,
        "orthographic_scale": float(height / frame.ortho_scale)
        if projection == "orthographic"
        else None,
        "near": frame.near,
        "far": frame.far,
        "projected_pixel_work": work,
    }
    if highlights:
        result["highlight_mode"] = "xray_overlay"
        result["highlights"] = highlight_summary
        result["diagnostics"].append(
            "Orange x-ray highlights ignore surface occlusion; section and camera clipping apply."
        )
    if render_pass == "depth":
        result["depth_path"] = sidecar
        result["depth_semantics"] = "camera-forward distance in scene units; infinity is background"
    if render_pass == "normal":
        result["normal_space"] = "world"
    return result


def turntable(
    engine: Engine,
    scene_id: str,
    output: str,
    frames: int,
    size: int,
    elevation: float,
    fps: int,
    projection: str,
    sample_textures: bool,
    only_object_ids: list[str] | None,
    renderer: str = "auto",
) -> dict[str, Any]:
    from scissors.core.export import preview

    path = engine.artifact_path(output)
    if path.suffix.lower() != ".gif":
        raise ValueError("Turntable output must be GIF")
    if frames * size * size * 16 + size * size * 128 > engine.settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Turntable frame cache exceeds memory budget")
    deadline = time.monotonic() + engine.settings.operation_timeout
    images = []
    used = "auto"
    with tempfile.TemporaryDirectory(prefix=".turntable-", dir=engine.root) as folder:
        for index in range(frames):
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Turntable timed out")
            frame_path = str(engine.path(f"{folder}/frame_{index}.png"))
            rendered = preview(
                engine,
                scene_id,
                frame_path,
                size,
                size,
                azimuth_degrees=index * 360 / frames,
                elevation_degrees=elevation,
                projection=projection,
                only_object_ids=only_object_ids,
                sample_textures=sample_textures,
                renderer=renderer,
            )
            used = rendered["renderer"]
            with Image.open(frame_path) as image:
                images.append(image.convert("RGB"))
    if time.monotonic() > deadline:
        raise ScissorsError("OPERATION_TIMEOUT", "Turntable timed out")
    buffer = io.BytesIO()
    duration = max(10, round(1000 / fps / 10) * 10)
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=2,
        optimize=False,
    )
    with Image.open(io.BytesIO(buffer.getvalue())) as encoded:
        encoded_frames = encoded.n_frames
    if time.monotonic() > deadline:
        raise ScissorsError("OPERATION_TIMEOUT", "Turntable encoding timed out")
    atomic_write(path, buffer.getvalue(), engine.settings.max_file_bytes)
    return {
        **artifact(
            path, "gif", ["GIF uses a palette and opaque background; scene geometry is unchanged."]
        ),
        "frames": encoded_frames,
        "requested_frames": frames,
        "fps": fps,
        "frame_duration_ms": duration,
        "duration_seconds": frames * duration / 1000,
        "projection": projection,
        "elevation_degrees": elevation,
        "renderer": used,
    }
