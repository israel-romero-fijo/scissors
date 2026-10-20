"""Fast preview backends: VTK offscreen and a Pillow painter fallback.

VTK draws the scene through OpenGL in a fraction of a second; the painter fills
depth-sorted triangles with Pillow when no OpenGL context exists.
"""

import io
import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject, compute_normals
from scissors.core.scene import Scene
from scissors.core.view import Frame, clipped_face_mask, normalized, project

BACKENDS = ("auto", "vtk", "pillow")


@dataclass(frozen=True)
class Surface:
    """One mesh prepared for a rasterizer: plain arrays, no engine state."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv: np.ndarray | None
    colors: np.ndarray | None
    base_color: tuple[float, float, float, float]
    emissive: tuple[float, float, float]
    alpha: float
    metallic: float
    roughness: float
    texture: np.ndarray | None


@dataclass(frozen=True)
class Style:
    background: np.ndarray
    light: np.ndarray
    ambient: float
    light_strength: float
    transparent: bool
    sample_textures: bool
    wireframe: bool


@dataclass(frozen=True)
class Rendered:
    """One rendered PNG plus the numeric buffers that passes may need."""

    data: bytes
    renderer: str
    detail: dict[str, Any]
    depth: np.ndarray | None = None
    coverage: np.ndarray | None = None


def _probe_vtk_once() -> tuple[bool, dict[str, Any]]:
    """Keep native driver failures inside a bounded child process."""
    try:
        process = subprocess.run(
            [sys.executable, "-m", "scissors.core.vtk_probe"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if process.returncode:
            return False, {"reason": "OpenGL probe failed", "exit_code": process.returncode}
        detail = json.loads(process.stdout)
        return True, {"device": str(detail["device"])}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
        return False, {"reason": f"OpenGL probe failed: {type(error).__name__}"}


@dataclass
class _ProbeState:
    result: tuple[bool, dict[str, Any]] | None = None
    expires: float = 0


_vtk_probe_state = _ProbeState()
_vtk_probe_lock = threading.Lock()
_probe_clock = time.monotonic


def vtk_capability() -> tuple[bool, dict[str, Any]]:
    """Cache successful probes; retry failures once and expire negative results after 30s."""
    with _vtk_probe_lock:
        cached = _vtk_probe_state.result
        if cached is not None and (cached[0] or _probe_clock() < _vtk_probe_state.expires):
            return cached[0], dict(cached[1])
        for _ in range(2):
            result = _probe_vtk_once()
            if result[0]:
                break
        _vtk_probe_state.result = result
        _vtk_probe_state.expires = _probe_clock() + 30
        return result[0], dict(result[1])


def collect_surfaces(
    engine: Engine,
    scene: Scene,
    objects: list[MeshObject],
    *,
    sample_textures: bool,
    clip_axis: str | None,
    clip_position: float | None,
    clip_keep: str,
) -> list[Surface]:
    """Copy engine meshes into arrays shared by both backends."""
    surfaces: list[Surface] = []
    for obj in objects:
        if not obj.faces:
            continue
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        faces = np.asarray(obj.faces, dtype=np.int64)
        mask = clipped_face_mask(vertices, faces, clip_axis, clip_position, clip_keep)
        if not mask.any():
            continue
        faces = np.ascontiguousarray(faces[mask])
        normals = (
            np.asarray(obj.normals, dtype=np.float64)
            if len(obj.normals) == len(obj.vertices)
            else compute_normals(obj.geometry())
        )
        material = scene.materials.get(obj.material_id or "")
        texture = (
            load_texture(engine, material.base_color_texture)
            if sample_textures and material is not None and material.base_color_texture
            else None
        )
        raw_base = material.base_color if material is not None else (0.32, 0.58, 0.83, 1.0)
        raw_emissive = material.emissive if material is not None else (0.0, 0.0, 0.0)
        surfaces.append(
            Surface(
                vertices=np.ascontiguousarray(vertices),
                faces=faces,
                normals=np.ascontiguousarray(normals),
                uv=np.asarray(obj.uv, dtype=np.float64) if obj.uv is not None else None,
                colors=(
                    np.asarray(obj.vertex_colors, dtype=np.float64) / 255.0
                    if obj.vertex_colors is not None
                    else None
                ),
                base_color=(
                    float(raw_base[0]),
                    float(raw_base[1]),
                    float(raw_base[2]),
                    float(raw_base[3]),
                ),
                emissive=(
                    float(raw_emissive[0]),
                    float(raw_emissive[1]),
                    float(raw_emissive[2]),
                ),
                alpha=float(material.alpha) if material is not None else 1.0,
                metallic=float(material.metallic) if material is not None else 0.0,
                roughness=float(material.roughness) if material is not None else 0.7,
                texture=texture,
            )
        )
    return surfaces


def load_texture(engine: Engine, texture_id: str) -> np.ndarray:
    """Decode one managed texture, bounded by the same limits as the CPU renderer."""
    source = engine.path(engine.textures[texture_id]["path"])
    if source.stat().st_size > engine.settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Preview texture exceeds file limit")
    with Image.open(source) as image:
        if (
            image.width * image.height > engine.settings.max_texture_pixels
            or image.width * image.height * 32 > engine.settings.max_memory_bytes
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Preview texture cache exceeds budget")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def render_fast(
    *,
    surfaces: list[Surface],
    frame: Frame,
    style: Style,
    requested: str,
    pass_name: str = "color",
) -> Rendered:
    """Render with VTK when possible, else with the Pillow painter.

    The depth pass reads the OpenGL z-buffer, so it needs VTK. The painter cannot
    produce numeric camera distance and reports BACKEND_UNAVAILABLE instead of
    returning a misleading image.
    """
    if pass_name in {"depth", "normal"}:
        available, detail = vtk_capability()
        if requested == "pillow" or not available:
            raise ScissorsError(
                "BACKEND_UNAVAILABLE",
                f"The {pass_name} pass needs the VTK offscreen renderer",
                {"backend": "vtk", "requirement": "OpenGL 2.1+ driver", **detail},
            )
        if pass_name == "depth":
            return _render_vtk_depth(surfaces, frame, style, detail)
        return Rendered(_render_vtk(surfaces, frame, style, "normal"), "vtk", dict(detail))
    if requested == "pillow":
        return Rendered(_render_painter(surfaces, frame, style), "pillow", {})
    available, detail = vtk_capability()
    if not available:
        if requested == "vtk":
            raise ScissorsError(
                "BACKEND_UNAVAILABLE",
                "VTK offscreen rendering needs an OpenGL context",
                {"backend": "vtk", "requirement": "OpenGL 2.1+ driver", **detail},
            )
        return Rendered(_render_painter(surfaces, frame, style), "pillow", dict(detail))
    return Rendered(_render_vtk(surfaces, frame, style), "vtk", dict(detail))


def _focal_point(frame: Frame) -> np.ndarray:
    """World point that the CPU renderer maps to the centre of the image.

    Panel offsets move the focal point inside the view plane instead of using
    vtkCamera.SetWindowCenter, which shifted the image even for its documented
    neutral value.
    """
    depth = float((frame.target - frame.eye) @ frame.rotation[2])
    if frame.projection == "perspective":
        distance = max(-depth, frame.near)
        shift_x = -frame.offset_x * frame.width * distance / frame.focal
        shift_y = frame.offset_y * frame.height * distance / frame.focal
        centre = np.zeros(2)
    else:
        shift_x = -frame.offset_x * frame.width / frame.ortho_scale
        shift_y = frame.offset_y * frame.height / frame.ortho_scale
        centre = frame.center
    return (
        frame.eye
        + frame.rotation[0] * (centre[0] + shift_x)
        + frame.rotation[1] * (centre[1] + shift_y)
        + frame.rotation[2] * depth
    )


def _vtk_scene(surfaces: list[Surface], frame: Frame, style: Style, mode: str = "color") -> Any:
    """Build, render and return a window that still owns its OpenGL context.

    mode="normal" writes world-space normals as vertex colors with lighting off,
    so antialiasing stays disabled and the encoded values stay exact.
    """
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*(float(value) for value in style.background))
    if style.transparent:
        renderer.SetBackgroundAlpha(0.0)
    if mode == "color" and hasattr(renderer, "SetUseFXAA"):
        renderer.SetUseFXAA(True)
    for surface in surfaces:
        if mode == "normal":
            actor = _vtk_normal_actor(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, surface)
        else:
            actor = _vtk_actor(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, surface, style)
        renderer.AddActor(actor)

    light = vtk.vtkLight()
    light.SetPosition(*(float(value) for value in frame.target + style.light * 1000.0))
    light.SetFocalPoint(*(float(value) for value in frame.target))
    light.SetPositional(False)
    light.SetIntensity(max(float(style.light_strength), 0.0))
    renderer.AddLight(light)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(*(float(value) for value in frame.eye))
    camera.SetFocalPoint(*(float(value) for value in _focal_point(frame)))
    camera.SetViewUp(*(float(value) for value in frame.up))
    if frame.projection == "perspective":
        camera.ParallelProjectionOff()
        camera.SetViewAngle(float(np.degrees(2.0 * np.arctan((frame.height / 2.0) / frame.focal))))
    else:
        camera.ParallelProjectionOn()
        camera.SetParallelScale(float((frame.height / 2.0) / frame.ortho_scale))
    camera.SetClippingRange(float(frame.near), float(frame.far))

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(frame.width, frame.height)
    window.SetAlphaBitPlanes(1 if style.transparent else 0)
    # Multisampling blends edge pixels. Diagnostic passes must keep exact values.
    window.SetMultiSamples(8 if mode == "color" else 0)
    window.AddRenderer(renderer)
    window.Render()
    return window


def _render_vtk(surfaces: list[Surface], frame: Frame, style: Style, mode: str = "color") -> bytes:
    import vtk

    window = _vtk_scene(surfaces, frame, style, mode)
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetScale(1)
    capture.ReadFrontBufferOff()
    if style.transparent:
        capture.SetInputBufferTypeToRGBA()
    else:
        capture.SetInputBufferTypeToRGB()
    capture.Update()
    try:
        with tempfile.TemporaryDirectory(prefix="scissors-preview-") as folder:
            target = Path(folder) / "frame.png"
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(target))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            return target.read_bytes()
    finally:
        window.Finalize()


def _render_vtk_depth(
    surfaces: list[Surface], frame: Frame, style: Style, detail: dict[str, Any]
) -> Rendered:
    """Read the OpenGL z-buffer and convert it to camera-forward distance."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    window = _vtk_scene(surfaces, frame, style)
    try:
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetScale(1)
        capture.ReadFrontBufferOff()
        capture.SetInputBufferTypeToZBuffer()
        capture.Update()
        image = capture.GetOutput()
        dimensions = image.GetDimensions()
        width, height = int(dimensions[0]), int(dimensions[1])
        scalars = vtk_to_numpy(image.GetPointData().GetScalars())
        # VTK images start at the bottom-left corner; PNG rows start at the top.
        window_depth = np.flipud(scalars.reshape(height, width)).astype(np.float64)
    finally:
        window.Finalize()
    near, far = float(frame.near), float(frame.far)
    if frame.projection == "perspective":
        normalised = 2.0 * window_depth - 1.0
        depth = (2.0 * near * far) / ((far + near) - normalised * (far - near))
    else:
        depth = near + window_depth * (far - near)
    coverage = window_depth < 1.0
    depth = np.where(coverage, depth, np.inf)
    encoded = np.where(coverage, np.clip((depth - near) / (far - near), 0.0, 1.0), 1.0)
    if style.transparent:
        rgb = np.repeat(encoded[..., None], 3, axis=2)
        picture = Image.fromarray(
            np.dstack(
                [
                    np.rint(rgb * 255).astype(np.uint8),
                    coverage.astype(np.uint8) * 255,
                ]
            )
        )
    else:
        picture = Image.fromarray(np.rint(encoded * 65535).astype(np.uint16))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return Rendered(
        buffer.getvalue(),
        "vtk",
        dict(detail),
        depth.astype(np.float32),
        coverage,
    )


def _vtk_polydata(
    vtk: Any,
    numpy_to_vtk: Any,
    numpy_to_vtkIdTypeArray: Any,
    surface: Surface,
) -> tuple[Any, Any, Any]:
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(surface.vertices, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, surface.faces.size + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(surface.faces.reshape(-1), deep=True),
    )
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    data.GetPointData().SetNormals(numpy_to_vtk(surface.normals, deep=True))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return data, mapper, actor


def _vtk_normal_actor(
    vtk: Any,
    numpy_to_vtk: Any,
    numpy_to_vtkIdTypeArray: Any,
    surface: Surface,
) -> Any:
    """Draw world-space normals as vertex colors with lighting switched off."""
    data, mapper, actor = _vtk_polydata(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, surface)
    encoded = np.clip((surface.normals + 1.0) * 0.5, 0.0, 1.0)
    colors = np.ascontiguousarray(np.rint(encoded * 255.0).astype(np.uint8))
    data.GetPointData().SetScalars(
        numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    )
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()
    property_ = actor.GetProperty()
    property_.LightingOff()
    property_.SetOpacity(1.0)
    return actor


def _vtk_actor(
    vtk: Any,
    numpy_to_vtk: Any,
    numpy_to_vtkIdTypeArray: Any,
    surface: Surface,
    style: Style,
) -> Any:
    data, mapper, actor = _vtk_polydata(vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray, surface)
    property_ = actor.GetProperty()
    base = surface.base_color
    ambient = float(np.clip(style.ambient, 0.0, 1.0))
    property_.SetColor(float(base[0]), float(base[1]), float(base[2]))
    property_.SetAmbientColor(
        *(
            float(np.clip(base[index] * ambient + surface.emissive[index], 0.0, 1.0))
            for index in range(3)
        )
    )
    property_.SetAmbient(1.0)
    property_.SetDiffuse(1.0)
    property_.SetSpecular(0.15)
    property_.SetSpecularPower(30.0)
    property_.SetInterpolationToPhong()
    property_.SetOpacity(float(np.clip(base[3] * surface.alpha, 0.0, 1.0)))
    if style.wireframe:
        property_.SetRepresentationToWireframe()
    if surface.colors is not None:
        colors = numpy_to_vtk(
            np.ascontiguousarray(np.rint(surface.colors * 255).astype(np.uint8)),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        data.GetPointData().SetScalars(colors)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
    elif surface.texture is not None and surface.uv is not None:
        pixels = np.flipud(surface.texture)
        image = vtk.vtkImageData()
        image.SetDimensions(pixels.shape[1], pixels.shape[0], 1)
        image.GetPointData().SetScalars(
            numpy_to_vtk(
                np.ascontiguousarray(pixels.reshape(-1, 3)),
                deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR,
            )
        )
        data.GetPointData().SetTCoords(numpy_to_vtk(surface.uv, deep=True))
        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOn()
        if hasattr(texture, "SetUseSRGBColorSpace"):
            texture.SetUseSRGBColorSpace(True)
        actor.SetTexture(texture)
        mapper.ScalarVisibilityOff()
    return actor


def _render_painter(surfaces: list[Surface], frame: Frame, style: Style) -> bytes:
    """Fill depth-sorted triangles with Pillow when OpenGL is not available."""
    factor = 2 if (2 * frame.width) * (2 * frame.height) <= 16_777_216 else 1
    big = frame.scaled(factor)
    background: tuple[int, int, int] = (
        int(round(float(style.background[0]) * 255)),
        int(round(float(style.background[1]) * 255)),
        int(round(float(style.background[2]) * 255)),
    )
    transparent = style.transparent
    fill: tuple[int, int, int] | tuple[int, int, int, int] = (
        (background[0], background[1], background[2], 0) if transparent else background
    )
    canvas = Image.new("RGBA" if transparent else "RGB", (big.width, big.height), fill)
    draw = ImageDraw.Draw(canvas)
    for surface in surfaces:
        _paint_surface(draw, surface, big, style)
    if factor > 1:
        canvas = canvas.resize((frame.width, frame.height), Image.LANCZOS)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _paint_surface(draw: ImageDraw.ImageDraw, surface: Surface, frame: Frame, style: Style) -> None:
    screen, depth = project(frame, surface.vertices)
    triangles = screen[surface.faces]
    order = np.argsort(-depth[surface.faces].mean(axis=1))
    world = surface.vertices[surface.faces]
    normals = normalized(np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0]))
    centroid = world.mean(axis=1)
    view = (
        normalized(frame.eye - centroid)
        if frame.projection == "perspective"
        else np.broadcast_to(frame.rotation[2], centroid.shape)
    )
    normals = normals * np.where(np.sum(normals * view, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    lambert = np.clip(normals @ style.light, 0.0, None)
    base = np.clip(np.asarray(surface.base_color[:3], dtype=float), 0.0, None) ** 2.2
    if surface.colors is not None:
        base = base * surface.colors[surface.faces].mean(axis=1)[:, :3]
    if surface.texture is not None and surface.uv is not None:
        uv = surface.uv[surface.faces].mean(axis=1)
        rows = np.clip(
            np.rint((1.0 - uv[:, 1]) * (surface.texture.shape[0] - 1)).astype(int),
            0,
            surface.texture.shape[0] - 1,
        )
        columns = np.clip(
            np.rint(uv[:, 0] * (surface.texture.shape[1] - 1)).astype(int),
            0,
            surface.texture.shape[1] - 1,
        )
        base = base * (surface.texture[rows, columns].astype(float) / 255.0) ** 2.2
    emissive = np.asarray(surface.emissive, dtype=float) ** 2.2
    shaded = style.ambient * base + style.light_strength * lambert[:, None] * base + emissive
    encoded = (np.clip(shaded, 0.0, 1.0) ** (1 / 2.2) * 255.0).astype(int)
    alpha = float(np.clip(surface.base_color[3] * surface.alpha, 0.0, 1.0))
    background = np.asarray(style.background[:3], dtype=float) * 255.0
    points = triangles.tolist()
    for index in order:
        polygon = points[index]
        colour = (
            int(encoded[index][0]),
            int(encoded[index][1]),
            int(encoded[index][2]),
        )
        if style.wireframe:
            draw.line(polygon + [polygon[0]], fill=colour, width=1)
        elif style.transparent:
            draw.polygon(polygon, fill=(colour[0], colour[1], colour[2], int(round(alpha * 255))))
        elif alpha < 1.0:
            blended = np.asarray(colour, dtype=float) * alpha + background * (1.0 - alpha)
            draw.polygon(
                polygon,
                fill=(int(blended[0]), int(blended[1]), int(blended[2])),
            )
        else:
            draw.polygon(polygon, fill=colour)
