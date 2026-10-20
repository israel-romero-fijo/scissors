"""Preview-backed review sheets with staged multi-file publication."""

import io
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.export import artifact, preview
from scissors.core.persistence import atomic_write, json_bytes
from scissors.core.rendering import prepare_frame

CAMERA_KEYS = (
    "azimuth_degrees",
    "elevation_degrees",
    "zoom",
    "offset_x",
    "offset_y",
    "projection",
    "camera_position",
    "camera_target",
    "camera_up",
    "fov",
    "near",
    "far",
    "orthographic_scale",
)
DEFAULT_VIEWS = [
    {"name": "Front", "azimuth_degrees": 0, "elevation_degrees": 0},
    {"name": "Right", "azimuth_degrees": 90, "elevation_degrees": 0},
    {"name": "Top", "azimuth_degrees": 0, "elevation_degrees": 90},
    {"name": "Back", "azimuth_degrees": 180, "elevation_degrees": 0},
    {"name": "Left", "azimuth_degrees": 270, "elevation_degrees": 0},
    {"name": "Perspective", "azimuth_degrees": 35, "elevation_degrees": 35},
]


def expand_view_sets(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len({value["name"] for value in sets}) != len(sets):
        raise ValueError("View set names must be unique")
    expanded = []
    for group in sets:
        if not group["stereo"] and (not group["include_center"] or "eye_separation" in group):
            raise ValueError("A non-stereo view set requires its center and no eye_separation")
        offsets = group["head_offsets"]
        if len({value["name"] for value in offsets}) != len(offsets):
            raise ValueError("Head offset names must be unique in each view set")
        position, target, up = (
            np.asarray(group[key], dtype=float) for key in ("position", "target", "up")
        )
        forward = target - position
        length = float(np.linalg.norm(forward))
        if not np.isfinite(length) or length <= 1e-12:
            raise ValueError("View set position and target must differ")
        right = np.cross(forward / length, up)
        length = float(np.linalg.norm(right))
        if not np.isfinite(length) or length <= 1e-12:
            raise ValueError("View set up must not be zero or parallel to its view")
        right /= length
        eyes = [("center", 0.0)] if group["include_center"] else []
        if group["stereo"]:
            separation = group.get("eye_separation", 0.064)
            eyes.extend([("left", -separation / 2), ("right", separation / 2)])
        common = {
            key: value
            for key, value in group.items()
            if key
            not in {
                "stereo",
                "eye_separation",
                "include_center",
                "head_offsets",
            }
        }
        for head in offsets:
            for eye_name, distance in eyes:
                offset = np.asarray(head["offset"]) + right * distance
                eye, aim = position + offset, target + offset
                if not np.isfinite([eye, aim]).all():
                    raise ValueError("Expanded eye position and target must remain finite")
                expanded.append(
                    {
                        **common,
                        "name": f"{group['name']}/{head['name']}/{eye_name}",
                        "position": eye.tolist(),
                        "target": aim.tolist(),
                        "view_set": group["name"],
                        "head_name": head["name"],
                        "eye": eye_name,
                        "eye_separation": group.get("eye_separation", 0.064)
                        if group["stereo"]
                        else None,
                    }
                )
    return expanded


def publish_files(engine: Engine, files: list[tuple[Path, Path]], staging: Path) -> None:
    """Use per-file atomic replacement and restore prior files on ordinary failure."""
    backups: dict[Path, Path | None] = {}
    attempted = []
    backup_size = sum(
        destination.stat().st_size for _, destination in files if destination.exists()
    )
    if backup_size > engine.settings.max_file_bytes:
        raise ScissorsError(
            "RESOURCE_LIMIT", "Render sheet backup files exceed aggregate size budget"
        )
    for index, (_, destination) in enumerate(files):
        previous = staging / f"backup_{index}.png" if destination.exists() else None
        if previous is not None:
            shutil.copy2(destination, previous)
        backups[destination] = previous
    try:
        for source, destination in files:
            attempted.append(destination)
            atomic_write(destination, source.read_bytes(), engine.settings.max_file_bytes)
    except Exception as exc:
        failures = []
        for destination in reversed(attempted):
            try:
                previous = backups[destination]
                if previous is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(previous, destination)
            except OSError:
                failures.append(str(destination))
        if failures:
            raise ScissorsError(
                "FILESYSTEM_ERROR",
                "Render sheet publication failed and some prior files could not be restored",
                {"unrestored_paths": failures},
            ) from exc
        if isinstance(exc, ScissorsError):
            raise
        raise ScissorsError(
            "FILESYSTEM_ERROR",
            "Render sheet publication failed; prior files were restored",
            {"reason": type(exc).__name__},
        ) from exc


def render_sheet(
    engine: Engine,
    scene_id: str,
    output: str,
    views: list[dict[str, Any]],
    view_sets: list[dict[str, Any]],
    columns: int,
    size: int,
    only: list[str] | None,
    exclude: list[str] | None,
    sample_textures: bool,
    projection: str | None,
    transparent: bool,
    renderer: str,
    background: list[float] | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + engine.settings.operation_timeout
    choices = [*views, *expand_view_sets(view_sets)]
    if not choices:
        choices = [dict(value, type="orbit") for value in DEFAULT_VIEWS]
    if len(choices) > 24:
        raise ScissorsError("RESOURCE_LIMIT", "Render sheet exceeds 24 expanded views")
    destination = engine.artifact_path(output)
    if destination.suffix.lower() != ".png":
        raise ValueError("Render sheet output must be PNG")
    background = background if background is not None else [242 / 255, 244 / 255, 248 / 255]
    prepared = []
    face_work = 0
    geometry_bytes = 0
    for index, view in enumerate(choices):
        camera = view.get("type", "orbit") == "camera"
        mode = view.get("projection", projection or ("perspective" if camera else "orthographic"))
        if camera and mode == "orthographic" and "orthographic_scale" not in view:
            raise ValueError("Explicit orthographic sheet cameras require orthographic_scale")
        if camera and mode == "orthographic" and "fov" in view:
            raise ValueError("Explicit orthographic cameras use orthographic_scale, not fov")
        parameters: dict[str, Any] = {
            "width": view.get("width", size),
            "height": view.get("height", size),
            "azimuth_degrees": view.get("azimuth_degrees", 35),
            "elevation_degrees": view.get("elevation_degrees", 35),
            "zoom": view.get("zoom", 1),
            "offset_x": view.get("offset_x", 0),
            "offset_y": view.get("offset_y", 0),
            "projection": mode,
            "camera_position": view.get("position"),
            "camera_target": view.get("target"),
            "camera_up": view.get("up"),
            "fov": view.get("fov", 45),
            "near": view.get("near", 0.001),
            "far": view.get("far"),
            "orthographic_scale": view.get("orthographic_scale"),
            "only_object_ids": only,
            "exclude_object_ids": exclude,
            "sample_textures": view.get("sample_textures", sample_textures),
            "transparent": transparent,
            "renderer": view.get("renderer", renderer),
            "wireframe": view.get("wireframe", False),
            "ambient": view.get("ambient", 0.35),
            "light_strength": view.get("light_strength", 0.65),
            "light_direction": view.get("light_direction"),
            "background": view.get("background", background),
        }
        objects, _ = prepare_frame(
            engine,
            scene_id,
            parameters["width"],
            parameters["height"],
            only,
            exclude,
            **{key: parameters[key] for key in CAMERA_KEYS},
        )
        face_work += sum(len(obj.faces) for obj in objects)
        geometry_bytes = max(
            geometry_bytes, sum(len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in objects)
        )
        target = engine.artifact_path(destination.with_name(f"{destination.stem}_view_{index}.png"))
        prepared.append((view, parameters, target))
    targets = [destination, *(value[2] for value in prepared)]
    if len(set(targets)) != len(targets):
        raise ValueError("Sheet and tile output paths must be distinct")
    for target in targets:
        if target.exists() and not target.is_file():
            raise ValueError("Render sheet targets must be files")
    columns = min(columns, len(prepared))
    rows = (len(prepared) + columns - 1) // columns
    tile_width = max(parameters["width"] for _, parameters, _ in prepared)
    tile_height = max(parameters["height"] for _, parameters, _ in prepared)
    dimensions = (columns * tile_width, rows * (tile_height + 24))
    sheet_pixels = dimensions[0] * dimensions[1]
    total_pixels = sum(parameters["width"] * parameters["height"] for _, parameters, _ in prepared)
    largest = max(parameters["width"] * parameters["height"] for _, parameters, _ in prepared)
    if (
        sheet_pixels + total_pixels > engine.settings.max_texture_pixels
        or sheet_pixels * 16 + largest * 64 + geometry_bytes > engine.settings.max_memory_bytes
    ):
        raise ScissorsError(
            "RESOURCE_LIMIT", "Render sheet exceeds aggregate pixel or memory budget"
        )
    if face_work > engine.settings.max_geometry_pairs:
        raise ScissorsError(
            "RESOURCE_LIMIT", "Render sheet face instances exceed geometry work budget"
        )

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Render sheet exceeded its time budget")

    check_time()
    results = []
    # The staging folder belongs to this call. No final artifacts are written until
    # all views, labels, image budgets, and result metadata have been checked.
    with tempfile.TemporaryDirectory(prefix=".render-sheet-", dir=engine.root) as folder:
        staging = Path(folder)
        rgb = (
            int(round(background[0] * 255)),
            int(round(background[1] * 255)),
            int(round(background[2] * 255)),
        )
        sheet = Image.new(
            "RGBA" if transparent else "RGB",
            dimensions,
            (*rgb, 0) if transparent else rgb,
        )
        draw = ImageDraw.Draw(sheet)
        outputs = []
        byte_count = 0
        for index, (view, parameters, target) in enumerate(prepared):
            check_time()
            tile = staging / f"tile_{index}.png"
            rendered = preview(engine, scene_id, str(tile), **parameters)
            byte_count += tile.stat().st_size
            if byte_count > engine.settings.max_file_bytes:
                raise ScissorsError(
                    "RESOURCE_LIMIT", "Render sheet view files exceed aggregate byte budget"
                )
            x = index % columns * tile_width + (tile_width - parameters["width"]) // 2
            y = (
                index // columns * (tile_height + 24)
                + 24
                + (tile_height - parameters["height"]) // 2
            )
            with Image.open(tile) as image:
                image.load()
                sheet.paste(image, (x, y))
            name = view.get("name", f"View {index + 1}")
            label = name
            while len(label) > 3 and draw.textlength(label) > tile_width - 12:
                label = label[:-4] + "..." if label.endswith("...") else label[:-1] + "..."
            draw.text(
                (index % columns * tile_width + 6, index // columns * (tile_height + 24) + 5),
                label,
                fill=(230, 235, 240) if sum(rgb) < 384 else (35, 45, 65),
            )
            outputs.append((tile, target))
            results.append(
                {
                    **rendered,
                    "path": str(target),
                    "name": name,
                    "label": label,
                    "index": index,
                    "width": parameters["width"],
                    "height": parameters["height"],
                    "preview_parameters": parameters,
                    "view_set": view.get("view_set"),
                    "head_name": view.get("head_name"),
                    "eye": view.get("eye"),
                    "eye_separation": view.get("eye_separation"),
                }
            )
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG")
        byte_count += len(buffer.getvalue())
        if byte_count > engine.settings.max_file_bytes:
            raise ScissorsError(
                "RESOURCE_LIMIT", "Render sheet and view files exceed aggregate byte budget"
            )
        sheet_path = staging / "sheet.png"
        atomic_write(sheet_path, buffer.getvalue(), engine.settings.max_file_bytes)
        outputs.append((sheet_path, destination))
        backends = sorted({value["renderer"] for value in results})
        result = {
            **artifact(sheet_path, "png"),
            "scene_id": scene_id,
            "path": str(destination),
            "view_count": len(results),
            "columns": columns,
            "rows": rows,
            "width": dimensions[0],
            "height": dimensions[1],
            "views": results,
            "renderer": backends[0] if len(backends) == 1 else "mixed",
            "renderers": backends,
            "work": {
                "rendered_pixels": total_pixels,
                "sheet_pixels": sheet_pixels,
                "face_instances": face_work,
                "total_file_bytes": byte_count,
            },
            "warnings": [
                "Stereo uses parallel cameras; "
                "headset runtime projection matrices are not simulated."
            ]
            if any(group["stereo"] for group in view_sets)
            else [],
        }
        if len(json_bytes(result)) > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Render sheet metadata exceeds result size limit")
        check_time()
        publish_files(engine, outputs, staging)
    return result
