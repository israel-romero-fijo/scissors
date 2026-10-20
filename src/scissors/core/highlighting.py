"""Backend-independent, revision-bound x-ray selection overlays."""

import io
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject
from scissors.core.selections import FIELDS, select_geometry
from scissors.core.view import Frame, project


def clip_polygon(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Clip a polygon or segment against one linear half-space."""
    output = []
    for i, point in enumerate(points):
        previous, a, b = points[i - 1], distances[i - 1], distances[i]
        if (a >= 0) != (b >= 0):
            output.append(previous + (point - previous) * a / (a - b))
        if b >= 0:
            output.append(point)
    return np.asarray(output).reshape(-1, points.shape[1])


def prepare_highlights(
    engine: Engine,
    objects: list[MeshObject],
    selections: list[dict[str, Any]],
    frame: Frame,
    clip_axis: str | None,
    clip_position: float | None,
    clip_keep: str,
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, Any]]]:
    available = {obj.id: obj for obj in objects}
    primitives: list[tuple[str, np.ndarray]] = []
    summaries = []
    deadline = time.monotonic() + engine.settings.operation_timeout
    work = 0
    for selection in selections:
        object_id = selection["object_id"]
        if object_id not in available:
            raise ValueError("Highlighted object must be visible and included in the preview scene")
        obj = available[object_id]
        if selection["geometry_revision"] != obj.geometry_revision:
            raise ValueError("Highlight selection belongs to another geometry revision")
        kind = selection.get("kind", "faces")
        requested = selection.get(FIELDS[kind])
        if requested is None:
            raise ValueError(f"Highlight selection requires {FIELDS[kind]}")
        canonical = select_geometry(obj, kind, requested, None, 0, engine.settings)
        vertices = np.asarray(obj.vertices)
        if kind in {"faces", "components"}:
            raw = vertices[np.asarray(obj.faces)[canonical["face_indices"]]]
            draw_kind = "face"
        elif kind == "edges":
            raw = vertices[np.asarray(canonical["edges"], dtype=int).reshape(-1, 2)]
            draw_kind = "edge"
        elif kind == "boundaries":
            edges = [
                [a, loop["vertex_indices"][(i + 1) % len(loop["vertex_indices"])]]
                for loop in canonical["loops"]
                for i, a in enumerate(loop["vertex_indices"])
            ]
            raw = vertices[np.asarray(edges, dtype=int).reshape(-1, 2)]
            draw_kind = "edge"
        else:
            raw = vertices[canonical["vertex_indices"]][:, None, :]
            draw_kind = "vertex"
        if len(primitives) + len(raw) > min(
            engine.settings.max_geometry_pairs, engine.settings.max_memory_bytes // 1024
        ):
            raise ScissorsError("RESOURCE_LIMIT", "Highlight geometry exceeds work/memory budget")
        drawn = 0
        for points in raw:
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Highlight projection timed out")
            if clip_axis is not None and clip_position is not None:
                along = points[:, "xyz".index(clip_axis)].mean()
                if (along >= clip_position) != (clip_keep == "positive"):
                    continue
            _, depth = project(frame, points)
            points = clip_polygon(points, depth - frame.near)
            if not len(points):
                continue
            _, depth = project(frame, points)
            points = clip_polygon(points, frame.far - depth)
            if not len(points):
                continue
            screen, _ = project(frame, points)
            for axis, bound, sign in [
                (0, 0, 1),
                (0, frame.width - 1, -1),
                (1, 0, 1),
                (1, frame.height - 1, -1),
            ]:
                screen = clip_polygon(screen, sign * (screen[:, axis] - bound))
                if not len(screen):
                    break
            if not len(screen):
                continue
            span = screen.max(axis=0) - screen.min(axis=0) + 1
            work += int(np.prod(span))
            if work > engine.settings.max_geometry_pairs * 64:
                raise ScissorsError("RESOURCE_LIMIT", "Highlight pixel work exceeds budget")
            primitives.append((draw_kind, screen))
            drawn += 1
        summaries.append(
            {
                "object_id": obj.id,
                "kind": kind,
                "count": canonical["count"],
                "drawn_primitives": drawn,
            }
        )
    return primitives, summaries


def draw_highlights(data: bytes, primitives: list[tuple[str, np.ndarray]], timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    base = Image.open(io.BytesIO(data)).convert("RGBA")
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    for index, (kind, points) in enumerate(primitives):
        if index % 256 == 0 and time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Highlight drawing timed out")
        xy = [tuple(point) for point in points]
        if kind == "face" and len(xy) >= 3:
            draw.polygon(xy, fill=(255, 80, 0, 96), outline=(255, 80, 0, 255), width=2)
        elif kind == "edge" and len(xy) >= 2:
            draw.line(xy, fill=(255, 80, 0, 255), width=3)
        elif kind == "vertex":
            x, y = xy[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 80, 0, 255))
    output = io.BytesIO()
    Image.alpha_composite(base, layer).save(output, format="PNG")
    return output.getvalue()
