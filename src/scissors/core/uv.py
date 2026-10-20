import importlib.util
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend, run_xatlas
from scissors.core.mesh import MeshObject


def generate_uv(obj: MeshObject, method: str, settings: Settings) -> None:
    mesh = obj.geometry()
    vertices = np.asarray(mesh.vertices)
    if method == "auto":
        method = {
            "sphere": "spherical",
            "cylinder": "cylindrical",
            "cone": "cylindrical",
            "plane": "planar",
        }.get(str(obj.metadata.get("primitive", "custom")), "box")
    if method == "xatlas":
        if importlib.util.find_spec("xatlas") is None:
            raise ScissorsError(
                "BACKEND_UNAVAILABLE",
                "Repair the Scissors installation to restore xatlas",
                {"backend": "xatlas", "requirement": "Scissors standard installation"},
            )
        result = run_xatlas(mesh, settings)
        obj.replace(
            trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False),
            result["uv"],
        )
        return
    if method == "box":
        # Per-triangle vertices preserve UV seams; topology diagnostics weld exact positions.
        points = vertices[np.asarray(mesh.faces)].reshape(-1, 3)
        projected = np.empty((len(points), 2))
        bounds = mesh.bounds
        span = np.maximum(bounds[1] - bounds[0], 1e-12)
        for i, normal in enumerate(mesh.face_normals):
            axis = int(np.argmax(np.abs(normal)))
            axes = [a for a in range(3) if a != axis]
            projected[i * 3 : i * 3 + 3] = (
                points[i * 3 : i * 3 + 3, axes] - bounds[0, axes]
            ) / span[axes]
        obj.replace(
            trimesh.Trimesh(
                vertices=points, faces=np.arange(len(points)).reshape(-1, 3), process=False
            ),
            projected.tolist(),
        )
        return
    centered = vertices - (mesh.bounds[0] + mesh.bounds[1]) / 2
    if method == "planar":
        uv = (vertices[:, :2] - mesh.bounds[0, :2]) / np.maximum(mesh.extents[:2], 1e-12)
    else:
        u = np.arctan2(centered[:, 1], centered[:, 0]) / (2 * np.pi) + 0.5
        if method == "spherical":
            radius = np.maximum(np.linalg.norm(centered, axis=1), 1e-12)
            v = np.arccos(np.clip(centered[:, 2] / radius, -1, 1)) / np.pi
        else:
            v = (vertices[:, 2] - mesh.bounds[0, 2]) / max(mesh.extents[2], 1e-12)
        uv = np.column_stack([u, v])
        # Split triangles at the seam so interpolation does not cross the texture.
        points = vertices[mesh.faces].reshape(-1, 3)
        per_face = uv[mesh.faces].copy()
        for i, coords in enumerate(per_face):
            if np.ptp(coords[:, 0]) > 0.5:
                coords[coords[:, 0] < 0.5, 0] += 1
            if method == "cylindrical" and abs(mesh.face_normals[i, 2]) > 0.99:
                coords[:] = (points[i * 3 : i * 3 + 3, :2] - mesh.bounds[0, :2]) / np.maximum(
                    mesh.extents[:2], 1e-12
                )
        obj.replace(
            trimesh.Trimesh(
                vertices=points, faces=np.arange(len(points)).reshape(-1, 3), process=False
            ),
            per_face.reshape(-1, 2).tolist(),
        )
        return
    obj.uv = uv.tolist()


def unwrap_atlas(
    obj: MeshObject,
    settings: Settings,
    face_groups: list[list[int]] | None,
    density_weights: list[float] | None,
    resolution: int,
    padding: int,
) -> dict[str, Any]:
    if importlib.util.find_spec("xatlas") is None:
        raise ScissorsError(
            "BACKEND_UNAVAILABLE",
            "Repair the Scissors installation to restore xatlas for atlas unwrapping",
            {"backend": "xatlas", "requirement": "Scissors standard installation"},
        )
    groups = face_groups if face_groups is not None else [list(range(len(obj.faces)))]
    flat = [index for group in groups for index in group]
    if (
        not groups
        or any(not group for group in groups)
        or sorted(flat) != list(range(len(obj.faces)))
    ):
        raise ValueError("Face groups must partition every face exactly once")
    weights = density_weights if density_weights is not None else [1.0] * len(groups)
    if len(weights) != len(groups):
        raise ValueError("Provide one density weight per face group")
    if len(obj.faces) * 3 > settings.max_vertices:
        raise ScissorsError("RESOURCE_LIMIT", "Atlas seam vertices exceed configured limit")
    if len(obj.faces) * 2304 + resolution**2 * 16 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Atlas exceeds estimated memory budget")
    result = run_backend(
        {
            "backend": "atlas",
            "vertices": obj.vertices,
            "faces": obj.faces,
            "groups": groups,
            "weights": weights,
            "resolution": resolution,
            "padding": padding,
        },
        settings,
    )
    obj.replace(
        trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False),
        result["uv"],
    )
    return {
        **obj.summary(),
        "chart_count": result["chart_count"],
        "atlas_width": result["width"],
        "atlas_height": result["height"],
        "distortion": uv_distortion(obj),
        "warnings": ["Group boundaries force cuts; xatlas may add cuts within each group."],
    }


def uv_distortion(obj: MeshObject) -> dict[str, Any]:
    """Report per-triangle anisotropy, independent of uniform texture density."""
    if obj.uv is None:
        return {"has_uv": False}
    points = np.asarray(obj.vertices)[obj.faces]
    uv = np.asarray(obj.uv)[obj.faces]
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    length = np.linalg.norm(a, axis=1)
    tangent = a / np.maximum(length[:, None], 1e-15)
    bx = np.sum(b * tangent, axis=1)
    by = np.linalg.norm(b - bx[:, None] * tangent, axis=1)
    local = np.zeros((len(points), 2, 2))
    local[:, 0, 0], local[:, 0, 1], local[:, 1, 1] = length, bx, by
    delta = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=2)
    jacobian = delta @ np.linalg.inv(local)
    values = np.linalg.svd(jacobian, compute_uv=False)
    degenerate = values[:, 1] <= 1e-12
    stretch = values[~degenerate, 0] / values[~degenerate, 1]
    return {
        "has_uv": True,
        "degenerate_triangles": int(np.count_nonzero(degenerate)),
        "max_anisotropy": float(stretch.max()) if len(stretch) else None,
        "p95_anisotropy": float(np.percentile(stretch, 95)) if len(stretch) else None,
    }


def build_atlas(payload: dict[str, Any]) -> dict[str, Any]:
    import xatlas

    vertices, faces = np.asarray(payload["vertices"]), np.asarray(payload["faces"])
    atlas = xatlas.Atlas()
    sources = []
    for group, weight in zip(payload["groups"], payload["weights"], strict=True):
        used, inverse = np.unique(faces[group], return_inverse=True)
        sources.append(vertices[used])
        # A weight is relative texel area, so lengths scale by its square root.
        atlas.add_mesh(
            (vertices[used] * np.sqrt(weight)).astype(np.float32),
            inverse.reshape(-1, 3).astype(np.uint32),
        )
    options = xatlas.PackOptions()
    options.resolution = payload["resolution"]
    options.padding = payload["padding"]
    options.bilinear = True
    atlas.generate(pack_options=options)
    if atlas.atlas_count != 1:
        raise ValueError("Packing produced multiple images; increase resolution or reduce padding")
    points, triangles, coordinates = [], [], []
    offset = 0
    for index, source in enumerate(sources):
        mapping, indices, uv = atlas[index]
        points.extend(source[mapping].tolist())
        triangles.extend((indices + offset).tolist())
        coordinates.extend(uv.tolist())
        offset += len(mapping)
    return {
        "vertices": points,
        "faces": triangles,
        "uv": coordinates,
        "chart_count": atlas.chart_count,
        "width": atlas.width,
        "height": atlas.height,
    }
