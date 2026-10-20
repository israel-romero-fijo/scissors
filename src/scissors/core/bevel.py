"""Bounded rolling-kernel edge treatment of existing closed solids."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.boolean_attributes import planar_uv
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, topology_mesh
from scissors.core.normals import split_normals
from scissors.core.simplification import topology_signature


def morphology(value: dict[str, Any]) -> dict[str, Any]:
    """Run native Minkowski operations in the time/memory limited geometry worker."""
    import manifold3d as m

    solid = m.Manifold(
        m.Mesh64(
            np.asarray(value["vertices"], dtype=np.float64),
            np.asarray(value["faces"], dtype=np.uint64),
        )
    )
    if solid.status() != m.Error.NoError or solid.is_empty():
        raise ValueError("Edge treatment requires a nonempty manifold solid")
    if value["segments"] == 1:
        kernel = m.Manifold.hull_points(np.asarray(value["directions"]) * value["width"])
    else:
        kernel = m.Manifold.sphere(value["width"], circular_segments=4 * value["segments"])
    if kernel.status() != m.Error.NoError or kernel.is_empty():
        raise ValueError("Cannot construct a volumetric edge kernel")
    # Manifold's difference reflects the swept solid. Reflect the kernel first
    # so erosion and dilation use the same shape even when it is asymmetric.
    erosion_kernel = kernel.scale([-1, -1, -1])
    steps = {
        "convex": ["erode", "dilate"],
        "concave": ["dilate", "erode"],
        "both": ["erode", "dilate", "dilate", "erode"],
    }[value["mode"]]
    for step in steps:
        pairs = solid.num_tri() * kernel.num_tri()
        if pairs > value["max_pairs"] or pairs * 128 > value["max_memory"]:
            return {"error": "RESOURCE_LIMIT", "message": "Minkowski work exceeds budget"}
        solid = (
            solid.minkowski_difference(erosion_kernel)
            if step == "erode"
            else solid.minkowski_sum(kernel)
        )
        if solid.status() != m.Error.NoError or solid.is_empty():
            return {
                "error": "GEOMETRY_ERROR",
                "message": "Edge width consumes the solid or produces invalid geometry",
            }
        if solid.num_tri() > value["max_faces"] or solid.num_vert() > value["max_vertices"]:
            return {"error": "RESOURCE_LIMIT", "message": "Edge result exceeds geometry limits"}
    # Boolean history can keep nearly coincident coplanar vertices. Collapse
    # them in Manifold, which preserves a closed solid, before splitting normals.
    scale = float(np.ptp(np.asarray(value["vertices"]), axis=0).max())
    cleanup = min(value["width"] * 1e-6, scale * 1e-10)
    solid = solid.as_original().simplify(cleanup)
    if solid.status() != m.Error.NoError or solid.is_empty():
        raise ValueError("Edge cleanup did not preserve a solid")
    mesh = solid.to_mesh64()
    return {
        "vertices": mesh.vert_properties[:, :3].tolist(),
        "faces": mesh.tri_verts.tolist(),
        "cleanup_tolerance": cleanup,
    }


def edge_treatment(
    obj: MeshObject,
    width: float,
    segments: int,
    mode: str,
    preserve_uv: bool,
    preserve_topology: bool,
    settings: Settings,
) -> MeshObject:
    source = topology_mesh(obj.geometry())
    if obj.kind != "mesh" or not source.is_volume:
        raise ValueError("Edge treatment requires a closed, consistently oriented positive solid")
    if len(source.faces) * 1024 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Edge input exceeds working memory budget")
    response = run_backend(
        {
            "backend": "bevel",
            "vertices": source.vertices.tolist(),
            "faces": source.faces.tolist(),
            "directions": np.unique(np.round(source.face_normals, 12), axis=0).tolist(),
            "width": width,
            "segments": segments,
            "mode": mode,
            "max_pairs": settings.max_geometry_pairs,
            "max_memory": settings.max_memory_bytes,
            "max_faces": settings.max_faces,
            "max_vertices": settings.max_vertices,
        },
        settings,
    )
    if "error" in response:
        raise ScissorsError(response["error"], response["message"])
    cleanup = response.pop("cleanup_tolerance")
    mesh = trimesh.Trimesh(**response, process=False)
    if not mesh.is_volume:
        raise ScissorsError("GEOMETRY_ERROR", "Edge result is not a closed positive solid")
    target = obj.model_copy(deep=True)
    target.replace(mesh)
    if preserve_topology and topology_signature(obj) != topology_signature(target):
        raise ScissorsError("GEOMETRY_ERROR", "Edge treatment changed topology")
    corners = mesh.triangles.reshape(-1, 3)
    if len(corners) * 768 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Edge attribute buffers exceed memory budget")
    uv = planar_uv(mesh.triangles, 1).reshape(-1, 2)
    colors = None
    has_uv = preserve_uv and obj.uv is not None
    if has_uv or obj.vertex_colors is not None:
        pairs = len(corners) * len(obj.faces)
        if pairs > settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Edge attribute projection exceeds pair budget")
        triangles = obj.geometry().triangles
        source_normals = obj.geometry().face_normals
        corner_normals = np.repeat(mesh.face_normals, 3, axis=0)
        if obj.vertex_colors is not None:
            colors = np.empty((len(corners), 4), dtype=np.uint8)
        faces = np.asarray(obj.faces)
        old_uv = np.asarray(obj.uv) if has_uv else None
        old_colors = np.asarray(obj.vertex_colors) if colors is not None else None
        deadline = time.monotonic() + settings.operation_timeout
        for index, point in enumerate(corners):
            if time.monotonic() > deadline:
                raise ScissorsError("RESOURCE_LIMIT", "Edge attribute projection timed out")
            closest = trimesh.triangles.closest_point(
                triangles, np.broadcast_to(point, (len(triangles), 3))
            )
            distances = np.sum((closest - point) ** 2, axis=1)
            candidates = np.flatnonzero(
                distances <= distances.min() + max(float(source.scale) ** 2 * 1e-24, 1e-30)
            )
            # At UV seams choose the closest face whose normal best matches this new corner.
            face = int(candidates[np.argmax(source_normals[candidates] @ corner_normals[index])])
            weights = trimesh.triangles.points_to_barycentric(
                triangles[face : face + 1], closest[face : face + 1]
            )[0]
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum()
            if old_uv is not None:
                uv[index] = weights @ old_uv[faces[face]]
            if colors is not None and old_colors is not None:
                colors[index] = np.clip(np.rint(weights @ old_colors[faces[face]]), 0, 255)
    target.replace(
        trimesh.Trimesh(
            vertices=corners, faces=np.arange(len(corners)).reshape(-1, 3), process=False
        ),
        uv.tolist(),
    )
    target.vertex_colors = colors.tolist() if colors is not None else None
    for key in ["primitive", "boundary_rings", "center_pairs", "instance_source", "tangents"]:
        target.metadata.pop(key, None)
    split_normals(target, 0 if segments == 1 else 60, settings)
    target.metadata["bevel_cleanup_tolerance"] = cleanup
    return target
