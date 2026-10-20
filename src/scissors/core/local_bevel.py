"""Finite edge prisms for attributed local solid chamfers and round bevels."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, topology_mesh
from scissors.core.profiles import triangulate
from scissors.core.selections import edge_table, indices
from scissors.core.simplification import topology_signature


def edge_prism(
    obj: MeshObject,
    directed_edge: np.ndarray,
    normals: np.ndarray,
    width: float,
    segments: int,
    radius: bool,
    settings: Settings,
) -> MeshObject:
    a, b = np.asarray(obj.vertices)[directed_edge]
    direction = b - a
    length = float(np.linalg.norm(direction))
    direction /= length
    first = np.cross(normals[0], direction)
    second = np.cross(normals[1], -direction)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(first @ second, -1, 1))
    angle = float(np.arccos(cosine))
    if angle <= 1e-6 or np.pi - angle <= 1e-6:
        raise ValueError("Selected edge must have a nonzero dihedral angle")
    distance = width / np.tan(angle / 2) if radius else width
    center = distance * (first + second) / (1 + cosine)
    start, end = distance * first - center, distance * second - center
    arc_angle = float(np.arccos(np.clip(start @ end / np.linalg.norm(start) ** 2, -1, 1)))
    points = [np.zeros(3)]
    for t in np.linspace(0, 1, segments + 1):
        if segments == 1:
            point = distance * ((1 - t) * first + t * second)
        else:
            point = center + (
                np.sin((1 - t) * arc_angle) * start + np.sin(t * arc_angle) * end
            ) / np.sin(arc_angle)
        points.append(point)
    axis2 = second - cosine * first
    axis2 /= np.linalg.norm(axis2)
    flat = [[float(point @ first), float(point @ axis2)] for point in points]
    polygon, cap, rings, _ = triangulate(flat, [], settings)
    lower = a + polygon[:, :1] * first + polygon[:, 1:] * axis2
    vertices = np.concatenate([lower, lower + (b - a)])
    count = len(lower)
    faces = [*cap[:, ::-1].tolist(), *(cap + count).tolist()]
    for ring in rings:
        for i, x in enumerate(ring):
            y = ring[(i + 1) % len(ring)]
            faces.extend([[x, y, y + count], [x, y + count, x + count]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_volume:
        raise ScissorsError("GEOMETRY_ERROR", "Selected edge cannot form a stable bevel prism")
    cutter = MeshObject(id="edge_prism", name="Edge prism", vertices=[], faces=[])
    cutter.replace(mesh)
    if obj.vertex_colors is not None:
        t = np.clip((vertices - a) @ direction / length, 0, 1)
        colors = np.asarray(obj.vertex_colors)[directed_edge]
        cutter.vertex_colors = (
            np.rint((1 - t[:, None]) * colors[0] + t[:, None] * colors[1]).astype(int).tolist()
        )
    return cutter


def selected_edge_treatment(
    obj: MeshObject,
    selection: dict[str, Any],
    width: float,
    segments: int,
    mode: str,
    preserve_uv: bool,
    preserve_topology: bool,
    settings: Settings,
    radius: bool = False,
) -> tuple[MeshObject, list[int]]:
    if selection["object_id"] != obj.id or selection["geometry_revision"] != obj.geometry_revision:
        raise ValueError("Edge selection belongs to another geometry revision")
    if obj.kind != "mesh" or not topology_mesh(obj.geometry()).is_volume:
        raise ValueError("Selected-edge bevel requires a closed oriented positive solid")
    deadline = time.monotonic() + settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Selected-edge bevel timed out")

    if (
        len(obj.vertices) > settings.max_vertices
        or len(obj.faces) > settings.max_faces
        or len(obj.vertices) * 1024 + len(obj.faces) * 4096 > settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Selected-edge bevel exceeds memory budget")
    edges, _ = edge_table(obj)
    chosen = indices(selection["edge_indices"], len(edges))
    if not len(chosen):
        raise ValueError("Selected-edge bevel requires at least one edge")
    _, welded = np.unique(obj.vertices, axis=0, return_inverse=True)
    keys = {tuple(sorted(welded[edges[i]])): int(i) for i in chosen}
    faces = np.asarray(obj.faces)
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    owners: dict[int, list[int]] = {int(i): [] for i in chosen}
    for row, edge in enumerate(directed):
        if row % 1024 == 0:
            check_time()
        index = keys.get(tuple(sorted(welded[edge])))
        if index is not None:
            owners[index].append(row)
    source_mesh = obj.geometry()
    source_vertices = np.asarray(obj.vertices)
    normals = np.asarray(source_mesh.face_normals)
    batches: dict[str, list[MeshObject]] = {"difference": [], "union": []}
    for index in chosen:
        check_time()
        rows = owners[int(index)]
        if len(rows) != 2:
            raise ValueError("Selected edges must have exactly two adjacent faces")
        a, b = rows[0] // 3, rows[1] // 3
        edge = directed[rows[0]]
        opposite = next(v for v in faces[b] if welded[v] not in welded[edge])
        side = float(normals[a] @ (source_vertices[opposite] - source_vertices[edge[0]]))
        if abs(side) <= max(float(source_mesh.scale), 1) * 1e-12:
            raise ValueError("Selected edge is coplanar; select a geometric crease")
        convex = side < 0
        if mode != "both" and convex != (mode == "convex"):
            raise ValueError("Selected edge does not match mode; use mode='both' for mixed creases")
        cutter = edge_prism(obj, edge, normals[[a, b]], width, segments, radius, settings)
        batches["difference" if convex else "union"].append(cutter)
    target = obj.model_copy(deep=True)
    if not preserve_uv:
        target.uv = None
    for boolean_mode, cutters in batches.items():
        if not cutters:
            continue
        check_time()
        pairs = sum(len(c.faces) for c in cutters) * len(target.faces)
        if pairs > settings.max_geometry_pairs or pairs * 128 > settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Selected-edge boolean work exceeds budget")
        objects = [target, *cutters]
        payload = {
            "backend": "boolean_attributes",
            "mode": boolean_mode,
            "cut_uv": "planar",
            "uv_scale": 1,
            "source_ids": [str(i) for i in range(len(objects))],
            "objects": [
                {
                    k: getattr(value, k)
                    for k in ("vertices", "faces", "uv", "vertex_colors", "normals")
                }
                for value in objects
            ],
        }
        remaining = deadline - time.monotonic()
        check_time()
        data = run_backend(payload, settings.model_copy(update={"operation_timeout": remaining}))
        properties, triangles = np.asarray(data["properties"]), np.asarray(data["faces"])
        if len(properties) > settings.max_vertices or len(triangles) > settings.max_faces:
            raise ScissorsError("RESOURCE_LIMIT", "Selected-edge result exceeds geometry limits")
        target.replace(
            trimesh.Trimesh(vertices=properties[:, :3], faces=triangles, process=False),
            properties[:, 3:5].tolist(),
        )
        target.vertex_colors = (
            np.rint(properties[:, 5:9]).astype(int).tolist()
            if obj.vertex_colors is not None
            else None
        )
    if not topology_mesh(target.geometry()).is_volume:
        raise ScissorsError(
            "GEOMETRY_ERROR", "Selected-edge bevel did not preserve a positive solid"
        )
    if preserve_topology and topology_signature(target) != topology_signature(obj):
        raise ScissorsError("GEOMETRY_ERROR", "Selected-edge bevel changed topology")
    for key in ("primitive", "boundary_rings", "center_pairs", "instance_source", "tangents"):
        target.metadata.pop(key, None)
    target.metadata.pop("bevel_cleanup_tolerance", None)
    check_time()
    return target, chosen.tolist()
