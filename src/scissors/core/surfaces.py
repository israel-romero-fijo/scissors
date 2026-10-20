"""Seam-aware surface thickness, offsets and attributed plane clipping."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, topology_mesh


@dataclass
class SurfaceMesh:
    mesh: trimesh.Trimesh
    uv: list[list[float]] | None
    colors: list[list[int]] | None
    normals: list[list[float]] | None = None
    details: dict[str, Any] = field(default_factory=dict)


def boundaries(vertices: np.ndarray, faces: np.ndarray) -> list[list[tuple[int, int]]]:
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    welded = inverse[edges]
    _, mapping, counts = np.unique(
        np.sort(welded, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ScissorsError("GEOMETRY_ERROR", "Surface contains nonmanifold edges")
    selected = edges[counts[mapping] == 1]
    following = {}
    for a, b in selected:
        key = int(inverse[a])
        if key in following:
            raise ScissorsError("GEOMETRY_ERROR", "Surface boundary branches")
        following[key] = (int(a), int(b))
    loops = []
    while following:
        start = min(following)
        current, loop = start, []
        while True:
            if current not in following:
                raise ScissorsError("GEOMETRY_ERROR", "Surface boundary is not a closed loop")
            a, b = following.pop(current)
            loop.append((a, b))
            current = int(inverse[b])
            if current == start:
                break
        loops.append(loop)
    return loops


def offset_vertices(
    obj: MeshObject, distance: float, method: str, miter_limit: float
) -> np.ndarray:
    vertices, inverse = np.unique(np.asarray(obj.vertices), axis=0, return_inverse=True)
    mesh = obj.geometry()
    faces = inverse[np.asarray(obj.faces)]
    matrices, vectors = np.zeros((len(vertices), 3, 3)), np.zeros_like(vertices)
    normals = mesh.face_normals
    if not mesh.is_winding_consistent:
        raise ScissorsError("GEOMETRY_ERROR", "Surface winding must be consistent before offset")
    for corner in range(3):
        np.add.at(matrices, faces[:, corner], normals[:, :, None] * normals[:, None, :])
        np.add.at(vectors, faces[:, corner], normals)
    if method == "normal":
        direction = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-15)
    else:
        direction = np.einsum("nij,nj->ni", np.linalg.pinv(matrices, rcond=1e-10), vectors)
        if np.any(np.linalg.norm(direction, axis=1) > miter_limit):
            raise ScissorsError("GEOMETRY_ERROR", "Offset exceeds the sharp-corner miter limit")
    if np.any(np.linalg.norm(direction, axis=1) < 1e-12):
        raise ScissorsError("GEOMETRY_ERROR", "Offset direction is undefined at a vertex")
    updated = vertices + direction * distance
    result = trimesh.Trimesh(vertices=updated, faces=faces, process=False)
    if np.any(np.sum(result.face_normals * normals, axis=1) <= 0):
        raise ScissorsError("GEOMETRY_ERROR", "Offset collapses or reverses a face")
    if topology_mesh(mesh).is_watertight and result.volume <= 0:
        raise ScissorsError("GEOMETRY_ERROR", "Offset collapses or inverts a closed volume")
    return updated[inverse]


def offset(obj: MeshObject, distance: float, method: str, miter_limit: float) -> SurfaceMesh:
    boundaries(np.asarray(obj.vertices), np.asarray(obj.faces))
    vertices = offset_vertices(obj, distance, method, miter_limit)
    return SurfaceMesh(
        trimesh.Trimesh(vertices=vertices, faces=obj.faces, process=False),
        obj.uv,
        obj.vertex_colors,
        details={"distance": distance, "method": method},
    )


def shell(
    obj: MeshObject,
    thickness: float,
    position: float,
    method: str,
    miter_limit: float,
    settings: Settings,
    opening_faces: np.ndarray | None = None,
) -> SurfaceMesh:
    original, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
    full_faces = faces
    if opening_faces is not None:
        keep = np.ones(len(faces), dtype=bool)
        keep[opening_faces] = False
        faces = faces[keep]
        if not len(faces):
            raise ValueError("Opening cannot remove every source face")
    loops = boundaries(original, faces)
    n = len(original)
    edges = sum(len(loop) for loop in loops)
    if 2 * n + edges * 4 > settings.max_vertices or 2 * len(faces) + edges * 2 > settings.max_faces:
        raise ScissorsError("RESOURCE_LIMIT", "Shell exceeds geometry limits")
    outer = offset_vertices(obj, thickness * (1 + position) / 2, method, miter_limit)
    inner = offset_vertices(obj, thickness * (position - 1) / 2, method, miter_limit)
    vertices = np.concatenate([outer, inner]).tolist()
    triangles = np.concatenate([faces, faces[:, ::-1] + n]).tolist()
    uv = [*obj.uv, *obj.uv] if obj.uv is not None else None
    colors = [*obj.vertex_colors, *obj.vertex_colors] if obj.vertex_colors is not None else None
    for loop in loops:
        lengths = [float(np.linalg.norm(original[b] - original[a])) for a, b in loop]
        cumulative = np.concatenate([[0], np.cumsum(lengths)]) / sum(lengths)
        for i, (a, b) in enumerate(loop):
            start = len(vertices)
            vertices.extend(
                [outer[a].tolist(), inner[a].tolist(), inner[b].tolist(), outer[b].tolist()]
            )
            triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
            if uv is not None:
                u0, u1 = float(cumulative[i]), float(cumulative[i + 1])
                uv.extend([[u0, 1], [u0, 0], [u1, 0], [u1, 1]])
            if colors is not None:
                assert obj.vertex_colors is not None
                colors.extend(
                    [
                        obj.vertex_colors[a],
                        obj.vertex_colors[a],
                        obj.vertex_colors[b],
                        obj.vertex_colors[b],
                    ]
                )
    return SurfaceMesh(
        trimesh.Trimesh(vertices=vertices, faces=triangles, process=False),
        uv,
        colors,
        details={
            "thickness": thickness,
            "offset": position,
            "boundary_walls": edges,
            "method": method,
            "opening_face_indices": opening_faces.tolist() if opening_faces is not None else [],
            "rim_face_indices": list(range(2 * len(faces), len(triangles))),
            "source_face_count": len(full_faces),
        },
    )


def cap_triangulation(value: dict[str, Any]) -> dict[str, Any]:
    """VTK contour triangulation in the bounded child, including multiple holes."""
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    points = vtkPoints()
    points.SetDataTypeToDouble()
    for x, y in value["points"]:
        points.InsertNextPoint(x, y, 0)
    lines = vtkCellArray()
    for loop in value["loops"]:
        for i, a in enumerate(loop):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(a)
            lines.InsertCellPoint(loop[(i + 1) % len(loop)])
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetLines(lines)
    polygons = vtkCellArray()
    success = vtkContourTriangulator.TriangulateContours(
        data, 0, lines.GetNumberOfCells(), polygons, [0, 0, 1]
    )
    if not success:
        raise ValueError("Plane cap contours could not be triangulated")
    ids, faces = vtkIdList(), []
    polygons.InitTraversal()
    while polygons.GetNextCell(ids):
        if ids.GetNumberOfIds() != 3:
            raise ValueError("Cap backend returned non-triangle geometry")
        faces.append([ids.GetId(i) for i in range(3)])
    return {"faces": faces}


def plane_clip(
    obj: MeshObject,
    normal: np.ndarray,
    point: np.ndarray,
    capped: bool,
    tolerance: float,
    settings: Settings,
) -> SurfaceMesh:
    from scissors.core.profiles import area, inside

    deadline = time.monotonic() + settings.operation_timeout

    def timed() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Plane clipping timed out")

    geometry = obj.geometry()
    if not geometry.is_winding_consistent:
        raise ScissorsError("GEOMETRY_ERROR", "Plane clipping requires consistent winding")
    if capped and topology_mesh(geometry).is_watertight and geometry.volume <= 0:
        raise ScissorsError("GEOMETRY_ERROR", "Flip inward closed geometry before capped clipping")
    normal = normal / max(float(np.linalg.norm(normal)), 1e-30)
    if np.linalg.norm(normal) < 0.5:
        raise ValueError("Plane normal must be nonzero")
    vertices = np.asarray(obj.vertices)
    distances = (vertices - point) @ normal
    if np.min(distances) >= -tolerance:
        return SurfaceMesh(
            obj.geometry(),
            obj.uv,
            obj.vertex_colors,
            obj.normals,
            {"cap_faces": 0, "unchanged": True},
        )
    if np.max(distances) <= tolerance:
        raise ScissorsError(
            "GEOMETRY_ERROR", "Plane removes the mesh or leaves only coplanar faces"
        )
    uv = np.asarray(obj.uv) if obj.uv is not None else np.zeros((len(vertices), 2))
    colors = (
        np.asarray(obj.vertex_colors)
        if obj.vertex_colors is not None
        else np.ones((len(vertices), 4)) * 255
    )
    attrs = np.column_stack([vertices, obj.normals, uv, colors])
    for snap_index in np.flatnonzero(np.abs(distances) <= tolerance):
        attrs[snap_index, :3] -= distances[snap_index] * normal
        distances[snap_index] = 0
    cache: dict[tuple, np.ndarray] = {}
    output: list[np.ndarray] = []
    faces: list[list[int]] = []
    for face_index, face in enumerate(obj.faces):
        if face_index % 128 == 0:
            timed()
        polygon = []
        previous, previous_d = attrs[face[-1]], distances[face[-1]]
        for index in face:
            current, current_d = attrs[index], distances[index]
            if (current_d >= 0) != (previous_d >= 0):
                t = previous_d / (previous_d - current_d)
                value = previous + t * (current - previous)
                key = tuple(sorted([tuple(previous[:3]), tuple(current[:3])]))
                if key not in cache:
                    a, b = np.asarray(key[0]), np.asarray(key[1])
                    da, db = float((a - point) @ normal), float((b - point) @ normal)
                    cache[key] = a + (b - a) * da / (da - db)
                if previous_d == 0:
                    value = previous.copy()
                elif current_d == 0:
                    value = current.copy()
                else:
                    value[:3] = cache[key]
                polygon.append(value)
            if current_d >= 0:
                polygon.append(current)
            previous, previous_d = current, current_d
        for index in range(1, len(polygon) - 1):
            triangle = np.asarray([polygon[0], polygon[index], polygon[index + 1]])
            if (
                np.linalg.norm(
                    np.cross(triangle[1, :3] - triangle[0, :3], triangle[2, :3] - triangle[0, :3])
                )
                <= 1e-14
            ):
                continue
            start = len(output)
            output.extend(triangle)
            faces.append([start, start + 1, start + 2])
    if not faces:
        raise ScissorsError("GEOMETRY_ERROR", "Plane leaves no mesh triangles")
    data = np.asarray(output)
    triangles = np.asarray(faces)
    cap_faces = 0
    if capped:
        loops = boundaries(data[:, :3], triangles)
        cut_loops = []
        for loop in loops:
            ids = [a for a, _ in loop]
            if np.max(np.abs((data[ids, :3] - point) @ normal)) <= tolerance * 2:
                cut_loops.append(ids[::-1])
            elif any(abs(float((data[i, :3] - point) @ normal)) <= tolerance * 2 for i in ids):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Cut crosses an open boundary; use capped=false"
                )
        if cut_loops:
            outward = -normal
            u = np.cross(np.eye(3)[int(np.argmin(np.abs(outward)))], outward)
            u /= np.linalg.norm(u)
            basis = np.vstack([u, np.cross(outward, u)])
            indices = [index for loop in cut_loops for index in loop]
            flat = (data[indices, :3] - point) @ basis.T
            if 4 * len(flat) ** 2 > settings.max_geometry_pairs:
                raise ScissorsError("RESOURCE_LIMIT", "Plane cap exceeds contour workload budget")
            rings, cursor = [], 0
            for cut_loop in cut_loops:
                rings.append(list(range(cursor, cursor + len(cut_loop))))
                cursor += len(cut_loop)
            for ring in rings:
                depth = sum(
                    inside(flat[ring[0]], flat[other]) for other in rings if other is not ring
                )
                if (area(flat[ring]) > 0) != (depth % 2 == 0):
                    ring.reverse()
            result = run_backend(
                {"backend": "plane_cap", "points": flat.tolist(), "loops": rings}, settings
            )
            if not result["faces"]:
                raise ScissorsError("GEOMETRY_ERROR", "Plane cap triangulation is empty")
            caps = []
            steps = 0
            for triangle in result["faces"]:
                pending = [triangle]
                while pending:
                    steps += 1
                    if steps % 128 == 0:
                        timed()
                    if steps > settings.max_geometry_pairs:
                        raise ScissorsError(
                            "RESOURCE_LIMIT", "Cap boundary refinement exceeds budget"
                        )
                    tri = pending.pop()
                    for i in range(3):
                        a, b, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
                        delta = flat[b] - flat[a]
                        length2 = float(delta @ delta)
                        t = (flat - flat[a]) @ delta / max(length2, 1e-30)
                        on = (
                            np.linalg.norm(flat - flat[a] - t[:, None] * delta, axis=1) <= tolerance
                        )
                        selected = np.flatnonzero(on & (t > 1e-10) & (t < 1 - 1e-10))
                        if len(selected):
                            chain = [a, *selected[np.argsort(t[selected])].tolist(), b]
                            pending.extend(
                                [[chain[j], chain[j + 1], c] for j in range(len(chain) - 1)]
                            )
                            break
                    else:
                        caps.append(tri)
            cap_data = data[indices].copy()
            cap_data[:, 3:6] = outward
            cap_data[:, 6:8] = (flat - flat.min(axis=0)) / np.maximum(np.ptp(flat, axis=0), 1e-12)
            start = len(data)
            caps_array = np.asarray(caps)
            # Enforce cap orientation independently of the triangulator's output winding.
            edge_a = flat[caps_array[:, 1]] - flat[caps_array[:, 0]]
            edge_b = flat[caps_array[:, 2]] - flat[caps_array[:, 0]]
            signs = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
            caps_array[signs < 0] = caps_array[signs < 0, ::-1]
            triangles = np.concatenate([triangles, caps_array + start])
            data = np.concatenate([data, cap_data])
            cap_faces = len(caps)
    timed()
    data[:, 3:6] /= np.maximum(np.linalg.norm(data[:, 3:6], axis=1)[:, None], 1e-15)
    unique, inverse = np.unique(data, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=unique[:, :3], faces=inverse[triangles], process=False)
    return SurfaceMesh(
        mesh,
        unique[:, 6:8].tolist() if obj.uv is not None else None,
        np.clip(np.rint(unique[:, 8:12]), 0, 255).astype(int).tolist()
        if obj.vertex_colors is not None
        else None,
        unique[:, 3:6].tolist(),
        {"cap_faces": cap_faces, "unchanged": False},
    )
