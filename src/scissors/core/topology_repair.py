"""Bounded triangle-soup reconstruction and verification of repaired surfaces."""

import time
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, face_components, topology_mesh
from scissors.core.validation import validate_mesh


def repair_surface(
    vertices: list[list[float]],
    faces: list[list[int]],
    voxel_size: float,
    mode: str,
    shell_thickness: float | None,
    settings: Settings,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    if (mode == "shell") != (shell_thickness is not None):
        raise ValueError("Provide shell_thickness only for shell mode")
    if shell_thickness is not None and shell_thickness / voxel_size < 3 - 1e-12:
        raise ValueError("Shell thickness must be at least three voxel sizes")
    if len(vertices) > settings.max_vertices or len(faces) > settings.max_faces:
        raise ScissorsError("RESOURCE_LIMIT", "Repair input exceeds geometry limits")
    if len(vertices) * 384 + len(faces) * 768 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Repair input exceeds memory budget")
    points, triangles = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("Repair face index is out of range")
    # Remove exact geometric duplicates and degenerates before sampling. Unlike
    # Engine objects, this temporary soup may contain non-manifold edges.
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    triangles = inverse[triangles]
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(unique)]
    mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=False)
    mesh.update_faces(mesh.area_faces > 1e-14)
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError("Repair requires nondegenerate triangles")
    padding = 3 * voxel_size + (shell_thickness or 0) / 2
    origin = mesh.bounds[0] - padding - voxel_size * 0.1732050807568877
    dimensions = np.ceil((mesh.bounds[1] + padding - origin) / voxel_size) + 1
    samples = float(np.prod(dimensions))
    memory = samples * 128 + len(mesh.faces) * 768 + len(mesh.vertices) * 384
    if not np.isfinite(memory) or memory > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Repair grid exceeds memory budget")
    if mode == "solid" and samples * len(mesh.faces) > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Winding reconstruction exceeds pair budget")
    response = run_backend(
        {
            "backend": "topology_repair",
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "origin": origin.tolist(),
            "dimensions": dimensions.astype(int).tolist(),
            "voxel_size": voxel_size,
            "mode": mode,
            "shell_thickness": shell_thickness,
            "max_vertices": settings.max_vertices,
            "max_faces": settings.max_faces,
        },
        settings,
    )
    if "error" in response:
        code = "RESOURCE_LIMIT" if response["error"] == "geometry_limit" else "GEOMETRY_ERROR"
        raise ScissorsError(code, "Surface reconstruction failed", response)
    result = trimesh.Trimesh(**response, process=False)
    report = verify_surface(result, settings)
    return result, {
        "validation": report,
        "grid_samples": int(samples),
        "removed_input_faces": len(faces) - len(mesh.faces),
        "voxel_size": voxel_size,
        "mode": mode,
        "shell_thickness": shell_thickness,
        "warnings": [
            "Reconstruction changes geometry; features and gaps near voxel size can disappear.",
            "UV and vertex colors are cleared; regenerate UV and transfer textures as needed.",
            *(
                [
                    "Solid mode uses absolute winding above 0.5; opposite orientations can cancel. "
                    "Open surfaces do not define a reliable solid interior."
                ]
                if mode == "solid"
                else ["Shell mode forms a thick envelope on both sides of every input triangle."]
            ),
        ],
    }


def verify_surface(mesh: trimesh.Trimesh, settings: Settings) -> dict[str, Any]:
    """Reject invalid output, pinched vertices and nonadjacent triangle contacts."""
    from scissors.core.inspection import triangle_distances

    mesh = topology_mesh(mesh)
    obj = MeshObject(id="repair", name="Repair", vertices=[], faces=[])
    obj.replace(mesh)
    report = validate_mesh(obj, settings, require_closed=True)
    if not report["valid"]:
        raise ScissorsError("GEOMETRY_ERROR", "Repair output is not a closed manifold", report)
    deadline = time.monotonic() + settings.operation_timeout
    # Each closed vertex link must be a single fan. Edge counts alone miss
    # two otherwise closed components touching at one exact vertex.
    incident: list[list[int]] = [[] for _ in mesh.vertices]
    for index, face in enumerate(mesh.faces):
        for vertex in face:
            incident[int(vertex)].append(index)
    for vertex, indices in enumerate(incident):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Repair verification timed out")
        links = np.asarray([mesh.faces[i][mesh.faces[i] != vertex] for i in indices])
        if len(links) and len(face_components(len(links), _link_adjacency(links))) != 1:
            raise ScissorsError("GEOMETRY_ERROR", "Repair output has a pinched vertex")
    triangles = mesh.triangles
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    order = np.argsort(low[:, 0], kind="stable")
    sorted_low = low[order, 0]
    checked = 0
    candidates_checked = 0
    for offset, index in enumerate(order):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Repair verification timed out")
        # Sorted bounds avoid an unconditional quadratic pair enumeration.
        stop = np.searchsorted(sorted_low, high[index, 0], side="right")
        candidates = order[offset + 1 : stop]
        candidates_checked += len(candidates)
        if candidates_checked > settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Repair broad phase exceeds pair budget")
        for start in range(0, len(candidates), 1024):
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Repair verification timed out")
            ids = candidates[start : start + 1024]
            overlap = np.all(high[ids] >= low[index], axis=1)
            overlap &= np.all(low[ids] <= high[index], axis=1)
            ids = ids[overlap]
            if not len(ids):
                continue
            adjacent = np.any(
                mesh.faces[index, None, :, None] == mesh.faces[ids, None, :], axis=(1, 2)
            )
            ids = ids[~adjacent]
            checked += len(ids)
            if checked > settings.max_geometry_pairs:
                raise ScissorsError("RESOURCE_LIMIT", "Repair verification exceeds pair budget")
            if len(ids) and np.any(
                triangle_distances(
                    np.repeat(triangles[index][None], len(ids), axis=0), triangles[ids]
                )
                <= 0
            ):
                raise ScissorsError("GEOMETRY_ERROR", "Repair output self-intersects")
    report["properties"]["self_intersections"] = {
        "status": "passed",
        "checked_pairs": checked,
        "adjacent_faces_excluded": True,
    }
    report["properties"]["vertex_links_manifold"] = True
    return report


def _link_adjacency(links: np.ndarray) -> np.ndarray:
    owners: dict[int, int] = {}
    adjacency = []
    for index, link in enumerate(links):
        for vertex in link:
            if int(vertex) in owners:
                adjacency.append([owners[int(vertex)], index])
            else:
                owners[int(vertex)] = index
    return np.asarray(adjacency, dtype=np.int64).reshape(-1, 2)


def reconstruct_soup(payload: dict[str, Any]) -> dict[str, Any]:
    """Sample a winding field or an unsigned-distance envelope in a child worker."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkImplicitPolyDataDistance
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction

    dims = np.asarray(payload["dimensions"])
    spacing = payload["voxel_size"]
    origin = np.asarray(payload["origin"])
    # Local voxel coordinates reduce cancellation for translated geometry.
    vertices = (np.asarray(payload["vertices"]) - origin) / spacing
    faces = np.asarray(payload["faces"], dtype=np.int64)
    if payload["mode"] == "solid":
        triangles = vertices[faces]
        field = np.empty(int(np.prod(dims)))
        for start in range(0, len(field), 128):
            ids = np.arange(start, min(start + 128, len(field)))
            points = np.column_stack(
                (ids % dims[0], ids // dims[0] % dims[1], ids // (dims[0] * dims[1]))
            )
            winding = np.zeros(len(ids))
            for first in range(0, len(triangles), 128):
                directions = triangles[None, first : first + 128] - points[:, None, None]
                lengths = np.linalg.norm(directions, axis=-1)
                directions /= np.maximum(lengths[..., None], 1e-30)
                a, b, c = np.moveaxis(directions, 2, 0)
                numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
                denominator = 1 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
                winding += (2 * np.arctan2(numerator, denominator)).sum(axis=1)
            field[ids] = 0.5 - np.abs(winding / (4 * np.pi))
    else:
        points_vtk = vtkPoints()
        points_vtk.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(faces.reshape(-1), deep=True),
        )
        poly = vtkPolyData()
        poly.SetPoints(points_vtk)
        poly.SetPolys(cells)
        distance = vtkImplicitPolyDataDistance()
        distance.SetInput(poly)
        sample = vtkSampleFunction()
        sample.SetImplicitFunction(distance)
        sample.SetModelBounds(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        sample.SetSampleDimensions(*dims)
        sample.ComputeNormalsOff()
        sample.Update()
        field = np.abs(vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()))
        field -= payload["shell_thickness"] / spacing / 2
    grid = field.reshape(tuple(dims[::-1]))
    if not np.isfinite(field).all() or not np.any(field < 0):
        return {"error": "empty_interior", "hint": "Use finer resolution or shell mode"}
    if any(np.any(np.take(grid, edge, axis=axis) <= 0) for axis in range(3) for edge in (0, -1)):
        return {"error": "interior_at_grid_boundary"}
    image = vtkImageData()
    image.SetDimensions(*dims)
    image.GetPointData().SetScalars(numpy_to_vtk(field, deep=True))
    contour = vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0)
    contour.ComputeNormalsOff()
    contour.Update()
    output = contour.GetOutput()
    if (
        output.GetNumberOfPoints() > payload["max_vertices"]
        or output.GetNumberOfCells() > payload["max_faces"]
    ):
        return {"error": "geometry_limit"}
    mesh = trimesh.Trimesh(
        vertices=vtk_to_numpy(output.GetPoints().GetData()).astype(float) * spacing + origin,
        faces=vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3),
        process=True,
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    # FlyingEdges points toward decreasing field. Reverse globally to orient
    # solid exteriors outward while preserving the orientation of cavities.
    if mesh.volume < 0:
        mesh.invert()
    return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
