"""CPU surface editing and bounded signed-distance reconstruction."""

from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, compute_normals, topology_mesh
from scissors.core.simplification import available


def brush_weights(vertices: np.ndarray, center: list[float], radius: float) -> np.ndarray:
    t = np.clip(1 - np.linalg.norm(vertices - center, axis=1) / radius, 0, 1)
    return t * t * (3 - 2 * t)


def edit_surface(
    obj: MeshObject,
    center: list[float],
    radius: float,
    mode: str,
    displacement: list[float],
    strength: float,
    iterations: int,
) -> dict[str, Any]:
    original = np.asarray(obj.vertices)
    # Work on exact welded positions so edits cannot open a UV seam.
    mesh = topology_mesh(obj.geometry())
    vertices = np.asarray(mesh.vertices).copy()
    _, inverse = np.unique(original, axis=0, return_inverse=True)
    weights = brush_weights(vertices, center, radius)
    affected = int(np.count_nonzero(weights))
    if not affected:
        raise ScissorsError("GEOMETRY_ERROR", "Brush does not contain any vertices")
    if mode == "move":
        vertices += weights[:, None] * np.asarray(displacement)
    elif mode == "inflate":
        vertices += weights[:, None] * strength * compute_normals(mesh)
    else:
        edges = np.asarray(mesh.edges_unique)
        counts = np.bincount(edges.reshape(-1), minlength=len(vertices))
        for _ in range(iterations):
            sums = np.zeros_like(vertices)
            np.add.at(sums, edges[:, 0], vertices[edges[:, 1]])
            np.add.at(sums, edges[:, 1], vertices[edges[:, 0]])
            average = sums / np.maximum(counts[:, None], 1)
            delta = average - vertices
            # Sharpen is an inverse Laplacian. Its strength is explicitly bounded.
            vertices += weights[:, None] * strength * delta
    obj.replace(trimesh.Trimesh(vertices=vertices[inverse], faces=obj.faces, process=False), obj.uv)
    obj.metadata.pop("primitive", None)
    return {
        **obj.summary(),
        "affected_vertices": affected,
        "warnings": [
            "Vertex brush; subdivide first for small details. Check for self-intersections."
        ],
    }


def reconstruct(
    meshes: list[trimesh.Trimesh],
    voxel_size: float,
    blend_radius: float,
    settings: Settings,
) -> trimesh.Trimesh:
    if not available():
        raise ScissorsError(
            "BACKEND_UNAVAILABLE",
            "Repair the Scissors installation to restore VTK for signed-distance surfaces",
            {"backend": "vtk", "requirement": "Scissors standard installation"},
        )
    meshes = [topology_mesh(mesh) for mesh in meshes]
    if not all(mesh.is_volume for mesh in meshes):
        raise ScissorsError("GEOMETRY_ERROR", "Reconstruction requires closed, oriented volumes")
    low = np.min([m.bounds[0] for m in meshes], axis=0)
    high = np.max([m.bounds[1] for m in meshes], axis=0)
    padding = voxel_size * 3 + blend_radius * len(meshes)
    low -= padding
    high += padding
    dimensions = np.ceil((high - low) / voxel_size) + 1
    # Estimate before conversion to integers and before allocating native buffers.
    samples = float(np.prod(dimensions))
    memory = samples * 96 + sum(len(m.vertices) * 768 + len(m.faces) * 384 for m in meshes)
    if not np.isfinite(memory) or memory > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Signed-distance grid exceeds memory budget")
    response = run_backend(
        {
            "backend": "organic",
            "meshes": [
                {"vertices": m.vertices.tolist(), "faces": m.faces.tolist()} for m in meshes
            ],
            "origin": low.tolist(),
            "dimensions": dimensions.astype(int).tolist(),
            "voxel_size": voxel_size,
            "blend_radius": blend_radius,
            "max_vertices": settings.max_vertices,
            "max_faces": settings.max_faces,
        },
        settings,
    )
    mesh = trimesh.Trimesh(**response, process=False)
    if len(mesh.faces) == 0 or not mesh.is_volume:
        raise ScissorsError("GEOMETRY_ERROR", "Reconstruction did not produce a closed volume")
    return mesh


def distance_surface(payload: dict[str, Any]) -> dict[str, Any]:
    # Native operations run in the existing timeout-controlled child worker.
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkImplicitPolyDataDistance
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction

    dims = payload["dimensions"]
    origin = np.asarray(payload["origin"])
    spacing = payload["voxel_size"]
    high = origin + (np.asarray(dims) - 1) * spacing
    bounds = np.column_stack((origin, high)).reshape(-1)
    field = None
    for source in payload["meshes"]:
        points = vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(source["vertices"], dtype=float), deep=True))
        faces = np.asarray(source["faces"], dtype=np.int64)
        cells = vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(faces.reshape(-1), deep=True),
        )
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        distance = vtkImplicitPolyDataDistance()
        distance.SetInput(poly)
        sample = vtkSampleFunction()
        sample.SetImplicitFunction(distance)
        sample.SetModelBounds(*bounds)
        sample.SetSampleDimensions(*dims)
        sample.ComputeNormalsOff()
        sample.Update()
        values = vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars())
        if field is None:
            field = values.copy()
        elif payload["blend_radius"] > 0:
            k = payload["blend_radius"]
            h = np.maximum(k - np.abs(field - values), 0) / k
            field = np.minimum(field, values) - h * h * k / 4
        else:
            field = np.minimum(field, values)
    image = vtkImageData()
    image.SetDimensions(*dims)
    image.SetOrigin(*origin)
    image.SetSpacing(spacing, spacing, spacing)
    image.GetPointData().SetScalars(numpy_to_vtk(field, deep=True))
    contour = vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0)
    contour.ComputeNormalsOff()
    contour.Update()
    output = contour.GetOutput()
    if output.GetNumberOfPoints() > payload["max_vertices"] or (
        output.GetNumberOfCells() > payload["max_faces"]
    ):
        raise ValueError("Reconstructed surface exceeds geometry limits")
    vertices = vtk_to_numpy(output.GetPoints().GetData())
    faces_out = vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_out, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
