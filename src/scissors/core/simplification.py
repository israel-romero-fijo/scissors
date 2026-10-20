"""Conservative triangle reduction with unchanged surviving positions and UV."""

import importlib.util
import math
from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mesh import MeshObject, topology_mesh
from scissors.core.validation import validate_mesh


def available() -> bool:
    return importlib.util.find_spec("vtkmodules") is not None


def topology_signature(obj: MeshObject) -> tuple[int, int, int]:
    mesh = topology_mesh(obj.geometry())
    used = np.unique(mesh.faces)
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    parent = np.arange(len(mesh.vertices))

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    components = len({root(int(vertex)) for vertex in used})
    return len(used) - len(edges) + len(mesh.faces), components, int(np.count_nonzero(counts == 1))


def verify_conservation(original: MeshObject, candidate: MeshObject, preserve_uv: bool) -> None:
    if topology_signature(original) != topology_signature(candidate):
        raise ScissorsError("GEOMETRY_ERROR", "Simplification changed topology or boundaries")
    vertices = np.asarray(original.vertices)
    faces = np.asarray(original.faces)
    keys = (
        np.column_stack((vertices, original.uv))
        if preserve_uv and original.uv is not None
        else vertices
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    render_faces = inverse[faces]
    edges = np.sort(
        np.concatenate((render_faces[:, :2], render_faces[:, 1:], render_faces[:, [2, 0]])), axis=1
    )
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(edges[counts == 1])
    if preserve_uv and original.uv is not None:
        if candidate.uv is None:
            raise ScissorsError("GEOMETRY_ERROR", "Simplification lost UV")
        result_keys = np.column_stack((candidate.vertices, candidate.uv))
    else:
        result_keys = np.asarray(candidate.vertices)
    result_set = set(map(tuple, result_keys))
    if (
        not result_set <= set(map(tuple, keys))
        or not set(map(tuple, keys[first[boundary_vertices]])) <= result_set
    ):
        raise ScissorsError("GEOMETRY_ERROR", "Simplification changed protected vertices or UV")


def simplify_mesh(
    obj: MeshObject,
    settings: Settings,
    reduction_percent: float,
    uv_policy: str,
    max_error_relative: float,
    require_target: bool,
) -> dict[str, Any]:
    if not available():
        raise ScissorsError(
            "BACKEND_UNAVAILABLE",
            "Repair the Scissors installation to restore VTK for mesh simplification",
            {"backend": "vtk", "requirement": "Scissors standard installation"},
        )
    validation = validate_mesh(obj, settings)
    if not validation["valid"]:
        raise ScissorsError("GEOMETRY_ERROR", "Input mesh is invalid", validation)
    before = len(obj.faces)
    target = max(1, math.ceil(before * (1 - reduction_percent / 100)))
    # Include temporary arrays and native data structures in the estimate.
    if len(obj.vertices) * 768 + before * 384 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Simplification exceeds estimated memory budget")
    closed = bool(topology_mesh(obj.geometry()).is_watertight)
    response = run_backend(
        {
            "backend": "simplify",
            "vertices": obj.vertices,
            "faces": obj.faces,
            "uv": obj.uv if uv_policy == "preserve" else None,
            "target_faces": target,
            "max_error_relative": max_error_relative,
        },
        settings,
    )
    candidate = obj.model_copy(deep=True)
    try:
        candidate.replace(
            trimesh.Trimesh(vertices=response["vertices"], faces=response["faces"], process=False),
            response["uv"],
        )
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise ScissorsError("GEOMETRY_ERROR", "Invalid simplification backend result") from exc
    validation = validate_mesh(candidate, settings, require_closed=closed)
    after = len(candidate.faces)
    if not validation["valid"] or after > before:
        raise ScissorsError("GEOMETRY_ERROR", "Simplified mesh is invalid", validation)
    verify_conservation(obj, candidate, uv_policy == "preserve")
    if obj.geometry().is_winding_consistent and not candidate.geometry().is_winding_consistent:
        raise ScissorsError("GEOMETRY_ERROR", "Simplification changed consistent winding")
    reached = after <= target
    if require_target and not reached:
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Conservation constraints prevent the requested reduction",
            {"target_faces": target, "achieved_faces": after},
        )
    warnings = []
    if not reached:
        warnings.append("TARGET_NOT_REACHED: topology, boundaries, UV seams or error limit")
    if obj.uv is not None and uv_policy == "discard":
        warnings.append("UV_DISCARDED: regenerate UV before textured export")
    obj.replace(candidate.geometry(), candidate.uv)
    return {
        **obj.summary(),
        "original_face_count": before,
        "target_face_count": target,
        "requested_reduction_percent": reduction_percent,
        "actual_reduction_percent": 100 * (before - after) / before,
        "target_reached": reached,
        "uv_policy": uv_policy,
        "backend": "vtk",
        "warnings": warnings,
    }


def reduce_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    # Only the child imports VTK. Do not load rendering modules or create a window.
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkDecimatePro

    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    uv = np.asarray(payload["uv"], dtype=np.float64) if payload["uv"] is not None else None
    # Weld exact duplicates within a UV chart, never across a UV discontinuity.
    # Existing UV generators can split every triangle into its own render vertices.
    keys = vertices if uv is None else np.column_stack((vertices, uv))
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = vertices[first]
    faces = inverse[faces]
    if uv is not None:
        uv = uv[first]
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(faces.reshape(-1), deep=True),
    )
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    if uv is not None:
        coords = numpy_to_vtk(uv, deep=True)
        coords.SetName("UV")
        mesh.GetPointData().SetTCoords(coords)
    decimator = vtkDecimatePro()
    decimator.SetInputData(mesh)
    decimator.SetTargetReduction(1 - payload["target_faces"] / len(faces))
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    decimator.PreSplitMeshOff()
    decimator.BoundaryVertexDeletionOff()
    decimator.AccumulateErrorOn()
    decimator.SetMaximumError(payload["max_error_relative"])
    # VTK 9.7 defaults this to 15 degrees, which stops decimation on smooth
    # surfaces long before the target reduction. Keep the classic 30-degree
    # crease so hard mechanical edges survive while curved areas reduce.
    decimator.SetFeatureAngle(30.0)
    decimator.SetOutputPointsPrecision(1)  # VTK DOUBLE_PRECISION
    decimator.Update()
    output = decimator.GetOutput()
    vertices_out = vtk_to_numpy(output.GetPoints().GetData())
    counts = vtk_to_numpy(output.GetPolys().GetOffsetsArray())
    if np.any(np.diff(counts) != 3):
        raise ValueError("Simplifier produced non-triangular faces")
    faces_out = vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    coords_out = output.GetPointData().GetTCoords()
    if uv is not None and coords_out is None:
        raise ValueError("Simplifier lost UV coordinates")
    # Compact using the same mapping for geometry and UV, including unchanged meshes.
    used, remap = np.unique(faces_out, return_inverse=True)
    return {
        "vertices": vertices_out[used].tolist(),
        "faces": remap.reshape(-1, 3).tolist(),
        "uv": vtk_to_numpy(coords_out)[used].tolist() if uv is not None else None,
    }
