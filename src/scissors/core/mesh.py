import hashlib
from typing import Any

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field


class MeshObject(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str
    name: str
    vertices: list[list[float]]
    faces: list[list[int]]
    normals: list[list[float]] = Field(default_factory=list)
    uv: list[list[float]] | None = None
    vertex_colors: list[list[int]] | None = None
    material_id: str | None = None
    transform: list[list[float]] = Field(default_factory=lambda: np.eye(4).tolist())
    metadata: dict[str, Any] = Field(default_factory=dict)
    kind: str = "mesh"
    parent_id: str | None = None
    hidden: bool = False
    pivot: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)

    def geometry_signature(self) -> str:
        digest = hashlib.sha256()
        for value, dtype in ((self.vertices, "<f8"), (self.faces, "<i8"), (self.transform, "<f8")):
            array = np.asarray(value, dtype=dtype)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def sync_geometry_revision(self) -> None:
        signature = self.geometry_signature()
        if self.metadata.get("_geometry_signature") != signature:
            self.metadata["_geometry_revision"] = (
                int(self.metadata.get("_geometry_revision", 0)) + 1
            )
            self.metadata["_geometry_signature"] = signature

    @property
    def geometry_revision(self) -> str:
        # Include the current digest so direct edits inside transactions also
        # invalidate queries before the outer state checkpoint is written.
        return f"{self.metadata.get('_geometry_revision', 0)}:{self.geometry_signature()}"

    def geometry(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False,
            vertex_normals=self.normals if len(self.normals) == len(self.vertices) else None,
        )

    def replace(self, mesh: trimesh.Trimesh, uv: list[list[float]] | None = None) -> None:
        self.vertex_colors = None
        self.metadata.pop("normal_method", None)
        self.metadata.pop("normal_angle_degrees", None)
        self.vertices = np.asarray(mesh.vertices).tolist()
        self.faces = np.asarray(mesh.faces).tolist()
        self.normals = compute_normals(mesh).tolist()
        self.uv = uv
        self.sync_geometry_revision()

    def summary(self) -> dict[str, Any]:
        if self.kind in {"group", "marker"}:
            return {
                "id": self.id,
                "object_id": self.id,
                "name": self.name,
                "kind": self.kind,
                "parent_id": self.parent_id,
                "hidden": self.hidden,
                "pivot": self.pivot,
                "vertex_count": 0,
                "face_count": 0,
                "bounds": [self.pivot, self.pivot],
                "centroid": self.pivot,
                "area": 0.0,
                "volume": 0.0,
                "watertight": False,
                "material_id": None,
                "has_uv": False,
            }
        mesh = self.geometry()
        return {
            "id": self.id,
            "object_id": self.id,
            "name": self.name,
            "vertex_count": len(self.vertices),
            "geometry_revision": self.geometry_revision,
            "face_count": len(self.faces),
            "bounds": mesh.bounds.tolist(),
            "centroid": mesh.centroid.tolist(),
            "area": float(mesh.area),
            "volume": float(mesh.volume) if np.all(mesh.extents > 1e-12) else 0.0,
            "watertight": bool(topology_mesh(mesh).is_watertight),
            "material_id": self.material_id,
            "has_uv": self.uv is not None,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "hidden": self.hidden,
            "pivot": self.pivot,
        }


def compute_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    # NumPy accumulation avoids requiring scipy merely to recompute normals.
    normals = np.zeros((len(mesh.vertices), 3))
    for corner in range(3):
        np.add.at(
            normals, mesh.faces[:, corner], mesh.face_normals * mesh.face_angles[:, corner, None]
        )
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-15)
    return normals


def topology_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # UV seam duplicates are welded by exact position only for topology diagnostics.
    vertices, inverse = np.unique(np.asarray(mesh.vertices), axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices=vertices, faces=inverse[np.asarray(mesh.faces)], process=False)


def face_components(face_count: int, adjacency: np.ndarray) -> list[np.ndarray]:
    """Find edge-connected faces without optional scipy or networkx backends."""
    parents = list(range(face_count))
    sizes = [1] * face_count

    def root(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for a, b in adjacency:
        first, second = root(int(a)), root(int(b))
        if first != second:
            if sizes[first] < sizes[second]:
                first, second = second, first
            parents[second] = first
            sizes[first] += sizes[second]
    groups: dict[int, list[int]] = {}
    for index in range(face_count):
        groups.setdefault(root(index), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
