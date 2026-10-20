import copy
from typing import Any

import numpy as np


class PluginContext:
    """Public extension API 1.0: plain geometry, staged changes, no engine internals.

    Mesh coordinates are already in scene space. All changes are validated and
    committed by the parent engine after the child succeeds. Subprocesses are not
    a security sandbox: run only packages you trust.
    """

    api_version = "1.0"

    def __init__(
        self, meshes: dict[str, dict[str, Any]], scene_ids: list[str], writable: bool = True
    ):
        self._meshes = copy.deepcopy(meshes)
        self._scene_ids = scene_ids
        self._writable = writable
        self.changes: list[dict[str, Any]] = []

    def get_mesh(self, object_id: str) -> dict[str, Any]:
        if object_id not in self._meshes:
            raise ValueError("Mesh was not supplied to this operation")
        return copy.deepcopy(self._meshes[object_id])

    def replace_mesh(self, object_id: str, vertices: Any, faces: Any, uv: Any = None) -> None:
        if not self._writable:
            raise PermissionError("Operation does not have scene.write permission")
        mesh = self.get_mesh(object_id)
        value = {
            "kind": "replace",
            "object_id": object_id,
            "vertices": np.asarray(vertices, dtype=float).tolist(),
            "faces": np.asarray(faces).tolist(),
            "uv": uv if uv is not None else mesh["uv"],
        }
        self.changes.append(value)
        self._meshes[object_id] = {**mesh, **value}

    def create_mesh(
        self, scene_id: str, vertices: Any, faces: Any, name: str = "Plugin mesh"
    ) -> None:
        if not self._writable or scene_id not in self._scene_ids:
            raise PermissionError("Scene is not writable by this operation")
        self.changes.append(
            {
                "kind": "create",
                "scene_id": scene_id,
                "name": name,
                "vertices": np.asarray(vertices, dtype=float).tolist(),
                "faces": np.asarray(faces).tolist(),
            }
        )

    def statistics(self, object_id: str) -> dict[str, Any]:
        mesh = self.get_mesh(object_id)
        vertices = np.asarray(mesh["vertices"])
        return {
            "object_id": object_id,
            "vertex_count": len(vertices),
            "face_count": len(mesh["faces"]),
            "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        }
