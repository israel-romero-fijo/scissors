"""Revision-bound geometry selections and persistent named selection storage."""

import copy
import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.bridging import boundary_loops
from scissors.core.face_queries import triangle_box
from scissors.core.mesh import MeshObject, face_components, topology_mesh
from scissors.core.persistence import json_bytes

NAMED_KEY = "_named_selections"
FIELDS = {
    "faces": "face_indices",
    "vertices": "vertex_indices",
    "edges": "edge_indices",
    "boundaries": "loop_indices",
    "components": "component_indices",
}


def indices(values: list[int] | None, count: int) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.int64)
    result = np.asarray(values, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= count):
        raise ValueError("Selection index is out of range")
    if len(np.unique(result)) != len(result):
        raise ValueError("Selection indices must be unique")
    return np.sort(result)


def edge_table(obj: MeshObject) -> tuple[np.ndarray, np.ndarray]:
    """Order welded edges lexicographically by their topology vertex pair."""
    _, representatives, inverse = np.unique(
        obj.vertices, axis=0, return_index=True, return_inverse=True
    )
    faces = inverse[np.asarray(obj.faces)]
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return representatives[unique], counts


def select_geometry(
    obj: MeshObject,
    kind: str,
    requested: list[int] | None,
    region: dict[str, Any] | None,
    tolerance: float,
    settings: Settings,
    boundary_only: bool = False,
    seed_faces: list[int] | None = None,
) -> dict[str, Any]:
    if obj.kind != "mesh":
        raise ValueError("Geometry selection requires a mesh")
    if kind not in FIELDS:
        raise ValueError("Unknown selection kind")
    if len(obj.vertices) * 512 + len(obj.faces) * 3072 > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Geometry selection exceeds working memory budget")
    deadline = time.monotonic() + settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Geometry selection timed out")

    vertices, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
    low = high = None
    rule = "fully_inside"
    if region is not None:
        low, high = np.asarray(region["bounds"], dtype=float)
        if np.any(low > high):
            raise ValueError("Region bounds are inverted")
        with np.errstate(over="ignore", invalid="ignore"):
            low, high = low - tolerance, high + tolerance
            span, midpoint = high - low, (high + low) / 2
        if not np.isfinite([low, high, span, midpoint]).all():
            raise ValueError("Expanded region must remain finite")
        rule = region["rule"]
    pairs = 0

    def inside(points: np.ndarray) -> np.ndarray:
        assert low is not None and high is not None
        return np.all((points >= low) & (points <= high), axis=-1)

    def segment_hits(points: np.ndarray) -> np.ndarray:
        # Slab intersection handles segments that cross a box without an inside endpoint.
        assert low is not None and high is not None
        start, delta = points[:, 0], points[:, 1] - points[:, 0]
        enter, leave = np.zeros(len(points)), np.ones(len(points))
        valid = np.ones(len(points), dtype=bool)
        for axis in range(3):
            moving = delta[:, axis] != 0
            valid &= moving | ((start[:, axis] >= low[axis]) & (start[:, axis] <= high[axis]))
            first = np.divide(
                low[axis] - start[:, axis],
                delta[:, axis],
                out=np.full(len(points), -np.inf),
                where=moving,
            )
            second = np.divide(
                high[axis] - start[:, axis],
                delta[:, axis],
                out=np.full(len(points), np.inf),
                where=moving,
            )
            enter = np.maximum(enter, np.minimum(first, second))
            leave = np.minimum(leave, np.maximum(first, second))
        return valid & (enter <= leave)

    def region_matches(points: np.ndarray, triangles: np.ndarray | None = None) -> bool:
        nonlocal pairs
        if region is None:
            return True
        if rule == "fully_inside":
            return bool(np.all(inside(points)))
        if rule == "centroid_inside":
            if triangles is None:
                center = points.mean(axis=0)
            else:
                areas = np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=1,
                )
                origin = triangles[0, 0]
                center = origin + np.average(
                    (triangles - origin).mean(axis=1), axis=0, weights=areas
                )
            return bool(inside(center))
        if triangles is None:
            segments = np.stack((points, np.roll(points, -1, axis=0)), axis=1)
            pairs += len(segments) * 3
            if pairs > settings.max_geometry_pairs:
                raise ScissorsError("RESOURCE_LIMIT", "Selection exceeds pair budget")
            return bool(np.any(segment_hits(segments)))
        pairs += len(triangles) * 13
        if pairs > settings.max_geometry_pairs:
            raise ScissorsError("RESOURCE_LIMIT", "Selection exceeds pair budget")
        assert low is not None and high is not None
        return bool(np.any(triangle_box(triangles, low, high)))

    result: dict[str, Any] = {
        "object_id": obj.id,
        "geometry_revision": obj.geometry_revision,
        "kind": kind,
    }
    selected_faces = np.array([], dtype=np.int64)
    selected_vertices = np.array([], dtype=np.int64)
    if kind == "vertices":
        chosen = indices(requested, len(vertices))
        if region is not None:
            chosen = chosen[inside(vertices[chosen])]
        selected_vertices = chosen
    elif kind == "faces":
        chosen = indices(requested, len(faces))
        chosen = np.asarray(
            [
                int(i)
                for i in chosen
                if region_matches(vertices[faces[i]], vertices[faces[i]][None])
            ],
            dtype=np.int64,
        )
        selected_faces = chosen
        selected_vertices = np.unique(faces[chosen])
    elif kind == "edges":
        edges, counts = edge_table(obj)
        chosen = indices(requested, len(edges))
        if boundary_only:
            chosen = chosen[counts[chosen] == 1]
        selected = []
        for i in chosen:
            check_time()
            points = vertices[edges[i]]
            if region_matches(points):
                selected.append(int(i))
        chosen = np.asarray(selected, dtype=np.int64)
        result["edges"] = edges[chosen].tolist()
        selected_vertices = np.unique(edges[chosen])
    elif kind == "boundaries":
        loops = boundary_loops(obj)
        selected = []
        for i in indices(requested, len(loops)):
            check_time()
            if region_matches(vertices[loops[i]]):
                selected.append(int(i))
        chosen = np.asarray(selected, dtype=np.int64)
        result["loops"] = [
            {"loop_index": int(i), "vertex_indices": loops[i], "edge_count": len(loops[i])}
            for i in chosen
        ]
        selected_vertices = np.unique([v for i in chosen for v in loops[i]]).astype(np.int64)
    else:
        groups = face_components(len(faces), topology_mesh(obj.geometry()).face_adjacency)
        seeds = set(indices(seed_faces, len(faces)).tolist()) if seed_faces is not None else None
        selected = []
        selected_groups = []
        for i in indices(requested, len(groups)):
            check_time()
            group = groups[i]
            if seeds is not None and not seeds.intersection(group.tolist()):
                continue
            used = np.unique(faces[group])
            if region_matches(vertices[used], vertices[faces[group]]):
                selected.append(int(i))
                selected_groups.append(group)
        chosen = np.asarray(selected, dtype=np.int64)
        selected_faces = (
            np.sort(np.concatenate(selected_groups)) if selected_groups else selected_faces
        )
        selected_vertices = np.unique(faces[selected_faces])
        result["components"] = [
            {"component_index": int(i), "face_indices": groups[i].tolist()} for i in chosen
        ]
    result[FIELDS[kind]] = chosen.tolist()
    result["vertex_indices"] = selected_vertices.tolist()
    if kind in {"faces", "components"}:
        result["face_indices"] = selected_faces.tolist()
    points = vertices[selected_vertices]
    result.update(
        {
            "count": len(chosen),
            "empty": not len(chosen),
            "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
            if len(points)
            else None,
        }
    )
    check_time()
    if len(json_bytes(result)) > min(settings.max_memory_bytes, settings.max_file_bytes):
        raise ScissorsError("RESOURCE_LIMIT", "Selection result exceeds byte budget")
    return result


def save_selection(
    obj: MeshObject, name: str, selection: dict[str, Any], overwrite: bool, settings: Settings
) -> dict[str, Any]:
    if selection["object_id"] != obj.id or selection["geometry_revision"] != obj.geometry_revision:
        raise ValueError("Selection belongs to another geometry revision")
    kind = selection["kind"]
    requested = selection.get(FIELDS[kind])
    if requested is None:
        raise ValueError(f"Selection requires {FIELDS[kind]}")
    result = select_geometry(obj, kind, requested, None, 0, settings)
    stored = copy.deepcopy(obj.metadata.get(NAMED_KEY, {}))
    if name in stored and not overwrite:
        raise ValueError("Named selection exists; set overwrite=true to replace it")
    stored[name] = result
    if len(stored) > 128 or len(json_bytes(stored)) > min(
        settings.max_file_bytes, settings.max_memory_bytes // 4
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Named selections exceed storage budget")
    obj.metadata[NAMED_KEY] = stored
    return {"name": name, "selection": copy.deepcopy(result)}


def get_selection(obj: MeshObject, name: str) -> dict[str, Any]:
    stored = obj.metadata.get(NAMED_KEY, {})
    if name not in stored:
        raise ValueError("Named selection does not exist")
    result = stored[name]
    if result["object_id"] != obj.id or result["geometry_revision"] != obj.geometry_revision:
        raise ScissorsError("INVALID_PARAMETER", "Named selection is stale; select geometry again")
    return copy.deepcopy(result)


def list_selections(obj: MeshObject) -> dict[str, Any]:
    revision = obj.geometry_revision
    return {
        "object_id": obj.id,
        "selections": [
            {
                "name": name,
                "kind": value["kind"],
                "count": value["count"],
                "geometry_revision": value["geometry_revision"],
                "stale": value["object_id"] != obj.id or value["geometry_revision"] != revision,
            }
            for name, value in sorted(obj.metadata.get(NAMED_KEY, {}).items())
        ],
    }


def delete_selection(obj: MeshObject, name: str) -> dict[str, Any]:
    stored = obj.metadata.get(NAMED_KEY, {})
    if name not in stored:
        raise ValueError("Named selection does not exist")
    del stored[name]
    if not stored:
        obj.metadata.pop(NAMED_KEY, None)
    return {"object_id": obj.id, "name": name, "deleted": True}
