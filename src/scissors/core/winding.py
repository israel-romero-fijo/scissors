"""Binary winding constraints with fixed outside faces and explicit volume orientation."""

import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.face_editing import flipped_faces
from scissors.core.face_queries import selection_indices
from scissors.core.persistence import json_bytes


def repair_winding(
    engine: Engine,
    object_id: str,
    selection: dict[str, Any] | None,
    orientation: str,
    dry_run: bool,
) -> dict[str, Any]:
    scene, obj = engine.object(object_id)
    if obj.kind != "mesh":
        raise ValueError("Winding repair requires a mesh")
    settings = engine.settings
    if (
        len(obj.vertices) > settings.max_vertices
        or len(obj.faces) > settings.max_faces
        or len(obj.vertices) * 512 + len(obj.faces) * 4096 > settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Winding repair exceeds geometry or memory budget")
    deadline = time.monotonic() + settings.operation_timeout

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Winding repair timed out")

    selected = (
        np.sort(selection_indices(obj, selection))
        if selection is not None
        else np.arange(len(obj.faces), dtype=np.int64)
    )
    mask = np.zeros(len(obj.faces), dtype=bool)
    mask[selected] = True
    vertices, faces = np.asarray(obj.vertices), np.asarray(obj.faces)
    _, welded = np.unique(vertices, axis=0, return_inverse=True)
    topology_faces = welded[faces]
    directed = topology_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2) or np.any(directed[:, 0] == directed[:, 1]):
        raise ScissorsError(
            "GEOMETRY_ERROR", "Winding repair requires nondegenerate manifold edges"
        )
    if len(counts) > settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Winding constraints exceed comparison budget")
    ordered = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = starts[counts == 2]
    first, second = ordered[paired], ordered[paired + 1]
    owners_a, owners_b = first // 3, second // 3
    conflict = directed[first, 0] == directed[second, 0]
    graph: list[list[tuple[int, int]]] = [[] for _ in faces]
    anchors: list[list[int]] = [[] for _ in faces]
    for i, (a, b, same) in enumerate(zip(owners_a, owners_b, conflict, strict=True)):
        if i % 1024 == 0:
            check_time()
        a, b, same = int(a), int(b), int(same)
        if mask[a] and mask[b]:
            graph[a].append((b, same))
            graph[b].append((a, same))
        elif mask[a]:
            anchors[a].append(same)
        elif mask[b]:
            anchors[b].append(same)
    has_boundary = np.zeros(len(faces), dtype=bool)
    has_boundary[ordered[starts[counts == 1]] // 3] = True
    assignment = np.full(len(faces), -1, dtype=np.int8)
    flips = np.zeros(len(faces), dtype=bool)
    regions = []
    for seed in selected:
        if assignment[seed] >= 0:
            continue
        assignment[seed] = 0
        stack = [int(seed)]
        group: list[int] = []
        phase: int | None = None
        while stack:
            if len(group) % 256 == 0:
                check_time()
            face = stack.pop()
            group.append(face)
            for required in anchors[face]:
                candidate = int(assignment[face]) ^ required
                if phase is not None and phase != candidate:
                    raise ScissorsError(
                        "GEOMETRY_ERROR",
                        "Selection cannot repair winding with fixed outside faces",
                        {
                            "face_index": face,
                            "suggestion": "Expand the selection to include conflicting neighbors.",
                        },
                    )
                phase = candidate
            for neighbor, parity in graph[face]:
                required = int(assignment[face]) ^ parity
                if assignment[neighbor] < 0:
                    assignment[neighbor] = required
                    stack.append(neighbor)
                elif assignment[neighbor] != required:
                    raise ScissorsError(
                        "GEOMETRY_ERROR",
                        "Selected surface has contradictory winding constraints",
                        {
                            "face_indices": [face, neighbor],
                        },
                    )
        group_indices = np.asarray(sorted(group), dtype=np.int64)
        anchored = phase is not None
        closed = not anchored and not bool(np.any(has_boundary[group_indices]))
        volume: float | None = None
        if orientation != "preserve":
            if not closed:
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Absolute orientation requires complete closed components"
                )
            candidate_faces = faces[group_indices].copy()
            reverse = assignment[group_indices] == 1
            candidate_faces[reverse] = candidate_faces[reverse, ::-1]
            points = vertices[np.unique(candidate_faces)]
            origin = points[0] + (points - points[0]).mean(axis=0)
            scale = float(np.ptp(points, axis=0).max())
            if not np.isfinite([*origin, scale]).all() or scale <= 0:
                raise ScissorsError("GEOMETRY_ERROR", "Component has no stable volume scale")
            triangles = (vertices[candidate_faces] - origin) / scale
            normalized_volume = float(
                np.einsum(
                    "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
                ).sum()
                / 6
            )
            if not np.isfinite(normalized_volume) or abs(normalized_volume) <= 1e-12:
                raise ScissorsError("GEOMETRY_ERROR", "Component signed volume is ambiguous")
            phase = int((normalized_volume > 0) != (orientation == "outward"))
            volume = normalized_volume * (-1 if phase else 1)
        elif phase is None:
            # Without a fixed neighbor, preserve the majority's orientation. Ties keep
            # the lowest selected face unchanged. No cavity orientation is inferred.
            phase = int(np.count_nonzero(assignment[group_indices]) > len(group_indices) / 2)
        flips[group_indices] = np.logical_xor(assignment[group_indices], phase)
        regions.append(
            {
                "face_indices": group_indices.tolist(),
                "anchored": anchored,
                "closed": closed,
                "planned_normalized_signed_volume": volume,
                "planned_flip_count": int(np.count_nonzero(flips[group_indices])),
            }
        )
    touched = mask[owners_a] | mask[owners_b]
    remaining = conflict ^ flips[owners_a] ^ flips[owners_b]
    if np.any(remaining & touched):
        raise ScissorsError("GEOMETRY_ERROR", "Winding constraints remain inconsistent")
    changed = np.flatnonzero(flips)
    target = obj
    if len(changed) and not dry_run:
        target = flipped_faces(obj, changed)
        engine.checked(target)
    result = {
        **target.summary(),
        "source_geometry_revision": obj.geometry_revision,
        "orientation": orientation,
        "dry_run": dry_run,
        "changed": bool(len(changed)) and not dry_run,
        "planned_flip_count": len(changed),
        "flipped_face_count": 0 if dry_run else len(changed),
        "flipped_face_indices": [] if dry_run else changed.tolist(),
        "planned_flip_face_indices": changed.tolist(),
        "regions": regions,
        "conflicts_before": int(np.count_nonzero(conflict)),
        "scope_conflicts_before": int(np.count_nonzero(conflict & touched)),
        "scope_conflicts_after": int(
            np.count_nonzero((conflict if dry_run else remaining) & touched)
        ),
        "global_conflicts_after": int(np.count_nonzero(conflict if dry_run else remaining)),
        "planned_scope_conflicts_after": int(np.count_nonzero(remaining & touched)),
        "planned_global_conflicts_after": int(np.count_nonzero(remaining)),
        "selection": {
            "object_id": obj.id,
            "geometry_revision": target.geometry_revision,
            "face_indices": selected.tolist(),
        },
    }
    if len(json_bytes(result)) > settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Winding repair response exceeds byte budget")
    check_time()
    if target is not obj:
        scene.objects[obj.id] = target
    return result
