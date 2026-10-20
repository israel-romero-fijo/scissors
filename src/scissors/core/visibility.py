"""Read-only sampled eye visibility and reproducible before/after comparisons."""

import hashlib
import json
import time
from collections import Counter
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject
from scissors.core.persistence import json_bytes
from scissors.core.ray_queries import TriangleBVH, WorkBudget

MAX_SAMPLES = 32768
MAX_EYES = 32


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def unit(value: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Visibility forward and up must be finite nonzero vectors")
    return vector / length


def angles(bounds: list[float], sampling: dict[str, Any], axis: str, limit: float) -> np.ndarray:
    low, high = map(float, bounds)
    if not -limit <= low <= high <= limit:
        raise ValueError(f"{axis} range must be ordered within [-{limit}, {limit}] degrees")
    if sampling["type"] == "counts":
        count = sampling[axis]
        if low == high and count != 1:
            raise ValueError("A zero-width angular range requires one sample")
        values = np.asarray([(low + high) / 2]) if count == 1 else np.linspace(low, high, count)
    else:
        step = float(sampling[axis])
        ratio = (high - low) / step
        if not np.isfinite(ratio) or ratio >= 1024:
            raise ScissorsError("RESOURCE_LIMIT", "Visibility angular axis exceeds 1024 samples")
        rounding = np.finfo(float).eps * 8 * max(1, abs(ratio))
        count = int(np.floor(ratio + rounding)) + 1
        if count > 1024:
            raise ScissorsError("RESOURCE_LIMIT", "Visibility angular axis exceeds 1024 samples")
        values = np.minimum(low + np.arange(count) * step, high)
    rounding_degrees = np.finfo(float).eps * 8 * max(abs(low), abs(high))
    if abs(values[-1] - high) <= rounding_degrees:
        values[-1] = high
    if low <= 0 <= high:
        values[np.abs(values) <= rounding_degrees] = 0
    if np.any(np.diff(values) <= 0):
        raise ValueError("Angular steps cannot be resolved as distinct floating-point samples")
    return values


def expand_eyes(
    eyes: list[dict[str, Any]], head_offsets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len({eye["name"] for eye in eyes}) != len(eyes):
        raise ValueError("Eye names must be unique")
    if len({head["name"] for head in head_offsets}) != len(head_offsets):
        raise ValueError("Head offset names must be unique")
    if len(eyes) * len(head_offsets) > MAX_EYES:
        raise ScissorsError("RESOURCE_LIMIT", "Visibility exceeds 32 eye/head combinations")
    result = []
    for eye in sorted(eyes, key=lambda value: value["name"]):
        for head in sorted(head_offsets, key=lambda value: value["name"]):
            position = np.asarray(eye["position"]) + np.asarray(head["offset"])
            if not np.isfinite(position).all():
                raise ValueError("Eye position plus head offset must remain finite")
            result.append(
                {
                    "name": f"{eye['name']}@{head['name']}",
                    "eye_name": eye["name"],
                    "head_name": head["name"],
                    "position": position.astype(float).tolist(),
                }
            )
    return result


def selected_objects(
    engine: Engine,
    scene_id: str,
    include_ids: list[str] | None,
    exclude_ids: list[str] | None,
    include_hidden: bool,
    deadline: float,
) -> list[MeshObject]:
    scene = engine.scene(scene_id)
    children: dict[str, list[str]] = {}
    for obj in scene.objects.values():
        if obj.parent_id is not None:
            children.setdefault(obj.parent_id, []).append(obj.id)

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Visibility object filtering timed out")

    def expand(ids: list[str]) -> set[str]:
        result: set[str] = set()
        for oid in sorted(set(ids)):
            check_time()
            if oid not in scene.objects:
                engine.object(oid)  # Preserve the normal missing-object error.
                raise ValueError("Visibility filters must reference objects in the selected scene")
        pending = list(ids)
        while pending:
            check_time()
            oid = pending.pop()
            if oid not in result:
                result.add(oid)
                pending.extend(children.get(oid, []))
        return result

    included = set(scene.objects) if include_ids is None else expand(include_ids)
    excluded = expand(exclude_ids or [])
    selected = []
    hidden_cache: dict[str, bool] = {}
    for oid in sorted(included - excluded):
        check_time()
        obj = scene.objects[oid]
        if obj.kind != "mesh":
            continue
        path: list[str] = []
        current: str | None = oid
        while current is not None and current not in hidden_cache:
            check_time()
            path.append(current)
            current = scene.objects[current].parent_id
        hidden = hidden_cache.get(current, False) if current is not None else False
        for entry in reversed(path):
            hidden |= scene.objects[entry].hidden
            hidden_cache[entry] = hidden
        if include_hidden or not hidden_cache[oid]:
            selected.append(obj)
    return selected


def clear_spans(values: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    spans = []
    start: int | None = None
    for index, clear in enumerate([*mask.tolist(), False]):
        if clear and start is None:
            start = index
        elif not clear and start is not None:
            spans.append([float(values[start]), float(values[index - 1])])
            start = None
    return spans


def eye_summary(
    eye: dict[str, Any],
    status: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    blockers: Counter[str],
) -> dict[str, Any]:
    clear, blocked, unresolved = (int(np.count_nonzero(status == code)) for code in (0, 1, -1))
    rows = []
    for index, value in enumerate(vertical):
        rows.append(
            {
                "vertical_degrees": float(value),
                "clear_horizontal_spans": clear_spans(horizontal, status[index] == 0),
            }
        )
    central = None
    horizontal_zero, vertical_zero = np.flatnonzero(horizontal == 0), np.flatnonzero(vertical == 0)
    if len(horizontal_zero) and len(vertical_zero):
        col, row = int(horizontal_zero[0]), int(vertical_zero[0])
        if status[row, col] == 0:
            left = right = col
            while left > 0 and status[row, left - 1] == 0:
                left -= 1
            while right + 1 < len(horizontal) and status[row, right + 1] == 0:
                right += 1
            central = [float(horizontal[left]), float(horizontal[right])]
    return {
        **eye,
        "clear": clear,
        "blocked": blocked,
        "unresolved": unresolved,
        "coverage": {
            "requested": int(status.size),
            "resolved": clear + blocked,
            "resolved_fraction": (clear + blocked) / status.size,
            "clear_fraction": clear / status.size,
            "clear_fraction_of_resolved": clear / (clear + blocked) if clear + blocked else None,
        },
        "blocking_objects": [
            {"object_id": oid, "sample_count": count} for oid, count in sorted(blockers.items())
        ],
        "angular_spans": rows,
        "central_horizontal_span_degrees": central,
        "central_span_status": "sampled"
        if central is not None
        else "blocked_or_unresolved"
        if len(horizontal_zero) and len(vertical_zero)
        else "zero_angles_not_sampled",
    }


def validate_baseline(
    baseline: dict[str, Any] | None, configuration: dict[str, Any], scene_id: str, sample_count: int
) -> None:
    if baseline is None:
        return
    current = fingerprint(configuration)
    if (
        baseline["scene_id"] != scene_id
        or baseline["configuration_fingerprint"] != current
        or fingerprint(baseline["configuration"]) != current
    ):
        raise ScissorsError(
            "INVALID_PARAMETER",
            "Visibility comparison requires the same scene and measurement configuration",
        )
    names = [eye["name"] for eye in baseline["eyes"]]
    if len(names) != len(set(names)) or set(names) != {
        eye["name"] for eye in configuration["eyes"]
    }:
        raise ValueError("Baseline eye names do not match this configuration")
    if any(
        eye["clear"] + eye["blocked"] + eye["unresolved"] != sample_count
        for eye in baseline["eyes"]
    ):
        raise ValueError("Baseline sample totals do not match this configuration")


def visibility_report(
    engine: Engine,
    scene_id: str,
    eyes: list[dict[str, Any]],
    forward: list[float],
    up: list[float],
    horizontal_range: list[float],
    vertical_range: list[float],
    sampling: dict[str, Any],
    near: float,
    max_distance: float,
    head_offsets: list[dict[str, Any]],
    include_ids: list[str] | None,
    exclude_ids: list[str] | None,
    include_hidden: bool,
    detail: str,
    tolerance: float,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = engine.scene(scene_id)
    deadline = time.monotonic() + engine.settings.operation_timeout
    if not near < max_distance:
        raise ValueError("Visibility near must be less than max_distance")
    fwd, up_reference = unit(forward), unit(up)
    lateral = np.cross(fwd, up_reference)
    if np.linalg.norm(lateral) <= 1e-10:
        raise ValueError("Visibility forward and up must not be parallel")
    right = unit(lateral)
    vertical_axis = np.cross(right, fwd)
    horizontal = angles(horizontal_range, sampling, "horizontal", 180)
    vertical = angles(vertical_range, sampling, "vertical", 90)
    views = expand_eyes(eyes, head_offsets)
    per_eye = len(horizontal) * len(vertical)
    total = per_eye * len(views)
    if total > MAX_SAMPLES:
        raise ScissorsError(
            "RESOURCE_LIMIT",
            "Visibility exceeds 32768 total samples",
            {"requested": total, "limit": MAX_SAMPLES},
        )
    configuration: dict[str, Any] = {
        "version": 1,
        "eyes": views,
        "forward": fwd.tolist(),
        "up": vertical_axis.tolist(),
        "right": right.tolist(),
        "horizontal_range": list(map(float, horizontal_range)),
        "vertical_range": list(map(float, vertical_range)),
        "sampling": sampling,
        "horizontal_angles": horizontal.tolist(),
        "vertical_angles": vertical.tolist(),
        "near": float(near),
        "max_distance": float(max_distance),
        "tolerance": float(tolerance),
        "include_ids": sorted(set(include_ids)) if include_ids is not None else None,
        "exclude_ids": sorted(set(exclude_ids or [])),
        "include_hidden": include_hidden,
        "obstruction_policy": "geometric_surfaces",
        "backfaces": "included",
        "coordinates": "right-handed Z-up; world meters; yaw/elevation degrees",
    }
    validate_baseline(baseline, configuration, scene_id, per_eye)
    objects = selected_objects(engine, scene_id, include_ids, exclude_ids, include_hidden, deadline)
    face_count = sum(len(obj.faces) for obj in objects)
    if (
        face_count > engine.settings.max_faces
        or face_count * 768 + total * 1536 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError(
            "RESOURCE_LIMIT", "Visibility geometry or working memory exceeds budget"
        )
    # This content revision tracks all scene geometry and inherited visibility. It
    # does not mistake the registry's catalog revision for a scene geometry revision.
    scene_revision = fingerprint(
        {
            "scene_id": scene_id,
            "objects": [
                {
                    "id": obj.id,
                    "kind": obj.kind,
                    "geometry_revision": obj.geometry_revision,
                    "parent_id": obj.parent_id,
                    "hidden": obj.hidden,
                }
                for obj in sorted(scene.objects.values(), key=lambda value: value.id)
            ],
        }
    )
    triangles = np.empty((face_count, 3, 3), dtype=float)
    owners = np.empty(face_count, dtype=np.int32)
    starts = []
    cursor = 0
    for index, obj in enumerate(objects):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Visibility geometry preparation timed out")
        starts.append(cursor)
        stop = cursor + len(obj.faces)
        triangles[cursor:stop] = np.asarray(obj.vertices)[np.asarray(obj.faces)]
        owners[cursor:stop] = index
        cursor = stop
    hierarchy = TriangleBVH(triangles, deadline)
    h, v = np.meshgrid(np.radians(horizontal), np.radians(vertical))
    directions = (
        np.cos(v)[..., None] * (np.cos(h)[..., None] * fwd + np.sin(h)[..., None] * right)
        + np.sin(v)[..., None] * vertical_axis
    ).reshape(-1, 3)
    budget = WorkBudget(engine.settings.max_geometry_pairs, deadline)
    summaries = []
    exhausted: str | None = None
    numeric_scale = max(
        max_distance,
        float(np.abs(triangles).max()) if triangles.size else 0,
        max(float(np.max(np.abs(view["position"]))) for view in views),
    )
    resolution = float(numeric_scale * np.finfo(float).eps * 16)
    if resolution > tolerance:
        exhausted = "requested_tolerance_below_numeric_resolution"
    for view in views:
        status = np.full(per_eye, -1, dtype=np.int8)
        blockers: Counter[str] = Counter()
        unresolved_reasons: Counter[str] = Counter()
        samples = []
        origin = np.asarray(view["position"])
        for index, direction in enumerate(directions):
            state, triangle_id, distance = "unresolved", None, None
            reason = exhausted
            if exhausted is None:
                try:
                    state, triangle_id, distance = hierarchy.query(
                        origin, direction, near, max_distance, tolerance, budget
                    )
                except ScissorsError as exc:
                    if exc.code not in {"RESOURCE_LIMIT", "OPERATION_TIMEOUT"}:
                        raise
                    exhausted = reason = exc.code
                else:
                    reason = "coplanar_or_near_parallel_contact" if state == "unresolved" else None
            status[index] = {"clear": 0, "blocked": 1, "unresolved": -1}[state]
            if state == "unresolved":
                unresolved_reasons[reason or "unknown"] += 1
            object_id = objects[int(owners[triangle_id])].id if triangle_id is not None else None
            if object_id is not None:
                blockers[object_id] += 1
            if detail == "samples":
                samples.append(
                    {
                        "horizontal_degrees": float(horizontal[index % len(horizontal)]),
                        "vertical_degrees": float(vertical[index // len(horizontal)]),
                        "direction": direction.tolist(),
                        "status": state,
                        "reason": reason,
                        "object_id": object_id,
                        "distance": distance,
                        "face_index": triangle_id - starts[int(owners[triangle_id])]
                        if triangle_id is not None
                        else None,
                    }
                )
        summary = eye_summary(
            view, status.reshape(len(vertical), len(horizontal)), horizontal, vertical, blockers
        )
        summary["unresolved_reasons"] = dict(sorted(unresolved_reasons.items()))
        if detail == "samples":
            summary["samples"] = samples
        summaries.append(summary)
    unresolved = sum(eye["unresolved"] for eye in summaries)
    result: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_revision": scene_revision,
        "scene_revision_semantics": (
            "content hash of object geometry revisions, IDs, hierarchy and hidden flags"
        ),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint(configuration),
        "status": "uncompleted" if unresolved else "completed",
        "eyes": summaries,
        "grid": {
            "horizontal_count": len(horizontal),
            "vertical_count": len(vertical),
            "samples_per_eye": per_eye,
            "total_samples": total,
            "sampled_horizontal_span": [float(horizontal[0]), float(horizontal[-1])],
            "sampled_vertical_span": [float(vertical[0]), float(vertical[-1])],
        },
        "coverage": {
            "clear": sum(eye["clear"] for eye in summaries),
            "blocked": sum(eye["blocked"] for eye in summaries),
            "unresolved": unresolved,
            "resolved_fraction": (total - unresolved) / total,
        },
        "objects_considered": [obj.id for obj in objects],
        "work": {
            "backend": "numpy_triangle_bvh",
            "triangle_count": face_count,
            "bvh_nodes": len(hierarchy.nodes),
            "triangle_tests": budget.triangle_tests,
            "node_tests": budget.node_tests,
            "triangle_test_limit": budget.limit,
            "node_test_limit": 2 * budget.limit,
            "stopped_reason": exhausted,
            "numeric_resolution": resolution,
        },
        "warnings": [
            "This is sampled geometric visibility, not a headset FOV or VR comfort certification.",
            "Alpha, textures and material names do not make a surface transparent.",
            "Angular spans connect clear sample endpoints; unsampled directions are not certified.",
        ]
        + (["Unresolved samples must not be counted as clear."] if unresolved else []),
    }
    if baseline is not None:
        previous = {eye["name"]: eye for eye in baseline["eyes"]}
        result["comparison"] = {
            "status": "partial"
            if unresolved or any(eye["unresolved"] for eye in previous.values())
            else "completed",
            "baseline_scene_revision": baseline["scene_revision"],
            "configuration_matched": True,
            "eyes": [
                {
                    "name": eye["name"],
                    "clear_delta": eye["clear"] - previous[eye["name"]]["clear"],
                    "blocked_delta": eye["blocked"] - previous[eye["name"]]["blocked"],
                    "unresolved_delta": eye["unresolved"] - previous[eye["name"]]["unresolved"],
                }
                for eye in summaries
            ],
        }
    if len(json_bytes(result)) > engine.settings.max_file_bytes:
        raise ScissorsError(
            "RESOURCE_LIMIT",
            "Visibility report exceeds result size limit; request summary or fewer samples",
        )
    return result
