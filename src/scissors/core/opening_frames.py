"""Opening selection and bounded frame sweeps over baked world geometry."""

import copy
import time
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.core.cut_volume import PARTS_KEY, combine_parts, cut_contours, source_parts
from scissors.core.engine import Engine, identifier
from scissors.core.inspection import point_segment, self_intersections
from scissors.core.mechanical import _tube_frames, _tube_tangents, sweep_rings
from scissors.core.mesh import MeshObject, compute_normals, topology_mesh
from scissors.core.profiles import area, inside, intersects, validate_curve_path

MAX_PATH_POINTS = 512
MAX_CONTOUR_POINTS = 4096


def spatial_path_check(path: np.ndarray, closed: bool, engine: Engine, tolerance: float) -> None:
    """Check finite 3D segment distances without projecting spatial paths to XY."""
    count = len(path) if closed else len(path) - 1
    if count * count > engine.settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Spatial path exceeds pair budget")
    deadline = time.monotonic() + engine.settings.operation_timeout
    for index in range(count):
        if time.monotonic() > deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Spatial path check timed out")
        others = np.arange(index + 2, count)
        if closed and index == 0:
            others = others[others != count - 1]
        if not len(others):
            continue
        a, b = path[index], path[(index + 1) % len(path)]
        c, d = path[others], path[(others + 1) % len(path)]
        distance = np.minimum(point_segment(a, c, d), point_segment(b, c, d))
        distance = np.minimum(distance, point_segment(c, a, b))
        distance = np.minimum(distance, point_segment(d, a, b))
        u, v, w = b - a, d - c, a - c
        aa, bb, cc = np.dot(u, u), v @ u, np.sum(v * v, axis=1)
        dd, ee = w @ u, np.sum(v * w, axis=1)
        denominator = aa * cc - bb * bb
        nonparallel = denominator > 1e-30
        first = np.divide(bb * ee - cc * dd, denominator, out=np.zeros_like(bb), where=nonparallel)
        second = np.divide(aa * ee - bb * dd, denominator, out=np.zeros_like(bb), where=nonparallel)
        interior = nonparallel & (first >= 0) & (first <= 1) & (second >= 0) & (second <= 1)
        distance = np.minimum(
            distance,
            np.where(
                interior,
                np.linalg.norm(w + first[:, None] * u - second[:, None] * v, axis=1),
                np.inf,
            ),
        )
        if np.any(distance <= tolerance):
            raise ScissorsError(
                "GEOMETRY_ERROR", "Opening path edges touch or cross", {"edge": index}
            )


def current_source(engine: Engine, object_id: str) -> tuple[MeshObject, str]:
    scene, source = engine.object(object_id)
    if source.kind == "mesh":
        return source, source.geometry_revision
    if source.kind == "group" and PARTS_KEY in source.metadata:
        parts = source_parts(engine, scene.id, source)
        if sum(len(part.faces) * 3 for part in parts) > engine.settings.max_vertices:
            raise ScissorsError("RESOURCE_LIMIT", "Opening source exceeds corner budget")
        combined, _ = combine_parts(parts, 1)
        return combined, combined.geometry_revision
    raise ValueError("Opening source must be a mesh or a cut-volume material group")


def unit(vector: np.ndarray, message: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError(message)
    return vector / length


def plane_basis(normal: np.ndarray) -> np.ndarray:
    first = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = unit(first - normal * np.dot(first, normal), "Invalid normal")
    return np.column_stack((first, np.cross(normal, first)))


def simplify_path(points: Any, closed: bool, tolerance: float) -> np.ndarray:
    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
        raise ValueError("Opening points must be finite world-space triples")
    if len(path) > MAX_CONTOUR_POINTS:
        raise ScissorsError("RESOURCE_LIMIT", "Opening contour exceeds point budget")
    if closed and len(path) > 1 and np.array_equal(path[0], path[-1]):
        path = path[:-1]
    if len(path) < (3 if closed else 2):
        raise ValueError("Opening path has too few points")
    segments = np.roll(path, -1, axis=0) - path if closed else np.diff(path, axis=0)
    if np.any(np.linalg.norm(segments, axis=1) <= tolerance):
        raise ScissorsError("GEOMETRY_ERROR", "Opening has repeated or too-close points")
    # Boolean triangulation often adds collinear points. Keep the actual corners.
    while len(path) > (3 if closed else 2):
        removed = False
        for index in range(len(path)):
            if not closed and index in (0, len(path) - 1):
                continue
            incoming = path[index] - path[index - 1]
            outgoing = path[(index + 1) % len(path)] - path[index]
            if np.dot(incoming, outgoing) > 0 and np.linalg.norm(
                np.cross(incoming, outgoing)
            ) <= tolerance * min(np.linalg.norm(incoming), np.linalg.norm(outgoing)):
                path = np.delete(path, index, axis=0)
                removed = True
                break
        if not removed:
            break
    if len(path) > MAX_PATH_POINTS:
        raise ScissorsError("RESOURCE_LIMIT", "Resolved opening path exceeds 512 corners")
    return path


def choose_contour(
    contours: list[dict[str, Any]],
    index: int | None,
    side: str,
    normal: np.ndarray | None,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    if not contours:
        raise ScissorsError("GEOMETRY_ERROR", "Opening has no usable contours")
    if side == "selected":
        if index is None:
            if len(contours) != 1:
                raise ScissorsError(
                    "INVALID_PARAMETER",
                    "Choose contour_index or an explicit inner/outer side",
                    {"contour_count": len(contours)},
                )
            index = 0
        if not 0 <= index < len(contours):
            raise ValueError("Opening contour_index is out of range")
        return simplify_path(contours[index]["points"], True, tolerance), index
    if index is not None:
        raise ValueError("Choose contour_index or contour_side, not both")
    if len(contours) != 2:
        raise ValueError("Inner/outer selection requires exactly two contours; use contour_index")
    paths = [simplify_path(contour["points"], True, tolerance) for contour in contours]
    if normal is None:
        raise ValueError("Inner/outer selection requires normal directed from inside to outside")
    basis = plane_basis(normal)
    projections = [path @ basis for path in paths]
    levels = [path @ normal for path in paths]
    if any(np.ptp(level) > tolerance for level in levels):
        raise ValueError("Inner/outer contours must each be planar and perpendicular to normal")
    difference = float(levels[1].mean() - levels[0].mean())
    if abs(difference) > tolerance:
        # Two parallel faces of one rim must have matching projected contours.
        # Checking vertices against segments also accepts different triangulation splits.
        for first, second in ((projections[0], projections[1]), (projections[1], projections[0])):
            distances = np.full(len(first), np.inf)
            for a, b in zip(second, np.roll(second, -1, axis=0), strict=True):
                distances = np.minimum(distances, point_segment(first, a, b))
            if np.any(distances > tolerance):
                raise ValueError("Rim contours do not correspond; choose contour_index explicitly")
        outer = 1 if difference > 0 else 0
    else:
        # In one plane, inner/outer means nested, disjoint polygon boundaries.
        for a, b in zip(projections[0], np.roll(projections[0], -1, axis=0), strict=True):
            for c, d in zip(projections[1], np.roll(projections[1], -1, axis=0), strict=True):
                if intersects(a, b, c, d):
                    raise ValueError("Inner/outer contours cross or touch")
        outer = int(abs(area(projections[1])) > abs(area(projections[0])))
        if not all(inside(point, projections[outer]) for point in projections[1 - outer]):
            raise ValueError("Contours are not nested; choose contour_index explicitly")
    chosen = outer if side == "outer" else 1 - outer
    return paths[chosen], chosen


def resolve_opening(
    engine: Engine,
    object_id: str,
    opening: dict[str, Any],
    side: str,
    normal: np.ndarray | None,
) -> tuple[np.ndarray, bool, str, float, int | None]:
    mesh, revision = current_source(engine, object_id)
    if opening["geometry_revision"] != revision:
        raise ScissorsError(
            "INVALID_PARAMETER",
            "Opening belongs to another geometry revision",
            {"object_id": object_id, "current_revision": revision},
        )
    extent = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
    tolerance = max(1e-9, extent * 1e-8)
    if opening["type"] == "points":
        if side != "selected":
            raise ValueError("Explicit points already select a contour; use contour_side=selected")
        return (
            simplify_path(opening["points"], opening["closed"], tolerance),
            opening["closed"],
            revision,
            tolerance,
            None,
        )
    if opening["type"] == "cut":
        metadata = engine.object(object_id)[1].metadata.get("_last_cut_opening")
        if (
            not isinstance(metadata, dict)
            or metadata.get("geometry_revision") != revision
            or metadata.get("status") != "contours"
        ):
            raise ScissorsError(
                "INVALID_PARAMETER",
                "No current cut contours; select rim faces or supply explicit points",
            )
    else:
        if engine.object(object_id)[1].kind != "mesh":
            raise ValueError(
                "Rim face indices require a mesh; use cut metadata for a material group"
            )
        indices = np.asarray(opening["face_indices"], dtype=np.int64)
        if (
            np.any(indices < 0)
            or np.any(indices >= len(mesh.faces))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Rim selection must contain unique valid face indices")
        if len(mesh.faces) * 512 > engine.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Rim selection exceeds memory budget")
        selected = np.zeros(len(mesh.faces), dtype=int)
        selected[indices] = 1
        metadata = cut_contours(np.asarray(mesh.vertices), np.asarray(mesh.faces), selected)
        if metadata["status"] != "contours":
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Selected rim faces have ambiguous or missing boundaries",
                metadata,
            )
    contours = metadata["contours"]
    count = sum(len(contour["points"]) for contour in contours)
    if count > MAX_CONTOUR_POINTS or count * count > engine.settings.max_geometry_pairs:
        raise ScissorsError("RESOURCE_LIMIT", "Opening contour selection exceeds pair budget")
    path, index = choose_contour(contours, opening.get("contour_index"), side, normal, tolerance)
    return path, True, revision, tolerance, index


def miter_vectors(path: np.ndarray, closed: bool, normal: np.ndarray, limit: float) -> np.ndarray:
    edges = np.roll(path, -1, axis=0) - path if closed else np.diff(path, axis=0)
    directions = edges / np.linalg.norm(edges, axis=1)[:, None]
    laterals = np.cross(normal, directions)
    vectors = []
    for index in range(len(path)):
        incoming = laterals[index - 1] if closed or index > 0 else laterals[0]
        outgoing = (
            laterals[index % len(laterals)] if closed or index < len(path) - 1 else laterals[-1]
        )
        summed = incoming + outgoing
        denominator = float(np.dot(summed, outgoing))
        if denominator <= 1e-10:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Opening path reverses at a corner", {"corner": index}
            )
        vector = summed / denominator
        if np.linalg.norm(vector) > limit + 1e-10:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Opening corner exceeds miter_limit", {"corner": index}
            )
        vectors.append(vector)
    return np.asarray(vectors)


def frame_opening(
    engine: Engine,
    object_id: str,
    opening: dict[str, Any],
    section: dict[str, Any],
    contour_side: str,
    normal: list[float] | None,
    placement: str,
    offset: float,
    clearance: float,
    joint_policy: str,
    end_policy: str,
    miter_limit: float,
    material_id: str | None,
    parent_id: str | None,
    name: str,
    allow_incomplete_checks: bool,
) -> dict[str, Any]:
    scene, source = engine.object(object_id)
    deadline = time.monotonic() + engine.settings.operation_timeout
    if material_id is not None and engine.material(material_id)[0].id != scene.id:
        raise ValueError("Frame material must belong to the source scene")
    if parent_id is not None:
        if engine.object(parent_id)[0].id != scene.id:
            raise ValueError("Frame parent must belong to the source scene")
    else:
        parent_id = source.parent_id
    supplied_normal = (
        unit(np.asarray(normal), "Frame normal must be nonzero") if normal is not None else None
    )
    path, closed, revision, tolerance, contour_index = resolve_opening(
        engine, object_id, opening, contour_side, supplied_normal
    )
    if closed and end_policy != "auto":
        raise ValueError("Closed contours join the seam; use end_policy=auto")
    if placement == "centered" and clearance != 0:
        raise ValueError("Nonzero clearance requires inside or outside placement")
    if not closed and placement != "centered":
        raise ValueError("Inside/outside placement requires a closed planar contour")
    if section["type"] == "circle":
        angles = np.arange(section["sides"]) * (2 * np.pi / section["sides"])
        polygon = section["radius"] * np.column_stack((np.cos(angles), np.sin(angles)))
        width = depth = 2 * section["radius"]
    else:
        width, depth = section["width"], section["depth"]
        polygon = np.asarray(
            [
                [-width / 2, -depth / 2],
                [width / 2, -depth / 2],
                [width / 2, depth / 2],
                [-width / 2, depth / 2],
            ]
        )
    if min(width, depth) <= 10 * tolerance:
        raise ValueError("Frame section is too small for the source coordinate tolerance")
    count, sides = len(path), len(polygon)
    vertex_count = count * sides + (0 if closed or end_policy == "open" else 2)
    face_count = 2 * sides * (count if closed else count - 1) + (
        0 if closed or end_policy == "open" else 2 * sides
    )
    if (
        vertex_count > engine.settings.max_vertices
        or face_count > engine.settings.max_faces
        or vertex_count * 384 + face_count * 192 > engine.settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Frame exceeds geometry or memory budget")
    fitted_normal = supplied_normal
    if fitted_normal is None:
        _, singular, axes = np.linalg.svd(path - path.mean(axis=0), full_matrices=False)
        if len(singular) >= 2 and singular[1] > tolerance:
            fitted_normal = axes[-1]
            if fitted_normal[int(np.argmax(np.abs(fitted_normal)))] < 0:
                fitted_normal = -fitted_normal
    planar = (
        fitted_normal is not None and np.max(np.abs((path - path[0]) @ fitted_normal)) <= tolerance
    )
    if joint_policy == "miter":
        if not planar or fitted_normal is None:
            raise ValueError(
                "Miter frames require a planar contour and a perpendicular normal; "
                "use transport for a spatial path"
            )
        basis = plane_basis(fitted_normal)
        validate_curve_path((path - path[0]) @ basis, closed, engine.settings)
        if closed and area((path - path[0]) @ basis) < 0:
            path = path[::-1].copy()
        vectors = miter_vectors(path, closed, fitted_normal, miter_limit)
        shift = (
            0.0
            if placement == "centered"
            else (width / 2 + clearance) * (1 if placement == "inside" else -1)
        )
        centerline = path + vectors * shift + fitted_normal * offset
        # Check both extreme lateral envelopes, not only the path centerline.
        for distance in (shift - width / 2, shift + width / 2):
            envelope = path + vectors * distance
            validate_curve_path((envelope - envelope[0]) @ basis, closed, engine.settings)
            edge = np.roll(path, -1, axis=0) - path if closed else np.diff(path, axis=0)
            changed_edge = (
                np.roll(envelope, -1, axis=0) - envelope if closed else np.diff(envelope, axis=0)
            )
            if np.any(np.einsum("ij,ij->i", edge, changed_edge) <= tolerance**2):
                raise ScissorsError("GEOMETRY_ERROR", "Frame section collapses or reverses an edge")
        rings = (
            centerline[:, None, :]
            + vectors[:, None, :] * polygon[None, :, 0, None]
            + fitted_normal[None, None, :] * polygon[None, :, 1, None]
        )
    else:
        spatial_path_check(path, closed, engine, tolerance)
        if placement != "centered" or clearance:
            raise ValueError("Transport frames use centered placement and zero lateral clearance")
        if offset and supplied_normal is None:
            raise ValueError("Spatial normal offset requires an explicit normal direction")
        edges = np.roll(path, -1, axis=0) - path if closed else np.diff(path, axis=0)
        directions = edges / np.linalg.norm(edges, axis=1)[:, None]
        turns = np.sum(directions[:-1] * directions[1:], axis=1)
        if closed:
            turns = np.append(turns, np.dot(directions[-1], directions[0]))
        if np.any(1 + turns < 2 / miter_limit**2 - 1e-10):
            raise ScissorsError("GEOMETRY_ERROR", "Spatial path corner exceeds miter_limit")
        frames = _tube_frames(_tube_tangents(directions, closed), closed)
        centerline = path + (supplied_normal * offset if supplied_normal is not None else 0)
        laterals, binormals = np.asarray(frames)[:, 0], np.asarray(frames)[:, 1]
        rings = (
            centerline[:, None, :]
            + laterals[:, None, :] * polygon[None, :, 0, None]
            + binormals[:, None, :] * polygon[None, :, 1, None]
        )
    mesh = sweep_rings(rings, end_policy != "open", closed, centers=centerline)
    obj = MeshObject(
        id=identifier("obj"),
        name=name,
        vertices=mesh.vertices.tolist(),
        faces=mesh.faces.tolist(),
        normals=compute_normals(mesh).tolist(),
        material_id=material_id,
        parent_id=parent_id,
        metadata={
            "primitive": "opening_frame",
            "source_object_id": object_id,
            "source_geometry_revision": revision,
            "joint_policy": joint_policy,
        },
    )
    engine.checked(obj)
    topology = topology_mesh(mesh)
    if closed or end_policy != "open":
        if not topology.is_volume:
            raise ScissorsError("GEOMETRY_ERROR", "Frame is not a closed oriented volume")
    checks: dict[str, Any] = {
        "path": "passed",
        "section": "passed",
        "planar_envelopes": "passed" if joint_policy == "miter" else "not_applicable",
    }
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ScissorsError("OPERATION_TIMEOUT", "Opening frame preparation timed out")
    try:
        check = self_intersections(
            obj, engine.settings.model_copy(update={"operation_timeout": remaining}), 0
        )
    except ScissorsError as exc:
        if not allow_incomplete_checks or exc.code not in {"RESOURCE_LIMIT", "OPERATION_TIMEOUT"}:
            raise
        checks["non_adjacent_self_intersections"] = {
            "status": "uncompleted",
            "error": exc.as_dict(),
        }
    else:
        if check["self_intersects"]:
            raise ScissorsError("GEOMETRY_ERROR", "Frame surfaces intersect", check)
        checks["non_adjacent_self_intersections"] = {
            "status": "passed",
            "checked_pairs": check["checked_pairs"],
            "adjacent_faces_excluded": True,
        }
    # Publish only after all required checks. The registry owns state validation and rollback.
    obj.metadata["frame_checks"] = copy.deepcopy(checks)
    scene.objects[obj.id] = obj
    return {
        **obj.summary(),
        "source_object_id": object_id,
        "source_geometry_revision": revision,
        "contour_index": contour_index,
        "contour_side": contour_side,
        "resolved_contour": {
            "points": path.tolist(),
            "closed": closed,
            "coordinate_space": "world",
        },
        "centerline": centerline.tolist(),
        "normal": (
            fitted_normal.tolist()
            if joint_policy == "miter" and fitted_normal is not None
            else supplied_normal.tolist()
            if supplied_normal is not None
            else None
        ),
        "planar": bool(planar),
        "winding": "counterclockwise_about_normal"
        if closed and joint_policy == "miter"
        else "input_order",
        "section": section,
        "section_dimensions": {"width": width, "depth": depth},
        "placement": placement,
        "offset": offset,
        "clearance": clearance,
        "tolerance": tolerance,
        "joint_policy": joint_policy,
        "end_policy": "joined" if closed else "open" if end_policy == "open" else "cap",
        "checks": checks,
        "warnings": (
            [
                "Non-adjacent surface checks are incomplete; "
                "do not treat this frame as checked for self-intersection."
            ]
            if checks["non_adjacent_self_intersections"]["status"] != "passed"
            else []
        )
        + [
            "Clearance is lateral to the selected planar contour; "
            "it is not a collision check against the source body.",
            "Frame/source contact and VR visibility are not checked by this operation.",
        ],
    }
