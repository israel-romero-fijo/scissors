"""Shared profile inspection and loft mesh construction."""

from typing import Any

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.mesh import MeshObject
from scissors.core.profiles import area, catmull, triangulate, validate_curve_path


def polygon_normal(points: np.ndarray) -> np.ndarray:
    shifted = points - points[0]
    normal = np.cross(shifted, np.roll(shifted, -1, axis=0)).sum(axis=0)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ScissorsError("GEOMETRY_ERROR", "Loft contour has zero signed area")
    return normal / length


def valid_stored_rings(stored: Any, recovered: list[list[int]], vertices: np.ndarray) -> bool:
    """Retain source indexing only when it still describes the actual boundary."""
    if not isinstance(stored, list) or len(stored) != len(recovered):
        return False
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)

    def edges(ring: list[int]) -> set[tuple[int, int]]:
        ids = inverse[ring].tolist()
        return {tuple(sorted((a, b))) for a, b in zip(ids, ids[1:] + ids[:1], strict=True)}

    available = [edges(ring) for ring in recovered]
    matched = []
    for ring in stored:
        if (
            not isinstance(ring, list)
            or len(ring) < 3
            or any(type(i) is not int or not 0 <= i < len(vertices) for i in ring)
            or len(set(inverse[ring])) != len(ring)
        ):
            return False
        value = edges(ring)
        if value not in available:
            return False
        index = available.index(value)
        if index in matched:
            return False
        matched.append(index)
    return matched[0] == 0


def _budget(settings: Settings, vertices: int, faces: int) -> None:
    if (
        vertices > settings.max_vertices
        or faces > settings.max_faces
        or vertices * 384 + faces * 192 > settings.max_memory_bytes
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Loft exceeds geometry budget")


def profile_source(obj: MeshObject, preserve_order: bool = False) -> MeshObject:
    if preserve_order:
        obj = obj.model_copy(deep=True)
    stored = obj.metadata.get("boundary_rings")
    primitive = obj.metadata.get("primitive", "custom")
    if primitive not in {"profile", "curve"}:
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Input must be a profile or curve",
            {"object_id": obj.id, "primitive": primitive},
        )
    vertices = np.asarray(obj.vertices)
    points, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    faces = inverse[np.asarray(obj.faces)]
    normal = obj.geometry().face_normals.mean(axis=0)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ScissorsError("GEOMETRY_ERROR", "Profile has inconsistent face orientation")
    normal /= length
    span = np.max(np.ptp(points, axis=0)) if preserve_order else np.ptp(points)
    if np.max(np.abs((points - points[0]) @ normal)) > max(span, 1) * 1e-8:
        raise ScissorsError("GEOMETRY_ERROR", "Profile is no longer planar")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, edge_inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = edges[counts[edge_inverse] == 1]
    following: dict[int, int] = {}
    for a, b in boundary:
        if int(a) in following:
            raise ScissorsError("GEOMETRY_ERROR", "Profile has a branched boundary")
        following[int(a)] = int(b)
    rings = []
    while following:
        start = min(following)
        ring = [start]
        current = following.pop(start)
        while current != start:
            if current not in following or current in ring:
                raise ScissorsError("GEOMETRY_ERROR", "Profile boundary is not a closed contour")
            ring.append(current)
            current = following.pop(current)
        rings.append([int(first[index]) for index in ring])
    if not rings:
        raise ScissorsError("GEOMETRY_ERROR", "Profile has no boundary contours")

    def ring_area(ring: list[int]) -> float:
        contour = vertices[ring]
        return abs(float(np.sum(np.cross(contour, np.roll(contour, -1, axis=0)), axis=0) @ normal))

    recovered = sorted(rings, key=ring_area, reverse=True)
    use_stored = preserve_order and valid_stored_rings(stored, recovered, vertices)
    obj.metadata["boundary_rings"] = stored if use_stored else recovered
    if preserve_order:
        obj.metadata["loft_boundary_order"] = "stored" if use_stored else "derived"
    return obj


def curve_path(obj: MeshObject) -> tuple[np.ndarray, list[list[int]]]:
    """Check ribbon indexing before recovering its centerline."""
    vertices = np.asarray(obj.vertices)
    pairs = obj.metadata.get("center_pairs")
    if (
        not isinstance(pairs, list)
        or len(pairs) < 2
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(i) is not int or not 0 <= i < len(vertices) for i in pair)
            for pair in pairs
        )
    ):
        raise ScissorsError(
            "GEOMETRY_ERROR", "Curve center-pair metadata is invalid", {"object_id": obj.id}
        )
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    ids = inverse[np.asarray(pairs)]
    expected = []
    for i in range(len(pairs) if obj.metadata["closed"] else len(pairs) - 1):
        j = (i + 1) % len(pairs)
        a, b = ids[i]
        c, d = ids[j]
        expected.extend([tuple(sorted((a, b, d))), tuple(sorted((a, d, c)))])
    actual = [tuple(sorted(face)) for face in inverse[np.asarray(obj.faces)]]
    if sorted(actual) != sorted(expected):
        raise ScissorsError(
            "GEOMETRY_ERROR",
            "Curve center-pair metadata no longer matches faces",
            {"object_id": obj.id},
        )
    return vertices[np.asarray(pairs)].mean(axis=1), pairs


def prepare_preserved(
    objects: list[MeshObject],
    correspondence: list[dict[str, Any]],
    settings: Settings,
    spine_closed: bool = False,
) -> tuple[np.ndarray, list[int], bool, dict[str, Any]]:
    """Map validated source rings without resampling or implicit alignment."""
    by_id = {item["object_id"]: item for item in correspondence}
    if len(by_id) != len(correspondence) or set(by_id) - {obj.id for obj in objects}:
        raise ScissorsError("INVALID_PARAMETER", "Correspondence requires unique loft input IDs")
    curves = objects[0].metadata["primitive"] == "curve"
    sections_closed = not curves or bool(objects[0].metadata["closed"])
    prepared = []
    sections = []
    counts: list[int] = []
    normals = []
    for obj in objects:
        hint = by_id.get(obj.id)
        if hint is not None and hint["geometry_revision"] != obj.geometry_revision:
            raise ScissorsError(
                "INVALID_PARAMETER",
                "Loft correspondence has a stale source revision",
                {"object_id": obj.id, "geometry_revision": obj.geometry_revision},
            )
        if curves:
            if bool(obj.metadata["closed"]) != sections_closed:
                raise ScissorsError("GEOMETRY_ERROR", "Curve sections must agree on closure")
            path, pairs = curve_path(obj)
            rings = [path]
            references = [pairs]
        else:
            indices = obj.metadata["boundary_rings"]
            rings = [np.asarray(obj.vertices)[ring] for ring in indices]
            references = indices
        if len(rings) > 1 and hint is None:
            raise ScissorsError(
                "INVALID_PARAMETER",
                "Profiles with holes require explicit ring correspondence",
                {"object_id": obj.id},
            )
        mappings: list[dict[str, Any]] = (
            hint["rings"] if hint else [{"source_ring": 0, "start_index": None, "reverse": False}]
        )
        selected = [mapping["source_ring"] for mapping in mappings]
        if (
            len(selected) != len(rings)
            or sorted(selected) != list(range(len(rings)))
            or selected[0] != 0
        ):
            raise ScissorsError(
                "INVALID_PARAMETER",
                "Map the outer ring first and each hole exactly once",
                {"object_id": obj.id},
            )
        section = []
        ring_details = []
        for mapping in mappings:
            source_index = mapping["source_ring"]
            ring = rings[source_index]
            count = len(ring)
            if count > 512:
                raise ScissorsError("RESOURCE_LIMIT", "Loft ring exceeds 512 points")
            if len(np.unique(ring, axis=0)) != count or count < (3 if sections_closed else 2):
                raise ScissorsError(
                    "GEOMETRY_ERROR", "Loft ring has repeated or insufficient points"
                )
            edges = np.roll(ring, -1, axis=0) - ring if sections_closed else np.diff(ring, axis=0)
            if np.any(np.linalg.norm(edges, axis=1) <= 1e-12):
                raise ScissorsError("GEOMETRY_ERROR", "Loft ring has degenerate edges")
            reverse = mapping["reverse"]
            start = mapping["start_index"]
            if start is None:
                start = 0 if sections_closed or not reverse else count - 1
            if not 0 <= start < count or (
                not sections_closed and start != (count - 1 if reverse else 0)
            ):
                raise ScissorsError(
                    "INVALID_PARAMETER",
                    "Invalid loft ring starting index",
                    {"object_id": obj.id, "source_ring": source_index},
                )
            order = (start + (-1 if reverse else 1) * np.arange(count)) % count
            section.append(ring[order])
            ring_details.append(
                {
                    "source_ring": source_index,
                    "start_index": start,
                    "reverse": reverse,
                    "source_indices": [references[source_index][i] for i in order],
                }
            )
        current_counts = [len(ring) for ring in section]
        if counts and counts != current_counts:
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Preserved loft rings require equal corresponding vertex counts",
                {"object_id": obj.id, "expected": counts, "actual": current_counts},
            )
        counts = current_counts
        points = np.concatenate(section)
        normal = polygon_normal(section[0]) if sections_closed else obj.geometry().face_normals[0]
        span = float(np.max(np.ptp(points, axis=0)))
        if np.max(np.abs((points - points[0]) @ normal)) > max(span, 1) * 1e-8:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Loft section is not planar", {"object_id": obj.id}
            )
        u = section[0][1] - section[0][0]
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        flat = [
            np.column_stack([(ring - points[0]) @ u, (ring - points[0]) @ v]) for ring in section
        ]
        for ring in flat:
            validate_curve_path(ring, sections_closed, settings)
        if sections_closed:
            if any(area(ring) >= -1e-12 for ring in flat[1:]):
                raise ScissorsError("GEOMETRY_ERROR", "Hole winding must oppose the outer ring")
            triangulate(flat[0].tolist(), [ring.tolist() for ring in flat[1:]], settings)
        normals.append(normal)
        prepared.append(points)
        sections.append(
            {
                "object_id": obj.id,
                "geometry_revision": obj.geometry_revision,
                "boundary_order": obj.metadata["loft_boundary_order"],
                "source_index_kind": "center_pairs" if curves else "render_vertices",
                "rings": ring_details,
            }
        )
    normal_pairs = list(zip(normals[:-1], normals[1:], strict=True))
    if spine_closed:
        normal_pairs.append((normals[-1], normals[0]))
    for i, (first, second) in enumerate(normal_pairs):
        if float(first @ second) < -1 + 1e-8:
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Parallel section winding disagrees; use reverse hints",
                {"section_pair": [i, (i + 1) % len(objects)]},
            )
    _budget(settings, sum(counts) * len(objects), 2 * sum(counts) * len(objects))
    return (
        np.asarray(prepared),
        counts,
        sections_closed,
        {
            "preserve_vertices": True,
            "sections_closed": sections_closed,
            "ring_vertex_counts": counts,
            "sections": sections,
            "planarity_relative_tolerance": 1e-8,
            "edge_tolerance": 1e-12,
        },
    )


def build_mesh(
    arrays: np.ndarray,
    counts: list[int],
    objects: list[MeshObject],
    sections_closed: bool,
    smooth: bool,
    caps: bool,
    closed: bool,
    settings: Settings,
    preserve_vertices: bool = False,
) -> trimesh.Trimesh:
    if np.any(np.linalg.norm(np.diff(arrays.mean(axis=1), axis=0), axis=1) <= 1e-12):
        raise ScissorsError(
            "GEOMETRY_ERROR", "Loft sections are coplanar or have coincident centers"
        )
    pairs = list(zip(arrays[:-1], arrays[1:], strict=True))
    if closed:
        pairs.append((arrays[-1], arrays[0]))
    for index, (first, second) in enumerate(pairs):
        points = np.concatenate([first, second])
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[-1] <= max(float(singular[0]), 1) * 1e-10:
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Loft sections are coplanar",
                {"section_pair": [index, (index + 1) % len(arrays)]},
            )
    n = sum(counts)
    section_count = len(arrays) * 4 if closed else (len(arrays) - 1) * 4 + 1
    _budget(
        settings,
        n * (section_count if smooth else len(arrays)),
        2 * n * (section_count if smooth else len(arrays)) + 4 * n,
    )
    if smooth:
        arrays = np.stack([catmull(arrays[:, i], 4, closed) for i in range(n)], axis=1)
    vertices = arrays.reshape(-1, 3)
    faces = []
    for layer in range(len(arrays) if closed else len(arrays) - 1):
        following = (layer + 1) % len(arrays)
        offset = 0
        for count in counts:
            for i in range(count if sections_closed else count - 1):
                j = (i + 1) % count
                a, b = layer * n + offset + i, layer * n + offset + j
                c, d = following * n + offset + j, following * n + offset + i
                faces.extend([[a, b, c], [a, c, d]])
            offset += count
    if caps and sections_closed and not closed:
        from scissors.core.profiles import triangulate

        for layer, flip in ((0, True), (len(arrays) - 1, False)):
            section = arrays[layer]
            origin = section[0]
            u = section[1] - origin
            u /= np.linalg.norm(u)
            normal = (
                polygon_normal(section[: counts[0]])
                if preserve_vertices
                else objects[0 if layer == 0 else -1].geometry().face_normals[0]
            )
            v = np.cross(normal, u)
            flat = np.column_stack([(section - origin) @ u, (section - origin) @ v])
            rings = []
            offset = 0
            for count in counts:
                rings.append(flat[offset : offset + count].tolist())
                offset += count
            cap_vertices, cap_faces, _, _ = triangulate(rings[0], rings[1:], settings)
            # Triangulation may reverse contour order; map its points to ring indices.
            mapping = [
                int(np.argmin(np.linalg.norm(flat - point, axis=1))) for point in cap_vertices
            ]
            for face in cap_faces:
                ids = [mapping[index] + layer * n for index in face]
                faces.append(ids[::-1] if flip else ids)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def preserved_metadata(
    mesh: trimesh.Trimesh,
    counts: list[int],
    sections_closed: bool,
    caps: bool,
    closed: bool,
    smooth: bool,
    details: dict[str, Any],
    settings: Settings,
) -> None:
    from scissors.core.inspection import self_intersections
    from scissors.core.mesh import topology_mesh
    from scissors.core.validation import validate_mesh

    _budget(settings, 3 * len(mesh.faces), len(mesh.faces))
    candidate = MeshObject(
        id="loft-candidate",
        name="Loft candidate",
        vertices=mesh.vertices.tolist(),
        faces=mesh.faces.tolist(),
    )
    report = validate_mesh(candidate, settings)
    if not report["valid"]:
        raise ScissorsError("GEOMETRY_ERROR", "Loft mesh validation failed", report)
    if np.any(mesh.area_faces <= 1e-14):
        raise ScissorsError("GEOMETRY_ERROR", "Loft correspondence produces degenerate faces")
    topology = topology_mesh(mesh)
    if (
        sections_closed
        and (caps or closed)
        and (not topology.is_watertight or not topology.is_winding_consistent)
    ):
        raise ScissorsError(
            "GEOMETRY_ERROR", "Loft correspondence does not form an oriented closed surface"
        )
    try:
        check = self_intersections(candidate, settings, 1e-9)
    except ScissorsError as exc:
        if exc.code not in {"RESOURCE_LIMIT", "OPERATION_TIMEOUT"}:
            raise
        check = {"status": "unresolved", "reason": exc.code}
    else:
        if check["self_intersects"]:
            raise ScissorsError("GEOMETRY_ERROR", "Loft surface intersects itself", check)
        check["status"] = "passed"
        check.pop("object_id", None)
    details.update(
        {
            "spine_closed": closed,
            "caps_applied": caps and sections_closed and not closed,
            "layers_per_source_segment": 4 if smooth else 1,
            "geometry_vertex_count": len(mesh.vertices),
            "layer_count": len(mesh.vertices) // sum(counts),
            "self_intersection_check": check,
            "render_vertex_to_geometry_vertex": mesh.faces.reshape(-1).tolist(),
            "seam_policy": (
                "box UV splits vertices per triangle; mapping indexes pre-UV loft vertices"
            ),
            "self_intersection_policy": (
                "nonadjacent triangle contact within 1e-9 world units; "
                "triangle pairs sharing any exact position are excluded"
            ),
            "warnings": (
                ["Surface intersection check did not complete."]
                if check["status"] == "unresolved"
                else []
            ),
        }
    )
    n = sum(counts)
    for i, section in enumerate(details["sections"]):
        layer = i * (4 if smooth else 1)
        section["layer"] = layer
        offset = 0
        for ring, count in zip(section["rings"], counts, strict=True):
            ring["geometry_vertex_indices"] = list(
                range(layer * n + offset, layer * n + offset + count)
            )
            offset += count
