"""Exact planar partitions for material-interior normal-chord bounds."""

import math
import time
from fractions import Fraction as Q
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.face_queries import selection_indices
from scissors.core.mesh import MeshObject, face_components, topology_mesh
from scissors.core.topology_repair import verify_surface

Vec = tuple[Q, ...]
Point = tuple[Q, Q]
Plane = Vec
Polygon = list[Point]
Cell = tuple[Polygon, int | None, Plane | None, bool]
ZERO = Q(0)
ONE = Q(1)
DOMAIN: Polygon = [(ZERO, ZERO), (ONE, ZERO), (ZERO, ONE)]


def sub(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def dot(a: Vec, b: Vec) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), ZERO)


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def value(plane: Plane, point: Point) -> Q:
    return plane[0] + plane[1] * point[0] + plane[2] * point[1]


def area2(polygon: Polygon) -> Q:
    return abs(
        sum(
            (
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True)
            ),
            ZERO,
        )
    )


def clip(polygon: Polygon, plane: Plane) -> Polygon:
    """Intersect a convex polygon with an exact closed half-plane."""
    if not polygon:
        return []
    if len(polygon) > 4096:
        raise ScissorsError("RESOURCE_LIMIT", "Exact thickness polygon exceeds vertex budget")
    output = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        va, vb = value(plane, a), value(plane, b)
        if va >= 0:
            output.append(a)
        if (va < 0 < vb) or (vb < 0 < va):
            weight = va / (va - vb)
            output.append((a[0] + weight * (b[0] - a[0]), a[1] + weight * (b[1] - a[1])))
    if len(output) < 3 or not area2(output):
        return []
    if any(
        c.numerator.bit_length() + c.denominator.bit_length() > 16384
        for point in output
        for c in point
    ):
        raise ScissorsError("RESOURCE_LIMIT", "Exact thickness exceeds rational precision budget")
    return output


def negative(plane: Plane) -> Plane:
    return (-plane[0], -plane[1], -plane[2])


def difference(polygon: Polygon, planes: list[Plane]) -> tuple[list[Polygon], Polygon]:
    """Split into disjoint interiors outside and inside a convex intersection."""
    inside = polygon
    for plane in planes:
        inside = clip(inside, plane)
        if not inside:
            return [polygon], []
    outside = []
    remaining = polygon
    for plane in planes:
        if not remaining:
            break
        # A zero plane imposes no restriction. Splitting it would double coverage.
        if not any(plane):
            continue
        part = clip(remaining, negative(plane))
        if part:
            outside.append(part)
        remaining = clip(remaining, plane)
    return outside, remaining


def ray_planes(
    source: tuple[Vec, Vec, Vec], target: tuple[Vec, Vec, Vec], direction: Vec
) -> tuple[list[Plane], Plane, bool] | None:
    """Barycentric target coverage and ray parameter as affine source functions."""
    origin, first, second = source
    target_origin, edge1, edge2 = target
    h = cross(direction, edge2)
    determinant = dot(edge1, h)
    if not determinant:
        return None
    offsets = (sub(origin, target_origin), first, second)
    u = tuple(dot(offset, h) / determinant for offset in offsets)
    qs = [cross(offset, edge1) for offset in offsets]
    v = tuple(dot(direction, q) / determinant for q in qs)
    depth = tuple(dot(edge2, q) / determinant for q in qs)
    planes = [u, v, (ONE - u[0] - v[0], -u[1] - v[1], -u[2] - v[2]), depth]
    return planes, depth, determinant < 0


class Budget:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.deadline = time.monotonic() + settings.operation_timeout
        self.pairs = 0

    def check(self, work: int = 0, cells: int = 0) -> None:
        self.pairs += work
        if self.pairs > self.settings.max_geometry_pairs or cells > 4096:
            raise ScissorsError("RESOURCE_LIMIT", "Exact thickness exceeds comparison/cell budget")
        if cells * 32768 > self.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Exact thickness partitions exceed memory budget")
        if time.monotonic() > self.deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Exact thickness inspection timed out")


def bounds(squared: Q) -> tuple[float, float]:
    """Enclose the square root with floats, checking rounding against exact squares."""
    if not squared:
        return 0.0, 0.0
    # Decimal scaling avoids overflow/underflow in float(squared).
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 40
        approximate = float((Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt())
    if not math.isfinite(approximate):
        raise ScissorsError("RESOURCE_LIMIT", "Thickness exceeds finite output range")
    low = high = approximate
    while Q(low) ** 2 > squared:
        low = math.nextafter(low, -math.inf)
    while Q(high) ** 2 < squared:
        high = math.nextafter(high, math.inf)
    return low, high


def exact_contacts(
    data: list[tuple[Vec, Vec, Vec]], normals: list[Vec], triangles: np.ndarray, budget: Budget
) -> None:
    """Allow only the common vertex/edge of a triangle pair as their intersection."""
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    order = np.argsort(low[:, 0], kind="stable")
    sorted_low = low[order, 0]
    zero = (ZERO, ZERO, ZERO)
    for offset, i in enumerate(order):
        stop = np.searchsorted(sorted_low, high[i, 0], side="right")
        candidates = order[offset + 1 : stop]
        budget.check(len(candidates))
        overlap = np.all(high[candidates] >= low[i], axis=1)
        overlap &= np.all(low[candidates] <= high[i], axis=1)
        for j in candidates[overlap]:
            budget.check()
            vertices = []
            for index in (i, j):
                a, b, c = data[int(index)]
                vertices.append(
                    [a, tuple(a[k] + b[k] for k in range(3)), tuple(a[k] + c[k] for k in range(3))]
                )
            common = set(vertices[0]) & set(vertices[1])
            parallel = not any(cross(normals[int(i)], normals[int(j)]))
            if parallel:
                if dot(sub(vertices[0][0], vertices[1][0]), normals[int(j)]):
                    continue
                hit = ray_planes(data[int(i)], data[int(j)], normals[int(i)])
                assert hit is not None
                polygon = DOMAIN
                for plane in hit[0][:3]:
                    polygon = clip(polygon, plane)
                if polygon:
                    raise ScissorsError("GEOMETRY_ERROR", "Overlapping coplanar faces")
                for side, target in ((0, int(j)), (1, int(i))):
                    for point in vertices[side]:
                        hit = ray_planes((point, zero, zero), data[target], normals[target])
                        assert hit is not None
                        if point not in common and all(p[0] >= 0 for p in hit[0][:3]):
                            raise ScissorsError("GEOMETRY_ERROR", "Non-topological face contact")
            elif len(common) < 2:
                for side, target in ((0, int(j)), (1, int(i))):
                    points = vertices[side]
                    for a, b in zip(points, points[1:] + points[:1], strict=True):
                        direction = sub(b, a)
                        hit = ray_planes((a, zero, zero), data[target], direction)
                        if hit is None:
                            continue
                        planes, depth, _ = hit
                        if 0 <= depth[0] <= 1 and all(p[0] >= 0 for p in planes[:3]):
                            point = tuple(a[k] + depth[0] * direction[k] for k in range(3))
                            if point not in common:
                                raise ScissorsError("GEOMETRY_ERROR", "Intersecting boundary faces")


def check_nesting(
    data: list[tuple[Vec, Vec, Vec]],
    normals: list[Vec],
    components: list[np.ndarray],
    budget: Budget,
) -> None:
    """Disjoint closed boundary shells must alternate orientation when nested."""
    for component in components:
        budget.check()
        origin, a, b = data[int(component[0])]
        point = tuple(origin[k] + (a[k] + b[k]) / 3 for k in range(3))
        containers = 0
        for other in components:
            if component is other:
                continue
            found = False
            for attempt in range(1, 17):
                direction = (Q(1), Q(attempt), Q(attempt * attempt + 1))
                count = 0
                ambiguous = False
                for index in other:
                    budget.check(1)
                    hit = ray_planes((point, (ZERO,) * 3, (ZERO,) * 3), data[int(index)], direction)
                    if hit is None:
                        continue
                    planes, depth, _ = hit
                    if depth[0] <= 0 or any(plane[0] < 0 for plane in planes[:3]):
                        continue
                    if any(plane[0] == 0 for plane in planes[:3]):
                        ambiguous = True
                        break
                    count += 1
                if not ambiguous:
                    containers += count % 2
                    found = True
                    break
            if not found:
                raise ScissorsError("GEOMETRY_ERROR", "Component nesting is ambiguous")
        volume6 = sum((dot(data[int(i)][0], normals[int(i)]) for i in component), ZERO)
        if (volume6 > 0) != (containers % 2 == 0):
            raise ScissorsError("GEOMETRY_ERROR", "Nested boundary orientation is inconsistent")


def check_thickness(
    obj: MeshObject,
    minimum: float,
    tolerance: float,
    settings: Settings,
    selection: dict[str, Any] | None = None,
    rim_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if obj.kind != "mesh":
        raise ValueError("Thickness inspection requires a mesh")
    indices = (
        selection_indices(obj, selection).tolist()
        if selection is not None
        else list(range(len(obj.faces)))
    )
    explicit_rims = None
    if rim_selection is not None:
        if (
            rim_selection["object_id"] != obj.id
            or rim_selection["geometry_revision"] != obj.geometry_revision
        ):
            raise ScissorsError(
                "INVALID_PARAMETER", "Rim selection belongs to another geometry revision"
            )
        explicit_rims = (
            set(selection_indices(obj, rim_selection).tolist())
            if rim_selection["face_indices"]
            else set()
        )
    result: dict[str, Any] = {
        "object_id": obj.id,
        "geometry_revision": obj.geometry_revision,
        "status": "inconclusive",
        "minimum": minimum,
        "tolerance": tolerance,
        "minimum_measured_thickness": None,
        "samples": [],
        "violations": [],
        "excluded_rim_regions": [],
        "excluded_rim_exit_regions": [],
        "unresolved_regions": [],
        "method": "exact_face_normal_chord_partitions",
        "scope": "selected_face_interiors_excluding_rims_and_first_rim_exits",
        "coverage": {
            "complete": False,
            "sampled_faces": 0,
            "total_faces": len(obj.faces),
            "selected_faces": len(indices),
            "measured_area": 0.0,
            "excluded_exit_area": 0.0,
        },
        "achieved_error_bounds": {"global_lower": None, "global_upper": None},
        "corner_handling": "Infimum over face interiors; edge limits count. "
        "Zero-area tangent domains are excluded.",
    }
    from scissors.core.validation import validate_mesh

    basic = validate_mesh(obj, settings, require_closed=True)
    if not basic["valid"]:
        result["unresolved_regions"].append(
            {"reason": "invalid_or_unoriented_wall_volume", "validation": basic}
        )
        return result
    certificate = obj.metadata.get("_thickness_certificate", {})
    certified = certificate.get("signature") == obj.geometry_signature()
    if explicit_rims is None and certificate and not certified:
        result["unresolved_regions"].append(
            {
                "reason": "rim_provenance_changed",
                "message": "Provide a current rim_selection or rebuild the opening rim map.",
            }
        )
        return result
    rims = (
        explicit_rims
        if explicit_rims is not None
        else (set(certificate.get("rim_faces", [])) if certified else set())
    )
    result["excluded_rim_regions"] = sorted(rims)
    if (
        selection is None
        and explicit_rims is None
        and certified
        and (certificate.get("method") == "rectangular_miter_shell")
    ):
        # Preserve the analytical shell contract and its conservative numeric bounds.
        mesh = obj.geometry()
        error = max(float(mesh.scale) * 1e-12, np.finfo(float).eps) * 16
        if error > tolerance:
            result["unresolved_regions"].append(
                {"reason": "requested_error_below_numeric_resolution"}
            )
            return result
        bound = float(certificate["minimum"])
        result["method"] = "rectangular_miter_shell_plane_bounds"
        result["scope"] = "certified_rectangular_wall_excluding_rims"
        result["coverage"]["complete"] = True
        result["coverage"]["certified_faces"] = sum(i not in rims for i in indices)
        result["coverage"]["measured_area"] = None
        result["coverage"]["excluded_exit_area"] = None
        result["minimum_measured_thickness"] = bound
        result["achieved_error_bounds"] = {
            "global_lower": bound - error,
            "global_upper": bound + error,
        }
        result["status"] = (
            "pass"
            if bound - error >= minimum
            else "fail"
            if bound + error < minimum
            else "inconclusive"
        )
        if result["status"] == "fail":
            result["violations"] = [
                {
                    "reason": "certified_wall_below_minimum",
                    "thickness": bound,
                    "face_indices": [i for i in indices if i not in rims],
                }
            ]
        return result
    try:
        verify_surface(obj.geometry(), settings)
    except ScissorsError as error:
        result["unresolved_regions"].append(
            {"reason": "invalid_or_unverified_wall_volume", "error": error.as_dict()}
        )
        return result
    if len(obj.faces) * 8192 + len(obj.vertices) * 4096 > settings.max_memory_bytes:
        result["unresolved_regions"].append({"reason": "exact_geometry_memory_budget"})
        return result
    mesh = obj.geometry()
    points = [tuple(Q(float(c)) for c in vertex) for vertex in obj.vertices]
    data = [
        (points[a], sub(points[b], points[a]), sub(points[c], points[a])) for a, b, c in obj.faces
    ]
    normals = [cross(a, b) for _, a, b in data]
    budget = Budget(settings)
    minimum_squared: Q | None = None
    try:
        components = face_components(len(obj.faces), topology_mesh(mesh).face_adjacency)
        exact_contacts(data, normals, mesh.triangles, budget)
        check_nesting(data, normals, components, budget)
        for face in indices:
            if face in rims:
                continue
            budget.check()
            if len(result["samples"]) >= 4096:
                raise ScissorsError("RESOURCE_LIMIT", "Thickness output exceeds face budget")
            direction = tuple(-c for c in normals[face])
            # Each cell holds its first hit. Its interior is disjoint from all other cells.
            cells: list[Cell] = [(DOMAIN, None, None, False)]
            for target, geometry in enumerate(data):
                if target == face:
                    continue
                budget.check(1, len(cells))
                hit = ray_planes(data[face], geometry, direction)
                if hit is None:
                    continue
                planes, depth, exiting = hit
                # No positive-volume interior can end on the starting plane.
                if all(value(depth, point) <= 0 for point in DOMAIN):
                    continue
                next_cells: list[Cell] = []
                for polygon, previous, previous_depth, previous_exiting in cells:
                    budget.check(1, len(next_cells) + len(cells))
                    outside, inside = difference(polygon, planes)
                    next_cells.extend(
                        (part, previous, previous_depth, previous_exiting) for part in outside
                    )
                    if not inside:
                        continue
                    if previous_depth is None:
                        next_cells.append((inside, target, depth, exiting))
                    else:
                        nearer = tuple(a - b for a, b in zip(previous_depth, depth, strict=True))
                        if not any(nearer):
                            next_cells.append((inside, previous, previous_depth, previous_exiting))
                        else:
                            new_part, old_part = (
                                clip(inside, nearer),
                                clip(inside, negative(nearer)),
                            )
                            if new_part:
                                next_cells.append((new_part, target, depth, exiting))
                            if old_part:
                                next_cells.append(
                                    (old_part, previous, previous_depth, previous_exiting)
                                )
                cells = next_cells
                partition_size = sum(len(cell[0]) + 3 for cell in cells)
                if partition_size * 8192 > settings.max_memory_bytes:
                    raise ScissorsError(
                        "RESOURCE_LIMIT", "Thickness partitions exceed memory budget"
                    )
            if sum((area2(cell[0]) for cell in cells), ZERO) != ONE:
                raise ScissorsError("GEOMETRY_ERROR", "Thickness partition coverage is incomplete")
            norm_squared = dot(direction, direction)
            face_minimum: tuple[Q, Point, int] | None = None
            measured_area = 0.0
            for polygon, cell_target, cell_depth, cell_exiting in cells:
                budget.check(0, len(cells))
                region_area = float(area2(polygon)) * float(mesh.area_faces[face])
                if cell_target is None or cell_depth is None or not cell_exiting:
                    result["unresolved_regions"].append(
                        {"face_index": face, "reason": "no_material_exit"}
                    )
                    continue
                if cell_target in rims:
                    result["excluded_rim_exit_regions"].append(
                        {
                            "face_index": face,
                            "opposing_face_index": cell_target,
                            "area": region_area,
                        }
                    )
                    result["coverage"]["excluded_exit_area"] += region_area
                    continue
                parameter, point = min((value(cell_depth, point), point) for point in polygon)
                squared = parameter * parameter * norm_squared
                measured_area += region_area
                if face_minimum is None or squared < face_minimum[0]:
                    face_minimum = squared, point, cell_target
            if face_minimum is not None:
                squared, point, target = face_minimum
                low, high = bounds(squared)
                location = [
                    float(
                        data[face][0][k] + point[0] * data[face][1][k] + point[1] * data[face][2][k]
                    )
                    for k in range(3)
                ]
                sample = {
                    "face_index": face,
                    "opposing_face_index": target,
                    "location": location,
                    "thickness": low + (high - low) / 2,
                    "lower_bound": low,
                    "upper_bound": high,
                    "error_bound": high - low,
                    "meaning": "measured_face_domain_infimum",
                }
                result["samples"].append(sample)
                if high < minimum:
                    result["violations"].append(sample)
                minimum_squared = (
                    squared if minimum_squared is None else min(minimum_squared, squared)
                )
                result["coverage"]["sampled_faces"] += 1
                result["coverage"]["measured_area"] += measured_area
        result["coverage"]["complete"] = not result["unresolved_regions"]
    except ScissorsError as error:
        result["unresolved_regions"].append(
            {"reason": "uncompleted_exact_check", "error": error.as_dict()}
        )
    result["comparison_count"] = budget.pairs
    if minimum_squared is None:
        result["coverage"]["complete"] = False
        result["unresolved_regions"].append({"reason": "empty_measured_domain"})
        return result
    low, high = bounds(minimum_squared)
    result["minimum_measured_thickness"] = low + (high - low) / 2
    result["achieved_error_bounds"]["global_upper"] = high
    if result["coverage"]["complete"]:
        result["achieved_error_bounds"]["global_lower"] = low
        if high - low <= tolerance:
            result["status"] = (
                "pass" if low >= minimum else "fail" if high < minimum else "inconclusive"
            )
        else:
            result["unresolved_regions"].append(
                {"reason": "requested_error_below_numeric_resolution"}
            )
    elif result["violations"]:
        result["status"] = "fail"
    return result
