"""Deterministic native panel, tube, louver and switch geometry."""

import math
from typing import Any

import numpy as np
import trimesh

# Fraction of a pitch or of an opening kept free so that separate shells never
# touch or overlap.
CLEARANCE = 0.02
# The switch grid has no side-count argument in its public contract, so the
# toggles use one fixed tessellation.
TOGGLE_SIDES = 12
# Every toggle is the +Z axis rotated around +X by this angle.
TOGGLE_TILT_DEGREES = 20.0


def _number(
    value: Any, name: Any, *, greater_than: Any = None, minimum: Any = None, maximum: Any = None
) -> Any:
    """Validate a numeric argument and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if greater_than is not None and number <= greater_than:
        raise ValueError(f"{name} must be greater than {greater_than}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _integer(value: Any, name: Any, minimum: Any, maximum: Any) -> Any:
    """Validate an integer argument inside an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _text(value: Any, name: Any) -> Any:
    """Validate a non-empty string argument."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: Any) -> Any:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _quads_to_triangles(quads: Any) -> Any:
    """Split every quad into two triangles without adding vertices."""
    triangles = []
    for first, second, third, fourth in quads:
        triangles.append([first, second, third])
        triangles.append([first, third, fourth])
    return triangles


def _fan(center: Any, ring: Any, flip: Any = False) -> Any:
    """Triangles joining one center vertex to every edge of a closed ring.

    The ring is counter-clockwise around its axis, so its polygon normal is the
    axis; ``flip`` reverses that to build the opposite cap.
    """
    triangles = []
    for index, current in enumerate(ring):
        following = ring[(index + 1) % len(ring)]
        triangles.append([center, following, current] if flip else [center, current, following])
    return triangles


def _oriented_box(center: Any, half_x: Any, half_y: Any, half_z: Any) -> Any:
    """Closed box on the right-handed basis (half_x, half_y, half_z), wound outward."""
    vertices: list[Any] = []
    index = {}
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            for sign_z in (-1, 1):
                index[(sign_x, sign_y, sign_z)] = len(vertices)
                vertices.append(center + sign_x * half_x + sign_y * half_y + sign_z * half_z)
    quads = [
        ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
        ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),
        ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),
    ]
    return vertices, _quads_to_triangles([[index[key] for key in quad] for quad in quads])


def _unit(vector: Any, message: Any) -> Any:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(message)
    return np.asarray(vector, dtype=float) / length


def _perpendicular(axis: Any) -> Any:
    """Deterministic unit vector perpendicular to a unit axis.

    The world axis with the smallest component is used as the reference, so the
    result never depends on floating-point tuning or on call order.
    """
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(axis)))] = 1.0
    return _unit(np.cross(axis, reference), "Axis has no usable perpendicular direction")


def _project(vector: Any, axis: Any) -> Any:
    """Unit projection of a vector into the plane perpendicular to a unit axis."""
    return _unit(vector - axis * float(np.dot(vector, axis)), "Cross-section frame collapsed")


def _rotate(vector: Any, axis: Any, angle: Any) -> Any:
    """Rodrigues rotation of a vector around a unit axis."""
    cross = np.cross(axis, vector)
    cosine, sine = math.cos(angle), math.sin(angle)
    return vector * cosine + cross * sine + axis * float(np.dot(axis, vector)) * (1.0 - cosine)


def _swing(source: Any, target: Any, vector: Any) -> Any:
    """Rotate a vector by the smallest rotation mapping unit source onto unit target."""
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-12:
        return vector.copy() if float(np.dot(source, target)) >= 0 else -vector
    return _rotate(vector, cross / sine, math.atan2(sine, float(np.dot(source, target))))


def _circle_offsets(normal: Any, binormal: Any, radius: Any, sides: Any) -> Any:
    """Offsets of a circle around the origin in the plane spanned by (normal, binormal)."""
    offsets = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        offsets.append(radius * (math.cos(angle) * normal + math.sin(angle) * binormal))
    return offsets


def _capped_cylinder(base: Any, axis: Any, length: Any, radius: Any, sides: Any) -> Any:
    """Closed cylinder whose base circle is centered on ``base`` and swept along ``axis``."""
    normal = _perpendicular(axis)
    binormal = np.cross(axis, normal)
    offsets = _circle_offsets(normal, binormal, radius, sides)
    top = base + axis * length
    vertices = [base]
    vertices.extend(base + offset for offset in offsets)
    vertices.extend(top + offset for offset in offsets)
    vertices.append(top)
    bottom_center, top_center = 0, len(vertices) - 1
    lower = list(range(1, sides + 1))
    upper = list(range(sides + 1, 2 * sides + 1))
    faces = _fan(bottom_center, lower, flip=True) + _fan(top_center, upper)
    faces += _quads_to_triangles(
        [[lower[i], lower[(i + 1) % sides], upper[(i + 1) % sides], upper[i]] for i in range(sides)]
    )
    return vertices, faces


def rounded_box(
    width: float,
    depth: float,
    height: float,
    corner_radius: float = 0,
    segments: int = 4,
    name: str = "Rounded box",
) -> trimesh.Trimesh:
    """Solid box with four filleted vertical edges, centered in XY with its base at z = 0."""
    name = _text(name, "name")
    width = _number(width, "width", greater_than=0)
    depth = _number(depth, "depth", greater_than=0)
    height = _number(height, "height", greater_than=0)
    corner_radius = _number(corner_radius, "corner_radius", minimum=0)
    segments = _integer(segments, "segments", 1, 16)
    limit = min(width, depth) / 2
    if corner_radius > limit:
        raise ValueError(f"corner_radius must not exceed min(width, depth) / 2 = {limit}")
    if corner_radius <= 0:
        # A zero radius degrades to a plain box: 8 vertices and 12 triangles.
        vertices, faces = _oriented_box(
            np.array([0.0, 0.0, height / 2]),
            np.array([width / 2, 0.0, 0.0]),
            np.array([0.0, depth / 2, 0.0]),
            np.array([0.0, 0.0, height / 2]),
        )
    else:
        outline = _rounded_outline(width, depth, corner_radius, segments)
        count = len(outline)
        vertices = [np.array([x, y, 0.0]) for x, y in outline]
        vertices += [np.array([x, y, height]) for x, y in outline]
        top_center = len(vertices)
        vertices.append(np.array([0.0, 0.0, height]))
        bottom_center = len(vertices)
        vertices.append(np.array([0.0, 0.0, 0.0]))
        lower = list(range(count))
        upper = list(range(count, 2 * count))
        faces = _fan(bottom_center, lower, flip=True) + _fan(top_center, upper)
        faces += _quads_to_triangles(
            [
                [lower[i], lower[(i + 1) % count], upper[(i + 1) % count], upper[i]]
                for i in range(count)
            ]
        )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _rounded_outline(width: Any, depth: Any, radius: Any, segments: Any) -> Any:
    """Counter-clockwise XY outline of a rectangle with four filleted corners.

    Every corner contributes ``segments`` chords: the points at both tangent ends
    are included, so the arc reaches the straight sides. A point that repeats its
    neighbour is dropped, which happens when a fillet consumes a whole side
    (``radius == min(width, depth) / 2``) and two corner centers coincide.
    """
    half_width, half_depth = width / 2, depth / 2
    corners = (
        (half_width - radius, half_depth - radius, 0.0),
        (-(half_width - radius), half_depth - radius, 90.0),
        (-(half_width - radius), -(half_depth - radius), 180.0),
        (half_width - radius, -(half_depth - radius), 270.0),
    )
    step = 90.0 / segments
    outline = []
    for center_x, center_y, start in corners:
        for index in range(segments + 1):
            angle = math.radians(start + index * step)
            outline.append(
                (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
            )
    tolerance = 1e-12 * max(width, depth)
    unique: list[Any] = []
    for point in outline:
        if not unique or math.dist(point, unique[-1]) > tolerance:
            unique.append(point)
    if len(unique) > 1 and math.dist(unique[0], unique[-1]) <= tolerance:
        unique.pop()
    return unique


def beveled_box(size: list[float], radius: float, segments: int, bevel: float) -> trimesh.Trimesh:
    """Inset the upper/lower outline while retaining the straight middle walls."""
    width, depth, height = size
    if bevel >= min(width, depth, height) / 2:
        raise ValueError("edge_bevel must be smaller than half of every box dimension")
    outer = np.asarray(_rounded_outline(width, depth, radius, segments))
    if radius == 0:
        inner = outer * [(width - 2 * bevel) / width, (depth - 2 * bevel) / depth]
    else:

        def raw_outline(w: float, d: float, r: float) -> np.ndarray:
            points = []
            for x, y, start in (
                (w / 2 - r, d / 2 - r, 0),
                (-w / 2 + r, d / 2 - r, 90),
                (-w / 2 + r, -d / 2 + r, 180),
                (w / 2 - r, -d / 2 + r, 270),
            ):
                for i in range(segments + 1):
                    angle = math.radians(start + i * 90 / segments)
                    points.append([x + r * math.cos(angle), y + r * math.sin(angle)])
            return np.asarray(points)

        raw = raw_outline(width, depth, radius)
        indices = np.argmin(np.linalg.norm(outer[:, None] - raw[None], axis=2), axis=1)
        inner = raw_outline(width - 2 * bevel, depth - 2 * bevel, max(0, radius - bevel))[indices]
    n = len(outer)
    vertices = []
    for outline, z in ((inner, 0), (outer, bevel), (outer, height - bevel), (inner, height)):
        vertices.extend([[float(x), float(y), z] for x, y in outline])
    vertices.extend([[0, 0, 0], [0, 0, height]])
    faces = _fan(4 * n, list(range(n)), flip=True) + _fan(4 * n + 1, list(range(3 * n, 4 * n)))
    for layer in range(3):
        for i in range(n):
            j = (i + 1) % n
            faces.extend(
                [
                    [layer * n + i, layer * n + j, (layer + 1) * n + j],
                    [layer * n + i, (layer + 1) * n + j, (layer + 1) * n + i],
                ]
            )
    points, mapping = np.unique(np.asarray(vertices), axis=0, return_inverse=True)
    triangles = mapping[np.asarray(faces)]
    mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=False)
    mesh.update_faces(mesh.area_faces > 1e-14)
    mesh.remove_unreferenced_vertices()
    return mesh


def tube(
    points: Any,
    radius: float,
    sides: int = 8,
    caps: bool = True,
    closed: bool = False,
    name: str = "Tube",
    max_points: int = 256,
    max_sides: int = 32,
) -> trimesh.Trimesh:
    """Closed or open tube swept along a polyline with parallel-transported frames."""
    name = _text(name, "name")
    radius = _number(radius, "radius", greater_than=0)
    sides = _integer(sides, "sides", 3, max_sides)
    caps = _boolean(caps, "caps")
    closed = _boolean(closed, "closed")
    path = _tube_points(points, max_points)
    count = len(path)
    directions = []
    for index in range(count if closed else count - 1):
        following = (index + 1) % count
        directions.append(
            _unit(path[following] - path[index], "Consecutive points must be distinct")
        )
    tangents = _tube_tangents(directions, closed)
    frames = _tube_frames(tangents, closed)
    vertices: list[Any] = []
    for point, frame in zip(path, frames, strict=True):
        normal, binormal = frame
        vertices.extend(
            point + offset for offset in _circle_offsets(normal, binormal, radius, sides)
        )
    return sweep_rings(np.asarray(vertices).reshape(count, sides, 3), caps, closed, centers=path)


def sweep_rings(
    sections: np.ndarray,
    caps: bool,
    closed: bool,
    centers: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Connect ordered world-space section rings; shared by tubes and opening frames."""
    count, sides, _ = sections.shape
    vertices = list(sections.reshape(-1, 3))
    rings = np.arange(count * sides).reshape(count, sides).tolist()
    segments = count if closed else count - 1
    faces = []
    for index in range(segments):
        current, next_ring = rings[index], rings[(index + 1) % count]
        for side in range(sides):
            step = (side + 1) % sides
            faces.append([current[side], current[step], next_ring[step]])
            faces.append([current[side], next_ring[step], next_ring[side]])
    if caps and not closed:
        start_center = len(vertices)
        vertices.append(sections[0].mean(axis=0) if centers is None else centers[0].copy())
        end_center = len(vertices)
        vertices.append(sections[-1].mean(axis=0) if centers is None else centers[-1].copy())
        # The start cap faces backwards along the first tangent; the end cap faces forward.
        faces += _fan(start_center, rings[0], flip=True)
        faces += _fan(end_center, rings[-1])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _tube_points(points: Any, max_points: int = 256) -> Any:
    """Validate a polyline and return it as an (n, 3) float array."""
    if not isinstance(points, (list, tuple)):
        raise ValueError("points must be an array of 3-element [x, y, z] arrays")
    if not 2 <= len(points) <= max_points:
        raise ValueError(f"points must contain between 2 and {max_points} points")
    path = np.empty((len(points), 3), dtype=float)
    for index, point in enumerate(points):
        try:
            values = np.asarray(point, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"points[{index}] must be a 3-element numeric array") from exc
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"points[{index}] must be a finite 3-element array")
        path[index] = values
    return path


def _tube_tangents(directions: Any, closed: Any) -> Any:
    """Unit tangents at every point; interior joints are mitered by averaging."""
    count = len(directions) if closed else len(directions) + 1
    tangents = []
    for index in range(count):
        if closed:
            incoming, outgoing = directions[index - 1], directions[index]
        elif index == 0:
            incoming = outgoing = directions[0]
        elif index == count - 1:
            incoming = outgoing = directions[-1]
        else:
            incoming, outgoing = directions[index - 1], directions[index]
        total = incoming + outgoing
        if float(np.linalg.norm(total)) < 1e-6:
            raise ValueError(
                f"points[{index}] reverses direction too sharply to miter a cross-section"
            )
        tangents.append(total / float(np.linalg.norm(total)))
    return tangents


def _tube_frames(tangents: Any, closed: Any) -> Any:
    """Parallel-transported (normal, binormal) frames, so the tube does not twist."""
    normals = [_perpendicular(tangents[0])]
    for index in range(1, len(tangents)):
        candidate = _swing(tangents[index - 1], tangents[index], normals[-1])
        normals.append(_project(candidate, tangents[index]))
    if closed:
        # Parallel transport around a loop returns a frame rotated by the loop
        # holonomy; spread the mismatch over every joint so the seam closes.
        wrapped = _project(_swing(tangents[-1], tangents[0], normals[-1]), tangents[0])
        angle = math.atan2(
            float(np.dot(np.cross(normals[0], wrapped), tangents[0])),
            float(np.dot(normals[0], wrapped)),
        )
        count = len(tangents)
        normals = [
            _rotate(normal, tangent, -angle * index / count)
            for index, (normal, tangent) in enumerate(zip(normals, tangents, strict=True))
        ]
    return [
        (normal, np.cross(tangent, normal))
        for normal, tangent in zip(normals, tangents, strict=True)
    ]


def louver(
    width: float,
    height: float,
    blade_count: int,
    blade_angle_degrees: float = 30,
    blade_thickness: float = 0.004,
    frame_thickness: float = 0.012,
    frame_depth: float = 0.02,
    depth: float = 0.01,
    name: str = "Louver",
) -> trimesh.Trimesh:
    """Rectangular grille in the XY plane: an optional frame ring plus angled slats."""
    name = _text(name, "name")
    width = _number(width, "width", greater_than=0)
    height = _number(height, "height", greater_than=0)
    blade_count = _integer(blade_count, "blade_count", 1, 40)
    angle_degrees = _number(blade_angle_degrees, "blade_angle_degrees", minimum=-80, maximum=80)
    thickness = _number(blade_thickness, "blade_thickness", greater_than=0)
    frame_thickness = _number(frame_thickness, "frame_thickness", minimum=0)
    frame_depth = _number(frame_depth, "frame_depth", greater_than=0)
    opening_depth = _number(depth, "depth", greater_than=0)
    if 2 * frame_thickness >= min(width, height):
        raise ValueError("frame_thickness must be smaller than half of min(width, height)")
    opening_width = width - 2 * frame_thickness
    opening_height = height - 2 * frame_thickness
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    clearance = CLEARANCE * opening_height
    # Solve the pitch so the tilted slats and the two clearances fit the opening.
    pitch = (opening_height - 2 * clearance - thickness * abs(sine)) / (
        (blade_count - 1) + (1 - CLEARANCE) * cosine * cosine
    )
    if pitch <= 0:
        raise ValueError(
            "blade_thickness and blade_count do not fit inside the opening height; "
            "reduce blade_thickness or blade_count"
        )
    blade_width = (1 - CLEARANCE) * pitch * cosine
    blade_length = (1 - CLEARANCE) * opening_width
    along = np.array([0.0, cosine, sine])
    normal = np.array([0.0, -sine, cosine])
    axis_x = np.array([1.0, 0.0, 0.0])
    vertices: list[Any] = []
    faces = []
    for index in range(blade_count):
        center = np.array([0.0, (index - (blade_count - 1) / 2) * pitch, opening_depth / 2])
        blade_vertices, blade_faces = _oriented_box(
            center, blade_length / 2 * axis_x, blade_width / 2 * along, thickness / 2 * normal
        )
        base = len(vertices)
        vertices.extend(blade_vertices)
        faces.extend([[index_ + base for index_ in face] for face in blade_faces])
    if frame_thickness > 0:
        frame_vertices, frame_faces = _frame_ring(
            width, height, opening_width, opening_height, frame_depth
        )
        base = len(vertices)
        vertices.extend(frame_vertices)
        faces.extend([[index_ + base for index_ in face] for face in frame_faces])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _frame_ring(
    width: Any, height: Any, opening_width: Any, opening_height: Any, frame_depth: Any
) -> Any:
    """Closed rectangular ring from z = 0 to z = frame_depth around an opening."""
    outer = (
        (width / 2, height / 2),
        (-width / 2, height / 2),
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
    )
    inner = (
        (opening_width / 2, opening_height / 2),
        (-opening_width / 2, opening_height / 2),
        (-opening_width / 2, -opening_height / 2),
        (opening_width / 2, -opening_height / 2),
    )
    vertices = [np.array([x, y, 0.0]) for x, y in outer]
    vertices += [np.array([x, y, frame_depth]) for x, y in outer]
    vertices += [np.array([x, y, 0.0]) for x, y in inner]
    vertices += [np.array([x, y, frame_depth]) for x, y in inner]
    outer_bottom = list(range(0, 4))
    outer_top = list(range(4, 8))
    inner_bottom = list(range(8, 12))
    inner_top = list(range(12, 16))
    quads = []
    for index in range(4):
        following = (index + 1) % 4
        quads.append(
            [
                outer_bottom[index],
                outer_bottom[following],
                outer_top[following],
                outer_top[index],
            ]
        )
        # The inner wall is reversed so it faces into the opening.
        quads.append(
            [inner_bottom[following], inner_bottom[index], inner_top[index], inner_top[following]]
        )
        quads.append(
            [outer_top[index], outer_top[following], inner_top[following], inner_top[index]]
        )
        quads.append(
            [
                outer_bottom[index],
                inner_bottom[index],
                inner_bottom[following],
                outer_bottom[following],
            ]
        )
    return vertices, _quads_to_triangles(quads)


def switch_grid(
    columns: int,
    rows: int,
    spacing: float,
    base_width: float,
    base_height: float,
    base_thickness: float = 0.004,
    toggle_length: float = 0.016,
    toggle_radius: float = 0.0035,
    toggle_angle_degrees: float = 20,
    name: str = "Switch grid",
) -> trimesh.Trimesh:
    """Base plate with its top surface at z = 0 plus a grid of tilted toggle switches."""
    name = _text(name, "name")
    columns = _integer(columns, "columns", 1, 12)
    rows = _integer(rows, "rows", 1, 12)
    spacing = _number(spacing, "spacing", greater_than=0)
    base_width = _number(base_width, "base_width", greater_than=0)
    base_height = _number(base_height, "base_height", greater_than=0)
    base_thickness = _number(base_thickness, "base_thickness", greater_than=0)
    toggle_length = _number(toggle_length, "toggle_length", greater_than=0)
    toggle_radius = _number(toggle_radius, "toggle_radius", greater_than=0)
    vertices, faces = _oriented_box(
        np.array([0.0, 0.0, -base_thickness / 2]),
        np.array([base_width / 2, 0.0, 0.0]),
        np.array([0.0, base_height / 2, 0.0]),
        np.array([0.0, 0.0, base_thickness / 2]),
    )
    tilt = math.radians(
        _number(toggle_angle_degrees, "toggle_angle_degrees", minimum=-80, maximum=80)
    )
    # +Z rotated around +X by the tilt angle.
    axis = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    for column in range(columns):
        for row in range(rows):
            base = np.array(
                [
                    (column - (columns - 1) / 2) * spacing,
                    (row - (rows - 1) / 2) * spacing,
                    0.0,
                ]
            )
            toggle_vertices, toggle_faces = _capped_cylinder(
                base, axis, toggle_length, toggle_radius, TOGGLE_SIDES
            )
            offset = len(vertices)
            vertices.extend(toggle_vertices)
            faces.extend([[index + offset for index in face] for face in toggle_faces])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
