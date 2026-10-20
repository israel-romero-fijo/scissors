"""Deterministic rounded, polygonal and helical figure geometry."""

from itertools import product
from math import gcd

import numpy as np
import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.geometry import run_backend
from scissors.core.mechanical import tube
from scissors.core.profiles import triangulate


def orient(mesh: trimesh.Trimesh, axis: str, pivot: str) -> trimesh.Trimesh:
    """Place a Z-axis figure on its base, center or original origin, then rotate its axis."""
    if pivot == "center":
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
    elif pivot == "base":
        mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])
    elif pivot != "origin":
        raise ValueError("Unknown figure pivot")
    if axis != "z":
        mesh.apply_transform(
            trimesh.geometry.align_vectors([0, 0, 1], np.eye(3)["xyz".index(axis)])
        )
    return mesh


def flat(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    return trimesh.Trimesh(
        vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False
    )


def open_poles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove the triangles incident to either axial pole, leaving two openings."""
    z = np.asarray(mesh.vertices)[:, 2]
    poles = (z == z.min()) | (z == z.max())
    mesh.update_faces(~np.any(poles[mesh.faces], axis=1))
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError("Resolution leaves no surface after opening the poles")
    return mesh


def open_flat_caps(mesh: trimesh.Trimesh, directions: list[list[float]]) -> trimesh.Trimesh:
    """Remove planar end faces parallel to the specified local component axes."""
    mask = np.ones(len(mesh.faces), dtype=bool)
    for direction in directions:
        normal = np.asarray(direction)
        mask &= np.abs(mesh.face_normals @ normal) < 1 - 1e-10
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def orient_planar(
    mesh: trimesh.Trimesh, plane: str, axis: str | None, pivot: str
) -> trimesh.Trimesh:
    """Orient a planar figure or its extrusion while preserving its input origin by default."""
    normal = np.eye(3)[{"xy": 2, "xz": 1, "yz": 0}[plane]]
    if pivot == "center":
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    elif pivot == "base":
        mesh.apply_translation(-normal * np.min(mesh.vertices @ normal))
    elif pivot != "origin":
        raise ValueError("Unknown planar figure pivot")
    if axis is not None:
        mesh.apply_transform(trimesh.geometry.align_vectors(normal, np.eye(3)["xyz".index(axis)]))
    return mesh


def revolve(profile: np.ndarray, sides: int, capped: bool) -> trimesh.Trimesh:
    """Revolve an increasing-height radius/height profile without duplicate pole vertices."""
    vertices: list[list[float]] = []
    rings: list[list[int]] = []
    angles = np.arange(sides) * 2 * np.pi / sides
    for radius, height in profile:
        start = len(vertices)
        if radius <= 1e-12:
            vertices.append([0, 0, float(height)])
            rings.append([start])
        else:
            vertices.extend(
                [
                    [float(radius * np.cos(angle)), float(radius * np.sin(angle)), float(height)]
                    for angle in angles
                ]
            )
            rings.append(list(range(start, start + sides)))
    faces = []
    for lower, upper in zip(rings[:-1], rings[1:], strict=True):
        for i in range(sides):
            j = (i + 1) % sides
            if len(lower) == 1:
                faces.append([lower[0], upper[j], upper[i]])
            elif len(upper) == 1:
                faces.append([lower[i], lower[j], upper[0]])
            else:
                faces.extend([[lower[i], lower[j], upper[j]], [lower[i], upper[j], upper[i]]])
    if capped:
        for ring, z, reverse in (
            (rings[0], profile[0, 1], True),
            (rings[-1], profile[-1, 1], False),
        ):
            if len(ring) == 1:
                continue
            center = len(vertices)
            vertices.append([0, 0, float(z)])
            for i in range(sides):
                j = (i + 1) % sides
                faces.append([center, ring[j], ring[i]] if reverse else [center, ring[i], ring[j]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def capsule(radius: float, height: float, segments: int) -> trimesh.Trimesh:
    if height < 2 * radius:
        raise ScissorsError("GEOMETRY_ERROR", "Capsule height must be at least twice its radius")
    steps = max(2, segments // 4)
    lower = np.linspace(-np.pi / 2, 0, steps + 1)
    upper = np.linspace(0, np.pi / 2, steps + 1)
    a = np.column_stack([radius * np.cos(lower), radius + radius * np.sin(lower)])
    b = np.column_stack([radius * np.cos(upper), height - radius + radius * np.sin(upper)])
    profile = np.concatenate([a, b[1:] if height == 2 * radius else b])
    profile[[0, -1], 0] = 0
    return revolve(profile, segments, True)


def dome(radius: float, height: float, segments: int, capped: bool) -> trimesh.Trimesh:
    angles = np.linspace(0, np.pi / 2, max(2, segments // 4) + 1)
    profile = np.column_stack([radius * np.cos(angles), height * np.sin(angles)])
    profile[-1, 0] = 0
    return revolve(profile, segments, capped)


def polygon_solid(
    points: list[list[float]],
    holes: list[list[list[float]]],
    thickness: float,
    settings: Settings,
    capped: bool = True,
) -> trimesh.Trimesh:
    vertices, caps, rings, _ = triangulate(points, holes, settings)
    n = len(vertices)
    xyz = np.concatenate(
        [
            np.column_stack([vertices, np.zeros(n)]),
            np.column_stack([vertices, np.full(n, thickness)]),
        ]
    )
    faces = []
    for ring in rings:
        for i, a in enumerate(ring):
            b = ring[(i + 1) % len(ring)]
            faces.extend([[a, b, b + n], [a, b + n, a + n]])
    if capped:
        faces.extend(caps[:, ::-1].tolist())
        faces.extend((caps + n).tolist())
    return flat(trimesh.Trimesh(vertices=xyz, faces=faces, process=False))


def gear(
    teeth: int,
    radius: float,
    thickness: float,
    tooth_depth: float,
    bore_radius: float,
    settings: Settings,
    capped: bool,
) -> trimesh.Trimesh:
    root = radius - tooth_depth
    if not 0 < root < radius or bore_radius >= root:
        raise ScissorsError(
            "GEOMETRY_ERROR", "Gear requires a positive root radius and a smaller bore"
        )
    angles = np.arange(4 * teeth) * np.pi / (2 * teeth)
    radii = np.where(np.arange(4 * teeth) % 4 < 2, root, radius)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).tolist()
    holes = []
    if bore_radius:
        theta = np.arange(min(128, 4 * teeth)) * 2 * np.pi / min(128, 4 * teeth)
        holes = [
            np.column_stack([bore_radius * np.cos(theta), bore_radius * np.sin(theta)]).tolist()
        ]
    return polygon_solid(points, holes, thickness, settings, capped)


def wedge(size: list[float], taper: float, capped: bool) -> trimesh.Trimesh:
    width, depth, height = size
    mesh = trimesh.creation.box(extents=size)
    vertices = np.asarray(mesh.vertices).copy() + [0, 0, height / 2]
    vertices[(vertices[:, 0] < 0) & (vertices[:, 2] > 0), 2] = taper * height
    unique, mapping = np.unique(vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=unique, faces=mapping[mesh.faces], process=False)
    mesh.update_faces(mesh.area_faces > 1e-14)
    if not capped:
        mesh.update_faces(np.abs(mesh.face_normals[:, 1]) < 0.9)
    mesh.remove_unreferenced_vertices()
    return flat(mesh)


def polyhedron(kind: str, radius: float, settings: Settings) -> trimesh.Trimesh:
    phi = (1 + np.sqrt(5)) / 2
    if kind == "tetrahedron":
        points = np.asarray([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
    elif kind == "octahedron":
        points = np.concatenate([np.eye(3), -np.eye(3)])
    else:
        values: list[tuple[float, ...]] = []
        if kind == "dodecahedron":
            values.extend(product((-1, 1), repeat=3))
            a, b = 1 / phi, phi
        else:
            a, b = 1.0, phi
        for one, two in product((-1, 1), repeat=2):
            values.extend([(0, one * a, two * b), (one * a, two * b, 0), (two * b, 0, one * a)])
        points = np.asarray(values, dtype=float)
    points *= radius / np.linalg.norm(points, axis=1).max()
    data = run_backend({"backend": "convex_hull", "vertices": points.tolist()}, settings)
    return flat(trimesh.Trimesh(**data, process=False))


def spring(
    radius: float,
    wire_radius: float,
    coils: float,
    height: float,
    segments: int,
    sides: int,
    capped: bool,
) -> trimesh.Trimesh:
    if radius <= wire_radius or height / coils <= 2 * wire_radius:
        raise ScissorsError("GEOMETRY_ERROR", "Spring wire must fit the radius and coil pitch")
    count = int(np.ceil(coils * segments)) + 1
    angles = np.linspace(0, 2 * np.pi * coils, count)
    points = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.linspace(wire_radius, height + wire_radius, count),
        ]
    )
    return tube(points.tolist(), wire_radius, sides, capped, False, "Spring", max_points=32768)


def torus(
    major: float,
    minor: float,
    major_sections: int,
    minor_sections: int,
    arc: float,
    twist: int,
    capped: bool,
) -> trimesh.Trimesh:
    """Circular torus arcs or a (2, twist) torus-knot wire."""
    if twist:
        components = gcd(2, twist)
        turns, winding = 2 // components, twist // components
        if major_sections < 16 * max(turns, winding):
            raise ScissorsError("GEOMETRY_ERROR", "Knots need at least 16 sections per winding")
        if minor >= major / 3:
            raise ScissorsError(
                "GEOMETRY_ERROR", "Knot wire radius must be smaller than one third of major radius"
            )
        t = np.linspace(
            0,
            np.radians(arc),
            major_sections if arc == 360 else major_sections + 1,
            endpoint=arc != 360,
        )
        strands = []
        for strand in range(components):
            angle = winding * t + 2 * np.pi * strand / components
            r = major * (1 + 0.5 * np.cos(angle))
            path = np.column_stack(
                [r * np.cos(turns * t), r * np.sin(turns * t), major * 0.5 * np.sin(angle)]
            )
            strands.append(
                tube(
                    path.tolist(),
                    minor,
                    minor_sections,
                    capped,
                    arc == 360,
                    "Torus knot",
                    max_points=513,
                    max_sides=512,
                )
            )
        result = trimesh.util.concatenate(strands)
        assert isinstance(result, trimesh.Trimesh)
        return result
    n = major_sections if arc == 360 else major_sections + 1
    u = np.linspace(0, np.radians(arc), n, endpoint=arc != 360)
    v = np.arange(minor_sections) * 2 * np.pi / minor_sections
    vertices = np.stack(
        [
            (major + minor * np.cos(v))[None, :] * np.cos(u)[:, None],
            (major + minor * np.cos(v))[None, :] * np.sin(u)[:, None],
            np.broadcast_to(minor * np.sin(v), (n, minor_sections)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for i in range(major_sections):
        k = (i + 1) % n
        for j in range(minor_sections):
            following = (j + 1) % minor_sections
            a, b = i * minor_sections + j, k * minor_sections + j
            c, d = k * minor_sections + following, i * minor_sections + following
            faces.extend([[a, b, c], [a, c, d]])
    if capped and arc != 360:
        centers = [
            [major * np.cos(u[0]), major * np.sin(u[0]), 0],
            [major * np.cos(u[-1]), major * np.sin(u[-1]), 0],
        ]
        count = len(vertices)
        vertices = np.concatenate([vertices, centers])
        for j in range(minor_sections):
            following = (j + 1) % minor_sections
            faces.extend(
                [
                    [count, j, following],
                    [count + 1, (n - 1) * minor_sections + following, (n - 1) * minor_sections + j],
                ]
            )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
