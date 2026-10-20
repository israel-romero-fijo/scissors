import numpy as np

from scissors.plugins.context import PluginContext


def twist_mesh(
    context: PluginContext,
    object_id: str,
    axis: str = "z",
    angle_degrees: float = 90,
    falloff: str = "linear",
) -> dict:
    mesh = context.get_mesh(object_id)
    vertices = np.asarray(mesh["vertices"], dtype=float)
    index = "xyz".index(axis)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    length = high[index] - low[index]
    if length <= 1e-12:
        raise ValueError("Twist axis has zero extent")
    t = (vertices[:, index] - low[index]) / length
    if falloff == "smoothstep":
        t = t * t * (3 - 2 * t)
    angles = np.radians(angle_degrees) * t
    a, b = [(1, 2), (2, 0), (0, 1)][index]
    center = (low + high) / 2
    x, y = vertices[:, a] - center[a], vertices[:, b] - center[b]
    vertices[:, a] = center[a] + x * np.cos(angles) - y * np.sin(angles)
    vertices[:, b] = center[b] + x * np.sin(angles) + y * np.cos(angles)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite deformation")
    context.replace_mesh(object_id, vertices, mesh["faces"])
    return {
        **context.statistics(object_id),
        "axis": axis,
        "angle_degrees": angle_degrees,
        "falloff": falloff,
        "warnings": [],
    }


OPERATIONS = {"deform.twist_mesh": twist_mesh}
