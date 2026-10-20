import importlib.util
from pathlib import Path

import numpy as np

from scissors.plugins.context import PluginContext


def test_known_rotation_and_axis_preservation():
    spec = importlib.util.spec_from_file_location(
        "twist_example", Path(__file__).parents[1] / "plugin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices = [[1, 0, 0], [-1, 0, 0], [1, 0, 1], [-1, 0, 1]]
    context = PluginContext(
        {"test": {"vertices": vertices, "faces": [[0, 1, 2], [1, 3, 2]], "uv": None}}, []
    )
    module.twist_mesh(context, "test", "z", 90)
    result = np.asarray(context.get_mesh("test")["vertices"])
    np.testing.assert_allclose(result[0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(result[2], [0, 1, 1], atol=1e-12)
    np.testing.assert_allclose(result[:, 2], np.asarray(vertices)[:, 2])
