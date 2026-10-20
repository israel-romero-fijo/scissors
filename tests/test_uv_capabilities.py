import importlib.util

import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


@pytest.mark.parametrize("available", [False, True])
def test_uv_discovery_matches_backend_availability(tmp_path, monkeypatch, available):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (object() if available else None) if name == "xatlas" else find_spec(name),
    )
    with Scissors(tmp_path / "workspace") as app:
        operation = app.registry.describe_operation("uv.generate")
        methods = operation["parameters"]["properties"]["method"]["enum"]
        assert ("xatlas" in methods) == available
        assert "auto" in methods
        if not available:
            assert "incomplete Scissors installation" in operation["description"]
        scene = app.scenes.create()
        cube = app.meshes.create_cube(scene_id=scene.id)
        if not available:
            before = app.call("mesh.inspect", object_id=cube.id)
            with pytest.raises(ScissorsError) as failure:
                app.call("uv.generate", object_id=cube.id, method="xatlas")
            assert failure.value.code == "BACKEND_UNAVAILABLE"
            assert failure.value.details["backend"] == "xatlas"
            assert failure.value.details["available_values"] == methods
            assert failure.value.details["parameter"] == ["method"]
            assert app.call("mesh.inspect", object_id=cube.id) == before
        invalid = app.registry.execute("uv.generate", {"object_id": cube.id, "method": "unknown"})
        assert invalid["error"]["code"] == "INVALID_PARAMETER"
        fallback = app.call("uv.generate", object_id=cube.id, method="auto")
        assert fallback["validation"]["valid"]
