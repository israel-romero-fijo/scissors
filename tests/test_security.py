import pytest
from pydantic import ValidationError

from scissors import Scissors, Settings
from scissors.api.errors import ScissorsError


def test_paths_and_limits(app, tmp_path):
    scene = app.scenes.create(name="Paths")
    mesh = app.meshes.create_cube(scene_id=scene.id)
    for path in ["../escape.glb", str(tmp_path / "escape.glb"), "state.json", "plugins/escape.glb"]:
        response = app.registry.execute("scene.export", {"scene_id": scene.id, "output": path})
        assert response["error"]["code"] == "PERMISSION_DENIED"
    assert not (tmp_path / "escape.glb").exists()
    assert app.meshes.inspect(object_id=mesh.id).face_count == 12
    with Scissors(settings=Settings(workspace=tmp_path / "limited", max_vertices=4)) as limited:
        sid = limited.scenes.create(name="Limited").id
        response = limited.registry.execute("mesh.create_cube", {"scene_id": sid})
        assert response["error"]["code"] == "RESOURCE_LIMIT"
        assert limited.scenes.inspect(scene_id=sid).objects == []


def test_atomic_rollback_and_nonatomic_success(app):
    scene = app.scenes.create(name="Batch")
    cube = app.meshes.create_cube(scene_id=scene.id)
    before = app.meshes.data(object_id=cube.id).vertices
    ops = [
        {"operation": "mesh.translate", "arguments": {"object_id": cube.id, "x": 10}},
        {"operation": "mesh.scale", "arguments": {"object_id": cube.id, "x": -1}},
    ]
    response = app.registry.execute("operations.batch", {"operations": ops, "atomic": True})
    assert response["error"]["code"] == "BATCH_FAILED"
    assert app.meshes.data(object_id=cube.id).vertices == before
    assert app.call("operations.batch", operations=ops, atomic=False)["success_count"] == 1
    assert app.meshes.inspect(object_id=cube.id).bounds[0][0] == 9.5
    response = app.registry.execute(
        "operations.batch",
        {
            "operations": [
                {
                    "operation": "scene.export",
                    "arguments": {"scene_id": scene.id, "output": "exports/batch.glb"},
                }
            ]
        },
    )
    assert response["error"]["code"] == "INVALID_PARAMETER"
    assert not app.engine.path("exports/batch.glb").exists()


def test_http_external_bind_requires_token(app):
    from scissors.http.server import create_app

    with pytest.raises(ScissorsError):
        create_app(app, host="0.0.0.0")


def test_nonfinite_and_geometry_failures(app):
    scene = app.scenes.create(name="Invalid")
    response = app.registry.execute(
        "mesh.create", {"scene_id": scene.id, "vertices": [[0, 0, 0]], "faces": [[0, 1, 2]]}
    )
    assert response["error"]["code"] == "GEOMETRY_ERROR"
    response = app.registry.execute(
        "mesh.create_cube", {"scene_id": scene.id, "size": float("nan")}
    )
    assert response["error"]["code"] == "INVALID_PARAMETER"
    assert app.scenes.inspect(scene_id=scene.id).objects == []


def test_settings_reject_removed_trusted_development(tmp_path):
    assert "trusted_development" not in Settings.model_fields
    with pytest.raises(ValidationError):
        Settings(workspace=tmp_path / "workspace", trusted_development=True)
