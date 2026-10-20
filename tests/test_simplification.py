import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.http.server import create_app
from scissors.plugins.runtime import limited_process

HAS_VTK = importlib.util.find_spec("vtkmodules") is not None
needs_vtk = pytest.mark.skipif(not HAS_VTK, reason="required VTK backend absent")


def sphere(app):
    scene = app.scenes.create(name="Simplify")
    obj = app.meshes.create_sphere(scene_id=scene.id, subdivisions=3)
    return scene, obj


def test_simplification_discovery(app):
    description = app.registry.describe_operation("mesh.simplify")
    assert description["plugin"] is None
    assert description["permissions"] == ["scene.write"]
    assert description["parameters"]["properties"]["uv_policy"]["default"] == "preserve"
    assert "actual_reduction_percent" in description["result_schema"]["required"]
    assert app.call("modeler.info")["backends"]["simplify"] == HAS_VTK


@pytest.mark.parametrize(
    "arguments",
    [
        {"reduction_percent": 0},
        {"reduction_percent": 100},
        {"reduction_percent": -1},
        {"reduction_percent": 50, "uv_policy": "guess"},
        {"reduction_percent": 50, "max_error_relative": -0.1},
        {"reduction_percent": 50, "extra": True},
    ],
)
def test_simplification_invalid_parameters_restore(app, arguments):
    _, obj = sphere(app)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, **arguments)
    assert error.value.code == "INVALID_PARAMETER"
    assert app.engine.snapshot() == before


def test_simplification_backend_missing(app, monkeypatch):
    _, obj = sphere(app)
    before = app.engine.snapshot()
    monkeypatch.setattr("scissors.core.simplification.available", lambda: False)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert error.value.code == "BACKEND_UNAVAILABLE"
    assert app.engine.snapshot() == before


@needs_vtk
@pytest.mark.integration
def test_http_simplification_uses_shared_registry(app):
    _, obj = sphere(app)
    with TestClient(create_app(app)) as client:
        description = client.get("/api/v1/operations/mesh.simplify").json()
        assert description["parameters"]["properties"]["uv_policy"]["default"] == "preserve"
        response = client.post(
            "/api/v1/operations/invoke",
            json={
                "name": "mesh.simplify",
                "arguments": {"object_id": obj.id, "reduction_percent": 50},
            },
        ).json()
    assert response["success"]
    result = response["result"]
    assert app.meshes.inspect(object_id=obj.id).face_count == result["face_count"]
    assert result["face_count"] < obj.face_count


def test_lost_uv_result_is_rejected(app, monkeypatch):
    _, obj = sphere(app)
    app.uv.generate(object_id=obj.id)
    before = app.engine.snapshot()
    original = app.engine.object(obj.id)[1]
    monkeypatch.setattr("scissors.core.simplification.available", lambda: True)
    monkeypatch.setattr(
        "scissors.core.simplification.run_backend",
        lambda *a: {
            "vertices": original.vertices,
            "faces": original.faces,
            "uv": None,
        },
    )
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


def test_real_worker_timeout_closes_child_handles(tmp_path):
    settings = Settings(workspace=tmp_path / "workspace", operation_timeout=1)
    # The child holds a separate file handle, so deleting the folder proves that
    # a redirected Windows interpreter was terminated along with its launcher.
    with tempfile.TemporaryDirectory(dir=tmp_path) as folder:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            with pytest.raises(ScissorsError) as error:
                limited_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; stream = open('held.log', 'w'); "
                        "stream.write('started'); stream.flush(); time.sleep(30)",
                    ],
                    Path(folder),
                    settings,
                    out,
                    err,
                )
            assert error.value.code == "OPERATION_TIMEOUT"
        assert (Path(folder) / "held.log").read_text() == "started"


@needs_vtk
@pytest.mark.integration
def test_simplify_closed_mesh_and_restart(app):
    _, obj = sphere(app)
    app.meshes.translate(object_id=obj.id, x=3)
    app.engine.object(obj.id)[1].metadata["user_label"] = {"part": "shell"}
    original = copy.deepcopy(app.engine.object(obj.id)[1])
    result = app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert 0 < result["face_count"] < obj.face_count
    assert result["original_face_count"] == obj.face_count
    assert result["actual_reduction_percent"] == pytest.approx(
        100 * (obj.face_count - result["face_count"]) / obj.face_count
    )
    assert app.meshes.validate(object_id=obj.id, require_closed=True).valid
    current = app.engine.object(obj.id)[1]
    assert current.transform == original.transform
    revision_keys = {"_geometry_revision", "_geometry_signature"}
    assert {k: v for k, v in current.metadata.items() if k not in revision_keys} == {
        k: v for k, v in original.metadata.items() if k not in revision_keys
    }
    assert current.geometry_revision != original.geometry_revision
    assert current.metadata["_geometry_revision"] > original.metadata["_geometry_revision"]
    assert current.metadata["_geometry_signature"] == current.geometry_signature()
    assert set(map(tuple, current.vertices)) <= set(map(tuple, original.vertices))
    expected = current.model_dump()
    workspace = app.engine.root
    app.close()
    with Scissors(workspace) as restarted:
        assert restarted.engine.object(obj.id)[1].model_dump() == expected


def chart_grid(app):
    # Two planar UV charts share a geometric seam, with different UV on each side.
    scene = app.scenes.create(name="UV charts")
    vertices, faces, uv = [], [], []
    for chart in range(2):
        offset = len(vertices)
        for y in range(11):
            for x in range(11):
                vertices.append([chart + x / 10, y / 10, 0])
                uv.append([x / 10, (y / 10 + chart) / 2])
        for y in range(10):
            for x in range(10):
                a = offset + y * 11 + x
                faces.extend([[a, a + 1, a + 12], [a, a + 12, a + 11]])
    obj = app.meshes.create(scene_id=scene.id, vertices=vertices, faces=faces)
    app.engine.object(obj.id)[1].uv = uv
    app.engine.commit()
    return scene, obj


@needs_vtk
@pytest.mark.integration
def test_uv_seams_boundaries_material_and_textured_export(app):
    scene, obj = chart_grid(app)
    mat = app.materials.create(scene_id=scene.id)
    texture = app.textures.generate_checker(width=32, height=32, cells=4)
    app.textures.assign(material_id=mat.id, texture_id=texture.id)
    app.materials.assign(object_id=obj.id, material_id=mat.id)
    original = copy.deepcopy(app.engine.object(obj.id)[1])
    pairs = {tuple(v + u) for v, u in zip(original.vertices, original.uv, strict=True)}
    seam = {p for p in pairs if p[0] == 1}
    boundary = {p for p in pairs if p[0] in {0, 2} or p[1] in {0, 1}}
    result = app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert result["face_count"] < obj.face_count
    assert result["has_uv"] and result["material_id"] == mat.id
    current = app.engine.object(obj.id)[1]
    reduced_pairs = {tuple(v + u) for v, u in zip(current.vertices, current.uv, strict=True)}
    assert reduced_pairs <= pairs
    assert seam | boundary <= reduced_pairs
    assert app.meshes.validate(object_id=obj.id).valid
    exported = app.export.glb(scene_id=scene.id)
    reopened = trimesh.load(exported.path, force="scene")
    mesh = next(iter(reopened.geometry.values()))
    assert len(mesh.faces) == result["face_count"]
    assert mesh.visual.uv.shape == (result["vertex_count"], 2)
    assert mesh.visual.material.baseColorTexture.size == (32, 32)


@needs_vtk
@pytest.mark.integration
def test_uv_discard_is_explicit(app):
    _, obj = sphere(app)
    app.uv.generate(object_id=obj.id)
    result = app.call("mesh.simplify", object_id=obj.id, reduction_percent=50, uv_policy="discard")
    assert result["face_count"] < obj.face_count
    assert not result["has_uv"]
    assert any("UV_DISCARDED" in w for w in result["warnings"])


@needs_vtk
@pytest.mark.integration
def test_unreachable_target_and_atomic_rollback(app):
    scene = app.scenes.create()
    obj = app.meshes.create_plane(scene_id=scene.id)
    result = app.call("mesh.simplify", object_id=obj.id, reduction_percent=90)
    assert not result["target_reached"] and result["face_count"] == obj.face_count
    assert any("TARGET_NOT_REACHED" in w for w in result["warnings"])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as error:
        app.call(
            "operations.batch",
            operations=[
                {"operation": "mesh.translate", "arguments": {"object_id": obj.id, "x": 10}},
                {
                    "operation": "mesh.simplify",
                    "arguments": {
                        "object_id": obj.id,
                        "reduction_percent": 90,
                        "require_target": True,
                    },
                },
            ],
        )
    assert error.value.code == "BATCH_FAILED"
    assert app.engine.snapshot() == before


@needs_vtk
@pytest.mark.integration
def test_disconnected_component_survives(app):
    scene, obj = sphere(app)
    cube = app.meshes.create_cube(scene_id=scene.id, size=0.1)
    app.meshes.translate(object_id=cube.id, x=4)
    merged = app.meshes.merge(scene_id=scene.id, object_ids=[obj.id, cube.id])
    result = app.call("mesh.simplify", object_id=merged.id, reduction_percent=80)
    assert result["face_count"] < merged.face_count
    data = app.meshes.data(object_id=merged.id, limit=5000)
    assert np.max(np.asarray(data.vertices)[:, 0]) > 4
    assert app.meshes.validate(object_id=merged.id, require_closed=True).valid


@needs_vtk
def test_child_failure_and_timeout_restore(app, monkeypatch):
    _, obj = sphere(app)
    before = app.engine.snapshot()

    def timeout(*args, **kwargs):
        raise ScissorsError("OPERATION_TIMEOUT", "Child operation timed out")

    monkeypatch.setattr("scissors.core.geometry.limited_process", timeout)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before


@needs_vtk
def test_invalid_backend_output_and_memory_restore(app, monkeypatch):
    _, obj = sphere(app)
    before = app.engine.snapshot()
    monkeypatch.setattr(
        "scissors.core.simplification.run_backend",
        lambda *a: {
            "vertices": [[0, 0, 0]],
            "faces": [[0, 0, 0]],
            "uv": None,
        },
    )
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert error.value.code == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before
    monkeypatch.setattr(app.engine.settings, "max_memory_bytes", 500_000)
    with pytest.raises(ScissorsError) as error:
        app.call("mesh.simplify", object_id=obj.id, reduction_percent=50)
    assert error.value.code == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before
