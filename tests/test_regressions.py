import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


def test_concurrent_thread_calls_are_serialized(app):
    scene = app.scenes.create(name="Concurrent")
    obj = app.meshes.create_cube(scene_id=scene.id)
    with ThreadPoolExecutor(max_workers=4) as threads:
        futures = [
            threads.submit(app.call, "mesh.translate", object_id=obj.id, x=1) for _ in range(12)
        ]
        for future in futures:
            assert future.result()["object_id"] == obj.id
    np.testing.assert_allclose(
        app.meshes.inspect(object_id=obj.id).bounds, [[11.5, -0.5, -0.5], [12.5, 0.5, 0.5]]
    )


def test_checkpoint_failure_restores_memory_and_disk(app, monkeypatch):
    scene = app.scenes.create(name="Checkpoint")
    obj = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    disk = (app.engine.root / "state.json").read_bytes()

    def failure(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr("scissors.core.binary_persistence.atomic_write", failure)
    response = app.registry.execute("mesh.translate", {"object_id": obj.id, "x": 2})
    assert response["error"]["code"] == "INTERNAL_ERROR"
    assert app.engine.snapshot() == before
    assert (app.engine.root / "state.json").read_bytes() == disk


@pytest.mark.integration
def test_parent_rejects_plugin_invalid_patch(app, twist_path, tmp_path):
    import shutil

    folder = tmp_path / "invalid_patch"
    shutil.copytree(twist_path, folder)
    (folder / "plugin.py").write_text(
        "def broken(context, object_id, **kwargs):\n"
        "    context.replace_mesh(object_id, [[0,0,0]], [[0,0,0]])\n"
        "    return {'object_id': object_id, 'vertex_count':1, 'face_count':1}\n"
        "OPERATIONS = {'deform.twist_mesh': broken}\n"
    )
    app.plugins.install(folder)
    scene = app.scenes.create(name="Invalid patch")
    obj = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    response = app.registry.execute("deform.twist_mesh", {"object_id": obj.id})
    assert response["error"]["code"] == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


@pytest.mark.integration
def test_new_procedural_plugin_uses_public_create_context(app, tmp_path):
    folder = tmp_path / "procedural"
    folder.mkdir()
    manifest = {
        "id": "procedural",
        "name": "Triangle",
        "version": "1.0.0",
        "api_version": "1.0",
        "description": "Create a triangle using only the public context",
        "entrypoint": "plugin.py",
        "permissions": ["scene.write"],
        "operations": [
            {
                "name": "procedural.triangle",
                "description": "Create one triangle",
                "input_schema": {
                    "type": "object",
                    "properties": {"scene_id": {"type": "string"}},
                    "required": ["scene_id"],
                    "additionalProperties": False,
                },
            }
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "plugin.py").write_text(
        "def triangle(context, scene_id):\n"
        "    context.create_mesh(scene_id, [[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]], 'Triangle')\n"
        "    return {'kind': 'triangle'}\n"
        "OPERATIONS = {'procedural.triangle': triangle}\n"
    )
    assert app.plugins.validate(folder)["valid"]
    app.plugins.install(folder)
    scene = app.scenes.create(name="Extension")
    response = app.call("procedural.triangle", scene_id=scene.id)
    obj = response["created_objects"][0]
    assert obj["face_count"] == 1
    assert app.meshes.validate(object_id=obj["id"]).valid


def test_texture_scene_restart_and_portability(tmp_path):
    import shutil

    source, moved = tmp_path / "source", tmp_path / "moved"
    with Scissors(source) as app:
        scene = app.scenes.create(name="portable")
        obj = app.meshes.create_cube(scene_id=scene.id)
        mat = app.materials.create(scene_id=scene.id)
        tex = app.textures.generate_checker(width=16, height=16)
        app.uv.generate(object_id=obj.id)
        app.textures.assign(material_id=mat.id, texture_id=tex.id)
        app.materials.assign(object_id=obj.id, material_id=mat.id)
    shutil.copytree(source, moved)
    with Scissors(moved) as app:
        exported = app.export.glb(scene_id=scene.id)
        import trimesh

        mesh = next(iter(trimesh.load(exported.path, force="scene").geometry.values()))
        assert mesh.visual.material.baseColorTexture.size == (16, 16)


def test_closed_service_cannot_modify(app):
    app.close()
    with pytest.raises(ScissorsError) as error:
        app.scenes.create(name="closed")
    assert error.value.code == "WORKSPACE_CLOSED"


def test_invalid_uv_normals_nonmanifold(app):
    from scissors.core.mesh import MeshObject
    from scissors.core.validation import validate_mesh

    scene = app.scenes.create(name="diagnostics")
    obj = app.meshes.create_cube(scene_id=scene.id)
    candidate = app.engine.object(obj.id)[1].model_copy(deep=True)
    candidate.normals = [[0, 0, 0]] * len(candidate.vertices)
    assert "INCORRECT_NORMALS" in validate_mesh(candidate, app.engine.settings)["warnings"]
    candidate.uv = [[0, 0]]
    assert "INVALID_UV" in validate_mesh(candidate, app.engine.settings)["errors"]
    candidate = MeshObject(
        id="nonmanifold",
        name="nonmanifold",
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    )
    assert "NON_MANIFOLD_EDGES" in validate_mesh(candidate, app.engine.settings)["errors"]
