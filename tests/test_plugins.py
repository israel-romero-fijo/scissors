import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from scissors import Scissors, Settings
from scissors.api.errors import ScissorsError


@pytest.mark.integration
def test_plugin_lifecycle_persistence(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        assert app.plugins.discover(twist_path.parent)["plugins"][0]["valid"]
        assert app.plugins.validate(twist_path)["valid"]
        app.plugins.install(twist_path)
        scene = app.scenes.create(name="Plugin")
        obj = app.meshes.create_cylinder(scene_id=scene.id, height_segments=16)
        app.meshes.scale(object_id=obj.id, x=2)
        before = app.meshes.data(object_id=obj.id, limit=5000).vertices
        stats = app.call("deform.twist_mesh", object_id=obj.id, angle_degrees=180)
        assert stats["vertex_count"] == len(before)
        assert not np.allclose(before, app.meshes.data(object_id=obj.id, limit=5000).vertices)
        assert app.meshes.validate(object_id=obj.id, require_closed=True).valid
        assert app.plugins.test("twist_mesh")["passed"]
        app.plugins.disable("twist_mesh")
        assert (
            app.registry.execute("deform.twist_mesh", {"object_id": obj.id})["error"]["code"]
            == "UNKNOWN_OPERATION"
        )
        app.plugins.reload("twist_mesh")
    with Scissors(workspace) as app:
        assert "deform.twist_mesh" in app.registry.operations
        assert app.call("deform.twist_mesh", object_id=obj.id)["face_count"] > 0


@pytest.mark.integration
def test_plugin_install_enables_and_edited_code_runs(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        assert app.plugins.install(twist_path)["state"]["enabled"] is True
        scene = app.scenes.create(name="Auto enable")
        obj = app.meshes.create_cube(scene_id=scene.id)
        before = app.meshes.data(object_id=obj.id, limit=5000).vertices
        app.call("deform.twist_mesh", object_id=obj.id)
        twisted = app.meshes.data(object_id=obj.id, limit=5000).vertices
        assert not np.allclose(before, twisted)
        with pytest.raises(ScissorsError) as error:
            app.plugins.install(twist_path)
        assert error.value.code == "PLUGIN_COLLISION"
        plugin = app.plugins.root / "twist_mesh" / "plugin.py"
        plugin.write_text(
            "def twist_mesh(context, object_id, **kwargs):\n"
            "    return context.statistics(object_id)\n"
            "OPERATIONS = {'deform.twist_mesh': twist_mesh}\n"
        )
        assert app.call("deform.twist_mesh", object_id=obj.id)["vertex_count"] == len(twisted)
        assert np.allclose(twisted, app.meshes.data(object_id=obj.id, limit=5000).vertices)


@pytest.mark.integration
def test_plugin_added_folder_is_enabled_on_open(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        shutil.copytree(twist_path, app.plugins.root / "twist_mesh")
    with Scissors(workspace) as app:
        entry = {plugin["id"]: plugin for plugin in app.plugins.list()["plugins"]}["twist_mesh"]
        assert entry["installed"] and entry["enabled"] and entry["active"]
        scene = app.scenes.create(name="Scanned")
        obj = app.meshes.create_cube(scene_id=scene.id)
        assert app.call("deform.twist_mesh", object_id=obj.id)["face_count"] == 12


@pytest.mark.integration
def test_plugin_disable_persists_across_restart(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        app.plugins.install(twist_path)
        assert app.plugins.disable("twist_mesh")["enabled"] is False
    with Scissors(workspace) as app:
        entry = {plugin["id"]: plugin for plugin in app.plugins.list()["plugins"]}["twist_mesh"]
        assert entry["enabled"] is False and entry["active"] is False
        assert "deform.twist_mesh" not in app.registry.operations
        assert app.plugins.enable("twist_mesh")["enabled"] is True


@pytest.mark.integration
def test_plugin_invalid_folder_is_reported(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        folder = app.plugins.root / "broken"
        shutil.copytree(twist_path, folder)
        (folder / "manifest.json").write_text("{not json")
        assert not app.plugins.validate(folder / "missing")["valid"]
    with Scissors(workspace) as app:
        entry = {plugin["id"]: plugin for plugin in app.plugins.list()["plugins"]}["broken"]
        assert entry["installed"] is False and entry["enabled"] is False
        assert entry["load_error"]["code"] == "PLUGIN_INVALID"


def test_plugin_invalid_manifest(app, twist_path, tmp_path):
    source = tmp_path / "bad"
    shutil.copytree(twist_path, source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["api_version"] = "999"
    (source / "manifest.json").write_text(json.dumps(manifest))
    assert not app.plugins.validate(source)["valid"]


def failing_plugin(twist_path, folder, body):
    shutil.copytree(twist_path, folder)
    path = folder / "plugin.py"
    path.write_text(
        "def broken(context, **kwargs):\n" + body + "\nOPERATIONS = {'deform.twist_mesh': broken}\n"
    )
    return folder


@pytest.mark.integration
def test_plugin_exception_does_not_mutate(app, tmp_path, twist_path):
    folder = failing_plugin(
        twist_path,
        tmp_path / "failing",
        "    mesh = context.get_mesh(kwargs['object_id'])\n"
        "    context.replace_mesh(kwargs['object_id'], [[0,0,0]], [[0,0,0]])\n"
        "    raise RuntimeError('expected failure')\n",
    )
    app.plugins.install(folder)
    scene = app.scenes.create(name="Failure")
    obj = app.meshes.create_cube(scene_id=scene.id)
    before = app.engine.snapshot()
    response = app.registry.execute("deform.twist_mesh", {"object_id": obj.id})
    assert response["error"]["code"] == "PLUGIN_FAILED"
    assert app.engine.snapshot() == before
    assert app.meshes.create_cube(scene_id=scene.id).face_count == 12


@pytest.mark.integration
def test_plugin_timeout_and_uninstall(tmp_path, twist_path):
    folder = failing_plugin(
        twist_path, tmp_path / "slow", "    import time\n    time.sleep(10)\n    return {}\n"
    )
    with Scissors(settings=Settings(workspace=tmp_path / "workspace", operation_timeout=5)) as app:
        app.plugins.install(folder)
        scene = app.scenes.create(name="timeout")
        obj = app.meshes.create_cube(scene_id=scene.id)
        app.engine.settings.operation_timeout = 0.5
        response = app.registry.execute("deform.twist_mesh", {"object_id": obj.id})
        assert response["error"]["code"] == "OPERATION_TIMEOUT"
        app.plugins.uninstall("twist_mesh")
        assert not app.plugins.list()["plugins"]


@pytest.mark.integration
def test_shared_packages_and_workspace_activation(tmp_path, twist_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    with Scissors(first) as app:
        app.plugins.install(twist_path)
        store = app.plugins.root
        assert store == Path(os.environ["SCISSORS_PLUGINS_DIR"]).resolve()
        assert app.call("modeler.info")["plugins_dir"] == str(store)
        assert app.call("plugins.list")["plugins_dir"] == str(store)
        assert not (first / "plugins").exists()
        app.plugins.disable("twist_mesh")
        with Scissors(second) as other:
            assert other.plugins.root == store
            assert "deform.twist_mesh" in other.registry.operations
            scene = other.scenes.create(name="Shared package")
            obj = other.meshes.create_cube(scene_id=scene.id)
            assert other.call("deform.twist_mesh", object_id=obj.id)["face_count"] == 12
            with pytest.raises(ScissorsError, match="already installed"):
                other.plugins.install(twist_path)
            assert "deform.twist_mesh" not in app.registry.operations
    with Scissors(first) as app:
        assert "deform.twist_mesh" not in app.registry.operations
        app.plugins.enable("twist_mesh")
    with Scissors(second) as other:
        other.plugins.uninstall("twist_mesh")
    with Scissors(first) as app:
        assert not app.plugins.list()["plugins"]
        assert "deform.twist_mesh" not in app.registry.operations


@pytest.mark.integration
def test_removed_shared_package_cannot_run_in_existing_session(tmp_path, twist_path):
    with Scissors(tmp_path / "first") as first:
        first.plugins.install(twist_path)
        with Scissors(tmp_path / "second") as second:
            scene = second.scenes.create(name="Removal")
            obj = second.meshes.create_cube(scene_id=scene.id)
            before = second.engine.snapshot()
            first.plugins.uninstall("twist_mesh")
            entry = second.plugins.list()["plugins"][0]
            assert not entry["installed"] and not entry["active"]
            assert entry["load_error"]["code"] == "PLUGIN_NOT_FOUND"
            with pytest.raises(ScissorsError) as error:
                second.call("deform.twist_mesh", object_id=obj.id)
            assert error.value.code == "PLUGIN_NOT_FOUND"
            with pytest.raises(ScissorsError) as error:
                second.plugins.uninstall("twist_mesh")
            assert error.value.code == "PLUGIN_NOT_FOUND"
            assert second.engine.snapshot() == before


def test_plugin_store_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("SCISSORS_PLUGINS_DIR")
    assert Settings().plugins_dir == Path.home() / ".scissors" / "plugins"
    monkeypatch.setenv("SCISSORS_PLUGINS_DIR", str(tmp_path / "environment"))
    assert Settings().plugins_dir == tmp_path / "environment"
    chosen = tmp_path / "chosen"
    with Scissors(settings=Settings(workspace=tmp_path / "scene", plugins_dir=chosen)) as app:
        assert app.plugins.root == chosen.resolve()


@pytest.mark.parametrize("location", ["same", "inside", "ancestor"])
def test_plugin_store_must_not_overlap_workspace(tmp_path, location):
    workspace = tmp_path / "workspace"
    store = {"same": workspace, "inside": workspace / "plugins", "ancestor": tmp_path}[location]
    with pytest.raises(ScissorsError) as error:
        Scissors(settings=Settings(workspace=workspace, plugins_dir=store))
    assert error.value.code == "INVALID_PARAMETER"
    # Initialization failure must release workspace ownership.
    with Scissors(workspace):
        pass


def test_workspace_plugin_folder_is_not_loaded(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(twist_path, workspace / "plugins" / "twist_mesh")
    with Scissors(workspace) as app:
        assert not app.plugins.list()["plugins"]
        assert "deform.twist_mesh" not in app.registry.operations
        assert (workspace / "plugins" / "twist_mesh" / "plugin.py").is_file()


def test_shared_store_lock_reports_busy(app):
    from filelock import FileLock

    app.engine.settings.operation_timeout = 0.05
    with FileLock(str(app.plugins.root / ".plugins.lock")):
        with pytest.raises(ScissorsError) as error:
            with app.plugins._locked():
                pass
    assert error.value.code == "PLUGIN_STORE_BUSY"


@pytest.mark.integration
def test_failed_shared_install_leaves_no_package_or_operations(app, tmp_path, twist_path):
    folder = tmp_path / "broken_import"
    shutil.copytree(twist_path, folder)
    (folder / "plugin.py").write_text("raise RuntimeError('import failed')\n")
    before = app.engine.snapshot()
    operations = set(app.registry.operations)
    with pytest.raises(ScissorsError) as error:
        app.plugins.install(folder)
    assert error.value.code == "PLUGIN_FAILED"
    assert app.engine.snapshot() == before
    assert set(app.registry.operations) == operations
    assert not (app.plugins.root / "twist_mesh").exists()
    assert app.plugins.install(twist_path)["state"]["enabled"]


def test_artifacts_cannot_write_shared_plugins(app):
    scene = app.scenes.create(name="Paths")
    target = app.plugins.root / "export.glb"
    with pytest.raises(ScissorsError) as error:
        app.call("scene.export", scene_id=scene.id, output=str(target))
    assert error.value.code == "PERMISSION_DENIED"
    assert not target.exists()
