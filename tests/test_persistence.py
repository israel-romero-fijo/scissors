import json

import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


def test_scene_restart_and_snapshot(tmp_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        scene = app.scenes.create(name="Persistent")
        cube = app.meshes.create_cube(scene_id=scene.id)
        app.meshes.translate(object_id=cube.id, x=2)
        saved = app.scenes.save(scene_id=scene.id)
        app.scenes.delete(scene_id=scene.id)
        loaded = app.scenes.load(path=saved.path)
        assert loaded.id == scene.id and loaded.objects[0]["id"] == cube.id
    with Scissors(workspace) as app:
        assert app.meshes.inspect(object_id=cube.id).bounds[0][0] == 1.5
        assert app.scenes.get(scene_id=scene.id).name == "Persistent"


def test_workspace_exclusivity_and_incompatible_version(tmp_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        app.scenes.create(name="test")
        with pytest.raises(ScissorsError, match="Another process"):
            Scissors(workspace)
    state = json.loads((workspace / "state.json").read_text())
    state["format_version"] = 999
    (workspace / "state.json").write_text(json.dumps(state))
    with pytest.raises(ScissorsError, match="Unsupported workspace version"):
        Scissors(workspace)
