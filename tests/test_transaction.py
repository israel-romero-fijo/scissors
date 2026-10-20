"""Deferred-checkpoint transactions for scripted builds."""

import json
from pathlib import Path

import pytest

from scissors.api.errors import ScissorsError


def state_file(app):
    return Path(app.engine.root) / "state.json"


def test_transaction_defers_state_writes_and_keeps_changes(app):
    assert not state_file(app).exists()
    with app.transaction():
        scene = app.scenes.create(name="Bulk")
        for index in range(5):
            app.meshes.create_cube(scene_id=scene.id, name=f"Cube {index}")
        # Inside the transaction the new scene is live in memory but not yet written.
        assert len(app.call("scene.list")["scenes"]) == 1
        assert not state_file(app).exists()
    written = json.loads(state_file(app).read_text(encoding="utf-8"))
    assert len(written["scenes"]) == 1
    assert len(next(iter(written["scenes"].values()))["objects"]) == 5


def test_transaction_rolls_back_every_change_on_error(app):
    with pytest.raises(RuntimeError):
        with app.transaction():
            scene = app.scenes.create(name="Doomed")
            app.meshes.create_cube(scene_id=scene.id, name="Cube")
            assert len(app.call("scene.list")["scenes"]) == 1
            raise RuntimeError("build failed")
    assert app.call("scene.list")["scenes"] == []
    assert not state_file(app).exists()


def test_transaction_rolls_back_after_a_failed_operation(app):
    scene = app.scenes.create(name="Committed")
    with pytest.raises(ScissorsError):
        with app.transaction():
            app.meshes.create_cube(scene_id=scene.id, name="First")
            app.call("mesh.mirror", object_id="obj_missing", axis="x")
    assert len(app.call("scene.list")["scenes"]) == 1
    assert app.call("scene.inspect", scene_id=scene.id)["objects"] == []


def test_transaction_can_be_reused_and_rejects_nesting(app):
    with app.transaction():
        app.scenes.create(name="One")
    with app.transaction():
        app.scenes.create(name="Two")
        with pytest.raises(ScissorsError) as error:
            with app.transaction():
                pass
        assert error.value.code == "INVALID_PARAMETER"
    assert len(app.call("scene.list")["scenes"]) == 2
    assert len(json.loads(state_file(app).read_text(encoding="utf-8"))["scenes"]) == 2
