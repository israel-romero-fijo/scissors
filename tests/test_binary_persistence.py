"""Binary persistence regressions; not executed under the user's no-tests request."""

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core import binary_persistence


def reference(workspace, scene_id, object_id):
    state = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
    return state, state["scenes"][scene_id]["objects"][object_id]["geometry"]


def test_binary_manifest_restart_and_json_scene_snapshot(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    app.uv.planar(object_id=cube.id)
    state, ref = reference(app.engine.root, scene.id, cube.id)
    record = state["scenes"][scene.id]["objects"][cube.id]
    assert state["format_version"] == 2 and "vertices" not in record
    assert (app.engine.root / ref["path"]).is_file()
    before = app.engine.snapshot()
    saved = app.scenes.save(scene_id=scene.id, output="scenes/portable.json")
    portable = json.loads(Path(saved.path).read_text())
    assert portable["format_version"] == 1 and portable["scene"]["objects"][cube.id]["vertices"]
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.snapshot() == before


def test_metadata_commit_reuses_existing_binary_arrays(app, monkeypatch):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    _, ref = reference(app.engine.root, scene.id, cube.id)
    path = app.engine.root / ref["path"]
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    def fail(_):
        raise AssertionError("Unchanged arrays must not be compressed again")

    monkeypatch.setattr(binary_persistence, "encoded_arrays", fail)
    app.meshes.rename(object_id=cube.id, name="Metadata only")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert reference(app.engine.root, scene.id, cube.id)[1] == ref


def test_legacy_workspace_upgrades_only_after_successful_mutation(tmp_path):
    workspace = tmp_path / "legacy"
    with Scissors(workspace) as app:
        scene = app.scenes.create()
        cube = app.meshes.create_cube(scene_id=scene.id)
        legacy = app.engine.snapshot()
    (workspace / "state.json").write_text(json.dumps(legacy), encoding="utf-8")
    with Scissors(workspace) as app:
        assert json.loads((workspace / "state.json").read_text())["format_version"] == 1
        app.meshes.rename(object_id=cube.id, name="Migrated")
    state, ref = reference(workspace, scene.id, cube.id)
    assert state["format_version"] == 2 and (workspace / ref["path"]).exists()


def test_manifest_failure_keeps_old_scene_and_disk_checkpoint(app, monkeypatch):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    before, disk = app.engine.snapshot(), (app.engine.root / "state.json").read_bytes()
    chunks = set((app.engine.root / ".geometry").iterdir())
    original = binary_persistence.atomic_write

    def fail_manifest(path, data, limit):
        if path.name == "state.json":
            raise OSError("manifest failure")
        original(path, data, limit)

    monkeypatch.setattr(binary_persistence, "atomic_write", fail_manifest)
    with pytest.raises(ScissorsError):
        app.meshes.translate(object_id=cube.id, x=2)
    assert app.engine.snapshot() == before and (app.engine.root / "state.json").read_bytes() == disk
    assert len(set((app.engine.root / ".geometry").iterdir()) - chunks) == 1


def test_changed_array_payload_is_rejected_even_with_valid_zip_crc(tmp_path):
    with Scissors(tmp_path) as app:
        scene = app.scenes.create()
        cube = app.meshes.create_cube(scene_id=scene.id)
    _, ref = reference(tmp_path, scene.id, cube.id)
    path = tmp_path / ref["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["vertices"][0, 0] += 0.01
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ScissorsError) as failure:
        Scissors(tmp_path)
    assert failure.value.code == "INCOMPATIBLE_FORMAT"


def test_declared_array_overflow_is_rejected_before_archive_decode(tmp_path, monkeypatch):
    with Scissors(tmp_path) as app:
        scene = app.scenes.create()
        cube = app.meshes.create_cube(scene_id=scene.id)
        maximum = app.engine.settings.max_vertices
    state, ref = reference(tmp_path, scene.id, cube.id)
    ref["arrays"]["vertices"]["shape"][0] = maximum + 1
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def unexpected(*args):
        raise AssertionError("Invalid counts must fail before reading compressed data")

    monkeypatch.setattr(binary_persistence, "decode", unexpected)
    with pytest.raises(ScissorsError) as failure:
        Scissors(tmp_path)
    assert failure.value.code == "RESOURCE_LIMIT"


def test_binary_workspace_relocation_preserves_ids_and_reserved_paths(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    with Scissors(source) as app:
        scene = app.scenes.create()
        app.meshes.create_cube(scene_id=scene.id)
        before = app.engine.snapshot()
    shutil.copytree(source, target)
    with Scissors(target) as app:
        assert app.engine.snapshot() == before
        with pytest.raises(ScissorsError):
            app.engine.path(".geometry/arbitrary.npz")


def test_duplicate_geometry_uses_one_content_file(app):
    scene = app.scenes.create()
    first = app.meshes.create_cube(scene_id=scene.id)
    second = app.meshes.duplicate(object_id=first.id)
    _, a = reference(app.engine.root, scene.id, first.id)
    _, b = reference(app.engine.root, scene.id, second.id)
    assert a == b
