"""Portable bundle publication contracts for the normal regression suite."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
import trimesh

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core import bundles


def assembly(app):
    sid = app.call("scene.create", name="Portable asset")["id"]
    first = app.call("mesh.create_cube", scene_id=sid)["id"]
    second = app.call("mesh.create_cube", scene_id=sid)["id"]
    app.call("mesh.translate", object_id=second, x=2)
    group = app.call("scene.group", scene_id=sid, object_ids=[first, second])["id"]
    app.call(
        "scene.marker", scene_id=sid, name="Player eye", position=[0, -2, 1.6], parent_id=group
    )
    return sid, first, second


def read_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    } == {"manifest.json", *[entry["path"] for entry in manifest["files"]]}
    for entry in manifest["files"]:
        content = (directory / entry["path"]).read_bytes()
        assert len(content) == entry["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    return manifest


def test_bundle_captures_one_scene_and_maps_material_batches(app):
    sid, first, second = assembly(app)
    before = app.engine.snapshot()
    result = app.call("scene.export_bundle", scene_id=sid, output_path="exports/asset")
    manifest = read_manifest(Path(result["path"]))
    assert app.engine.snapshot() == before
    hierarchy, batches = manifest["variants"]
    assert hierarchy["hierarchy_preserved"] and not batches["hierarchy_preserved"]
    assert hierarchy["source_to_export"][first] != hierarchy["source_to_export"][second]
    assert batches["source_to_export"][first] == batches["source_to_export"][second]
    for variant in manifest["variants"]:
        reopened = trimesh.load_scene(Path(result["path"]) / variant["path"], process=False)
        assert len(reopened.to_mesh().faces) == variant["triangle_count"] == 24
        assert variant["verification"]["reopen"]["status"] == "passed"
    assert manifest["markers"][0]["world_position"] == [0, -2, 1.6]


def test_bundle_scene_and_texture_references_survive_relocation(app, tmp_path):
    sid, first, _ = assembly(app)
    app.call("uv.generate", object_id=first, method="box")
    texture = app.call("texture.generate_checker", width=8, height=8)["id"]
    material = app.call("material.create", scene_id=sid)["id"]
    app.call("texture.assign", texture_id=texture, material_id=material)
    app.call("material.assign", object_id=first, material_id=material)
    unused = app.call("texture.generate_checker", width=4, height=4)["id"]
    result = app.call("scene.export_bundle", scene_id=sid, output_path="exports/asset")
    moved = tmp_path / "moved"
    shutil.copytree(result["path"], moved)
    with Scissors(moved) as restored:
        restored.call("scene.load", path="scene.json")
        assert restored.engine.scene(sid).model_dump() == app.engine.scene(sid).model_dump()
        assert unused not in restored.engine.textures
        assert restored.engine.path(restored.engine.textures[texture]["path"]).is_file()
        restored.call("scene.export", scene_id=sid, output="relocated.glb")
        assert len(trimesh.load_scene(moved / "relocated.glb").to_mesh().faces) == 24


def test_zip_bundle_members_match_manifest(app, tmp_path):
    sid, _, _ = assembly(app)
    result = app.call(
        "scene.export_bundle", scene_id=sid, output_path="exports/asset.zip", package_format="zip"
    )
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.testzip() is None
        assert all(
            not name.startswith("/") and ".." not in Path(name).parts for name in archive.namelist()
        )
        archive.extractall(tmp_path / "unpacked")
    read_manifest(tmp_path / "unpacked")
    assert result["manifest_in_archive"] and result["manifest_path"] == "manifest.json"
    assert result["verification"]["archive_entry_hashes"]["status"] == "passed"


def test_failed_export_retains_previous_bundle(app, monkeypatch):
    sid, _, _ = assembly(app)
    args = dict(scene_id=sid, output_path="exports/asset")
    app.call("scene.export_bundle", **args)
    output = app.engine.root / "exports/asset"
    original = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    real_export = bundles.export_scene
    count = 0

    def fail_second(*values, **options):
        nonlocal count
        count += 1
        if count == 2:
            raise ScissorsError("GEOMETRY_ERROR", "Injected export failure")
        return real_export(*values, **options)

    monkeypatch.setattr(bundles, "export_scene", fail_second)
    with pytest.raises(ScissorsError):
        app.call("scene.export_bundle", **args, overwrite=True)
    assert count == 2
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original
    assert not list(output.parent.glob(".scissors-bundle-*"))


def test_failed_directory_publication_restores_previous_files(app, monkeypatch):
    sid, _, _ = assembly(app)
    args = dict(scene_id=sid, output_path="exports/asset")
    app.call("scene.export_bundle", **args)
    output = app.engine.root / "exports/asset"
    original = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    replace = bundles.os.replace

    def fail_rename(source, destination):
        if Path(source).name == "bundle" and Path(destination) == output:
            raise OSError("Injected publication failure")
        return replace(source, destination)

    monkeypatch.setattr(bundles.os, "replace", fail_rename)
    with pytest.raises(ScissorsError) as error:
        app.call("scene.export_bundle", **args, overwrite=True)
    assert error.value.code == "FILESYSTEM_ERROR"
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original


@pytest.mark.parametrize("path", ["../outside", "plugins/export", "."])
def test_bundle_rejects_protected_output_paths(app, path):
    sid, _, _ = assembly(app)
    with pytest.raises(ScissorsError) as error:
        app.call("scene.export_bundle", scene_id=sid, output_path=path)
    assert error.value.code == "PERMISSION_DENIED"


def test_bundle_refuses_undeclared_files_on_overwrite(app):
    sid, _, _ = assembly(app)
    args = dict(scene_id=sid, output_path="exports/asset")
    result = app.call("scene.export_bundle", **args)
    personal = Path(result["path"]) / "personal.txt"
    personal.write_text("retain")
    with pytest.raises(ScissorsError):
        app.call("scene.export_bundle", **args, overwrite=True)
    assert personal.read_text() == "retain"


def test_bundle_rejects_atomic_scene_transactions(app):
    sid, _, _ = assembly(app)
    with app.transaction(), pytest.raises(ScissorsError) as error:
        app.call("scene.export_bundle", scene_id=sid, output_path="exports/asset")
    assert error.value.code == "INVALID_PARAMETER"


def test_structure_only_bundle_reports_skipped_reopening(app):
    sid, _, _ = assembly(app)
    result = app.call(
        "scene.export_bundle",
        scene_id=sid,
        output_path="exports/asset",
        include_scene=False,
        include_textures=False,
        report=False,
        verification="structure",
    )
    manifest = read_manifest(Path(result["path"]))
    assert manifest["checks"]["glb_reopen"] == "skipped"
    assert manifest["checks"]["editable_scene"]["status"] == "skipped"
    assert manifest["variants"][0]["verification"]["reopen"]["status"] == "skipped"


def test_bundle_reserves_base_frame_and_duplicate_node_names(app):
    sid, first, second = assembly(app)
    app.call("scene.update", scene_id=sid, name="world")
    app.call("mesh.rename", object_id=first, name="world")
    app.call("mesh.rename", object_id=second, name="world")
    result = app.call("scene.export_bundle", scene_id=sid, output_path="exports/asset")
    manifest = read_manifest(Path(result["path"]))
    names = manifest["variants"][0]["source_to_export"]
    assert names[first]["node_name"] == "world_2"
    assert names[second]["node_name"] == "world_3"


def test_bundle_advertises_file_effects_without_postpublication_scene_commit(app, monkeypatch):
    sid, _, _ = assembly(app)
    descriptor = app.registry.describe_operation("scene.export_bundle")
    assert descriptor["side_effects"] and not descriptor["scene_mutation"]

    def unexpected_commit():
        raise AssertionError("Bundle must not commit the unchanged source scene")

    monkeypatch.setattr(app.engine, "commit", unexpected_commit)
    before = app.engine.snapshot()
    result = app.call("scene.export_bundle", scene_id=sid, output_path="exports/asset")
    assert Path(result["path"]).is_dir() and app.engine.snapshot() == before
