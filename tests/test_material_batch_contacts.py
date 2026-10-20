"""Render batches preserve independent source topology at exact contacts."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError
from scissors.core.delivery import consolidate_materials


@pytest.mark.parametrize("offset", [0, 1])
def test_contact_batch_reopens_without_moving_sources(app, offset):
    sid = app.call("scene.create")["id"]
    a = app.call("mesh.create_cube", scene_id=sid)["id"]
    b = app.call("mesh.create_cube", scene_id=sid)["id"]
    app.call("mesh.translate", object_id=b, x=offset)
    before = app.engine.snapshot()
    result = app.call("scene.export", scene_id=sid, output="batch.glb", split_materials=True)
    scene = trimesh.load(result["path"], force="scene", process=False)
    assert len(scene.geometry) == 1
    mesh = next(iter(scene.geometry.values()))
    assert len(mesh.faces) == 24
    assert len(np.unique(mesh.faces[:12])) == 8
    assert not set(mesh.faces[:12].flat) & set(mesh.faces[12:].flat)
    assert app.engine.snapshot() == before
    assert {a, b} == set(app.engine.scene(sid).objects)


def test_batch_still_rejects_invalid_source(app):
    sid = app.call("scene.create")["id"]
    oid = app.call("mesh.create_cube", scene_id=sid)["id"]
    obj = app.engine.object(oid)[1]
    obj.faces.append(obj.faces[0])
    with pytest.raises(ScissorsError, match="validation"):
        consolidate_materials(app.engine, app.engine.scene(sid))


def test_bundle_uses_same_contact_batch_validation(app):
    sid = app.call("scene.create")["id"]
    for _ in range(2):
        app.call("mesh.create_cube", scene_id=sid)
    result = app.call(
        "scene.export_bundle",
        scene_id=sid,
        output_path="bundle",
        variants=[{"name": "game", "preset": "material_batches"}],
    )
    assert Path(result["path"]).is_dir()
    glb = next(Path(result["path"]).rglob("*.glb"))
    assert len(trimesh.load(glb, force="scene").geometry) == 1


def test_textured_contact_batch_retains_uv_colors_and_normals(app):
    sid = app.call("scene.create")["id"]
    tid = app.call("texture.generate_checker", width=16, height=16)["id"]
    mid = app.call("material.from_texture", scene_id=sid, texture_id=tid)["id"]
    for _ in range(2):
        oid = app.call("mesh.create_cube", scene_id=sid)["id"]
        app.call("uv.generate", object_id=oid)
        app.call("material.assign", object_id=oid, material_id=mid)
        obj = app.engine.object(oid)[1]
        obj.vertex_colors = [[255, 128, 64, 255] for _ in obj.vertices]
    before = app.engine.snapshot()
    result = app.call("scene.export", scene_id=sid, output="texture.glb", split_materials=True)
    mesh = next(iter(trimesh.load(result["path"], force="scene", process=False).geometry.values()))
    assert len(mesh.faces) == 24 and mesh.visual.kind == "texture"
    assert len(mesh.visual.uv) == len(mesh.vertices)
    assert mesh.visual.material.baseColorTexture.size == (16, 16)
    assert np.isfinite(mesh.vertex_normals).all()
    np.testing.assert_array_equal(mesh.visual.vertex_attributes["color"][0], [255, 128, 64, 255])
    assert app.engine.snapshot() == before


def test_contact_batch_lod_simplifies_parts_before_concatenating(app):
    sid = app.call("scene.create")["id"]
    for _ in range(2):
        app.call("mesh.create_sphere", scene_id=sid, subdivisions=2)
    before = app.engine.snapshot()
    result = app.call(
        "scene.export",
        scene_id=sid,
        output="lod.glb",
        split_materials=True,
        lod_reductions=[30],
        lod_max_error_relative=0.2,
    )
    level = result["lods"][0]
    assert level["face_count"] < result["source_face_count"]
    mesh = next(iter(trimesh.load(level["path"], force="scene", process=False).geometry.values()))
    assert len(mesh.faces) == level["face_count"]
    assert app.engine.snapshot() == before
