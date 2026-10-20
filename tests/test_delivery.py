"""Export-delivery regression cases."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scissors.api.errors import ScissorsError


def glb_tree(path):
    data = Path(path).read_bytes()
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    return json.loads(data[20 : 20 + length])


def test_subset_keeps_ancestor_frames_and_bakes_export_transform_once(app):
    scene = app.scenes.create()
    a = app.meshes.create_cube(scene_id=scene.id)
    b = app.meshes.create_cube(scene_id=scene.id)
    group = app.scenes.group(scene_id=scene.id, object_ids=[a.id, b.id])
    app.meshes.translate(object_id=group.id, x=2)
    matrix = np.eye(4)
    matrix[:3, 3] = [3, 0, 0]
    before = app.engine.snapshot()
    result = app.call(
        "scene.export",
        scene_id=scene.id,
        output="exports/subset.stl",
        format="stl",
        only_object_ids=[a.id],
        export_transform=matrix.tolist(),
        root_name="Playable",
    )
    reopened = trimesh.load_mesh(result["path"], process=False)
    assert len(reopened.faces) == 12
    np.testing.assert_allclose(reopened.bounds, [[4.5, -0.5, -0.5], [5.5, 0.5, 0.5]])
    assert app.engine.snapshot() == before


def test_stl_keeps_every_instance_and_its_world_placement(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    other = app.call("scene.instance", object_id=cube.id)
    app.meshes.translate(object_id=other["object_id"], x=3)
    result = app.call(
        "scene.export", scene_id=scene.id, output="exports/instances.stl", format="stl"
    )
    reopened = trimesh.load_mesh(result["path"], process=False)
    assert len(reopened.faces) == 24
    np.testing.assert_allclose(reopened.bounds, [[-0.5, -0.5, -0.5], [3.5, 0.5, 0.5]])


def test_root_name_and_material_split_are_real_export_structure(app):
    scene = app.scenes.create()
    a = app.meshes.create_cube(scene_id=scene.id)
    b = app.meshes.create_cube(scene_id=scene.id)
    app.meshes.translate(object_id=b.id, x=2)
    mat = app.call("material.create", scene_id=scene.id, name="Metal")
    for obj in [a, b]:
        app.call("material.assign", object_id=obj.id, material_id=mat["id"])
    result = app.call(
        "scene.export",
        scene_id=scene.id,
        output="exports/grouped.glb",
        root_name="Playable",
        split_materials=True,
    )
    tree = glb_tree(result["path"])
    assert any(node.get("name") == "Playable" for node in tree["nodes"])
    assert len(tree["meshes"]) == 1 and result["exported_mesh_count"] == 1
    assert len(app.engine.scene(scene.id).objects) == 2


def test_split_export_by_material_excludes_other_material_descendants(app):
    scene = app.scenes.create()
    parent = app.meshes.create_cube(scene_id=scene.id)
    child = app.meshes.create_cube(scene_id=scene.id)
    app.objects.parent(object_id=child.id, parent_id=parent.id)
    for obj in [parent, child]:
        mat = app.call("material.create", scene_id=scene.id, name="Repeated name")
        app.call("material.assign", object_id=obj.id, material_id=mat["id"])
    result = app.call(
        "scene.split_export", scene_id=scene.id, output_dir="exports/parts", by="material"
    )
    assert result["part_count"] == 2
    assert len({part["path"] for part in result["files"]}) == 2
    for part in result["files"]:
        assert len(glb_tree(part["path"])["meshes"]) == 1
    assert len(json.loads(Path(result["manifest"]).read_text())["parts"]) == 2


def test_ply_writer_has_real_ply_header_and_geometry(app):
    scene = app.scenes.create()
    app.meshes.create_cube(scene_id=scene.id)
    result = app.call("scene.export", scene_id=scene.id, output="exports/cube.ply", format="ply")
    assert Path(result["path"]).read_bytes().startswith(b"ply\n")
    reopened = trimesh.load_mesh(result["path"], process=False)
    assert len(reopened.faces) == 12


def test_lod_export_reduces_faces_and_preserves_source(app):
    scene = app.scenes.create()
    sphere = app.meshes.create_sphere(scene_id=scene.id, subdivisions=2)
    before = app.engine.snapshot()
    result = app.call(
        "scene.export",
        scene_id=scene.id,
        output="exports/asset.glb",
        lod_reductions=[30, 60],
        lod_max_error_relative=0.2,
    )
    counts = [result["source_face_count"], *[level["face_count"] for level in result["lods"]]]
    assert counts[0] > counts[1] > counts[2] > 0
    for level in result["lods"]:
        assert len(trimesh.load_scene(level["path"]).to_mesh().faces) == level["face_count"]
    assert Path(result["lod_manifest"]).exists()
    assert app.engine.snapshot() == before and sphere.id in app.engine.scene(scene.id).objects


def test_export_rejects_invalid_subset_and_lod_order(app):
    scene = app.scenes.create()
    app.meshes.create_cube(scene_id=scene.id)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.export", scene_id=scene.id, output="exports/a.glb", only_object_ids=["missing"]
        )
    with pytest.raises(ScissorsError):
        app.call("scene.export", scene_id=scene.id, output="exports/a.glb", lod_reductions=[60, 30])
