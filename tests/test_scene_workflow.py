"""Scene workflow regression cases; unrun under the no-tests request."""

import numpy as np
import pytest

from scissors import Scissors
from scissors.api.errors import ScissorsError


def test_clone_remaps_hierarchy_materials_and_preserves_shared_textures(app):
    scene = app.scenes.create(name="Source")
    cube = app.meshes.create_cube(scene_id=scene.id)
    group = app.scenes.group(scene_id=scene.id, object_ids=[cube.id])
    texture = app.call("texture.create", width=4, height=4)
    mat = app.call("material.from_texture", scene_id=scene.id, texture_id=texture["texture_id"])
    app.call("material.assign", object_id=cube.id, material_id=mat["id"])
    result = app.call("scene.clone", scene_id=scene.id, name="Variant")
    cloned = app.engine.scene(result["scene_id"])
    copied = cloned.objects[result["object_mapping"][cube.id]]
    assert copied.parent_id == result["object_mapping"][group.id]
    assert copied.material_id == result["material_mapping"][mat["id"]]
    assert cloned.materials[copied.material_id].base_color_texture == texture["texture_id"]
    np.testing.assert_allclose(copied.vertices, app.engine.object(cube.id)[1].vertices)
    app.meshes.translate(object_id=copied.id, x=3)
    np.testing.assert_allclose(
        app.engine.object(cube.id)[1].summary()["centroid"], [0, 0, 0], atol=1e-12
    )


def test_copy_selected_subtree_detaches_external_parent_and_rejects_invalid_ids(app):
    source, target = app.scenes.create(), app.scenes.create()
    cube = app.meshes.create_cube(scene_id=source.id)
    group = app.scenes.group(scene_id=source.id, object_ids=[cube.id])
    child = app.scenes.marker(scene_id=source.id, parent_id=cube.id, position=[0, 0, 2])
    result = app.call(
        "scene.copy_objects", from_scene=source.id, to_scene=target.id, object_ids=[cube.id]
    )
    assert group.id not in result["object_mapping"]
    copied = app.engine.object(result["object_mapping"][cube.id])[1]
    assert copied.parent_id is None
    assert app.engine.object(result["object_mapping"][child.id])[1].parent_id == copied.id
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "scene.copy_objects", from_scene=source.id, to_scene=target.id, object_ids=["unknown"]
        )
    assert app.engine.snapshot() == before


def test_checkpoint_diff_revert_and_reserved_paths(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    first = app.call("scene.checkpoint", scene_id=scene.id, name="shape")
    assert not first["replaced"]
    app.meshes.translate(object_id=cube.id, x=2)
    marker = app.scenes.marker(scene_id=scene.id)
    diff = app.call("scene.compare_snapshot", scene_id=scene.id, name="shape")
    assert diff["objects"]["added"] == [marker.id]
    assert "vertices" in diff["objects"]["changed_fields"][cube.id]
    app.call("scene.revert_to", scene_id=scene.id, name="shape")
    np.testing.assert_allclose(
        app.engine.object(cube.id)[1].summary()["centroid"], [0, 0, 0], atol=1e-12
    )
    assert marker.id not in app.engine.scene(scene.id).objects
    with pytest.raises(ScissorsError):
        app.call("scene.checkpoint", scene_id=scene.id, name="../outside")
    with pytest.raises(ScissorsError):
        app.engine.path(".checkpoints/file.json")


def test_checkpoint_survives_restart_and_restores_deleted_scene(tmp_path):
    with Scissors(workspace=tmp_path) as app:
        scene = app.scenes.create()
        cube = app.meshes.create_cube(scene_id=scene.id)
        app.call("scene.checkpoint", scene_id=scene.id, name="saved")
        app.call("scene.delete", scene_id=scene.id)
    with Scissors(workspace=tmp_path) as app:
        app.call("scene.revert_to", scene_id=scene.id, name="saved")
        assert cube.id in app.engine.scene(scene.id).objects


def test_checkpoint_rejects_transaction_filesystem_effects(app):
    scene = app.scenes.create()
    with pytest.raises(ScissorsError), app.transaction():
        app.call("scene.checkpoint", scene_id=scene.id, name="blocked")
    with pytest.raises(ScissorsError) as failure:
        app.call("scene.revert_to", scene_id=scene.id, name="missing")
    assert failure.value.code == "SNAPSHOT_NOT_FOUND"


def test_figure_table_uses_one_commit_and_places_assigns_parents(app, monkeypatch):
    scene = app.scenes.create()
    marker = app.scenes.marker(scene_id=scene.id)
    material = app.call("material.create", scene_id=scene.id)
    original_commit = app.engine.commit
    commits = []

    def commit():
        commits.append(True)
        original_commit()

    monkeypatch.setattr(app.engine, "commit", commit)
    result = app.call(
        "scene.create_table",
        scene_id=scene.id,
        rows=[
            {
                "figure": "prism",
                "parameters": {"sides": 6, "radius": 0.2, "height": 0.4},
                "name": "Bolt",
                "transform": {"position": [2, 0, 0]},
                "material_id": material["id"],
                "parent_id": marker.id,
            },
            {"figure": "mesh.create_wedge", "parameters": {"size": [1, 1, 1]}, "name": "Ramp"},
        ],
    )
    assert len(commits) == 1 and result["created_count"] == 2
    bolt = app.engine.object(result["objects"][0]["object_id"])[1]
    assert bolt.name == "Bolt" and bolt.material_id == material["id"]
    assert bolt.parent_id == marker.id
    assert bolt.transform[0][3] == pytest.approx(2)


def test_table_geometry_failure_restores_previous_rows_and_rejects_other_operations(app):
    scene = app.scenes.create()
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "scene.create_table",
            scene_id=scene.id,
            rows=[
                {"figure": "cube"},
                {"figure": "capsule", "parameters": {"radius": 1, "height": 0.5}},
            ],
        )
    assert app.engine.snapshot() == before
    assert app.registry.batch_depth == 0
    with pytest.raises(ScissorsError):
        app.call("scene.create_table", scene_id=scene.id, rows=[{"figure": "scene.delete"}])


def test_delete_accepts_a_batch_and_rolls_back_a_bad_id(app):
    scene = app.scenes.create()
    ids = [app.meshes.create_cube(scene_id=scene.id).id for _ in range(3)]
    result = app.call("mesh.delete", object_ids=ids)
    assert result["deleted_object_ids"] == ids
    assert app.engine.scene(scene.id).objects == {}

    kept = [app.meshes.create_cube(scene_id=scene.id).id for _ in range(2)]
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError) as failure:
        app.call("mesh.delete", object_ids=[kept[0], "obj_missing", kept[1]])
    assert failure.value.code == "OBJECT_NOT_FOUND"
    assert app.engine.snapshot() == before
    assert set(kept) <= set(app.engine.scene(scene.id).objects)

    single = app.meshes.create_cube(scene_id=scene.id).id
    deleted = app.call("mesh.delete", object_id=single)
    assert deleted["deleted"] == single


def test_create_table_has_no_hundred_row_cap_and_supports_repeat(app):
    scene = app.scenes.create()
    rows = [{"figure": "cube", "parameters": {"size": 0.1}}] * 100
    first = app.call("scene.create_table", scene_id=scene.id, rows=rows)
    second = app.call("scene.create_table", scene_id=scene.id, rows=rows)
    assert first["created_count"] == 100 and second["created_count"] == 100
    assert len(app.engine.scene(scene.id).objects) == 200

    repeated = app.call(
        "scene.create_table",
        scene_id=scene.id,
        rows=[{"figure": "cube", "parameters": {"size": 0.1}, "repeat": 150}],
    )
    assert repeated["created_count"] == 150
    assert repeated["objects"][0]["repeat_index"] == 0
    assert repeated["objects"][-1]["repeat_index"] == 149
    assert len(app.engine.scene(scene.id).objects) == 350
