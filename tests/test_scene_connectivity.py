"""Spatial attachment contracts, unrun under the no-tests instruction."""

import pytest

from scissors.api.errors import ScissorsError
from scissors.core.inspection import contact_components


def test_components_follow_indirect_contacts_and_multiple_anchors():
    ids = ["base", "bracket", "switch", "loose", "other_base"]
    contacts = [
        {"object_ids": ["base", "bracket"]},
        {"object_ids": ["bracket", "switch"]},
    ]
    result = contact_components(ids, contacts, ["base", "other_base"])
    assert result["floating_object_ids"] == ["loose"]
    assert result["connected_components"] == [
        ["base", "bracket", "switch"],
        ["loose"],
        ["other_base"],
    ]
    assert result["isolated_object_ids"] == ["loose", "other_base"]


def test_no_anchor_does_not_guess_which_part_is_supported():
    result = contact_components(["a", "b"], [], None)
    assert result["floating_object_ids"] is None
    assert result["isolated_object_ids"] == ["a", "b"]
    assert result["connectivity_warnings"]


def test_scene_collisions_reports_touching_chain_and_separate_part_without_mutation(app):
    scene = app.scenes.create()
    ids = []
    for position in [0, 1, 2, 5]:
        cube = app.meshes.create_cube(scene_id=scene.id)
        app.meshes.translate(object_id=cube.id, x=position)
        ids.append(cube.id)
    before = app.engine.snapshot()
    result = app.call("scene.collisions", scene_id=scene.id, anchor_object_ids=[ids[0]])
    assert result["floating_object_ids"] == [ids[3]]
    assert result["connected_components"] == [ids[:3], ids[3:]]
    assert result["pair_count"] == 2
    assert app.engine.snapshot() == before


def test_collision_anchors_must_belong_to_selection(app):
    scene = app.scenes.create()
    first = app.meshes.create_cube(scene_id=scene.id)
    second = app.meshes.create_cube(scene_id=scene.id)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.collisions",
            scene_id=scene.id,
            object_ids=[first.id],
            anchor_object_ids=[second.id],
        )
