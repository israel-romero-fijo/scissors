"""Atlas regression cases remain unrun under the no-tests instruction."""

import numpy as np
import pytest
from PIL import Image

from scissors.api.errors import ScissorsError


def part(app, scene_id, color, offset):
    image = app.call("texture.create", width=4, height=4, color_a=color, color_b=color)
    material = app.call("material.from_texture", scene_id=scene_id, texture_id=image["texture_id"])
    plane = app.meshes.create_plane(scene_id=scene_id)
    app.uv.planar(object_id=plane.id)
    app.meshes.translate(object_id=plane.id, x=offset)
    app.call("material.assign", object_id=plane.id, material_id=material["id"])
    return plane.id, material["id"]


def test_atlas_padding_uv_material_and_draw_call_reduction(app):
    scene = app.scenes.create()
    a, ma = part(app, scene.id, [255, 0, 0, 255], 0)
    b, mb = part(app, scene.id, [0, 255, 0, 255], 2)
    before = sum(len(app.engine.object(oid)[1].faces) for oid in [a, b])
    result = app.call("texture.atlas_pack", scene_id=scene.id, size=[32, 16], padding=2)
    assert result["draw_calls_before"] == 2 and result["draw_calls_after"] == 1
    assert result["object_mapping"] == {a: a, b: a}
    obj = app.engine.object(a)[1]
    assert len(obj.faces) == before and b not in app.engine.scene(scene.id).objects
    assert result["material_mapping"][ma] == result["material_mapping"][mb] == obj.material_id
    with Image.open(result["textures"]["base_color_texture"]["path"]) as image:
        for mid, color in [(ma, (255, 0, 0, 255)), (mb, (0, 255, 0, 255))]:
            u0, _, _, v1 = result["rectangles_by_material"][mid]
            x, y = round(u0 * 32 - 0.5), round((1 - v1) * 16 - 0.5)
            assert image.getpixel((x, y)) == color
            assert image.getpixel((x - 1, y - 1)) == color
    assert app.call("scene.budget", scene_id=scene.id)["draw_calls_estimate"] == 1


def test_atlas_can_preserve_objects_and_align_channels(app):
    scene = app.scenes.create()
    a, ma = part(app, scene.id, [255, 0, 0, 255], 0)
    b, _ = part(app, scene.id, [0, 255, 0, 255], 2)
    normal = app.call(
        "texture.create",
        width=4,
        height=4,
        color_a=[128, 128, 255, 255],
        color_b=[128, 128, 255, 255],
    )
    app.call("texture.assign", material_id=ma, texture_id=normal["texture_id"], slot="normal")
    result = app.call("texture.atlas_pack", scene_id=scene.id, size=[32, 16], merge=False)
    assert result["object_mapping"] == {a: a, b: b}
    assert set(result["textures"]) == {"base_color_texture", "normal_texture"}
    assert app.engine.object(a)[1].material_id == app.engine.object(b)[1].material_id
    for oid in [a, b]:
        uv = np.asarray(app.engine.object(oid)[1].uv)
        assert np.all(uv >= 0) and np.all(uv <= 1)
        assert np.ptp(uv[:, 0]) > 0 and np.ptp(uv[:, 1]) > 0


def test_atlas_failure_restores_scene_and_rejects_atomic_use(app):
    scene = app.scenes.create()
    a, _ = part(app, scene.id, [255, 0, 0, 255], 0)
    state = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("texture.atlas_pack", scene_id=scene.id, size=[2, 2])
    assert app.engine.snapshot() == state
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[{"operation": "texture.atlas_pack", "arguments": {"scene_id": scene.id}}],
        )
    obj = app.engine.object(a)[1]
    obj.uv[0][0] = 2
    with pytest.raises(ScissorsError):
        app.call("texture.atlas_pack", scene_id=scene.id)


def test_budget_uses_export_sharing_after_instance_translation(app):
    scene = app.scenes.create()
    cube = app.meshes.create_cube(scene_id=scene.id)
    instance = app.call("scene.instance", object_id=cube.id)
    app.meshes.translate(object_id=instance["object_id"], x=3)
    budget = app.call("scene.budget", scene_id=scene.id)
    assert budget["unique_meshes"] == 1 and budget["draw_calls_estimate"] == 2
