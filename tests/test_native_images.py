"""Image/PBR regression cases. Not executed under the user's no-tests instruction."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scissors.api.errors import ScissorsError


def test_edits_create_new_textures_and_preserve_source(app):
    source = app.call(
        "texture.create", width=8, height=4, color_a=[255, 0, 0, 255], color_b=[255, 0, 0, 255]
    )
    original = Path(source["path"]).read_bytes()
    resized = app.call("texture.resize", texture_id=source["texture_id"], size=[16, 8])
    cropped = app.call("texture.crop", texture_id=resized["texture_id"], rectangle=[0, 0, 4, 2])
    rotated = app.call("texture.rotate", texture_id=cropped["texture_id"], angle=90)
    assert [rotated["width"], rotated["height"]] == [2, 4]
    assert Path(source["path"]).read_bytes() == original
    with pytest.raises(ScissorsError):
        app.call(
            "texture.resize", texture_id=source["texture_id"], size=[8, 8], output=source["path"]
        )


def test_alpha_compose_and_export_flattening(app):
    source = app.call(
        "texture.create", width=4, height=4, color_a=[255, 0, 0, 255], color_b=[255, 0, 0, 255]
    )
    result = app.call(
        "texture.compose",
        width=8,
        height=8,
        layers=[{"texture_id": source["texture_id"], "position": [2, 2], "opacity": 0.5}],
    )
    with Image.open(result["path"]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        assert image.getpixel((3, 3)) == (255, 0, 0, 128)
    exported = app.call(
        "texture.export", texture_id=result["texture_id"], output="artifacts/image.jpg"
    )
    with Image.open(exported["path"]) as image:
        assert image.mode == "RGB" and image.size == (8, 8)
    assert "JPEG alpha flattened over white" in exported["diagnostics"]


def test_text_bounds_and_atomic_filesystem_rejection(app):
    result = app.call(
        "texture.draw_text", text="PRESS", width=128, height=64, font_size=16, position=[4, 4]
    )
    assert not result["clipped"] and result["text_bounds"][2] > 4
    with Image.open(result["path"]) as image:
        assert np.asarray(image)[..., 3].max() == 255
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[{"operation": "texture.draw_text", "arguments": {"text": "X"}}],
        )
    with pytest.raises(ScissorsError), app.transaction():
        app.call("texture.draw_text", text="No filesystem effects")


def test_pbr_slots_and_standard_tangent_accessor_export(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id)
    app.uv.planar(object_id=plane.id)
    mat = app.call("material.create", scene_id=scene.id, emissive=[0.2, 0.4, 1])
    texture = app.call("texture.create", width=4, height=4)
    for slot in ["normal", "emissive", "occlusion"]:
        app.call(
            "texture.assign", material_id=mat["id"], texture_id=texture["texture_id"], slot=slot
        )
    app.call("material.assign", object_id=plane.id, material_id=mat["id"])
    frames = np.asarray(app.call("mesh.tangents", object_id=plane.id)["tangents"])
    normals = np.asarray(app.engine.object(plane.id)[1].normals)
    np.testing.assert_allclose(np.linalg.norm(frames[:, :3], axis=1), 1)
    np.testing.assert_allclose(np.sum(frames[:, :3] * normals, axis=1), 0, atol=1e-12)
    result = app.call("scene.export", scene_id=scene.id, output="artifacts/pbr.glb", format="glb")
    data = Path(result["path"]).read_bytes()
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    tree = json.loads(data[20 : 20 + length])
    material = tree["materials"][0]
    assert material["emissiveFactor"] == [0.2, 0.4, 1]
    assert "emissiveTexture" in material and "occlusionTexture" in material
    attributes = tree["meshes"][0]["primitives"][0]["attributes"]
    assert "TANGENT" in attributes and "_SCISSORS_TANGENT" not in attributes
    assert tree["accessors"][attributes["TANGENT"]]["type"] == "VEC4"


def test_invalid_pbr_factor_and_unknown_texture_restore_state(app):
    scene = app.scenes.create()
    material = app.call("material.create", scene_id=scene.id)
    with pytest.raises(ScissorsError):
        app.call("material.update", material_id=material["id"], properties={"emissive": [-1, 0, 0]})
    with pytest.raises(ScissorsError):
        app.call(
            "texture.assign", material_id=material["id"], texture_id="missing", slot="occlusion"
        )
    assert app.call("material.get", material_id=material["id"])["emissive"] == [0, 0, 0]
