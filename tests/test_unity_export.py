"""Engine-facing export contracts: glTF up axis, readable nodes, mirroring, preview sections."""

import json
import struct

import numpy as np
import pytest
import trimesh
from PIL import Image

BACKGROUND = (242, 244, 248)


def glb_json(path: str) -> dict:
    data = open(path, "rb").read()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67
    _chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(data[20 : 20 + _chunk_length].decode("utf-8"))


def test_glb_is_y_up_for_engines(app):
    scene = app.scenes.create(name="Upright")
    cube = app.meshes.create_cube(scene_id=scene.id, size=0.5, name="Marker")
    app.meshes.translate(object_id=cube.id, z=3.0)
    exported = app.export.glb(scene_id=scene.id)
    assert any("Y-up" in line for line in exported.diagnostics)
    bounds = trimesh.load(exported.path, force="scene").bounds
    # Scene +Z is the height axis, so the marker at z=3 must arrive as glTF y=3.
    assert np.allclose(bounds[0], [-0.25, 2.75, -0.25], atol=1e-6)
    assert np.allclose(bounds[1], [0.25, 3.25, 0.25], atol=1e-6)
    assert np.allclose(bounds[1] - bounds[0], [0.5, 0.5, 0.5], atol=1e-6)


def test_gltf_converts_up_axis_and_other_formats_keep_scene_axes(app):
    scene = app.scenes.create(name="Formats")
    cube = app.meshes.create_cube(scene_id=scene.id, size=0.5, name="Marker")
    app.meshes.translate(object_id=cube.id, z=3.0)
    gltf = app.call("scene.export", scene_id=scene.id, output="exports/model.gltf", format="gltf")
    gltf_height = trimesh.load(gltf["path"], force="scene").bounds[1][1]
    assert abs(gltf_height - 3.25) < 1e-6
    for name in ("obj", "stl"):
        result = app.call(
            "scene.export", scene_id=scene.id, output=f"exports/model.{name}", format=name
        )
        bounds = trimesh.load(result["path"], force="scene").bounds
        assert abs(bounds[1][2] - 3.25) < 1e-6
        assert abs(bounds[1][1] - 0.25) < 1e-6


def test_glb_uses_one_root_node_for_the_whole_scene(app):
    scene = app.scenes.create(name="AH-64 Apache pilot station")
    app.meshes.create_cube(scene_id=scene.id, name="Panel")
    app.meshes.create_cube(scene_id=scene.id, name="Seat")
    document = glb_json(app.export.glb(scene_id=scene.id).path)
    roots = document["scenes"][0]["nodes"]
    assert len(roots) == 1
    root = document["nodes"][roots[0]]
    assert root["name"] == "AH_64_Apache_pilot_station"
    assert len(root["children"]) == 2
    assert "matrix" in root
    child_names = [document["nodes"][index]["name"] for index in root["children"]]
    assert child_names == ["Panel", "Seat"]
    # The conversion lives on the root only, so children keep identity transforms.
    assert all("matrix" not in document["nodes"][index] for index in root["children"])


def test_exported_nodes_use_readable_unique_names(app):
    scene = app.scenes.create(name="Names")
    panel = app.meshes.create_cube(scene_id=scene.id, name="Instrument panel")
    app.meshes.duplicate(object_id=panel.id, name="Instrument panel")
    duplicated = app.meshes.create_cube(scene_id=scene.id, name="Collective stick!")
    assert duplicated.name.startswith("Collective")
    names = [node["name"] for node in glb_json(app.export.glb(scene_id=scene.id).path)["nodes"]]
    # The scene name becomes the single root node; objects keep readable unique names.
    assert names == ["Names", "Instrument_panel", "Instrument_panel_2", "Collective_stick"]


def test_mirror_duplicate_keeps_source_and_restores_winding(app):
    scene = app.scenes.create(name="Symmetry")
    material = app.materials.create(scene_id=scene.id, name="Metal")
    source = app.meshes.create_cube(scene_id=scene.id, size=0.5, name="Left console")
    app.meshes.translate(object_id=source.id, x=1.0, z=0.4)
    app.uv.generate(object_id=source.id, method="box")
    app.materials.assign(object_id=source.id, material_id=material.id)
    before = app.meshes.stats(object_id=source.id)
    mirrored = app.call("mesh.mirror", object_id=source.id, axis="x", plane=0, name="Right console")
    assert mirrored["duplicated"] is True
    assert mirrored["mirrored_object_id"] != source.id
    assert mirrored["mirrored_object_id"] == mirrored["object_id"]
    assert np.allclose(app.meshes.stats(object_id=source.id)["bounds"], before["bounds"])
    bounds = np.asarray(mirrored["bounds"])
    assert np.allclose(bounds[0], [-1.25, -0.25, 0.15], atol=1e-6)
    assert np.allclose(bounds[1], [-0.75, 0.25, 0.65], atol=1e-6)
    # A reflected mesh keeps outward normals only if its winding is reversed.
    assert mirrored["volume"] > 0
    assert abs(mirrored["volume"] - before["volume"]) < 1e-9
    assert mirrored["has_uv"] is True
    assert mirrored["material_id"] == material.id
    assert any("winding" in line for line in mirrored["warnings"])


def test_mirror_in_place_uses_plane_offset(app):
    scene = app.scenes.create(name="In place")
    cube = app.meshes.create_cube(scene_id=scene.id, size=0.5, name="Bracket")
    app.meshes.translate(object_id=cube.id, x=0.6)
    result = app.call("mesh.mirror", object_id=cube.id, axis="x", plane=1.0, duplicate=False)
    assert result["duplicated"] is False
    assert result["mirrored_object_id"] == cube.id
    bounds = np.asarray(result["bounds"])
    # The cube spans x 0.35..0.85; reflecting across x=1 gives 1.15..1.65.
    assert np.allclose(bounds[0], [1.15, -0.25, -0.25], atol=1e-6)
    assert np.allclose(bounds[1], [1.65, 0.25, 0.25], atol=1e-6)
    assert result["volume"] > 0


def test_preview_camera_and_section_are_controllable(app):
    scene = app.scenes.create(name="Views")
    cube = app.meshes.create_cube(scene_id=scene.id, size=1.0, name="Block")
    app.meshes.translate(object_id=cube.id, z=0.5)
    default = app.scenes.preview(
        scene_id=scene.id, output="previews/default.png", width=128, height=128
    )
    bottom = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/bottom.png",
        width=128,
        height=128,
        azimuth_degrees=180,
        elevation_degrees=-65,
    )
    assert open(default.path, "rb").read() != open(bottom["path"], "rb").read()
    assert any("azimuth" in line for line in default.diagnostics)
    above = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/above.png",
        width=64,
        height=64,
        clip_axis="z",
        clip_position=50.0,
    )
    with Image.open(above["path"]) as image:
        assert image.getcolors(maxcolors=4096) == [(64 * 64, BACKGROUND)]
    assert any("Section view" in line for line in above["diagnostics"])
    kept = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/kept.png",
        width=64,
        height=64,
        clip_axis="z",
        clip_position=50.0,
        clip_keep="negative",
    )
    with Image.open(kept["path"]) as image:
        assert len(image.getcolors(maxcolors=4096)) > 1


@pytest.mark.parametrize("renderer", ["auto", "pillow"])
def test_preview_can_sample_base_colour_textures(app, renderer):
    scene = app.scenes.create(name="Sampling")
    source = app.engine.path("textures_src/two_tone.png")
    source.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:32] = (220, 30, 30)
    pixels[32:] = (10, 20, 200)
    Image.fromarray(pixels).save(source)
    imported = app.textures.load(path="textures_src/two_tone.png")
    material = app.materials.create(scene_id=scene.id, name="Two tone")
    app.textures.assign(material_id=material.id, texture_id=imported.texture_id)
    plane = app.meshes.create_plane(scene_id=scene.id, size=1.0, name="Label")
    app.uv.generate(object_id=plane.id, method="planar")
    app.materials.assign(object_id=plane.id, material_id=material.id)
    flat = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/flat.png",
        width=64,
        height=64,
        elevation_degrees=90.0,
        renderer=renderer,
        background=[1.0, 1.0, 1.0],
    )
    textured = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/textured.png",
        width=64,
        height=64,
        elevation_degrees=90.0,
        sample_textures=True,
        renderer=renderer,
        background=[1.0, 1.0, 1.0],
    )
    assert any("no texture sampling" in line for line in flat["diagnostics"])
    assert any("textures sampled" in line for line in textured["diagnostics"])
    with Image.open(flat["path"]) as image:
        flat_pixels = np.asarray(image.convert("RGB"), dtype=int)
    with Image.open(textured["path"]) as image:
        rendered = np.asarray(image, dtype=int)
    # A neutral background keeps antialiased silhouette pixels neutral too.
    # Without sampling the part must have no red or blue material tint.
    background = np.full(3, 255)
    flat_covered = np.abs(flat_pixels - background).sum(axis=2) > 12
    flat_shades = flat_pixels[flat_covered]
    assert flat_covered.sum() > 200
    assert np.abs(flat_shades[:, 0] - flat_shades[:, 2]).max() <= 3
    # The image is red on top and blue below. Looking straight down, screen up is -Y,
    # so the part's -Y edge (v = 0) samples the image's bottom row (blue) and appears
    # at the top: the engine's bottom-left UV convention shows through either way.
    covered = np.abs(rendered - background).sum(axis=2) > 12
    rows, _ = np.nonzero(covered)
    assert covered.sum() > 200
    middle = (int(rows.min()) + int(rows.max())) // 2
    row_index = np.arange(rendered.shape[0])[:, None]
    top = rendered[covered & (row_index <= middle)].reshape(-1, 3).mean(axis=0)
    bottom = rendered[covered & (row_index > middle)].reshape(-1, 3).mean(axis=0)
    assert top[2] > top[0] + 15, top
    assert bottom[0] > bottom[2] + 15, bottom


def test_preview_camera_never_rolls_and_top_view_uses_forward_up(app):
    scene = app.scenes.create(name="Camera")
    high_material = app.materials.create(
        scene_id=scene.id, name="High", base_color=[1.0, 0.0, 0.0, 1]
    )
    low_material = app.materials.create(
        scene_id=scene.id, name="Low", base_color=[0.0, 0.0, 1.0, 1]
    )
    high = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="High")
    app.meshes.translate(object_id=high.id, z=1.5)
    app.materials.assign(object_id=high.id, material_id=high_material.id)
    low = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="Low")
    app.meshes.translate(object_id=low.id, z=-1.5)
    app.materials.assign(object_id=low.id, material_id=low_material.id)

    def halves(result):
        with Image.open(result["path"]) as image:
            pixels = np.asarray(image, dtype=int)
        red = (pixels[:, :, 0] > pixels[:, :, 2] + 40) & (pixels[:, :, 0] > 80)
        blue = (pixels[:, :, 2] > pixels[:, :, 0] + 40) & (pixels[:, :, 2] > 80)
        rows = pixels.shape[0]
        upper = np.arange(rows)[:, None] < rows // 2
        return int((red & upper).sum()) > int((red & ~upper).sum()), int(
            (blue & ~upper).sum()
        ) > int((blue & upper).sum())

    # World +Z must stay above for every horizontal view: the camera never rolls.
    for azimuth in (0.0, 90.0, 180.0, 270.0, 35.0):
        result = app.call(
            "scene.preview",
            scene_id=scene.id,
            output=f"previews/upright_{int(azimuth)}.png",
            width=128,
            height=128,
            azimuth_degrees=azimuth,
            elevation_degrees=0.0,
        )
        red_above, blue_below = halves(result)
        assert red_above and blue_below, (azimuth, red_above, blue_below)

    # Looking straight down, -Y (the direction a model faces) is screen up.
    app.meshes.delete(object_id=high.id)
    app.meshes.delete(object_id=low.id)
    forward = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="Forward")
    app.meshes.translate(object_id=forward.id, y=-1.5)
    app.materials.assign(object_id=forward.id, material_id=high_material.id)
    aft = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="Aft")
    app.meshes.translate(object_id=aft.id, y=1.5)
    app.materials.assign(object_id=aft.id, material_id=low_material.id)
    top = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/top.png",
        width=128,
        height=128,
        azimuth_degrees=180.0,
        elevation_degrees=90.0,
    )
    red_above, blue_below = halves(top)
    assert red_above and blue_below


def test_preview_section_clips_in_world_space(app):
    scene = app.scenes.create(name="Section")
    top_material = app.materials.create(
        scene_id=scene.id, name="Top", base_color=[1.0, 0.0, 0.0, 1]
    )
    bottom_material = app.materials.create(
        scene_id=scene.id, name="Bottom", base_color=[0.0, 0.0, 1.0, 1]
    )
    top = app.meshes.create_cube(scene_id=scene.id, size=0.6, name="Top")
    app.meshes.translate(object_id=top.id, z=1.0)
    app.materials.assign(object_id=top.id, material_id=top_material.id)
    bottom = app.meshes.create_cube(scene_id=scene.id, size=0.6, name="Bottom")
    app.meshes.translate(object_id=bottom.id, z=-1.0)
    app.materials.assign(object_id=bottom.id, material_id=bottom_material.id)

    def counts(result):
        with Image.open(result["path"]) as image:
            pixels = np.asarray(image, dtype=int)
        red = ((pixels[:, :, 0] > 90) & (pixels[:, :, 0] > pixels[:, :, 2] + 40)).sum()
        blue = ((pixels[:, :, 2] > 90) & (pixels[:, :, 2] > pixels[:, :, 0] + 40)).sum()
        return int(red), int(blue)

    # A section must follow world axes whatever the camera looks at, so the same
    # plane is used from a horizontal view and from an oblique one.
    for azimuth, elevation in ((0.0, 0.0), (35.0, 40.0), (215.0, 58.0)):
        keep_upper = app.call(
            "scene.preview",
            scene_id=scene.id,
            output=f"previews/section_up_{int(azimuth)}.png",
            width=96,
            height=96,
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            clip_axis="z",
            clip_position=0.0,
            clip_keep="positive",
        )
        red, blue = counts(keep_upper)
        assert red > 0 and blue == 0, (azimuth, elevation, red, blue)
        keep_lower = app.call(
            "scene.preview",
            scene_id=scene.id,
            output=f"previews/section_down_{int(azimuth)}.png",
            width=96,
            height=96,
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            clip_axis="z",
            clip_position=0.0,
            clip_keep="negative",
        )
        red, blue = counts(keep_lower)
        assert blue > 0 and red == 0, (azimuth, elevation, red, blue)


def test_preview_object_filters_narrow_the_render(app):
    scene = app.scenes.create(name="Filters")
    near = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="Near")
    app.meshes.translate(object_id=near.id, x=-4.0)
    far = app.meshes.create_cube(scene_id=scene.id, size=0.4, name="Far")
    app.meshes.translate(object_id=far.id, x=4.0)
    only = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/only.png",
        width=64,
        height=64,
        only_object_ids=[near.id],
    )
    assert any("limited to 1" in line for line in only["diagnostics"])
    with Image.open(only["path"]) as image:
        colors = image.getcolors(maxcolors=4096)
    # Only the near cube is drawn, so the far cube cannot add pixels far to the right.
    assert len(colors) > 1
    excluded = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/excluded.png",
        width=64,
        height=64,
        exclude_object_ids=[near.id, far.id],
    )
    with Image.open(excluded["path"]) as image:
        assert image.getcolors(maxcolors=4096) == [(64 * 64, BACKGROUND)]
    try:
        app.call(
            "scene.preview",
            scene_id=scene.id,
            output="previews/bad.png",
            only_object_ids=["obj_missing"],
        )
    except Exception as error:  # noqa: BLE001 - the registry raises ScissorsError
        assert "OBJECT_NOT_FOUND" in getattr(error, "code", "")
    else:
        raise AssertionError("Unknown object ID must be rejected")
