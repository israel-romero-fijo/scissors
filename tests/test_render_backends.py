"""Fast preview backends: backend selection, framing, sections and limits."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core import rendering_fast

BACKGROUND = (242, 244, 248)


def covered(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=int)
    return np.abs(pixels - np.asarray(BACKGROUND)).sum(axis=2) > 12


def test_preview_reports_the_backend_that_drew_it(app):
    scene = app.scenes.create(name="Backends")
    app.meshes.create_cube(scene_id=scene.id)
    automatic = app.call("scene.preview", scene_id=scene.id, width=64, height=64)
    assert automatic["renderer"] in {"vtk", "pillow"}
    assert isinstance(automatic["renderer_detail"], dict)
    assert any("Renderer:" in line for line in automatic["diagnostics"])

    painter = app.call("scene.preview", scene_id=scene.id, width=64, height=64, renderer="pillow")
    assert painter["renderer"] == "pillow"


def test_removed_renderer_names_are_rejected(app):
    scene = app.scenes.create(name="Rejected")
    app.meshes.create_cube(scene_id=scene.id)
    for name in ("cpu", "legacy"):
        with pytest.raises(ScissorsError) as error:
            app.call("scene.preview", scene_id=scene.id, width=32, height=32, renderer=name)
        assert error.value.code == "INVALID_PARAMETER"


def test_depth_pass_writes_numeric_camera_distance(app):
    scene = app.scenes.create(name="Depth")
    app.meshes.create_plane(scene_id=scene.id, size=2.0)
    result = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/depth.png",
        width=64,
        height=64,
        projection="perspective",
        camera_position=[0, 0, 3],
        camera_target=[0, 0, 0],
        near=0.1,
        far=10,
        render_pass="depth",
    )
    assert result["renderer"] == "vtk"
    assert result["render_pass"] == "depth"
    assert result["depth_path"] and Path(result["depth_path"]).exists()
    assert "camera-forward" in result["depth_semantics"]
    with np.load(result["depth_path"], allow_pickle=False) as data:
        depth = data["depth"]
        coverage = data["coverage"]
        assert data["coverage"].dtype == bool
        np.testing.assert_allclose(depth[coverage], 3.0, atol=1e-4)
        assert np.isinf(depth[~coverage]).all()
        assert float(data["near"]) == pytest.approx(0.1)
        assert float(data["far"]) == pytest.approx(10.0)
    with Image.open(result["path"]) as image:
        assert np.asarray(image).dtype == np.uint16


def test_depth_pass_tracks_the_colour_silhouette(app):
    scene = app.scenes.create(name="Silhouette")
    plane = app.meshes.create_plane(scene_id=scene.id, size=2.0)
    app.meshes.translate(object_id=plane.id, y=1.0)
    app.meshes.create_sphere(scene_id=scene.id, radius=0.5, subdivisions=3)
    camera = {
        "width": 128,
        "height": 128,
        "projection": "perspective",
        "camera_position": [2.0, -2.5, 2.0],
        "camera_target": [0, 0, 0],
        "near": 0.1,
        "far": 20,
    }
    colour = app.call("scene.preview", scene_id=scene.id, output="previews/colour.png", **camera)
    depth = app.call(
        "scene.preview",
        scene_id=scene.id,
        output="previews/depth_compare.png",
        render_pass="depth",
        **camera,
    )
    with Image.open(colour["path"]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=int)
    colour_mask = np.abs(pixels - np.asarray(BACKGROUND)).sum(axis=2) > 12
    with np.load(depth["depth_path"], allow_pickle=False) as data:
        depth_mask = data["coverage"]
    intersection = int((colour_mask & depth_mask).sum())
    union = int((colour_mask | depth_mask).sum())
    assert intersection / union > 0.9


def test_depth_pass_keeps_a_transparent_background(app):
    scene = app.scenes.create(name="DepthAlpha")
    app.meshes.create_cube(scene_id=scene.id)
    result = app.call(
        "scene.preview",
        scene_id=scene.id,
        width=64,
        height=64,
        transparent=True,
        render_pass="depth",
    )
    with Image.open(result["path"]) as image:
        assert image.mode == "RGBA"
        pixels = np.asarray(image)
    assert pixels[0, 0, 3] == 0
    assert pixels[32, 32, 3] == 255


def test_normal_pass_encodes_world_normals(app):
    scene = app.scenes.create(name="Normals")
    app.meshes.create_plane(scene_id=scene.id)
    result = app.call(
        "scene.preview",
        scene_id=scene.id,
        width=64,
        height=64,
        projection="perspective",
        camera_position=[0, 0, 3],
        camera_target=[0, 0, 0],
        render_pass="normal",
        transparent=True,
    )
    assert result["renderer"] == "vtk"
    assert result["render_pass"] == "normal"
    assert result["normal_space"] == "world"
    with Image.open(result["path"]) as image:
        assert image.mode == "RGBA"
        pixels = np.asarray(image)
    assert pixels[0, 0, 3] == 0
    painted = pixels[pixels[..., 3] > 0]
    assert len(painted) > 200
    np.testing.assert_array_equal(np.unique(painted[:, :3], axis=0), [[128, 128, 255]])


def test_normal_pass_encodes_one_colour_per_face(app):
    scene = app.scenes.create(name="Faces")
    cube = app.meshes.create_cube(scene_id=scene.id, size=1.0)
    app.call("mesh.split_normals", object_id=cube.id, angle=30)
    result = app.call(
        "scene.preview",
        scene_id=scene.id,
        width=96,
        height=96,
        azimuth_degrees=35,
        elevation_degrees=65,
        render_pass="normal",
    )
    with Image.open(result["path"]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=int)
    shades = {
        tuple(int(value) for value in colour) for colour in np.unique(pixels.reshape(-1, 3), axis=0)
    }
    # +Z, +Y and -X faces of the cube seen from the default camera.
    assert {(128, 128, 255), (128, 255, 128), (0, 128, 128)} <= shades


def test_diagnostic_passes_require_the_vtk_backend(app, monkeypatch):
    scene = app.scenes.create(name="DiagnosticNoGL")
    app.meshes.create_cube(scene_id=scene.id)
    for render_pass in ("depth", "normal"):
        with pytest.raises(ScissorsError) as error:
            app.call(
                "scene.preview",
                scene_id=scene.id,
                width=32,
                height=32,
                renderer="pillow",
                render_pass=render_pass,
            )
        assert error.value.code == "BACKEND_UNAVAILABLE"
    monkeypatch.setattr(rendering_fast, "vtk_capability", lambda: (False, {"reason": "test"}))
    with pytest.raises(ScissorsError) as error:
        app.call("scene.preview", scene_id=scene.id, width=32, height=32, render_pass="normal")
    assert error.value.code == "BACKEND_UNAVAILABLE"


def test_unknown_render_pass_is_rejected(app):
    scene = app.scenes.create(name="Pass")
    app.meshes.create_cube(scene_id=scene.id)
    with pytest.raises(ScissorsError) as error:
        app.call("scene.preview", scene_id=scene.id, width=32, height=32, render_pass="ao")
    assert error.value.code == "INVALID_PARAMETER"


def test_preview_draws_a_centred_object_on_every_backend(app):
    scene = app.scenes.create(name="Centred")
    app.meshes.create_cube(scene_id=scene.id)
    for renderer in ("auto", "pillow"):
        result = app.call(
            "scene.preview", scene_id=scene.id, width=128, height=128, renderer=renderer
        )
        with Image.open(result["path"]) as image:
            mask = covered(image)
        rows, columns = np.nonzero(mask)
        assert mask.sum() > 500
        assert abs((columns.min() + columns.max()) / 2 - 64) < 6
        assert abs((rows.min() + rows.max()) / 2 - 64) < 6


def test_fast_preview_honours_world_axis_sections(app):
    scene = app.scenes.create(name="Section")
    top = app.materials.create(scene_id=scene.id, name="Top", base_color=[1.0, 0.0, 0.0, 1])
    bottom = app.materials.create(scene_id=scene.id, name="Bottom", base_color=[0.0, 0.0, 1.0, 1])
    upper = app.meshes.create_cube(scene_id=scene.id, size=0.6)
    app.meshes.translate(object_id=upper.id, z=1.0)
    app.materials.assign(object_id=upper.id, material_id=top.id)
    lower = app.meshes.create_cube(scene_id=scene.id, size=0.6)
    app.meshes.translate(object_id=lower.id, z=-1.0)
    app.materials.assign(object_id=lower.id, material_id=bottom.id)
    for renderer in ("auto", "pillow"):
        kept = app.call(
            "scene.preview",
            scene_id=scene.id,
            width=96,
            height=96,
            azimuth_degrees=0,
            elevation_degrees=0,
            clip_axis="z",
            clip_position=0.0,
            clip_keep="positive",
            renderer=renderer,
        )
        with Image.open(kept["path"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=int)
        red = (pixels[:, :, 0] > 90) & (pixels[:, :, 0] > pixels[:, :, 2] + 40)
        blue = (pixels[:, :, 2] > 90) & (pixels[:, :, 2] > pixels[:, :, 0] + 40)
        assert red.any() and not blue.any(), renderer
        assert any("Section view" in line for line in kept["diagnostics"])


def test_fast_preview_supports_a_transparent_background(app):
    scene = app.scenes.create(name="Alpha")
    app.meshes.create_cube(scene_id=scene.id)
    for renderer in ("auto", "pillow"):
        result = app.call(
            "scene.preview",
            scene_id=scene.id,
            width=64,
            height=64,
            transparent=True,
            renderer=renderer,
        )
        with Image.open(result["path"]) as image:
            assert image.mode == "RGBA"
            pixels = np.asarray(image)
        assert pixels[0, 0, 3] == 0
        assert pixels[32, 32, 3] == 255


def test_fast_preview_samples_base_colour_textures(app):
    scene = app.scenes.create(name="Texture")
    source = app.engine.path("textures_src/two_tone.png")
    source.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:32] = (220, 30, 30)
    pixels[32:] = (10, 20, 200)
    Image.fromarray(pixels).save(source)
    imported = app.textures.load(path="textures_src/two_tone.png")
    material = app.materials.create(scene_id=scene.id, name="Two tone")
    app.textures.assign(material_id=material.id, texture_id=imported.texture_id)
    plane = app.meshes.create_plane(scene_id=scene.id, size=1.0)
    app.uv.generate(object_id=plane.id, method="planar")
    app.materials.assign(object_id=plane.id, material_id=material.id)
    for renderer in ("auto", "pillow"):
        result = app.call(
            "scene.preview",
            scene_id=scene.id,
            width=64,
            height=64,
            elevation_degrees=90.0,
            sample_textures=True,
            renderer=renderer,
        )
        with Image.open(result["path"]) as image:
            rendered = np.asarray(image.convert("RGB"), dtype=int)
        mask = np.abs(rendered - np.asarray(BACKGROUND)).sum(axis=2) > 12
        rows = np.nonzero(mask)[0]
        middle = (int(rows.min()) + int(rows.max())) // 2
        row_index = np.arange(rendered.shape[0])[:, None]
        top = rendered[mask & (row_index <= middle)].reshape(-1, 3).mean(axis=0)
        bottom = rendered[mask & (row_index > middle)].reshape(-1, 3).mean(axis=0)
        assert top[2] > top[0] + 15, (renderer, top)
        assert bottom[0] > bottom[2] + 15, (renderer, bottom)


def test_emissive_material_glows_without_direct_light(app):
    scene = app.scenes.create(name="Emissive")
    plane = app.meshes.create_plane(scene_id=scene.id)
    material = app.call(
        "material.create", scene_id=scene.id, base_color=[0, 0, 0, 1], emissive=[1, 0, 0]
    )
    app.call("material.assign", object_id=plane.id, material_id=material["id"])
    result = app.call(
        "scene.preview",
        scene_id=scene.id,
        width=64,
        height=64,
        transparent=True,
        ambient=0,
        light_strength=0,
        azimuth_degrees=0,
        elevation_degrees=90,
    )
    with Image.open(result["path"]) as image:
        pixels = np.asarray(image)
    painted = pixels[..., 3] > 0
    assert painted.sum() > 200
    # Antialiased backends blend the silhouette, so only the interior is exact.
    interior = painted.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        interior &= np.roll(painted, (dy, dx), axis=(0, 1))
    assert interior.sum() > 0
    assert np.all(pixels[interior, :3] == [255, 0, 0])


def test_turntable_is_animated_and_leaves_geometry_intact(app):
    scene = app.scenes.create(name="Turntable")
    app.call("mesh.create_wedge", scene_id=scene.id, size=[1, 0.7, 0.8])
    before = app.engine.snapshot()
    result = app.call("scene.turntable", scene_id=scene.id, frames=4, size=64, fps=10)
    assert result["renderer"] in {"vtk", "pillow"}
    with Image.open(result["path"]) as image:
        assert image.format == "GIF" and image.n_frames > 1
        assert image.n_frames == result["frames"]
    assert app.engine.snapshot() == before


def test_preview_does_not_rewrite_workspace_state(app):
    scene = app.scenes.create(name="State")
    app.meshes.create_cube(scene_id=scene.id)
    app.call("material.create", scene_id=scene.id, base_color=[0.4, 0.5, 0.6, 1])
    state_path = app.engine.root / "state.json"
    before_state = app.engine.snapshot()
    before_mtime = state_path.stat().st_mtime_ns
    app.call("scene.preview", scene_id=scene.id, width=64, height=64)
    assert app.engine.snapshot() == before_state
    assert state_path.stat().st_mtime_ns == before_mtime


def test_camera_errors_are_structured(app):
    scene = app.scenes.create(name="Camera")
    app.meshes.create_cube(scene_id=scene.id)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.preview",
            scene_id=scene.id,
            projection="perspective",
            camera_position=[0, 0, 0],
            camera_target=[0, 0, 0],
        )


def test_explicit_vtk_without_opengl_raises_backend_unavailable(app, monkeypatch):
    monkeypatch.setattr(rendering_fast, "vtk_capability", lambda: (False, {"reason": "test"}))
    scene = app.scenes.create(name="NoGL")
    app.meshes.create_cube(scene_id=scene.id)
    with pytest.raises(ScissorsError) as error:
        app.call("scene.preview", scene_id=scene.id, width=64, height=64, renderer="vtk")
    assert error.value.code == "BACKEND_UNAVAILABLE"
    fallback = app.call("scene.preview", scene_id=scene.id, width=64, height=64, renderer="auto")
    assert fallback["renderer"] == "pillow"
