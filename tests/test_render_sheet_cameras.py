"""Explicit review-camera, stereo, image, and publication contracts."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core import render_sheets


def setup(app):
    sid = app.scenes.create().id
    app.meshes.create_cube(scene_id=sid)
    return sid


def files(app):
    root = app.engine.root / "previews"
    return {str(path.relative_to(root)): path.read_bytes() for path in root.glob("*.png")}


def test_legacy_sheet_and_explicit_parallel_stereo_match_preview(app):
    sid = setup(app)
    old = app.call("scene.render_sheet", scene_id=sid, size=64, renderer="pillow")
    with Image.open(old["path"]) as image:
        assert image.size == (192, 176)
    before = app.engine.snapshot()
    result = app.call(
        "scene.render_sheet",
        scene_id=sid,
        size=64,
        renderer="pillow",
        view_sets=[
            {
                "name": "Player",
                "position": [2, -3, 1],
                "target": [0, 0, 0.5],
                "stereo": True,
                "eye_separation": 0.064,
            }
        ],
    )
    assert result["view_count"] == 3 and before == app.engine.snapshot()
    by_eye = {view["eye"]: view for view in result["views"]}
    left, right = by_eye["left"], by_eye["right"]
    np.testing.assert_allclose(
        np.linalg.norm(np.array(left["camera_position"]) - right["camera_position"]), 0.064
    )
    np.testing.assert_allclose(
        np.array(left["camera_target"]) - left["camera_position"],
        np.array(right["camera_target"]) - right["camera_position"],
    )
    for index, view in enumerate(result["views"]):
        direct = app.call(
            "scene.preview",
            scene_id=sid,
            output=f"previews/match_{index}.png",
            **view["preview_parameters"],
        )
        assert Path(direct["path"]).read_bytes() == Path(view["path"]).read_bytes()


def test_orthographic_scale_and_mixed_output_sizes(app):
    sid = setup(app)
    result = app.call(
        "scene.render_sheet",
        scene_id=sid,
        renderer="pillow",
        views=[
            {
                "type": "camera",
                "position": [0, -3, 0.5],
                "target": [0, 0, 0.5],
                "projection": "orthographic",
                "orthographic_scale": 2,
                "width": 128,
                "height": 96,
            },
            {"name": "Orbit", "width": 64, "height": 64},
        ],
    )
    assert result["views"][0]["orthographic_scale"] == pytest.approx(2)
    assert result["views"][0]["effective_fov"] is None
    with Image.open(result["views"][0]["path"]) as image:
        assert image.size == (128, 96)
    with Image.open(result["path"]) as image:
        assert image.size == (256, 120)


def test_invalid_second_camera_and_pixel_limit_do_not_publish(app):
    sid = setup(app)
    app.call("scene.render_sheet", scene_id=sid, size=64, renderer="pillow")
    before = files(app)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.render_sheet",
            scene_id=sid,
            renderer="pillow",
            views=[
                {"name": "Good"},
                {
                    "type": "camera",
                    "position": [0, -3, 0.5],
                    "target": [0, 0, 0.5],
                    "up": [0, 1, 0],
                },
            ],
        )
    assert files(app) == before
    budget = app.engine.settings.max_texture_pixels
    app.engine.settings.max_texture_pixels = 20000
    try:
        with pytest.raises(ScissorsError) as error:
            app.call("scene.render_sheet", scene_id=sid, size=64, renderer="pillow")
        assert error.value.code == "RESOURCE_LIMIT"
    finally:
        app.engine.settings.max_texture_pixels = budget
    assert files(app) == before
    assert not list(app.engine.root.glob(".render-sheet-*"))


def test_conflicting_camera_fields_are_rejected_by_the_typed_contract(app):
    sid = setup(app)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.render_sheet",
            scene_id=sid,
            views=[
                {
                    "position": [0, -3, 0.5],
                    "target": [0, 0, 0.5],
                    "azimuth_degrees": 30,
                }
            ],
        )


def test_publication_failure_restores_existing_tiles(app, monkeypatch):
    sid = setup(app)
    args = dict(scene_id=sid, size=64, renderer="pillow")
    app.call("scene.render_sheet", **args)
    before = files(app)
    original = render_sheets.atomic_write

    def fail(path, data, limit):
        if path == app.engine.root / "previews/sheet_view_0.png":
            assert data != before["sheet_view_0.png"]
        if path == app.engine.root / "previews/sheet_view_1.png":
            raise OSError("Injected write failure")
        return original(path, data, limit)

    monkeypatch.setattr(render_sheets, "atomic_write", fail)
    with pytest.raises(ScissorsError) as error:
        app.call("scene.render_sheet", **args, background=[0.1, 0.1, 0.1])
    assert error.value.code == "FILESYSTEM_ERROR"
    assert before == files(app)
    assert not list(app.engine.root.glob(".render-sheet-*"))
