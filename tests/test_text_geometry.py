"""Typography and font-reader regression cases; unrun under the no-tests request."""

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.mesh import topology_mesh
from scissors.core.text import font_characters


def test_text_is_closed_geometry_with_holes_uv_and_material(app):
    scene = app.scenes.create()
    material = app.call("material.create", scene_id=scene.id)
    result = app.call(
        "mesh.create_text",
        scene_id=scene.id,
        text="BO8",
        size=0.5,
        depth=0.05,
        material_id=material["id"],
        segments=6,
    )
    obj = app.engine.object(result["object_id"])[1]
    assert result["watertight"] and result["volume"] > 0 and result["contour_count"] >= 6
    assert obj.material_id == material["id"] and len(obj.uv) == len(obj.vertices)
    np.testing.assert_allclose(np.ptp(obj.vertices, axis=0)[2], 0.05)
    assert topology_mesh(obj.geometry()).is_winding_consistent
    np.testing.assert_allclose(np.linalg.norm(obj.normals, axis=1), 1, atol=1e-6)


def test_text_is_deterministic_and_discovered_as_one_family(app):
    scene = app.scenes.create()
    first = app.call(
        "mesh.create_text", scene_id=scene.id, text="PRESS", font="Courier", segments=4
    )
    second = app.call(
        "mesh.create_text", scene_id=scene.id, text="PRESS", font="Courier", segments=4
    )
    a, b = app.engine.object(first["object_id"])[1], app.engine.object(second["object_id"])[1]
    assert a.vertices == b.vertices and a.faces == b.faces and a.uv == b.uv
    table = app.call("mesh.figures")["figures"]
    assert sum(row["kind"] == "text" for row in table) == 1


def test_text_axis_center_and_uncapped_walls(app):
    scene = app.scenes.create()
    result = app.call(
        "mesh.create_text",
        scene_id=scene.id,
        text="I",
        depth=0.2,
        axis="x",
        pivot="center",
        capped=False,
    )
    obj = app.engine.object(result["object_id"])[1]
    bounds = np.asarray(result["bounds"])
    np.testing.assert_allclose(bounds.mean(axis=0), [0, 0, 0], atol=1e-10)
    assert np.ptp(obj.vertices, axis=0)[0] == pytest.approx(0.2)
    assert not result["watertight"]


def test_text_can_join_figure_table_and_atomic_rollback(app):
    scene = app.scenes.create()
    result = app.call(
        "scene.create_table",
        scene_id=scene.id,
        rows=[{"figure": "text", "parameters": {"text": "ON", "size": 0.2, "depth": 0.02}}],
    )
    assert result["created_count"] == 1
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[
                {"operation": "mesh.create_text", "arguments": {"scene_id": scene.id, "text": "A"}},
                {
                    "operation": "mesh.create_text",
                    "arguments": {"scene_id": scene.id, "text": "   "},
                },
            ],
        )
    assert app.engine.snapshot() == before


def test_custom_font_unicode_inside_transaction(app):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        pytest.skip("No system Unicode test font is available")
    target = app.engine.root / "font.ttf"
    shutil.copyfile(source, target)
    scene = app.scenes.create()
    with app.transaction():
        result = app.call("mesh.create_text", scene_id=scene.id, text="CAFÉ", font="font.ttf")
    assert result["watertight"]


def test_font_cmap_validation_rejects_missing_glyphs_and_out_of_bounds(tmp_path):
    maxp = struct.pack(">IH", 0x00005000, 4)
    mapping = struct.pack(">HHHHHHH", 4, 32, 0, 4, 4, 1, 0)
    mapping += struct.pack(">HHHHHHHHH", 65, 65535, 0, 65, 65535, (1 - 65) & 65535, 1, 0, 0)
    cmap = struct.pack(">HHHHI", 0, 1, 3, 1, 12) + mapping
    header = struct.pack(">IHHHH", 0x00010000, 2, 0, 0, 0)
    records = struct.pack(">4sIII", b"maxp", 0, 44, len(maxp))
    records += struct.pack(">4sIII", b"cmap", 0, 44 + len(maxp), len(cmap))
    path = tmp_path / "cmap.ttf"
    path.write_bytes(header + records + maxp + cmap)
    font_characters(path, {65}, 4096)
    with pytest.raises(ValueError):
        font_characters(path, {66}, 4096)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError):
        font_characters(path, {65}, 4096)


def test_unsupported_font_and_builtin_unicode_fail_without_mutation(app):
    scene = app.scenes.create()
    before = app.engine.snapshot()
    for arguments in [{"text": "Á"}, {"text": "A", "font": "Unknown"}, {"text": "\x00"}]:
        with pytest.raises(ScissorsError):
            app.call("mesh.create_text", scene_id=scene.id, **arguments)
    assert app.engine.snapshot() == before
