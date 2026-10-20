"""Import and reference-plane cases. Unrun under the user's no-tests instruction."""

from pathlib import Path

import numpy as np
import pytest

from scissors.api.errors import ScissorsError
from scissors.core.importing import WorkspaceResolver


@pytest.mark.parametrize(
    "extension,content",
    [
        ("obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
        (
            "stl",
            "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\n"
            "vertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n",
        ),
        (
            "ply",
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
            "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
            "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
        ),
    ],
)
def test_triangle_import_and_numeric_placement(app, extension, content):
    source = app.engine.root / f"triangle.{extension}"
    source.write_text(content, encoding="utf-8")
    result = app.call("scene.import_mesh", path=str(source), scale=2, position=[3, 0, 0])
    assert len(result["object_ids"]) == 1 and result["source_up"] == "z"
    obj = app.engine.object(result["object_ids"][0])[1]
    np.testing.assert_allclose(np.min(obj.vertices, axis=0), [3, 0, 0])
    np.testing.assert_allclose(np.max(obj.vertices, axis=0), [5, 2, 0])
    assert obj.parent_id in result["node_ids"].values()


def test_glb_roundtrip_restores_z_up_uv_pbr_and_texture(app):
    scene = app.scenes.create()
    plane = app.meshes.create_plane(scene_id=scene.id)
    app.uv.planar(object_id=plane.id)
    app.meshes.translate(object_id=plane.id, z=2)
    texture = app.call("texture.create", width=4, height=4)
    material = app.call(
        "material.from_texture", scene_id=scene.id, texture_id=texture["texture_id"]
    )
    app.call("material.assign", object_id=plane.id, material_id=material["id"])
    exported = app.call("scene.export", scene_id=scene.id, output="assets/source.glb")
    result = app.call("scene.import_mesh", path=exported["path"])
    obj = app.engine.object(result["object_ids"][0])[1]
    assert result["source_up"] == "y" and obj.uv is not None
    assert result["texture_ids"] and obj.material_id in result["material_ids"]
    np.testing.assert_allclose(
        np.sort(np.asarray(obj.vertices), axis=0),
        np.sort(app.engine.object(plane.id)[1].vertices, axis=0),
        atol=1e-6,
    )


def test_reference_plane_axes_uv_and_scale(app):
    scene = app.scenes.create()
    texture = app.call("texture.create", width=8, height=4)
    result = app.call(
        "reference.plane",
        scene_id=scene.id,
        path=texture["path"],
        axis="y",
        size=[4, 2],
        position=[1, 3, 2],
    )
    obj = app.engine.object(result["object_id"])[1]
    np.testing.assert_allclose(obj.summary()["bounds"], [[-1, 3, 1], [3, 3, 3]])
    np.testing.assert_allclose(obj.geometry().face_normals, [[0, -1, 0], [0, -1, 0]])
    assert obj.uv == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert app.engine.material(obj.material_id)[1].double_sided


def test_resolver_rejects_remote_reserved_and_outside_paths(app):
    resolver = WorkspaceResolver(app.engine.root, app.engine.root, 1024)
    for key in ["https://example.com/a.png", "../outside.png", "state.json", ".runtime/a.png"]:
        with pytest.raises((ValueError, ScissorsError)):
            resolver.get(key)
    assert resolver.errors


def test_import_error_restores_state_and_rejects_atomic_batch(app):
    source = app.engine.root / "broken.glb"
    source.write_bytes(b"invalid glb")
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        app.call("scene.import_mesh", path=str(source))
    assert app.engine.snapshot() == before
    with pytest.raises(ScissorsError):
        app.call(
            "scene.edit",
            operations=[{"operation": "scene.import_mesh", "arguments": {"path": str(source)}}],
        )
    assert Path(source).read_bytes() == b"invalid glb"


def test_ply_vertex_colors_survive_affine_changes(app):
    source = app.engine.root / "colored.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n3 0 1 2\n",
        encoding="utf-8",
    )
    result = app.call("scene.import_mesh", path=str(source))
    oid = result["object_ids"][0]
    before = app.engine.object(oid)[1].vertex_colors
    assert before is not None and len(before) == 3
    assert {tuple(color) for color in before} == {
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
    }
    app.meshes.translate(object_id=oid, x=2)
    assert app.engine.object(oid)[1].vertex_colors == before
