"""Rig retention and sampled animation round trips through the shared registry."""

from pathlib import Path

import numpy as np
import pytest
import trimesh
import ufbx

from scissors import Scissors
from scissors.api.errors import ScissorsError
from scissors.core.exchange import ExchangeContext
from scissors.core.fbx import binary_file, node, prop
from scissors.core.fbx_document import child, read_binary
from tests.test_fbx_import import fixture_fbx

pytestmark = pytest.mark.integration
KT = 46186158000


def rig_fixture(app):
    path = fixture_fbx(app, embedded=True)
    nodes = read_binary(path.read_bytes(), app.engine.settings.max_file_bytes)
    root = node("root", children=nodes)
    objects, connections = child(root, "Objects"), child(root, "Connections")
    for identifier, name in [(200, "Root"), (201, "Tip")]:
        objects.children.append(
            node(
                "Model",
                identifier,
                f"{name}\x00\x01Model",
                "LimbNode",
                children=[
                    node("Version", np.int32(232)),
                    node(
                        "Properties70",
                        children=[prop("Lcl Translation", "Lcl Translation", 0.0, 0.0, 0.0)],
                    ),
                ],
            )
        )
        objects.children.append(
            node(
                "NodeAttribute",
                identifier + 10,
                f"{name}\x00\x01NodeAttribute",
                "LimbNode",
                children=[node("TypeFlags", "Skeleton")],
            )
        )
        connections.children.extend(
            [
                node("C", "OO", identifier, 200 if identifier == 201 else 0),
                node("C", "OO", identifier + 10, identifier),
            ]
        )
    objects.children.append(
        node(
            "Deformer",
            300,
            "Skin\x00\x01Deformer",
            "Skin",
            children=[node("Version", np.int32(101)), node("SkinningType", "Linear")],
        )
    )
    connections.children.append(node("C", "OO", 300, 1))
    for index, vertices in enumerate([[0, 3], [1, 2]]):
        identifier = 301 + index
        objects.children.append(
            node(
                "Deformer",
                identifier,
                f"Cluster{index}\x00\x01SubDeformer",
                "Cluster",
                children=[
                    node("Version", np.int32(100)),
                    node("Indexes", np.array(vertices, dtype=np.int32)),
                    node("Weights", np.ones(2)),
                    node("Transform", np.eye(4).ravel()),
                    node("TransformLink", np.eye(4).ravel()),
                ],
            )
        )
        connections.children.extend(
            [node("C", "OO", identifier, 300), node("C", "OO", 200 + index, identifier)]
        )
    objects.children.extend(
        [
            node(
                "AnimationStack",
                400,
                "Move\x00\x01AnimStack",
                "",
                children=[
                    node(
                        "Properties70",
                        children=[
                            prop("LocalStart", "KTime", 10 * KT),
                            prop("LocalStop", "KTime", 11 * KT),
                            prop("ReferenceStart", "KTime", 10 * KT),
                            prop("ReferenceStop", "KTime", 11 * KT),
                        ],
                    )
                ],
            ),
            node("AnimationLayer", 401, "Base\x00\x01AnimLayer", ""),
            node(
                "AnimationCurveNode",
                402,
                "T\x00\x01AnimCurveNode",
                "",
                children=[
                    node(
                        "Properties70",
                        children=[
                            prop("d|X", "Number", 0.0),
                            prop("d|Y", "Number", 0.0),
                            prop("d|Z", "Number", 0.0),
                        ],
                    )
                ],
            ),
            node(
                "AnimationCurve",
                403,
                "\x00\x01AnimCurve",
                "",
                children=[
                    node("Default", 0.0),
                    node("KeyVer", np.int32(4008)),
                    node("KeyTime", np.array([10 * KT, 11 * KT], dtype=np.int64)),
                    node("KeyValueFloat", np.array([0.0, 0.5])),
                    node("KeyAttrFlags", np.array([4, 4], dtype=np.int32)),
                    node("KeyAttrDataFloat", np.zeros(8)),
                    node("KeyAttrRefCount", np.ones(2, dtype=np.int32)),
                ],
            ),
        ]
    )
    connections.children.extend(
        [
            node("C", "OO", 401, 400),
            node("C", "OO", 402, 401),
            node("C", "OP", 402, 201, "Lcl Translation"),
            node("C", "OP", 403, 402, "d|Y"),
        ]
    )
    path.write_bytes(
        binary_file(nodes, ExchangeContext(trimesh.Scene(), app.engine.settings, "fixture"))
    )
    return path


def sampled_skin(path: Path, times=(10.0, 10.5, 11.0)):
    scene = ufbx.load_file(str(path), evaluate_skinning=True)
    refs = [scene, list(scene.nodes), list(scene.meshes), list(scene.anim_stacks)]
    stack = refs[-1][0]
    anim = stack.anim
    refs.extend([stack, anim])
    result = []
    for time in times:
        evaluated = scene.evaluate(anim, time, evaluate_skinning=True)
        meshes = list(evaluated.meshes)
        mesh = meshes[0]
        attribute = mesh.skinned_position
        values = attribute.values
        refs.extend([evaluated, meshes, mesh, attribute, values])
        result.append(np.asarray(values, dtype=float).copy())
    return result


def test_skin_and_animation_roundtrip_after_restart(app):
    source = rig_fixture(app)
    before = sampled_skin(source)
    assert np.max(np.abs(before[0] - before[-1])) == pytest.approx(0.5)
    assert np.max(np.abs(before[0] - before[1])) == pytest.approx(0.25)
    imported = app.call("scene.import_mesh", path=str(source))
    assert imported["rig"]["bone_count"] == 2
    assert imported["rig"]["animation_clips"] == [
        {"name": "Move", "start_seconds": 10.0, "end_seconds": 11.0}
    ]
    workspace = app.engine.root
    app.close()
    with Scissors(workspace) as reopened:
        result = reopened.call(
            "scene.export", scene_id=imported["scene_id"], output="exports/rigged.fbx", format="fbx"
        )
        after = sampled_skin(Path(result["path"]))
        for original, retained in zip(before, after, strict=True):
            np.testing.assert_allclose(original, retained, atol=1e-12)
        assert len(result["files"]) == 1
        assert (
            reopened.call("scene.inspect", scene_id=imported["scene_id"])["rigs"][0]["bone_count"]
            == 2
        )


def test_surviving_vertices_keep_exact_weights_and_animation(app):
    source = rig_fixture(app)
    original = sampled_skin(source)
    imported = app.call("scene.import_mesh", path=str(source))
    oid = imported["object_ids"][0]
    selection = {
        "object_id": oid,
        "geometry_revision": app.call("mesh.inspect", object_id=oid)["geometry_revision"],
        "face_indices": [1],
    }
    app.call("mesh.delete_faces", object_id=oid, selection=selection)
    result = app.call(
        "scene.export", scene_id=imported["scene_id"], output="exports/reduced.fbx", format="fbx"
    )
    after = sampled_skin(Path(result["path"]))
    # The retained first triangle uses control points 0,1,2, including two UV seams.
    for expected, actual in zip(original, after, strict=True):
        np.testing.assert_allclose(expected[[0, 1, 2]], actual, atol=1e-12)


def test_unsupported_exports_never_silently_drop_rig(app):
    imported = app.call("scene.import_mesh", path=str(rig_fixture(app)))
    for options in [
        dict(format="glb", output="exports/lost.glb"),
        dict(format="fbx", output="exports/lost.fbx", split_materials=True),
        dict(format="fbx", output="exports/lost.fbx", export_transform=np.eye(4).tolist()),
    ]:
        with pytest.raises(ScissorsError, match="rig|Rig"):
            app.call("scene.export", scene_id=imported["scene_id"], **options)
    assert not (app.engine.root / "exports/lost.fbx").exists()
    app.call("mesh.translate", object_id=imported["object_ids"][0], z=1)
    with pytest.raises(ScissorsError):
        app.call(
            "scene.export",
            scene_id=imported["scene_id"],
            format="fbx",
            output="exports/changed.fbx",
        )


def test_retained_asset_is_immutable_and_verified(app):
    imported = app.call("scene.import_mesh", path=str(rig_fixture(app)))
    document = app.engine.scene(imported["scene_id"]).fbx_documents[0]
    with pytest.raises(ScissorsError):
        app.call("scene.export", scene_id=imported["scene_id"], format="fbx", output=document.path)
    (app.engine.root / document.path).write_bytes(b"corrupt")
    with pytest.raises(ScissorsError, match="integrity"):
        app.call(
            "scene.export",
            scene_id=imported["scene_id"],
            format="fbx",
            output="exports/corrupt.fbx",
        )


def test_explicit_static_import_and_atomic_rejection(app):
    source = rig_fixture(app)
    static = app.call("scene.import_mesh", path=str(source), fbx_mode="static")
    assert static["rig"] is None
    before = app.call("scene.list")
    with pytest.raises(ScissorsError):
        app.call("scene.import_mesh", path=str(source), scene_id=static["scene_id"])
    assert app.call("scene.list") == before
    with pytest.raises(ScissorsError):
        app.call("scene.import_mesh", path=str(source), scale=2)
    assert app.call("scene.list") == before


def test_import_rollback_removes_new_document(app, monkeypatch):
    source = rig_fixture(app)
    before = app.call("scene.list")

    def fail_commit():
        raise OSError("Injected persistence failure")

    monkeypatch.setattr(app.engine, "commit", fail_commit)
    with pytest.raises(ScissorsError):
        app.call("scene.import_mesh", path=str(source))
    assert app.call("scene.list") == before
    assert not list((app.engine.root / "assets/fbx_documents").glob("*.fbx"))


def test_binary_parser_rejects_inflated_arrays_and_invalid_offsets(app):
    from scissors.core.fbx_document import document_bytes

    source = rig_fixture(app)
    raw = source.read_bytes()
    with pytest.raises(ValueError, match="budget"):
        read_binary(raw, len(raw), memory_limit=1)
    damaged = bytearray(raw)
    damaged[27:31] = (len(raw) + 1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="bounds"):
        read_binary(bytes(damaged), len(damaged))
    nodes = read_binary(raw, len(raw))
    written = document_bytes(nodes, app.engine.settings)
    curve = next(
        o
        for o in child(
            node("root", children=read_binary(written, len(written))), "Objects"
        ).children
        if o.name == "AnimationCurve"
    )
    np.testing.assert_array_equal(child(curve, "KeyTime").properties[0], [10 * KT, 11 * KT])


def test_nonplanar_quad_preserves_original_corners():
    from scissors.core.fbx_import import triangle_corners

    points = np.array(
        [
            [0.60446191, -0.02369125, 1.27795863],
            [0.60737878, -0.02149612, 1.27517354],
            [0.60490632, -0.03721823, 1.23681092],
            [0.60342616, -0.02756096, 1.22913265],
        ]
    )
    corners, owners = triangle_corners([[0, 4]], points)
    assert corners.shape == (2, 3)
    assert set(corners.ravel()) == {0, 1, 2, 3}
    assert owners.tolist() == [0, 0]
    triangles = points[corners]
    assert np.all(
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        > 1e-10
    )


def test_rig_copy_is_rejected_and_clear_removes_retention(app):
    imported = app.call("scene.import_mesh", path=str(rig_fixture(app)))
    before = app.call("scene.list")
    with pytest.raises(ScissorsError):
        app.call("scene.clone", scene_id=imported["scene_id"])
    assert app.call("scene.list") == before
    result = app.call("scene.clear", scene_id=imported["scene_id"])
    assert result["rigs"] == []


def test_float_tangent_payload_and_64bit_time_arrays_keep_their_types():
    from scissors.config import Settings
    from scissors.core.fbx_document import document_bytes

    bits = np.array([0x3F800000, 0x7FC12345, 0x80000000], dtype=np.uint32)
    nodes = [
        node("KeyAttrDataFloat", bits.view(np.float32)),
        node("KeyTime", np.array([10 * KT, 11 * KT], dtype=np.int64)),
    ]
    result = read_binary(document_bytes(nodes, Settings()), 10000)
    assert result[0].properties[0].dtype == np.dtype("<f4")
    np.testing.assert_array_equal(result[0].properties[0].view(np.uint32), bits)
    assert result[1].properties[0].dtype == np.dtype("<i8")
    np.testing.assert_array_equal(result[1].properties[0], [10 * KT, 11 * KT])
