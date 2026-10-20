"""Projected UV and five-channel texture transfer with real rollback and reopened artifacts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from scissors import Scissors
from scissors.api.errors import ScissorsError


def setup_surfaces(app):
    scene = app.scenes.create()
    ids = []
    materials = []
    for name in ("Source", "Target"):
        obj = app.meshes.create_plane(scene_id=scene.id, name=name, size=2)
        app.call("uv.planar", object_id=obj.id)
        material = app.call(
            "material.create",
            scene_id=scene.id,
            name=name,
            base_color=[1, 1, 1, 1],
            metallic=1,
            roughness=1,
            emissive=[1, 1, 1],
        )
        app.call("material.assign", object_id=obj.id, material_id=material["id"])
        ids.append(obj.id)
        materials.append(material["id"])
    source, target = ids
    app.call("mesh.translate", object_id=target, z=0.01)
    return scene.id, source, target, materials


def image_slot(app, material, slot, color):
    image = app.call("texture.create", width=4, height=4, color_a=color, color_b=color)
    app.call("texture.assign", material_id=material, texture_id=image["texture_id"], slot=slot)
    return image


def pbr_source(app, material):
    app.call(
        "material.update",
        material_id=material,
        properties={
            "base_color": [0.25, 1, 1, 0.5],
            "metallic": 0.6,
            "roughness": 0.4,
            "emissive": [0.5, 0.25, 0],
            "alpha": 0.8,
        },
    )
    for slot, color in [
        ("base_color", [255, 0, 0, 255]),
        ("normal", [218, 128, 218, 255]),
        ("metallic_roughness", [255, 128, 255, 255]),
        ("emissive", [255, 255, 255, 255]),
        ("occlusion", [64, 64, 64, 255]),
    ]:
        image_slot(app, material, slot, color)


def test_uv_projection_transfers_material_maps_and_preserves_world_geometry(app):
    _, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    before_source = app.engine.object(source)[1].model_dump()
    before_target = app.engine.object(target)[1].model_dump()
    result = app.call(
        "mesh.transfer_uv", object_id=target, source_object_id=source, max_distance=0.02
    )
    obj = app.engine.object(target)[1]
    np.testing.assert_allclose(
        np.asarray(obj.vertices)[obj.faces],
        np.asarray(before_target["vertices"])[before_target["faces"]],
    )
    np.testing.assert_allclose(
        np.asarray(obj.uv)[obj.faces], (np.asarray(obj.vertices)[obj.faces][:, :, :2] + 1) / 2
    )
    assert obj.material_id == materials[0]
    assert result["max_measured_distance"] == pytest.approx(0.01)
    assert obj.transform == before_target["transform"]
    assert app.engine.object(source)[1].model_dump() == before_source
    assert len(result["selection"]["face_indices"]) == 2


def test_partial_uv_transfer_preserves_unselected_uv_and_rejects_material_change(app):
    _, source, target, _ = setup_surfaces(app)
    app.call("uv.planar", object_id=target, u1=0.5)
    selected = app.call("mesh.select_faces", object_id=target)
    selected["face_indices"] = [0]
    before = app.engine.object(target)[1].model_dump()
    with pytest.raises(ScissorsError, match="different materials"):
        app.call(
            "mesh.transfer_uv",
            object_id=target,
            source_object_id=source,
            max_distance=0.1,
            selection=selected,
        )
    result = app.call(
        "mesh.transfer_uv",
        object_id=target,
        source_object_id=source,
        max_distance=0.1,
        selection=selected,
        material_policy="target",
    )
    obj = app.engine.object(target)[1]
    np.testing.assert_array_equal(
        np.asarray(obj.uv)[obj.faces[1]], np.asarray(before["uv"])[before["faces"][1]]
    )
    assert obj.material_id == before["material_id"] and result["transferred_face_count"] == 1


def test_five_channel_bake_factors_normal_basis_and_glb_reopening(app):
    scene, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    app.call("uv.planar", object_id=target, flip_u=True)
    before_source = app.engine.object(source)[1].model_dump()
    before_material = app.engine.scene(scene).materials[materials[0]].model_dump()
    result = app.call(
        "texture.transfer",
        object_id=target,
        source_object_id=source,
        max_distance=0.02,
        resolution=16,
        padding=0,
    )
    assert set(result["textures"]) == {
        "base_color",
        "normal",
        "metallic_roughness",
        "emissive",
        "occlusion",
    }
    expected = {
        "base_color": [137, 0, 0, 102],
        "metallic_roughness": [255, 51, 153, 255],
        "emissive": [188, 137, 0, 255],
        "occlusion": [64, 64, 64, 255],
        "normal": [37, 128, 218, 255],
    }
    for channel, texture in result["textures"].items():
        with Image.open(texture["path"]) as image:
            np.testing.assert_allclose(image.getpixel((8, 8)), expected[channel], atol=1)
    mat = app.engine.scene(scene).materials[result["material_id"]]
    assert mat.base_color == [1, 1, 1, 1] and mat.alpha == 1 and mat.metallic == mat.roughness == 1
    assert mat.emissive == [1, 1, 1]
    assert app.engine.object(source)[1].model_dump() == before_source
    assert app.engine.scene(scene).materials[materials[0]].model_dump() == before_material
    exported = app.call(
        "scene.export",
        scene_id=scene,
        output="exports/transferred.glb",
        format="glb",
        only_object_ids=[target],
    )
    loaded = trimesh.load_scene(exported["path"])
    visual = next(iter(loaded.geometry.values())).visual
    assert (
        visual.material.normalTexture is not None
        and visual.material.metallicRoughnessTexture is not None
    )
    assert (
        visual.material.emissiveTexture is not None and visual.material.occlusionTexture is not None
    )


def test_partial_bake_keeps_other_texels_and_shared_material_unchanged(app):
    scene, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    old = image_slot(app, materials[1], "base_color", [0, 0, 255, 255])
    old_bytes = Path(old["path"]).read_bytes()
    selected = app.call("mesh.select_faces", object_id=target)
    selected["face_indices"] = [0]
    before = app.engine.object(target)[1].model_dump()
    result = app.call(
        "texture.transfer",
        object_id=target,
        source_object_id=source,
        max_distance=0.1,
        selection=selected,
        channels=["base_color"],
        resolution=16,
    )
    assert result["material_id"] != materials[1]
    coords = np.asarray(before["uv"])[before["faces"][1]].mean(axis=0)
    x, y = np.floor(coords * [16, -16] + [0, 16]).astype(int)
    with Image.open(result["textures"]["base_color"]["path"]) as image:
        assert image.getpixel((int(x), int(y))) == (0, 0, 255, 255)
    assert app.engine.scene(scene).materials[materials[1]].base_color_texture == old["texture_id"]
    assert Path(old["path"]).read_bytes() == old_bytes


@pytest.mark.parametrize("operation", ["mesh.transfer_uv", "texture.transfer"])
def test_distance_and_stale_selection_failures_restore_state(app, operation):
    _, source, target, _ = setup_surfaces(app)
    selected = app.call("mesh.select_faces", object_id=target)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="unmatched"):
        app.call(operation, object_id=target, source_object_id=source, max_distance=0.001)
    with pytest.raises(ScissorsError, match="revision"):
        app.call(
            operation,
            object_id=target,
            source_object_id=source,
            max_distance=0.1,
            selection=dict(selected, geometry_revision="old"),
        )
    assert app.engine.snapshot() == before


def test_bake_commit_failure_removes_owned_maps(app, monkeypatch):
    _, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    before = app.engine.snapshot()
    files = {path: path.read_bytes() for path in (app.engine.root / "textures").glob("*.png")}

    def fail_commit():
        raise OSError("Checkpoint unavailable")

    monkeypatch.setattr(app.engine, "commit", fail_commit)
    with pytest.raises(ScissorsError):
        app.call(
            "texture.transfer",
            object_id=target,
            source_object_id=source,
            max_distance=0.1,
            resolution=16,
        )
    assert app.engine.snapshot() == before
    assert {
        path: path.read_bytes() for path in (app.engine.root / "textures").glob("*.png")
    } == files
    assert app.registry._rollback_stack == []


def test_bake_persistence_and_transaction_permissions(app):
    _, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError):
        with app.transaction():
            app.call(
                "texture.transfer",
                object_id=target,
                source_object_id=source,
                max_distance=0.1,
                resolution=16,
            )
    assert app.engine.snapshot() == before
    app.call(
        "texture.transfer",
        object_id=target,
        source_object_id=source,
        max_distance=0.1,
        resolution=16,
    )
    saved = app.engine.snapshot()
    root = app.engine.root
    app.close()
    with Scissors(root) as reopened:
        assert reopened.engine.snapshot() == saved


def test_uv_transfer_supports_different_triangle_topology(app):
    _, source, target, _ = setup_surfaces(app)
    app.call("mesh.subdivide", object_id=target, iterations=1)
    assert len(app.engine.object(target)[1].faces) != len(app.engine.object(source)[1].faces)
    result = app.call(
        "mesh.transfer_uv", object_id=target, source_object_id=source, max_distance=0.02
    )
    obj = app.engine.object(target)[1]
    np.testing.assert_allclose(
        np.asarray(obj.uv)[obj.faces], (np.asarray(obj.vertices)[obj.faces][:, :, :2] + 1) / 2
    )
    assert result["transferred_face_count"] == 8


@pytest.mark.parametrize("operation", ["mesh.transfer_uv", "texture.transfer"])
@pytest.mark.parametrize("field,value", [("max_geometry_pairs", 1), ("max_memory_bytes", 1)])
def test_transfer_resource_limits_restore_state(app, operation, field, value):
    _, source, target, _ = setup_surfaces(app)
    before = app.engine.snapshot()
    setattr(app.engine.settings, field, value)
    with pytest.raises(ScissorsError) as error:
        app.call(operation, object_id=target, source_object_id=source, max_distance=0.1)
    assert error.value.code == "RESOURCE_LIMIT" and app.engine.snapshot() == before


def test_bake_rejects_overlapping_uv_and_uv_transfer_remains_atomic(app):
    scene, source, target, _ = setup_surfaces(app)
    obj = app.engine.object(target)[1]
    copy = app.call(
        "mesh.create",
        scene_id=scene,
        vertices=[*obj.vertices, *np.add(obj.vertices, [0, 0, 0.01]).tolist()],
        faces=[*obj.faces, *np.add(obj.faces, len(obj.vertices)).tolist()],
    )
    app.call("uv.planar", object_id=copy["id"])
    app.call("material.assign", object_id=copy["id"], material_id=obj.material_id)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="overlap"):
        app.call(
            "texture.transfer",
            object_id=copy["id"],
            source_object_id=source,
            max_distance=0.1,
            resolution=16,
        )
    assert app.engine.snapshot() == before
    selected = app.call("mesh.select_faces", object_id=target)
    with pytest.raises(ScissorsError):
        app.call(
            "operations.batch",
            operations=[
                {
                    "operation": "mesh.transfer_uv",
                    "arguments": {
                        "object_id": target,
                        "source_object_id": source,
                        "max_distance": 0.1,
                    },
                },
                {
                    "operation": "mesh.flip",
                    "arguments": {"object_id": target, "selection": selected},
                },
            ],
        )
    assert app.engine.snapshot() == before


def test_nested_registry_failure_cleans_bake_files_and_preserves_checkpoint(app):
    _, source, target, _ = setup_surfaces(app)
    before = app.engine.snapshot()
    checkpoint = (app.engine.root / "state.json").read_bytes()

    def outer() -> dict:
        result = app.registry.execute(
            "texture.transfer",
            {
                "object_id": target,
                "source_object_id": source,
                "max_distance": 0.1,
                "resolution": 16,
            },
        )
        assert result["success"]
        raise ScissorsError("GEOMETRY_ERROR", "Reject enclosing operation")

    app.registry.add_function(
        "test.transfer_then_fail",
        outer,
        "Test nested transfer rollback",
        permissions=["scene.write", "filesystem.write"],
    )
    with pytest.raises(ScissorsError, match="enclosing operation"):
        app.call("test.transfer_then_fail")
    assert app.engine.snapshot() == before
    assert (app.engine.root / "state.json").read_bytes() == checkpoint
    assert not list((app.engine.root / "textures").glob("*.png"))
    assert app.registry._rollback_stack == []


def test_filled_hole_receives_pbr_without_changing_the_surrounding_atlas(app):
    scene, source, _, materials = setup_surfaces(app)
    app.call("mesh.scale", object_id=source, x=2, y=2)
    pbr_source(app, materials[0])
    n = 5
    coords = np.linspace(-2, 2, n)
    vertices = [[x, y, 0] for y in coords for x in coords]
    faces = []
    for j in range(4):
        for i in range(4):
            if i == j == 1:
                continue
            a = j * n + i
            faces.extend([[a, a + 1, a + n + 1], [a, a + n + 1, a + n]])
    target = app.call("mesh.create", scene_id=scene, vertices=vertices, faces=faces)["id"]
    app.call("uv.planar", object_id=target)
    app.call("material.assign", object_id=target, material_id=materials[1])
    image_slot(app, materials[1], "base_color", [0, 0, 255, 255])
    opening = app.call(
        "mesh.select_boundaries",
        object_id=target,
        region={"bounds": [[-1.1, -1.1, -0.1], [0.1, 0.1, 0.1]], "rule": "fully_inside"},
    )
    patch = app.call("mesh.fill_holes", object_id=target, selection=opening)
    before = app.engine.object(target)[1].model_dump()
    baked = app.call(
        "texture.transfer",
        object_id=target,
        source_object_id=source,
        max_distance=0,
        selection=patch["selection"],
        resolution=16,
    )
    obj = app.engine.object(target)[1]
    np.testing.assert_array_equal(
        np.asarray(obj.vertices)[obj.faces], np.asarray(before["vertices"])[before["faces"]]
    )
    with Image.open(baked["textures"]["base_color"]["path"]) as image:
        np.testing.assert_allclose(image.getpixel((6, 10)), [137, 0, 0, 102], atol=1)
        assert image.getpixel((1, 1)) == (0, 0, 255, 255)


def test_mid_publication_failure_removes_all_owned_maps(app, monkeypatch):
    from scissors.core import texture_transfer

    _, source, target, materials = setup_surfaces(app)
    pbr_source(app, materials[0])
    before = app.engine.snapshot()
    files = {path: path.read_bytes() for path in (app.engine.root / "textures").glob("*.png")}
    original = texture_transfer.store_image
    calls = 0

    def failed_second_image(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Map publication unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(texture_transfer, "store_image", failed_second_image)
    with pytest.raises(ScissorsError):
        app.call(
            "texture.transfer",
            object_id=target,
            source_object_id=source,
            max_distance=0.1,
            resolution=16,
        )
    assert calls == 2 and app.engine.snapshot() == before
    assert {
        path: path.read_bytes() for path in (app.engine.root / "textures").glob("*.png")
    } == files


def test_correspondence_timeout_and_normal_filter_are_explicit(app, monkeypatch):
    from scissors.core import surface_transfer

    _, source, target, _ = setup_surfaces(app)
    before = app.engine.snapshot()
    ticks = iter([0, 31])
    with monkeypatch.context() as patch:
        patch.setattr(surface_transfer, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        with pytest.raises(ScissorsError) as error:
            app.call(
                "mesh.transfer_uv", object_id=target, source_object_id=source, max_distance=0.1
            )
        assert error.value.code == "OPERATION_TIMEOUT"
    assert app.engine.snapshot() == before
    app.call("mesh.flip", object_id=source)
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="unmatched"):
        app.call("mesh.transfer_uv", object_id=target, source_object_id=source, max_distance=0.1)
    assert app.engine.snapshot() == before
    result = app.call(
        "mesh.transfer_uv",
        object_id=target,
        source_object_id=source,
        max_distance=0.1,
        min_normal_dot=-1,
    )
    assert result["transferred_face_count"] == 2


def test_uv_transfer_cross_scene_material_clone_and_source_selection(app):
    _, source, target, materials = setup_surfaces(app)
    other = app.scenes.create()
    cross = app.meshes.create_plane(scene_id=other.id, size=2)
    source_selection = app.call("mesh.select_faces", object_id=source)
    result = app.call(
        "mesh.transfer_uv",
        object_id=cross.id,
        source_object_id=source,
        source_selection=source_selection,
        max_distance=0,
    )
    material = app.engine.scene(other.id).materials[result["material_id"]]
    assert material.id != materials[0] and material.name == "Source"
    bad = dict(source_selection, geometry_revision="old")
    before = app.engine.snapshot()
    with pytest.raises(ScissorsError, match="revision"):
        app.call(
            "texture.transfer",
            object_id=target,
            source_object_id=source,
            source_selection=bad,
            max_distance=0.1,
            resolution=16,
        )
    assert app.engine.snapshot() == before


def test_normal_bake_compacts_unused_vertices_before_tangent_evaluation(app):
    scene, source, target, materials = setup_surfaces(app)
    old = app.engine.object(target)[1]
    obj = app.call(
        "mesh.create", scene_id=scene, vertices=[*old.vertices, [0, 0, 0.01]], faces=old.faces
    )
    target = obj["id"]
    app.call("uv.planar", object_id=target)
    app.call("material.assign", object_id=target, material_id=materials[1])
    selected = app.call("mesh.select_faces", object_id=target)
    result = app.call(
        "texture.transfer",
        object_id=target,
        source_object_id=source,
        selection=selected,
        max_distance=0.1,
        channels=["normal"],
        resolution=16,
    )
    assert result["vertex_count"] == 4 and result["new_to_old_vertices"] == [0, 1, 2, 3]
    assert result["geometry_revision"] != selected["geometry_revision"]
    exported = app.call(
        "scene.export",
        scene_id=scene,
        only_object_ids=[target],
        output="exports/compacted.glb",
        format="glb",
    )
    assert Path(exported["path"]).is_file()


def test_interrupted_enclosing_operation_cleans_maps_and_registry_frames(app):
    _, source, target, _ = setup_surfaces(app)
    before = app.engine.snapshot()
    checkpoint = (app.engine.root / "state.json").read_bytes()

    def interrupted() -> dict:
        assert app.registry.execute(
            "texture.transfer",
            {
                "object_id": target,
                "source_object_id": source,
                "max_distance": 0.1,
                "resolution": 16,
            },
        )["success"]
        raise KeyboardInterrupt()

    app.registry.add_function(
        "test.interrupted_transfer",
        interrupted,
        "Test interrupted map cleanup",
        permissions=["scene.write", "filesystem.write"],
    )
    with pytest.raises(KeyboardInterrupt):
        app.call("test.interrupted_transfer")
    assert app.engine.snapshot() == before
    assert (app.engine.root / "state.json").read_bytes() == checkpoint
    assert not list((app.engine.root / "textures").glob("*.png"))
    assert app.registry._rollback_stack == []
