import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from scissors import Scissors
from scissors.core.mesh import topology_mesh
from scissors.core.simplification import topology_signature


def sphere(app, subdivisions=2):
    scene = app.scenes.create()
    return scene, app.meshes.create_sphere(scene_id=scene.id, subdivisions=subdivisions)


def test_local_deform_falloff_uv_seams_and_restart(app):
    _, obj = sphere(app)
    app.uv.generate(object_id=obj.id, method="spherical")
    original = app.engine.object(obj.id)[1].model_copy(deep=True)
    before = np.asarray(original.vertices)
    center = [0, 0, 1]
    result = app.deform.local(object_id=obj.id, center=center, radius=0.7, displacement=[0.2, 0, 0])
    edited = app.engine.object(obj.id)[1]
    after = np.asarray(edited.vertices)
    outside = np.linalg.norm(before - center, axis=1) >= 0.7
    assert np.array_equal(before[outside], after[outside])
    assert result.affected_vertices > 0
    assert after[:, 0].max() <= before[:, 0].max() + 0.2
    assert original.uv == edited.uv and original.faces == edited.faces
    assert topology_mesh(edited.geometry()).is_watertight
    assert original.transform == edited.transform
    workspace = app.engine.root
    expected = edited.model_dump()
    app.close()
    with Scissors(workspace) as reopened:
        assert reopened.engine.object(obj.id)[1].model_dump() == expected


@pytest.mark.parametrize("mode", ["inflate", "deflate", "move", "smooth", "sharpen"])
def test_sculpt_changes_surface_without_opening_seams(app, mode):
    _, obj = sphere(app)
    app.uv.generate(object_id=obj.id, method="spherical")
    original = app.engine.object(obj.id)[1].model_copy(deep=True)
    result = app.sculpt.brush(
        object_id=obj.id,
        center=[0, 0, 1],
        radius=0.7,
        mode=mode,
        strength=0.05,
        displacement=[0.03, 0, 0],
    )
    edited = app.engine.object(obj.id)[1]
    assert result.affected_vertices > 0
    assert not np.array_equal(edited.vertices, original.vertices)
    assert edited.uv == original.uv and edited.faces == original.faces
    assert app.meshes.validate(object_id=obj.id, require_closed=True).valid
    outside = np.linalg.norm(np.asarray(original.vertices) - [0, 0, 1], axis=1) >= 0.7
    assert np.array_equal(
        np.asarray(edited.vertices)[outside], np.asarray(original.vertices)[outside]
    )


def test_failed_brush_and_atomic_edits_restore_state(app):
    _, obj = sphere(app)
    before = app.engine.snapshot()
    result = app.registry.execute(
        "operations.batch",
        {
            "operations": [
                {
                    "operation": "deform.local",
                    "arguments": {
                        "object_id": obj.id,
                        "center": [0, 0, 1],
                        "radius": 1,
                        "displacement": [0.1, 0, 0],
                    },
                },
                {
                    "operation": "sculpt.brush",
                    "arguments": {"object_id": obj.id, "center": [100, 0, 0], "radius": 0.1},
                },
            ]
        },
    )
    assert not result["success"]
    assert app.engine.snapshot() == before
    invalid = app.registry.execute(
        "deform.local", {"object_id": obj.id, "center": [0, 0], "radius": 1}
    )
    assert invalid["error"]["code"] == "INVALID_PARAMETER"


def textured_plane(app):
    scene = app.scenes.create()
    obj = app.meshes.create_plane(scene_id=scene.id)
    app.uv.generate(object_id=obj.id, method="planar")
    material = app.materials.create(scene_id=scene.id, base_color=[0.1, 0.2, 0.3, 1])
    app.materials.assign(object_id=obj.id, material_id=material.id)
    source = app.textures.generate_checker(
        width=64, height=64, cells=1, color_a=[255, 0, 0, 255], color_b=[255, 0, 0, 255]
    )
    return scene, obj, material, source


def test_texture_projection_feather_export_and_reopen(app):
    scene, obj, material, source = textured_plane(app)
    result = app.textures.project(
        object_id=obj.id,
        texture_id=source.id,
        center=[0, 0, 0],
        radius=1,
        size=[2, 2],
        resolution=64,
    )
    with Image.open(result.path) as image:
        center = np.asarray(image)[32, 32]
        corner = np.asarray(image)[0, 0]
    assert center[0] > 240 and center[1] < 10
    assert np.allclose(corner[:3], [26, 51, 76], atol=1)
    mat = app.materials.get(material_id=material.id)
    assert mat.base_color_texture == result.texture_id
    assert mat.base_color == [1, 1, 1, 1]
    exported = app.export.glb(scene_id=scene.id)
    reopened = trimesh.load(exported.path, force="scene")
    assert len(reopened.geometry) == 1
    geometry = next(iter(reopened.geometry.values()))
    assert geometry.visual.material.baseColorTexture is not None
    assert len(geometry.visual.uv) == 4
    original = app.engine.snapshot()
    # Painting is a filesystem operation and cannot enter an atomic batch.
    rejected = app.registry.execute(
        "operations.batch",
        {
            "operations": [
                {
                    "operation": "texture.project",
                    "arguments": {
                        "object_id": obj.id,
                        "texture_id": source.id,
                        "center": [0, 0, 0],
                        "radius": 1,
                        "size": [2, 2],
                    },
                }
            ]
        },
    )
    assert not rejected["success"] and app.engine.snapshot() == original


def test_projection_rejects_overlapping_uv_and_invalid_axes(app):
    _, obj, _, source = textured_plane(app)
    before = app.engine.snapshot()
    arguments = {
        "object_id": obj.id,
        "texture_id": source.id,
        "center": [0, 0, 0],
        "radius": 1,
        "size": [2, 2],
        "resolution": 64,
    }
    invalid = app.registry.execute("texture.project", {**arguments, "v_axis": [1, 0, 0]})
    assert not invalid["success"] and app.engine.snapshot() == before
    mesh = app.engine.object(obj.id)[1]
    mesh.replace(
        trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices)[mesh.faces].reshape(-1, 3),
            faces=[[0, 1, 2], [3, 4, 5]],
            process=False,
        ),
        [[0, 0], [1, 0], [0, 1]] * 2,
    )
    before = app.engine.snapshot()
    rejected = app.registry.execute("texture.project", arguments)
    assert rejected["error"]["code"] == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


def test_reference_comparison_artifacts_and_containment(app):
    scene, _ = sphere(app)
    reference = app.engine.root / "references/animal.png"
    reference.parent.mkdir()
    Image.new("RGB", (100, 80), (255, 0, 0)).save(reference)
    result = app.scenes.compare_reference(
        scene_id=scene.id, reference_path="references/animal.png", width=128, height=128
    )
    with Image.open(result.path) as image:
        assert image.size == (256, 128)
    overlay = app.scenes.compare_reference(
        scene_id=scene.id,
        reference_path="references/animal.png",
        width=128,
        height=128,
        mode="overlay",
        opacity=0,
        zoom=1.2,
        offset_x=0.1,
    )
    with Image.open(overlay.path) as image:
        assert image.size == (128, 128)
        assert image.getpixel((64, 64)) == (255, 0, 0)
    rejected = app.registry.execute(
        "scene.compare_reference", {"scene_id": scene.id, "reference_path": "../outside.png"}
    )
    assert rejected["error"]["code"] == "PERMISSION_DENIED"


@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("vtkmodules") is None, reason="required VTK backend absent"
)
def test_organic_union_remesh_and_export(app):
    scene, first = sphere(app, 1)
    second = app.meshes.create_sphere(scene_id=scene.id, subdivisions=1)
    app.meshes.translate(object_id=second.id, x=1)
    result = app.meshes.organic_union(
        scene_id=scene.id, object_ids=[first.id, second.id], voxel_size=0.15, blend_radius=0.3
    )
    assert result.watertight and result.volume > first.volume
    assert app.meshes.validate(object_id=result.id, require_closed=True).valid
    assert topology_signature(app.engine.object(result.id)[1])[1] == 1
    assert len(app.engine.scene(scene.id).objects) == 3
    app.uv.generate(object_id=result.id)
    remeshed = app.meshes.remesh(object_id=result.id, voxel_size=0.12, reduction_percent=20)
    assert remeshed.watertight and not remeshed.has_uv and remeshed.id == result.id
    app.meshes.delete(object_id=first.id)
    app.meshes.delete(object_id=second.id)
    exported = app.export.stl(scene_id=scene.id)
    assert trimesh.load(exported.path).is_watertight


def test_organic_limits_and_invalid_sources(app, monkeypatch):
    import scissors.core.organic as organic

    scene, first = sphere(app, 1)
    second = app.meshes.create_sphere(scene_id=scene.id, subdivisions=1)
    arguments = {
        "scene_id": scene.id,
        "object_ids": [first.id, second.id],
        "voxel_size": 1e-10,
        "blend_radius": 0.2,
    }
    before = app.engine.snapshot()
    monkeypatch.setattr(organic, "available", lambda: True)
    result = app.registry.execute("mesh.organic_union", arguments)
    assert result["error"]["code"] == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before
    monkeypatch.setattr(organic, "available", lambda: False)
    result = app.registry.execute("mesh.organic_union", arguments)
    assert result["error"]["code"] == "BACKEND_UNAVAILABLE"
    duplicate = app.registry.execute(
        "mesh.organic_union", {**arguments, "object_ids": [first.id] * 2}
    )
    assert duplicate["error"]["code"] == "INVALID_PARAMETER"


@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("xatlas") is None, reason="required xatlas backend absent"
)
def test_atlas_region_cuts_and_density(app):
    scene = app.scenes.create()
    # Two equal planar squares with distinct density requirements.
    obj = app.meshes.create(
        scene_id=scene.id,
        vertices=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
        ],
        faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
    )
    result = app.uv.unwrap(
        object_id=obj.id, face_groups=[[0, 1], [2, 3]], density_weights=[4, 1], resolution=256
    )
    assert result.has_uv and result.chart_count >= 2
    edited = app.engine.object(obj.id)[1]
    uv = np.asarray(edited.uv)[edited.faces]
    edges_a, edges_b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    areas = np.abs(edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0]) / 2
    assert areas[:2].sum() > areas[2:].sum() * 3
    assert np.min(uv) >= 0 and np.max(uv) <= 1
    before = app.engine.snapshot()
    invalid = app.registry.execute("uv.unwrap", {"object_id": obj.id, "face_groups": [[0], [0]]})
    assert invalid["error"]["code"] == "INVALID_PARAMETER"
    assert app.engine.snapshot() == before


def test_native_modeling_catalog(app):
    for name in [
        "mesh.organic_union",
        "deform.local",
        "sculpt.brush",
        "mesh.remesh",
        "uv.unwrap",
        "texture.project",
        "scene.compare_reference",
    ]:
        operation = app.registry.describe_operation(name)
        assert operation["plugin"] is None
        assert operation["parameters"]["additionalProperties"] is False
        assert operation["examples"]
    assert "filesystem.write" in app.registry.describe_operation("texture.project")["permissions"]
    assert Path(app.engine.root).is_dir()


def test_collapsing_brush_restores_geometry(app):
    scene = app.scenes.create()
    obj = app.meshes.create(
        scene_id=scene.id, vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    )
    before = app.engine.snapshot()
    failed = app.registry.execute(
        "sculpt.brush",
        {
            "object_id": obj.id,
            "center": [0, 0, 0],
            "radius": 0.1,
            "mode": "move",
            "displacement": [1, 0, 0],
        },
    )
    assert failed["error"]["code"] == "GEOMETRY_ERROR"
    assert app.engine.snapshot() == before


@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("xatlas") is None, reason="required xatlas backend absent"
)
def test_unwrap_distortion_limit_restores_original_uv(app):
    _, obj = sphere(app, 1)
    app.uv.generate(object_id=obj.id)
    before = app.engine.snapshot()
    failed = app.registry.execute("uv.unwrap", {"object_id": obj.id, "max_anisotropy": 1})
    assert failed["error"]["code"] == "GEOMETRY_ERROR"
    assert "max_anisotropy" in failed["error"]["details"]
    assert app.engine.snapshot() == before


def test_missing_atlas_backend_and_projection_limit(app, monkeypatch):
    import scissors.core.uv as uv

    _, obj, _, source = textured_plane(app)
    before = app.engine.snapshot()
    spec = uv.importlib.util.find_spec
    monkeypatch.setattr(
        uv.importlib.util, "find_spec", lambda name: None if name == "xatlas" else spec(name)
    )
    missing = app.registry.execute("uv.unwrap", {"object_id": obj.id})
    assert missing["error"]["code"] == "BACKEND_UNAVAILABLE"
    app.engine.settings.max_texture_pixels = 100
    limited = app.registry.execute(
        "texture.project",
        {
            "object_id": obj.id,
            "texture_id": source.id,
            "center": [0, 0, 0],
            "radius": 1,
            "size": [2, 2],
            "resolution": 64,
        },
    )
    assert limited["error"]["code"] == "RESOURCE_LIMIT"
    assert app.engine.snapshot() == before
