from pathlib import Path

import trimesh
from PIL import Image


def test_material_texture_export_roundtrip(app):
    scene = app.scenes.create(name="Textured")
    mesh = app.meshes.create_cube(scene_id=scene.id)
    mat = app.materials.create(scene_id=scene.id, metallic=0.3, roughness=0.4, double_sided=True)
    texture = app.textures.generate_checker(width=64, height=64, cells=4)
    app.uv.generate(object_id=mesh.id)
    app.materials.assign(object_id=mesh.id, material_id=mat.id)
    app.textures.assign(material_id=mat.id, texture_id=texture.id)
    exported = app.export.glb(scene_id=scene.id)
    reopened = trimesh.load(exported.path, force="scene")
    geometry = next(iter(reopened.geometry.values()))
    assert len(geometry.faces) == 12
    assert geometry.visual.uv.shape == (36, 2)
    assert geometry.visual.material.baseColorTexture.size == (64, 64)
    assert abs(geometry.visual.material.metallicFactor - 0.3) < 1e-6
    assert abs(geometry.visual.material.roughnessFactor - 0.4) < 1e-6
    preview = app.scenes.preview(scene_id=scene.id)
    with Image.open(preview.path) as image:
        assert image.size == (512, 512)
    for format in ["gltf", "obj", "stl"]:
        result = app.call(
            "scene.export",
            scene_id=scene.id,
            output=f"exports/{format}/model.{format}",
            format=format,
        )
        assert Path(result["path"]).stat().st_size > 0
        loaded = trimesh.load(result["path"], force="scene")
        assert len(loaded.geometry) > 0
        if format in {"gltf", "obj"}:
            assert next(iter(loaded.geometry.values())).visual.material is not None
