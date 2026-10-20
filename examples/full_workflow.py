import json

import trimesh
from PIL import Image

from scissors import Scissors


def main() -> None:
    with Scissors("workspaces/full_workflow") as app:
        scene = app.scenes.create(name="Textured cube")
        mesh = app.meshes.create_cube(scene_id=scene.id, size=2)
        app.meshes.scale(object_id=mesh.id, x=1, y=0.7, z=1.4)
        app.meshes.rotate(object_id=mesh.id, axis="z", angle_degrees=25)
        material = app.materials.create(scene_id=scene.id, name="Checker", roughness=0.65)
        texture = app.textures.generate_checker(width=256, height=256, cells=8)
        app.uv.generate(object_id=mesh.id)
        app.textures.assign(material_id=material.id, texture_id=texture.id)
        app.materials.assign(object_id=mesh.id, material_id=material.id)
        validation = app.meshes.validate(object_id=mesh.id, require_closed=True)
        assert validation.valid
        preview = app.scenes.preview(scene_id=scene.id, output="previews/model.png")
        exported = app.export.glb(scene_id=scene.id, output="exports/model.glb")
        saved = app.scenes.save(scene_id=scene.id)
        reopened = trimesh.load(exported.path, force="scene")
        assert len(reopened.geometry) == 1
        assert next(iter(reopened.geometry.values())).visual.material.baseColorTexture is not None
        with Image.open(preview.path) as image:
            image.verify()
        print(
            json.dumps(
                {
                    "scene_id": scene.id,
                    "object_id": mesh.id,
                    "validation": validation.as_dict(),
                    "preview": preview.as_dict(),
                    "export": exported.as_dict(),
                    "saved": saved.as_dict(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
