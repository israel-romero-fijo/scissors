"""Demonstrate surface tools on a head/muzzle blockout; uses the standard geometry backends."""

import argparse
from pathlib import Path

from scissors import Scissors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspaces/anatomy"))
    args = parser.parse_args()
    with Scissors(args.workspace) as app:
        scene = app.scenes.create(name="Anatomical surface tools")
        head = app.meshes.create_sphere(scene_id=scene.id, subdivisions=2, name="Head volume")
        muzzle = app.meshes.create_sphere(scene_id=scene.id, subdivisions=2, name="Muzzle volume")
        app.meshes.scale(object_id=muzzle.id, x=0.8, y=0.5, z=0.45)
        app.meshes.translate(object_id=muzzle.id, x=0.8, z=-0.2)
        surface = app.meshes.organic_union(
            scene_id=scene.id,
            object_ids=[head.id, muzzle.id],
            voxel_size=0.08,
            blend_radius=0.25,
            name="Head surface",
        )
        app.meshes.delete(object_id=head.id)
        app.meshes.delete(object_id=muzzle.id)
        app.deform.local(
            object_id=surface.id, center=[1.4, 0, -0.2], radius=0.5, displacement=[0.1, 0, 0]
        )
        app.sculpt.brush(
            object_id=surface.id,
            center=[1.4, 0.2, -0.1],
            radius=0.25,
            mode="deflate",
            strength=0.08,
        )
        # Remove thin reconstruction triangles before xatlas converts positions to float32.
        simplified = app.meshes.simplify(
            object_id=surface.id,
            reduction_percent=30,
            uv_policy="discard",
            max_error_relative=0.001,
        )
        # Separate face charts keep the projection atlas free of overlapping triangles.
        app.uv.unwrap(
            object_id=surface.id,
            face_groups=[[index] for index in range(simplified.face_count)],
            resolution=512,
            padding=2,
        )
        material = app.materials.create(scene_id=scene.id, base_color=[0.5, 0.35, 0.2, 1])
        app.materials.assign(object_id=surface.id, material_id=material.id)
        checker = app.textures.generate_checker(width=64, height=64, cells=8)
        app.textures.project(
            object_id=surface.id,
            texture_id=checker.id,
            center=[0.5, -0.4, 0],
            radius=1.5,
            size=[2, 2],
            u_axis=[1, 0, 0],
            v_axis=[0, 0, 1],
            resolution=256,
            opacity=0.6,
        )
        print(app.meshes.validate(object_id=surface.id, require_closed=True).as_dict())
        print(app.scenes.preview(scene_id=scene.id).path)
        print(app.export.glb(scene_id=scene.id, output="exports/head.glb").path)


if __name__ == "__main__":
    main()
