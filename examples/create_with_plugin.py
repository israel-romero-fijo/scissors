import json
from pathlib import Path

from scissors import Scissors


def main() -> None:
    with Scissors("workspaces/twist_example") as app:
        if "twist_mesh" not in app.engine.plugins:
            # This repository's example package; install enables it.
            app.plugins.install(Path(__file__).resolve().parents[1] / "plugins/examples/twist_mesh")
        else:
            app.call("plugins.enable", plugin_id="twist_mesh")
        scene = app.scenes.create(name="Twisted elliptical cylinder")
        mesh = app.meshes.create_cylinder(
            scene_id=scene.id, radius=0.6, height=3, sections=48, height_segments=16
        )
        app.meshes.scale(object_id=mesh.id, x=1.7)
        stats = app.call(
            "deform.twist_mesh",
            object_id=mesh.id,
            axis="z",
            angle_degrees=180,
            falloff="smoothstep",
        )
        assert app.meshes.validate(object_id=mesh.id, require_closed=True).valid
        exported = app.export.glb(scene_id=scene.id, output="exports/twisted.glb")
        preview = app.scenes.preview(scene_id=scene.id, output="previews/twisted.png")
        print(
            json.dumps(
                {"deformation": stats, "export": exported.as_dict(), "preview": preview.as_dict()},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
