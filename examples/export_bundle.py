"""Create a portable GLB and editable-scene package with a player camera."""

from pathlib import Path

from scissors import Scissors


def main() -> None:
    with Scissors(Path("workspaces/bundle-example")) as app:
        sid = app.call("scene.create", name="Hollow frame asset")["id"]
        profile = app.call(
            "mesh.create_profile",
            scene_id=sid,
            points=[[-1, -0.7], [1, -0.7], [1, 0.7], [-1, 0.7]],
            holes=[[[-0.8, -0.5], [-0.8, 0.5], [0.8, 0.5], [0.8, -0.5]]],
        )["id"]
        app.call("mesh.extrude", object_id=profile, distance=0.1, name="Frame")
        app.call("mesh.hide", object_id=profile)
        app.call("scene.marker", scene_id=sid, name="Player eye", position=[0, -2, 1.4])
        result = app.call(
            "scene.export_bundle",
            scene_id=sid,
            output_path=f"exports/{sid}.zip",
            package_format="zip",
            preview_renderer="pillow",
            preview_size=128,
            preview_views=[
                {
                    "type": "camera",
                    "name": "Player",
                    "position": [0, -2, 1.4],
                    "target": [0, 0, 0],
                    "fov": 75,
                }
            ],
        )
        print(result["path"])
        print(result["manifest_sha256"])


if __name__ == "__main__":
    main()
