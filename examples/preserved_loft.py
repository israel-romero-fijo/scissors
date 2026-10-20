"""Loft section corners through the public registry in a temporary workspace."""

import tempfile
from pathlib import Path

from scissors import Scissors


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="scissors-preserved-loft-") as directory:
        with Scissors(Path(directory)) as app:
            scene_id = app.call("scene.create", name="Preserved section corners")["id"]
            sections = []
            for height, width in [(0, 1.0), (0.8, 0.9), (1.4, 0.65)]:
                section = app.call(
                    "mesh.create_profile",
                    scene_id=scene_id,
                    points=[[-width, -0.5], [width, -0.5], [width, 0.5], [-width, 0.5]],
                )
                app.call("mesh.translate", object_id=section["id"], z=height)
                sections.append(section["id"])
            result = app.call(
                "mesh.loft",
                object_ids=sections,
                preserve_vertices=True,
                smooth=True,
                name="Tapered body",
            )
            print(result["id"], result["loft"]["ring_vertex_counts"])
            print(result["loft"]["self_intersection_check"]["status"])


if __name__ == "__main__":
    main()
