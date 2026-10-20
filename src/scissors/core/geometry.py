import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import trimesh

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.plugins.runtime import limited_process


def run_backend(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    from scissors.core.persistence import contained, json_bytes, read_json

    runtime = contained(settings.workspace.resolve(), ".runtime")
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        folder = Path(temporary)
        source, result = folder / "input.json", folder / "result.json"
        data = json_bytes(payload)
        if len(data) > settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Backend input exceeds size limit")
        source.write_bytes(data)
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            code = limited_process(
                [
                    sys.executable,
                    "-m",
                    "scissors.core.geometry",
                    str(source),
                    str(result),
                    str(settings.max_file_bytes),
                ],
                folder,
                settings,
                out,
                err,
            )
        if code or not result.exists():
            raise ScissorsError(
                "GEOMETRY_ERROR",
                "Geometry backend failed",
                {"backend": payload["backend"], "exit_code": code},
            )
        return read_json(result, settings.max_file_bytes)


def run_boolean(meshes: list[trimesh.Trimesh], mode: str, settings: Settings) -> trimesh.Trimesh:
    if importlib.util.find_spec("manifold3d") is None:
        raise ScissorsError(
            "BACKEND_UNAVAILABLE",
            "Repair the Scissors installation to restore the manifold3d backend",
            {"backend": "manifold3d", "requirement": "Scissors standard installation"},
        )
    result = run_backend(
        {
            "backend": "boolean",
            "meshes": [
                {"vertices": m.vertices.tolist(), "faces": m.faces.tolist()} for m in meshes
            ],
            "mode": mode,
        },
        settings,
    )
    return trimesh.Trimesh(**result, process=False)


def run_xatlas(mesh: trimesh.Trimesh, settings: Settings) -> dict[str, Any]:
    return run_backend(
        {"backend": "xatlas", "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()},
        settings,
    )


def main() -> None:
    from scissors.core.persistence import atomic_write, json_bytes, read_json

    source, result = map(Path, sys.argv[1:3])
    limit = int(sys.argv[3])
    value = read_json(source, limit)
    if value["backend"] == "bevel":
        from scissors.core.bevel import morphology

        response = morphology(value)
    elif value["backend"] == "text":
        from scissors.core.text import text_geometry

        response = text_geometry(value)
    elif value["backend"] == "boolean_attributes":
        from scissors.core.boolean_attributes import attributed_boolean

        response = attributed_boolean(value)
    elif value["backend"] == "plane_cap":
        from scissors.core.surfaces import cap_triangulation

        response = cap_triangulation(value)
    elif value["backend"] == "import":
        from scissors.core.importing import read_import

        response = read_import(value)
    elif value["backend"] == "boolean":
        meshes = [trimesh.Trimesh(**m, process=False) for m in value["meshes"]]
        operation: Any = {
            "union": trimesh.boolean.union,
            "difference": trimesh.boolean.difference,
            "intersection": trimesh.boolean.intersection,
        }[value["mode"]]
        mesh = operation(meshes, engine="manifold", check_volume=True)
        response = {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
    elif value["backend"] == "convex_hull":
        import manifold3d
        import numpy as np

        mesh_data = manifold3d.Manifold.hull_points(
            np.asarray(value["vertices"], dtype=np.float64)
        ).to_mesh64()
        response = {
            "vertices": mesh_data.vert_properties[:, :3].tolist(),
            "faces": mesh_data.tri_verts.tolist(),
        }
    elif value["backend"] == "simplify":
        from scissors.core.simplification import reduce_geometry

        response = reduce_geometry(value)
    elif value["backend"] == "organic":
        from scissors.core.organic import distance_surface

        response = distance_surface(value)
    elif value["backend"] == "topology_repair":
        from scissors.core.topology_repair import reconstruct_soup

        response = reconstruct_soup(value)
    elif value["backend"] == "atlas":
        from scissors.core.uv import build_atlas

        response = build_atlas(value)
    elif value["backend"] == "xatlas":
        import numpy as np
        import xatlas

        vertices, faces = np.asarray(value["vertices"]), np.asarray(value["faces"])
        mapping, indices, uv = xatlas.parametrize(vertices, faces)
        response = {
            "vertices": vertices[mapping].tolist(),
            "faces": indices.tolist(),
            "uv": uv.tolist(),
        }
    else:
        raise ValueError("Unknown geometry backend")
    atomic_write(result, json_bytes(response), limit)


if __name__ == "__main__":
    main()
