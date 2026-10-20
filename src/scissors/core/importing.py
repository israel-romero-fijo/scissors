"""Local mesh readers with bounded sidecar resolution and worker serialization."""

import base64
import io
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import trimesh
from PIL import Image

from scissors.core.persistence import contained


def reference_geometry(
    width: float, height: float, axis: str, position: list[float]
) -> tuple[trimesh.Trimesh, np.ndarray]:
    vertices = [
        [-width / 2, -height / 2, 0],
        [width / 2, -height / 2, 0],
        [width / 2, height / 2, 0],
        [-width / 2, height / 2, 0],
    ]
    matrix = np.eye(4)
    matrix[:3, :3] = {
        "x": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        "y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }[axis]
    matrix[:3, 3] = position
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False), matrix


class WorkspaceResolver(trimesh.resolvers.Resolver):
    def __init__(self, root: Path, folder: Path, limit: int, ledger: dict[str, int] | None = None):
        self.root, self.folder, self.limit = root, folder, limit
        self.ledger = ledger if ledger is not None else {}
        self.errors: list[str] = []

    def get(self, key: str) -> bytes:
        try:
            decoded = unquote(key).replace("\\", "/")
            if ":" in decoded or decoded.startswith("/"):
                raise ValueError("External import resources must use workspace-relative paths")
            path = contained(self.root, self.folder / decoded)
            if path.relative_to(self.root).parts[0] in {
                "plugins",
                ".runtime",
                ".checkpoints",
                "state.json",
                ".workspace.lock",
            }:
                raise ValueError("Import resource path is reserved")
            size = path.stat().st_size
            self.ledger[str(path)] = size
            if sum(self.ledger.values()) > self.limit:
                raise ValueError("Import source and sidecars exceed byte limit")
            return path.read_bytes()
        except Exception as exc:
            self.errors.append(str(exc))
            raise

    def write(self, name: str, data: Any) -> None:
        raise ValueError("Import resolver is read-only")

    def namespaced(self, namespace: str) -> "WorkspaceResolver":
        folder = (self.folder / namespace).resolve()
        if not folder.is_relative_to(self.root):
            self.errors.append("Import namespace must stay inside the workspace")
            raise ValueError("Import namespace must stay inside the workspace")
        result = WorkspaceResolver(self.root, folder, self.limit, self.ledger)
        result.errors = self.errors
        return result

    def keys(self) -> list[str]:
        # Reader lookups use get(); never recursively enumerate the workspace.
        return []

    def __contains__(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except Exception:
            return False


def read_import(value: dict[str, Any]) -> dict[str, Any]:
    """Run only in the bounded geometry child process."""
    root, source = Path(value["root"]), Path(value["path"])
    limit, pixels = value["max_file_bytes"], value["max_texture_pixels"]
    source = contained(root, source)
    if source.stat().st_size > limit:
        raise ValueError("Import file exceeds byte limit")
    resolver = WorkspaceResolver(root, source.parent, limit, {str(source): source.stat().st_size})
    Image.MAX_IMAGE_PIXELS = pixels
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    loaded = trimesh.load_scene(
        io.BytesIO(source.read_bytes()),
        file_type=source.suffix[1:].lower(),
        resolver=resolver,
        allow_remote=False,
        process=False,
    )
    if resolver.errors:
        raise ValueError(f"Import sidecar could not be read: {resolver.errors[0]}")
    textures: dict[str, str] = {}
    materials: list[dict[str, Any]] = []
    material_cache: dict[int, int] = {}
    diagnostics = []
    image_bytes = 0

    def image_data(image: Image.Image | None) -> str | None:
        nonlocal image_bytes
        if image is None:
            return None
        if image.width * image.height > pixels:
            raise ValueError("Imported image exceeds texture pixel limit")
        image_bytes += image.width * image.height * 4
        if image_bytes > value["max_memory_bytes"]:
            raise ValueError("Imported image memory exceeds budget")
        stream = io.BytesIO()
        image.convert("RGBA").save(stream, format="PNG")
        data = stream.getvalue()
        if len(data) > limit:
            raise ValueError("Imported image exceeds encoded byte limit")
        key = f"image_{len(textures)}"
        textures[key] = base64.b64encode(data).decode("ascii")
        return key

    def material_data(material: Any) -> int:
        key = id(material)
        if key in material_cache:
            return material_cache[key]
        pbr = material.to_pbr() if hasattr(material, "to_pbr") else material
        color = getattr(pbr, "baseColorFactor", None)
        roughness = getattr(pbr, "roughnessFactor", None)
        metallic = getattr(pbr, "metallicFactor", None)
        pbr_defaults = isinstance(material, trimesh.visual.material.PBRMaterial)
        result = {
            "name": getattr(pbr, "name", None) or "Imported material",
            "base_color": (np.asarray(color, dtype=float) / 255).tolist()
            if color is not None
            else [1, 1, 1, 1],
            "metallic": float(metallic if metallic is not None else (1 if pbr_defaults else 0)),
            "roughness": float(
                roughness if roughness is not None else (1 if pbr_defaults else 0.7)
            ),
            "double_sided": bool(getattr(pbr, "doubleSided", False)),
            "emissive": np.asarray(
                getattr(pbr, "emissiveFactor", None)
                if getattr(pbr, "emissiveFactor", None) is not None
                else [0, 0, 0]
            ).tolist(),
        }
        for slot, attribute in [
            ("base_color_texture", "baseColorTexture"),
            ("normal_texture", "normalTexture"),
            ("metallic_roughness_texture", "metallicRoughnessTexture"),
            ("emissive_texture", "emissiveTexture"),
            ("occlusion_texture", "occlusionTexture"),
        ]:
            result[slot] = image_data(getattr(pbr, attribute, None))
        alpha_mode = getattr(pbr, "alphaMode", None)
        if alpha_mode == "MASK":
            diagnostics.append("Alpha MASK is approximated as BLEND; cutoff is not preserved.")
        index = len(materials)
        material_cache[key] = index
        materials.append(result)
        return index

    geometries = {}
    vertices_total = faces_total = 0
    for name, geometry in loaded.geometry.items():
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError("Import contains non-triangle geometry")
        vertices_total += len(geometry.vertices)
        faces_total += len(geometry.faces)
        if vertices_total > value["max_vertices"] or faces_total > value["max_faces"]:
            raise ValueError("Import geometry exceeds vertex/face limits")
        visual: Any = geometry.visual
        material = getattr(visual, "material", None)
        all_materials = getattr(material, "materials", None)
        face_materials = getattr(visual, "face_materials", None)
        if all_materials is not None and face_materials is not None:
            selections = [
                (index, np.flatnonzero(np.asarray(face_materials) == index), mat)
                for index, mat in enumerate(all_materials)
            ]
        else:
            selections = [(0, np.arange(len(geometry.faces)), material)]
        parts = []
        for index, selection, mat in selections:
            if len(selection) == 0:
                continue
            faces = np.asarray(geometry.faces)[selection]
            indices, inverse = np.unique(faces, return_inverse=True)
            uv = getattr(visual, "uv", None)
            colors = (
                np.asarray(visual.vertex_colors)[indices].tolist()
                if visual.kind in {"vertex", "face"}
                else None
            )
            if visual.kind == "face":
                indices = faces.reshape(-1)
                inverse = np.arange(len(indices))
                colors = np.repeat(np.asarray(visual.face_colors)[selection], 3, axis=0).tolist()
            parts.append(
                {
                    "name": f"{name}" if len(selections) == 1 else f"{name} material {index}",
                    "vertices": np.asarray(geometry.vertices)[indices].tolist(),
                    "faces": inverse.reshape((-1, 3)).tolist(),
                    "normals": np.asarray(geometry.vertex_normals)[indices].tolist(),
                    "uv": np.asarray(uv)[indices].tolist() if uv is not None else None,
                    "vertex_colors": colors,
                    "material": material_data(mat) if mat is not None else None,
                }
            )
        geometries[str(name)] = parts
    if not any(geometries.values()):
        raise ValueError("Import contains no triangle meshes")
    edges = loaded.graph.to_edgelist()
    parents = {str(child): str(parent) for parent, child, _ in edges}
    nodes = []
    for node in loaded.graph.nodes:
        matrix, geometry_name = loaded.graph.get(node)
        nodes.append(
            {
                "name": str(node),
                "parent": parents.get(str(node)),
                "geometry": str(geometry_name) if geometry_name is not None else None,
                "matrix": np.asarray(matrix).tolist(),
            }
        )
    if (vertices_total * 384 + faces_total * 192 + image_bytes) > value["max_memory_bytes"]:
        raise ValueError("Import geometry and textures exceed memory budget")
    return {
        "nodes": nodes,
        "geometries": geometries,
        "materials": materials,
        "textures": textures,
        "diagnostics": sorted(set(diagnostics)),
    }
