"""Consistent asset snapshots, portable dependencies and staged bundle publication."""

import copy
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from scissors.api.errors import ScissorsError
from scissors.core.delivery import consolidate_materials, selection_view
from scissors.core.engine import Engine
from scissors.core.export import Z_UP_TO_Y_UP, export_scene, node_names
from scissors.core.persistence import atomic_write, json_bytes, read_json
from scissors.core.render_sheets import render_sheet
from scissors.core.scene import Scene
from scissors.core.validation import validate_mesh
from scissors.core.visibility import fingerprint

MAX_FILES = 4096
MAX_OBJECTS = 20000
MAX_TEXTURES = 1024
RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *[f"COM{i}" for i in range(1, 10)],
    *[f"LPT{i}" for i in range(1, 10)],
}


def no_links(root: Path, path: Path) -> None:
    """Check lexical components before resolve can hide links or Windows junctions."""
    if not path.is_absolute():
        path = root / path
    if not path.is_relative_to(root):
        raise ScissorsError("PERMISSION_DENIED", "Bundle paths must be inside the workspace")
    current = root
    for part in path.relative_to(root).parts:
        if part in {"..", "."}:
            raise ScissorsError("PERMISSION_DENIED", "Bundle paths cannot traverse directories")
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ScissorsError(
                "PERMISSION_DENIED", "Bundle paths cannot contain links", {"path": str(current)}
            )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BundleBudget:
    def __init__(self, engine: Engine):
        self.settings = engine.settings
        self.deadline = time.monotonic() + self.settings.operation_timeout

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Asset bundle exceeded its time budget")

    def files(self, root: Path) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        total = 0
        for directory, directories, files in os.walk(root, followlinks=False):
            self.check()
            for name in [*directories, *files]:
                no_links(root, Path(directory) / name)
            for name in sorted(files):
                path = Path(directory) / name
                if not path.is_file():
                    raise ScissorsError("PERMISSION_DENIED", "Bundle entries must be regular files")
                total += path.stat().st_size
                if total > self.settings.max_file_bytes or len(result) >= MAX_FILES:
                    raise ScissorsError("RESOURCE_LIMIT", "Bundle exceeds aggregate file budget")
                result.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": file_hash(path),
                    }
                )
        return sorted(result, key=lambda value: value["path"])


def destination_path(engine: Engine, output: str, package_format: str, overwrite: bool) -> Path:
    no_links(engine.root, Path(output))
    destination = engine.artifact_path(output)
    if package_format == "zip" and destination.suffix.lower() != ".zip":
        raise ValueError("ZIP bundles require a .zip output_path")
    if package_format == "directory" and destination.suffix.lower() == ".zip":
        raise ValueError("Directory bundles cannot use a .zip output_path")
    for texture in engine.textures.values():
        if engine.path(texture["path"]).is_relative_to(destination):
            raise ScissorsError("PERMISSION_DENIED", "A bundle cannot replace texture dependencies")
    if destination.exists():
        if not overwrite:
            raise ScissorsError("ALREADY_EXISTS", "Bundle output exists; set overwrite=true")
        if package_format == "zip" and not destination.is_file():
            raise ValueError("ZIP output must replace a file")
        if package_format == "directory":
            if not destination.is_dir():
                raise ValueError("Directory output must replace a directory")
            no_links(engine.root, destination / "manifest.json")
            if not (destination / "manifest.json").is_file():
                raise ValueError("Overwrite requires an Scissors bundle directory")
            manifest = read_json(destination / "manifest.json", engine.settings.max_file_bytes)
            if not isinstance(manifest, dict):
                raise ValueError("Bundle manifest must be an object")
            if manifest.get("kind") != "scissors.bundle" or manifest.get("format_version") != 1:
                raise ValueError("Overwrite requires an Scissors bundle directory")
            existing = BundleBudget(engine).files(destination)
            declared = {value["path"] for value in manifest.get("files", [])} | {"manifest.json"}
            if {value["path"] for value in existing} != declared:
                raise ValueError(
                    "Bundle directory contains undeclared files; preserve them before overwrite"
                )
    return destination


def snapshot_engine(engine: Engine, root: Path, scene: Scene) -> Engine:
    """Use existing read-only services on detached scene data without opening a workspace."""
    result = copy.copy(engine)
    result.root = root
    result.settings = engine.settings.model_copy(update={"workspace": root})
    result.scenes = {scene.id: scene}
    result.textures = {}
    result.plugins = {}
    result._geometry_files = {}
    return result


def copy_textures(
    engine: Engine,
    staged: Engine,
    scene: Scene,
    payload: Path,
    include_scene: bool,
    include_textures: bool,
    budget: BundleBudget,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    used = {
        obj.material_id
        for obj in scene.objects.values()
        if obj.kind == "mesh" and staged.visible(scene, obj)
    }
    materials = (
        scene.materials.values()
        if include_scene
        else (material for mid, material in scene.materials.items() if mid in used)
    )
    tids = sorted(
        {
            value
            for material in materials
            for key, value in material.model_dump().items()
            if key.endswith("_texture") and value
        }
    )
    if len(tids) > MAX_TEXTURES:
        raise ScissorsError("RESOURCE_LIMIT", "Bundle exceeds texture dependency limit")
    if tids and include_scene and not include_textures:
        raise ValueError("A textured editable scene requires include_textures=true")
    portable = {}
    dependencies = []
    total_bytes = 0
    for index, tid in enumerate(tids):
        budget.check()
        if tid not in engine.textures:
            raise ScissorsError(
                "INCOMPATIBLE_FORMAT", "Scene references a missing texture", {"texture_id": tid}
            )
        original = engine.textures[tid]
        no_links(engine.root, Path(original["path"]))
        source = engine.path(original["path"])
        if not source.is_file():
            raise ScissorsError(
                "FILESYSTEM_ERROR", "Texture dependency is missing", {"texture_id": tid}
            )
        total_bytes += source.stat().st_size
        if total_bytes > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Texture dependencies exceed bundle byte budget")
        suffix = source.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".tif", ".tiff"}:
            suffix = ".image"
        target = (
            payload / "textures" if include_textures else staged.root / "inputs"
        ) / f"texture_{index:04d}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, target.open("wb") as output_stream:
            copied = 0
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                budget.check()
                copied += len(block)
                if copied > engine.settings.max_file_bytes:
                    raise ScissorsError("RESOURCE_LIMIT", "Texture dependency exceeds byte budget")
                output_stream.write(block)
        with Image.open(target) as image:
            if (
                image.width * image.height > engine.settings.max_texture_pixels
                or image.width * image.height * 32 > engine.settings.max_memory_bytes
            ):
                raise ScissorsError("RESOURCE_LIMIT", "Bundle texture exceeds decoded image budget")
            image.load()
            metadata = {
                "id": tid,
                "texture_id": tid,
                "path": target.relative_to(payload).as_posix()
                if include_textures
                else target.relative_to(staged.root).as_posix(),
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "size_bytes": target.stat().st_size,
            }
        portable[tid] = metadata
        staged.textures[tid] = {**metadata, "path": target.relative_to(staged.root).as_posix()}
        dependencies.append(
            {
                "texture_id": tid,
                "sha256": file_hash(target),
                "size_bytes": target.stat().st_size,
                "path": metadata["path"] if include_textures else None,
                "included_as_file": include_textures,
            }
        )
    return portable, dependencies


def _glb_structure(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    data = path.read_bytes()
    if len(data) < 20 or struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ScissorsError("ARTIFACT_INVALID", "Exported GLB has an invalid header")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A or 20 + length > len(data):
        raise ScissorsError("ARTIFACT_INVALID", "Exported GLB has an invalid JSON chunk")
    tree = json.loads(data[20 : 20 + length])
    start = 20 + length
    binary_length, binary_kind = struct.unpack_from("<II", data, start)
    if (
        binary_kind != 0x004E4942
        or start + 8 + binary_length != len(data)
        or len(tree.get("buffers", [])) != 1
        or not 0 <= binary_length - tree["buffers"][0]["byteLength"] <= 3
    ):
        raise ScissorsError("ARTIFACT_INVALID", "Exported GLB has an invalid binary chunk")
    if any("uri" in item for key in ["buffers", "images"] for item in tree.get(key, [])):
        raise ScissorsError("ARTIFACT_INVALID", "Bundle GLB must embed its texture and buffer data")
    triangles = 0
    instances = 0
    for node in tree.get("nodes", []):
        if "mesh" not in node:
            continue
        instances += 1
        for primitive in tree["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise ScissorsError("ARTIFACT_INVALID", "Bundle GLB contains non-triangle geometry")
            count = tree["accessors"][primitive["indices"]]["count"]
            if count % 3:
                raise ScissorsError(
                    "ARTIFACT_INVALID", "Bundle GLB has incomplete triangle indices"
                )
            triangles += count // 3
    return tree, {
        "triangle_count": triangles,
        "mesh_instance_count": instances,
        "unique_mesh_count": len(tree.get("meshes", [])),
        "node_count": len(tree.get("nodes", [])),
    }


def glb_structure(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        return _glb_structure(path)
    except (ValueError, IndexError, KeyError, TypeError, struct.error) as exc:
        raise ScissorsError(
            "ARTIFACT_INVALID", "Exported GLB structure is invalid", {"reason": type(exc).__name__}
        ) from exc


def variant_mapping(source: Scene, view: Scene) -> dict[str, Any]:
    names = node_names(view.objects.values())
    batches = {
        obj.material_id: obj
        for obj in view.objects.values()
        if obj.metadata.get("export_material_split")
    }
    result = {}
    for oid, obj in source.objects.items():
        if oid in view.objects:
            exported = view.objects[oid]
        elif obj.kind == "mesh" and obj.material_id in batches and not obj.hidden:
            exported = batches[obj.material_id]
        else:
            result[oid] = {"status": "omitted", "export_object_id": None, "node_name": None}
            continue
        result[oid] = {
            "status": "exported",
            "export_object_id": exported.id,
            "node_name": names[exported.id],
            "kind": obj.kind,
        }
    return result


def export_variant(
    staged: Engine,
    source: Scene,
    variant: dict[str, Any],
    payload: Path,
    verification: str,
    budget: BundleBudget,
) -> dict[str, Any]:
    budget.check()
    view = selection_view(staged, source.id, None, None)
    selected = view
    if variant["preset"] == "material_batches":
        view = consolidate_materials(staged, view)
    target = payload / "variants" / f"{variant['name']}.glb"
    artifact = export_scene(staged, source.id, str(target), "glb", source_scene=view)
    tree, counts = glb_structure(target)
    expected = {
        "triangle_count": sum(len(obj.faces) for obj in view.objects.values()),
        "mesh_instance_count": sum(obj.kind == "mesh" for obj in view.objects.values()),
    }
    if any(counts[key] != value for key, value in expected.items()):
        raise ScissorsError("ARTIFACT_INVALID", "GLB geometry counts differ from the snapshot")
    mapping = variant_mapping(selected, view)
    for oid in source.objects.keys() - mapping.keys():
        mapping[oid] = {"status": "omitted", "export_object_id": None, "node_name": None}
    node_name_set = {node.get("name") for node in tree.get("nodes", [])}
    if any(
        item["node_name"] not in node_name_set
        for item in mapping.values()
        if item["status"] == "exported"
    ):
        raise ScissorsError("ARTIFACT_INVALID", "GLB is missing a declared source node")
    reopened: dict[str, Any] = {"status": "skipped", "reason": "verification=structure"}
    if verification == "reopen":
        budget.check()
        try:
            loaded = trimesh.load_scene(target, process=False)
        except Exception as exc:
            raise ScissorsError(
                "ARTIFACT_INVALID",
                "GLB reopening failed",
                {"backend": "trimesh", "reason": type(exc).__name__},
            ) from exc
        faces = sum(
            len(loaded.geometry[loaded.graph[node][1]].faces)
            for node in loaded.graph.nodes_geometry
        )
        if (
            faces != counts["triangle_count"]
            or len(loaded.graph.nodes_geometry) != counts["mesh_instance_count"]
        ):
            raise ScissorsError("ARTIFACT_INVALID", "Reopened GLB has different geometry counts")
        expected_bounds = np.vstack(
            [
                np.asarray(obj.vertices) @ Z_UP_TO_Y_UP[:3, :3].T
                for obj in view.objects.values()
                if obj.kind == "mesh"
            ]
        )
        expected_bounds = np.array([expected_bounds.min(axis=0), expected_bounds.max(axis=0)])
        tolerance = max(float(np.max(np.abs(expected_bounds))), 1) * 1e-6
        if not np.allclose(loaded.bounds, expected_bounds, rtol=1e-6, atol=tolerance):
            raise ScissorsError(
                "ARTIFACT_INVALID", "Reopened GLB bounds differ from baked source geometry"
            )
        reopened = {
            "status": "passed",
            "backend": "trimesh",
            "triangle_count": faces,
            "bounds": loaded.bounds.tolist(),
            "absolute_bounds_tolerance": tolerance,
        }
    budget.check()
    return {
        **variant,
        "path": target.relative_to(payload).as_posix(),
        **counts,
        "object_count": len(view.objects),
        "source_to_export": mapping,
        "hierarchy_preserved": variant["preset"] == "hierarchy",
        "verification": {
            "structure": {"status": "passed", "textures_embedded": True},
            "reopen": reopened,
        },
        "diagnostics": artifact["diagnostics"],
    }


def preview_bundle(
    staged: Engine,
    scene: Scene,
    payload: Path,
    views: list[dict[str, Any]],
    sets: list[dict[str, Any]],
    size: int,
    renderer: str,
) -> dict[str, Any] | None:
    if not views and not sets:
        return None
    result = render_sheet(
        staged,
        scene.id,
        str(payload / "previews" / "sheet.png"),
        views,
        sets,
        3,
        size,
        None,
        None,
        True,
        None,
        False,
        renderer,
        None,
    )
    result["path"] = Path(result["path"]).relative_to(payload).as_posix()
    for view in result["views"]:
        view["path"] = Path(view["path"]).relative_to(payload).as_posix()
    return result


def publish(payload: Path, destination: Path, stage: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise ScissorsError("ALREADY_EXISTS", "Bundle output was created during preparation")
    previous = stage / "previous"
    if destination.is_dir():
        os.replace(destination, previous)
    try:
        os.replace(payload, destination)
    except OSError as exc:
        if previous.exists():
            try:
                os.replace(previous, destination)
            except OSError as restoration:
                raise ScissorsError(
                    "FILESYSTEM_ERROR",
                    "Bundle publication and restoration failed",
                    {"recovery_path": str(previous), "preserve_staging": True},
                ) from restoration
        raise ScissorsError(
            "FILESYSTEM_ERROR", "Bundle publication failed; prior output was retained"
        ) from exc


def export_bundle(
    engine: Engine,
    scene_id: str,
    output_path: str,
    variants: list[dict[str, Any]],
    include_scene: bool,
    include_textures: bool,
    views: list[dict[str, Any]],
    sets: list[dict[str, Any]],
    size: int,
    renderer: str,
    report: bool,
    package_format: str,
    overwrite: bool,
    verification: str,
) -> dict[str, Any]:
    budget = BundleBudget(engine)
    try:
        destination = destination_path(engine, output_path, package_format, overwrite)
    except OSError as exc:
        raise ScissorsError(
            "FILESYSTEM_ERROR", "Bundle output preflight failed", {"reason": type(exc).__name__}
        ) from exc
    names = [variant["name"].casefold() for variant in variants]
    if len(names) != len(set(names)) or any(name.upper() in RESERVED_NAMES for name in names):
        raise ValueError(
            "Variant names must be unique ignoring case and cannot be reserved device names"
        )
    original = engine.scene(scene_id)
    geometry_bytes = sum(
        len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in original.objects.values()
    )
    faces = sum(len(obj.faces) for obj in original.objects.values())
    if (
        len(original.objects) > MAX_OBJECTS
        or geometry_bytes * 4 > engine.settings.max_memory_bytes
        or faces * len(variants) > engine.settings.max_geometry_pairs
    ):
        raise ScissorsError(
            "RESOURCE_LIMIT", "Bundle snapshot exceeds object, memory or work budget"
        )
    source = original.model_copy(deep=True)
    scene_revision = fingerprint(source.model_dump())
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".scissors-bundle-", dir=destination.parent))
    except OSError as exc:
        raise ScissorsError(
            "FILESYSTEM_ERROR",
            "Bundle staging could not be created",
            {"reason": type(exc).__name__},
        ) from exc
    keep_staging = False
    published = False
    warnings: list[str] = []
    try:
        payload = stage / "bundle"
        payload.mkdir()
        staged = snapshot_engine(engine, stage, source)
        textures, dependencies = copy_textures(
            engine, staged, source, payload, include_scene, include_textures, budget
        )
        snapshot_hash = fingerprint(
            {
                "scene": source.model_dump(),
                "dependencies": [
                    {key: value[key] for key in ["texture_id", "sha256", "size_bytes"]}
                    for value in dependencies
                ],
            }
        )
        variants_result = [
            export_variant(staged, source, variant, payload, verification, budget)
            for variant in variants
        ]
        scene_check: dict[str, Any] = {"status": "skipped", "reason": "include_scene=false"}
        if include_scene:
            document = {"format_version": 1, "scene": source.model_dump(), "textures": textures}
            atomic_write(
                payload / "scene.json", json_bytes(document), engine.settings.max_file_bytes
            )
            reopened = read_json(payload / "scene.json", engine.settings.max_file_bytes)
            restored = snapshot_engine(staged, payload, Scene.model_validate(reopened["scene"]))
            restored.textures = reopened["textures"]
            restored.check_state()
            if fingerprint(restored.scene(scene_id).model_dump()) != scene_revision:
                raise ScissorsError(
                    "ARTIFACT_INVALID", "Editable scene differs from the source snapshot"
                )
            scene_check = {
                "status": "passed",
                "path": "scene.json",
                "backend": "Scene.model_validate",
                "texture_references": "relative to package root",
            }
        preview = preview_bundle(staged, source, payload, views, sets, size, renderer)
        budget.check()
        markers = [
            {
                "object_id": obj.id,
                "name": obj.name,
                "parent_id": obj.parent_id,
                "world_position": obj.pivot,
                "gltf_world_position": (Z_UP_TO_Y_UP[:3, :3] @ obj.pivot).tolist(),
                "transform": obj.transform,
                "metadata": obj.metadata,
                "visible": staged.visible(source, obj),
            }
            for obj in source.objects.values()
            if obj.kind == "marker"
        ]
        checks = {
            "editable_scene": scene_check,
            "glb_structure": "passed",
            "glb_reopen": "passed" if verification == "reopen" else "skipped",
            "source_state": "detached snapshot; source scene is not mutated",
            "texture_dependencies": {"status": "passed", "decoded_image_count": len(dependencies)},
            "file_hashes": {"status": "passed", "manifest_excluded": True},
        }
        if report:
            geometry_reports = []
            for obj in source.objects.values():
                budget.check()
                geometry_reports.append(validate_mesh(obj, engine.settings))
            atomic_write(
                payload / "report.json",
                json_bytes(
                    {
                        "scene_id": scene_id,
                        "scene_revision": scene_revision,
                        "checks": checks,
                        "mesh_validation": geometry_reports,
                        "limitations": [
                            "Geometry validation does not certify VR comfort or gameplay quality.",
                            "Self-intersection checks are not requested by this report.",
                        ],
                    }
                ),
                engine.settings.max_file_bytes,
            )
        notes = (
            "Scissors asset bundle\n\n"
            "GLB variants use right-handed Y-up coordinates. One export root converts scene Z-up.\n"
            f"Numeric distances retain source units: {source.units}. "
            "No unit rescaling is applied.\n"
            "The hierarchy preset retains visible object parenting. Material batches flatten it.\n"
            "Read manifest.json for source IDs, node mappings, markers, cameras and checks.\n"
            "Checks do not certify headset projection, comfort or game readiness.\n"
        )
        if include_scene:
            notes += (
                "\nUse the extracted package directory as an Scissors workspace for editing.\n"
                "call scene.load with path=scene.json. Texture references use the package root.\n"
                "Stable IDs are retained. Do not load into a workspace with conflicting IDs.\n"
            )
        atomic_write(payload / "README.txt", notes.encode(), engine.settings.max_file_bytes)
        expected_paths = {"README.txt", *[variant["path"] for variant in variants_result]}
        if include_scene:
            expected_paths.add("scene.json")
        if report:
            expected_paths.add("report.json")
        expected_paths.update(
            dependency["path"] for dependency in dependencies if dependency["included_as_file"]
        )
        if preview:
            expected_paths.update([preview["path"], *[view["path"] for view in preview["views"]]])
        files = budget.files(payload)
        if {value["path"] for value in files} != expected_paths:
            raise ScissorsError("ARTIFACT_INVALID", "Bundle contains undeclared generated files")
        manifest: dict[str, Any] = {
            "kind": "scissors.bundle",
            "format_version": 1,
            "scene_id": scene_id,
            "scene_revision": scene_revision,
            "scene_revision_semantics": "SHA256 of canonical source Scene JSON",
            "snapshot_hash": snapshot_hash,
            "snapshot_hash_semantics": "canonical scene and captured texture dependency hashes",
            "coordinates": {
                "source": source.coordinates,
                "glb": "right-handed Y-up",
                "conversion": Z_UP_TO_Y_UP.tolist(),
                "source_units": source.units,
                "unit_rescaling": False,
            },
            "settings": {
                "variants": variants,
                "include_scene": include_scene,
                "include_textures": include_textures,
                "report": report,
                "verification": verification,
                "package_format": package_format,
                "preview_views": views,
                "preview_view_sets": sets,
                "preview_size": size,
                "preview_renderer": renderer,
            },
            "variants": variants_result,
            "dependencies": dependencies,
            "markers": markers,
            "previews": preview,
            "checks": checks,
            "files": files,
            "hash_policy": "SHA256 of file bytes; manifest excludes itself",
            "warnings": (
                ["Material batches flatten mesh hierarchy."]
                if any(value["preset"] == "material_batches" for value in variants)
                else []
            ),
        }
        manifest_path = payload / "manifest.json"
        atomic_write(manifest_path, json_bytes(manifest), engine.settings.max_file_bytes)
        all_files = budget.files(payload)
        if [value for value in all_files if value["path"] != "manifest.json"] != files:
            raise ScissorsError(
                "ARTIFACT_INVALID", "Bundle files changed while creating the manifest"
            )
        manifest_sha256 = file_hash(manifest_path)
        publication = payload
        if package_format == "zip":
            publication = stage / "bundle.zip"
            with zipfile.ZipFile(publication, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for entry in all_files:
                    budget.check()
                    archive.write(payload / entry["path"], entry["path"])
                    if publication.stat().st_size > engine.settings.max_file_bytes:
                        raise ScissorsError("RESOURCE_LIMIT", "ZIP exceeds bundle byte budget")
            if publication.stat().st_size > engine.settings.max_file_bytes:
                raise ScissorsError("RESOURCE_LIMIT", "ZIP exceeds bundle byte budget")
            with zipfile.ZipFile(publication) as archive:
                if (
                    archive.namelist() != [entry["path"] for entry in all_files]
                    or archive.testzip() is not None
                ):
                    raise ScissorsError("ARTIFACT_INVALID", "ZIP entries or checksums differ")
                for entry in all_files:
                    budget.check()
                    digest = hashlib.sha256()
                    with archive.open(entry["path"]) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if (
                        archive.getinfo(entry["path"]).file_size != entry["size_bytes"]
                        or digest.hexdigest() != entry["sha256"]
                    ):
                        raise ScissorsError(
                            "ARTIFACT_INVALID", "ZIP entry differs from captured file bytes"
                        )
        budget.check()
        warnings.extend(manifest["warnings"])
        result = {
            "scene_id": scene_id,
            "path": str(destination),
            "package_format": package_format,
            "manifest_path": str(destination / "manifest.json")
            if package_format == "directory"
            else "manifest.json",
            "manifest_in_archive": package_format == "zip",
            "manifest_sha256": manifest_sha256,
            "scene_revision": scene_revision,
            "snapshot_hash": snapshot_hash,
            "files": all_files,
            "size_bytes": (
                publication.stat().st_size
                if package_format == "zip"
                else sum(entry["size_bytes"] for entry in all_files)
            ),
            "package_sha256": file_hash(publication) if package_format == "zip" else None,
            "verification": {
                **checks,
                "archive_entry_hashes": {
                    "status": "passed" if package_format == "zip" else "skipped",
                    "reason": "directory package" if package_format == "directory" else None,
                },
            },
            "warnings": warnings,
        }
        if len(json_bytes(result)) > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Bundle result exceeds metadata budget")
        destination_path(engine, output_path, package_format, overwrite)
        budget.check()
        publish(publication, destination, stage, overwrite)
        published = True
        return result
    except ScissorsError as exc:
        keep_staging = bool(exc.details.get("preserve_staging"))
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ScissorsError(
            "FILESYSTEM_ERROR",
            "Bundle preparation failed; previous output was retained",
            {"reason": type(exc).__name__},
        ) from exc
    finally:
        if not keep_staging:
            try:
                no_links(engine.root, stage)
                if not stage.resolve().is_relative_to(engine.root) or not stage.name.startswith(
                    ".scissors-bundle-"
                ):
                    raise ScissorsError("PERMISSION_DENIED", "Refusing unsafe staging cleanup")
                shutil.rmtree(stage)
            except (OSError, ScissorsError) as exc:
                if published:
                    warnings.append(f"Owned staging cleanup could not complete: {stage}")
                else:
                    raise ScissorsError(
                        "FILESYSTEM_ERROR",
                        "Bundle preparation failed and staging cleanup is incomplete",
                        {"staging_path": str(stage)},
                    ) from exc
