"""Versioned, bounded scene checkpoints with relocatable texture references."""

import hashlib
import re
from pathlib import Path
from typing import Any

from scissors.api.errors import ScissorsError
from scissors.core.engine import Engine
from scissors.core.persistence import atomic_write, contained, json_bytes, read_json
from scissors.core.scene import Scene


def checkpoint_path(engine: Engine, scene_id: str, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Checkpoint name needs 1-64 ASCII letters, digits, underscores or hyphens")
    key = hashlib.sha256(scene_id.encode()).hexdigest()
    return contained(engine.root, f".checkpoints/{key}/{name}.json")


def referenced_textures(engine: Engine, scene: Scene) -> dict[str, Any]:
    ids = {
        value
        for mat in scene.materials.values()
        for key, value in mat.model_dump().items()
        if key.endswith("_texture") and value
    }
    return {tid: engine.textures[tid] for tid in sorted(ids)}


def save(engine: Engine, scene_id: str, name: str) -> dict[str, Any]:
    scene = engine.scene(scene_id)
    engine.check_state()
    path = checkpoint_path(engine, scene_id, name)
    replaced = path.exists()
    data = {
        "format_version": 1,
        "kind": "scissors-checkpoint",
        "name": name,
        "scene": scene.model_dump(),
        "textures": referenced_textures(engine, scene),
    }
    atomic_write(path, json_bytes(data), engine.settings.max_file_bytes)
    return {
        "scene_id": scene_id,
        "name": name,
        "size_bytes": path.stat().st_size,
        "replaced": replaced,
        "texture_policy": "immutable workspace files referenced by relative path",
    }


def load(engine: Engine, scene_id: str, name: str) -> tuple[Scene, dict[str, Any]]:
    path = checkpoint_path(engine, scene_id, name)
    if not path.exists():
        raise ScissorsError("SNAPSHOT_NOT_FOUND", "Named checkpoint does not exist", {"name": name})
    data = read_json(path, engine.settings.max_file_bytes)
    if (
        not isinstance(data, dict)
        or data.get("format_version") != 1
        or data.get("kind") != "scissors-checkpoint"
        or data.get("name") != name
    ):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Unsupported checkpoint format")
    scene = Scene.model_validate(data["scene"])
    if scene.id != scene_id or scene.format_version != 1:
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Checkpoint belongs to another scene or version")
    textures = data.get("textures", {})
    if not isinstance(textures, dict):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Checkpoint texture table is invalid")
    for texture in textures.values():
        source = engine.path(texture["path"])
        if not source.exists():
            raise ScissorsError("TEXTURE_NOT_FOUND", "Checkpoint texture file is missing")
        if source.stat().st_size > engine.settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Checkpoint texture exceeds file limit")
    return scene, textures


def revert(engine: Engine, scene_id: str, name: str) -> dict[str, Any]:
    scene, textures = load(engine, scene_id, name)
    for tid, metadata in textures.items():
        if tid in engine.textures and engine.textures[tid] != metadata:
            raise ScissorsError(
                "INCOMPATIBLE_FORMAT", "Checkpoint conflicts with a texture ID", {"texture_id": tid}
            )
    engine.textures.update(textures)
    engine.scenes[scene_id] = scene
    engine.check_state()
    return {
        "scene_id": scene_id,
        "name": name,
        "object_ids": list(scene.objects),
        "material_ids": list(scene.materials),
    }


def compare(engine: Engine, scene_id: str, name: str) -> dict[str, Any]:
    saved, _ = load(engine, scene_id, name)
    current = engine.scene(scene_id)

    def difference(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        common = before.keys() & after.keys()
        changed = {}
        for key in sorted(common):
            a, b = before[key].model_dump(), after[key].model_dump()
            fields = [field for field in a if a[field] != b[field]]
            if fields:
                changed[key] = fields
        return {
            "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys()),
            "changed_fields": changed,
        }

    fields = [
        key
        for key in ["name", "units", "metadata", "coordinates"]
        if getattr(saved, key) != getattr(current, key)
    ]
    return {
        "scene_id": scene_id,
        "checkpoint": name,
        "direction": "checkpoint to current scene",
        "scene_changed_fields": fields,
        "objects": difference(saved.objects, current.objects),
        "materials": difference(saved.materials, current.materials),
    }
