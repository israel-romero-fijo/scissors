import copy
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from filelock import FileLock, Timeout

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.materials import Material
from scissors.core.mesh import MeshObject
from scissors.core.persistence import contained
from scissors.core.scene import Scene
from scissors.core.validation import validate_mesh


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.lock = FileLock(str(self.root / ".workspace.lock"))
        try:
            self.lock.acquire(timeout=0)
        except Timeout as exc:
            raise ScissorsError("WORKSPACE_BUSY", "Another process owns this workspace") from exc
        self.scenes: dict[str, Scene] = {}
        self.textures: dict[str, dict[str, Any]] = {}
        self.plugins: dict[str, dict[str, Any]] = {}
        self._geometry_files: dict[str, tuple[int, int]] = {}
        self.closed = False
        try:
            path = self.root / "state.json"
            if path.exists():
                from scissors.core.binary_persistence import read_workspace

                state = read_workspace(self.root, settings, self._geometry_files)
                self.restore(state)
                self.check_state()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self.mutex:
            if not self.closed:
                self.lock.release()
                self.closed = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "scenes": {key: scene.model_dump() for key, scene in self.scenes.items()},
            "textures": copy.deepcopy(self.textures),
            "plugins": copy.deepcopy(self.plugins),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.scenes = {key: Scene.model_validate(value) for key, value in state["scenes"].items()}
        self.textures = copy.deepcopy(state["textures"])
        self.plugins = copy.deepcopy(state["plugins"])

    def commit(self) -> None:
        from scissors.core.binary_persistence import write_workspace

        write_workspace(
            self.root, self.settings, self.scenes, self.textures, self.plugins, self._geometry_files
        )

    def path(self, value: str | Path) -> Path:
        path = contained(self.root, value)
        relative = path.relative_to(self.root)
        if relative.parts[0] in {
            "state.json",
            ".workspace.lock",
            "plugins",
            ".runtime",
            ".checkpoints",
            ".geometry",
        }:
            raise ScissorsError("PERMISSION_DENIED", "Path is reserved for engine state")
        return path

    def scene(self, scene_id: str) -> Scene:
        try:
            return self.scenes[scene_id]
        except KeyError as exc:
            raise ScissorsError("SCENE_NOT_FOUND", "Scene does not exist") from exc

    def artifact_path(self, value: str | Path) -> Path:
        """Keep artifact writers from changing managed or retained default texture files."""
        path = self.path(value)
        if (
            any(self.path(texture["path"]) == path for texture in self.textures.values())
            or (path.exists() and path.is_relative_to(self.root / "textures"))
            or path.is_relative_to(self.root / "assets" / "fbx_documents")
        ):
            raise ScissorsError(
                "PERMISSION_DENIED", "Artifact output cannot overwrite texture files"
            )
        return path

    def object(self, object_id: str) -> tuple[Scene, MeshObject]:
        for scene in self.scenes.values():
            if object_id in scene.objects:
                return scene, scene.objects[object_id]
        raise ScissorsError("OBJECT_NOT_FOUND", "Object does not exist")

    def remove_objects(self, scene_id: str, object_ids: list[str]) -> list[str]:
        """Delete a batch after validating every ID, so one bad ID changes nothing."""
        scene = self.scene(scene_id)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Delete selection must contain unique object IDs")
        missing = [object_id for object_id in object_ids if object_id not in scene.objects]
        if missing:
            raise ScissorsError(
                "OBJECT_NOT_FOUND",
                "Delete selection contains unknown objects",
                {"object_ids": missing},
            )
        for object_id in object_ids:
            obj = scene.objects[object_id]
            for child in scene.objects.values():
                if child.parent_id == object_id:
                    child.parent_id = obj.parent_id
            del scene.objects[object_id]
        return object_ids

    def material(self, material_id: str) -> tuple[Scene, Material]:
        for scene in self.scenes.values():
            if material_id in scene.materials:
                return scene, scene.materials[material_id]
        raise ScissorsError("MATERIAL_NOT_FOUND", "Material does not exist")

    def checked(self, obj: MeshObject, *, track_revision: bool = True) -> None:
        report = validate_mesh(obj, self.settings)
        if not report["valid"]:
            code = "RESOURCE_LIMIT" if "RESOURCE_LIMIT" in report["errors"] else "GEOMETRY_ERROR"
            raise ScissorsError(code, "Mesh validation failed", report)
        if track_revision:
            obj.sync_geometry_revision()

    def check_state(self) -> None:
        ids: set[str] = set()
        memory = 0
        for sid, scene in self.scenes.items():
            if sid != scene.id or scene.format_version != 1:
                raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid scene identifier or version")
            for document in scene.fbx_documents:
                if document.path != f"assets/fbx_documents/{document.sha256}.fbx" or (
                    len(document.sha256) != 64
                    or any(c not in "0123456789abcdef" for c in document.sha256)
                ):
                    raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid retained FBX reference")
                if not self.path(document.path).is_file():
                    raise ScissorsError("INCOMPATIBLE_FORMAT", "Missing retained FBX document")
                memory += sum(
                    len(binding.world_vertices) * 256 for binding in document.bindings.values()
                )
            for oid, obj in scene.objects.items():
                if oid != obj.id or oid in ids:
                    raise ScissorsError("INCOMPATIBLE_FORMAT", "Duplicate or mismatched object ID")
                ids.add(oid)
                self.checked(obj, track_revision=False)
                if obj.parent_id is not None:
                    parent = scene.objects.get(obj.parent_id)
                    if parent is None or parent.id == obj.id:
                        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid parent reference")
                    seen = {obj.id}
                    while parent is not None:
                        if parent.id in seen:
                            raise ScissorsError(
                                "INCOMPATIBLE_FORMAT", "Object hierarchy has a cycle"
                            )
                        seen.add(parent.id)
                        parent = scene.objects.get(parent.parent_id) if parent.parent_id else None
                memory += len(obj.vertices) * 384 + len(obj.faces) * 192
                if obj.material_id and obj.material_id not in scene.materials:
                    raise ScissorsError("INCOMPATIBLE_FORMAT", "Missing material reference")
            for mid, mat in scene.materials.items():
                if (
                    mid != mat.id
                    or mid in ids
                    or any(c < 0 or c > 1 for c in [*mat.base_color, *mat.emissive])
                ):
                    raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid material")
                ids.add(mid)
                for tid in (
                    mat.base_color_texture,
                    mat.normal_texture,
                    mat.metallic_roughness_texture,
                    mat.emissive_texture,
                    mat.occlusion_texture,
                ):
                    if tid and tid not in self.textures:
                        raise ScissorsError("INCOMPATIBLE_FORMAT", "Missing texture reference")
        for texture in self.textures.values():
            self.path(texture["path"])
        if memory > self.settings.max_memory_bytes:
            raise ScissorsError(
                "RESOURCE_LIMIT", "Estimated workspace geometry memory exceeds limit"
            )

    def create_scene(
        self, name: str, units: str = "meters", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        scene = Scene(id=identifier("scene"), name=name, units=units, metadata=metadata or {})
        self.scenes[scene.id] = scene
        return scene.summary()

    def add_mesh(
        self, scene_id: str, mesh: trimesh.Trimesh, name: str, primitive: str = "custom"
    ) -> dict[str, Any]:
        scene = self.scene(scene_id)
        obj = MeshObject(
            id=identifier("obj"),
            name=name,
            vertices=[],
            faces=[],
            metadata={"primitive": primitive},
        )
        obj.replace(mesh)
        self.checked(obj)
        scene.objects[obj.id] = obj
        return obj.summary()

    def transform(self, object_id: str, matrix: np.ndarray) -> dict[str, Any]:
        scene, obj = self.object(object_id)
        targets = self.descendants(scene, object_id)
        for target in [obj, *targets]:
            if target.kind == "mesh":
                normal_method = target.metadata.get("normal_method")
                normal_angle = target.metadata.get("normal_angle_degrees")
                old_normals = np.asarray(target.normals)
                old_colors = target.vertex_colors
                # Apply every finite delta, including changes below the geometry
                # library's identity tolerance. Vertices already contain prior transforms.
                vertices = np.asarray(target.vertices) @ matrix[:3, :3].T + matrix[:3, 3]
                faces = np.asarray(target.faces)
                if np.linalg.slogdet(matrix[:3, :3])[0] < 0:
                    faces = np.fliplr(faces)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                target.replace(mesh, target.uv)
                target.vertex_colors = old_colors
                if normal_method == "split_angle":
                    normals = old_normals @ np.linalg.inv(matrix[:3, :3])
                    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
                    target.normals = normals.tolist()
                    target.metadata["normal_method"] = normal_method
                    target.metadata["normal_angle_degrees"] = normal_angle
            if target.kind != "mesh" or target.metadata.get("pivot_explicit", False):
                target.pivot = (matrix[:3, :3] @ np.asarray(target.pivot) + matrix[:3, 3]).tolist()
            target.transform = (matrix @ np.asarray(target.transform)).tolist()
            self.checked(target)
        return obj.summary()

    def descendants(self, scene: Scene, object_id: str) -> list[MeshObject]:
        children: dict[str, list[MeshObject]] = {}
        for obj in scene.objects.values():
            if obj.parent_id is not None:
                children.setdefault(obj.parent_id, []).append(obj)
        result: list[MeshObject] = []
        queue = [object_id]
        seen = {object_id}
        while queue:
            parent = queue.pop()
            for child in children.get(parent, []):
                if child.id in seen:
                    raise ScissorsError("GEOMETRY_ERROR", "Object hierarchy has a cycle")
                seen.add(child.id)
                result.append(child)
                queue.append(child.id)
        return result

    def clone_object(self, object_id: str, name: str | None = None) -> MeshObject:
        """Copy one subtree with fresh IDs and preserved world geometry."""
        scene, obj = self.object(object_id)
        sources = [obj, *self.descendants(scene, object_id)]
        if scene.fbx_documents:
            raise ScissorsError("UNSUPPORTED_OPERATION", "Retained FBX rigs cannot copy subtrees")
        extra = sum(len(source.vertices) * 384 + len(source.faces) * 192 for source in sources)
        used = sum(
            len(source.vertices) * 384 + len(source.faces) * 192
            for other in self.scenes.values()
            for source in other.objects.values()
        )
        if used + extra > self.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Object copy exceeds workspace geometry budget")
        mapping = {source.id: identifier("obj") for source in sources}
        for source in sources:
            duplicate = source.model_copy(deep=True)
            duplicate.id = mapping[source.id]
            duplicate.parent_id = (
                mapping.get(source.parent_id, source.parent_id) if source.parent_id else None
            )
            if source.kind == "mesh":
                duplicate.metadata["instance_source"] = source.metadata.get(
                    "instance_source", source.id
                )
            if source.id == object_id:
                duplicate.name = name or f"{source.name} copy"
            scene.objects[duplicate.id] = duplicate
        return scene.objects[mapping[object_id]]

    def visible(self, scene: Scene, obj: MeshObject) -> bool:
        while True:
            if obj.hidden:
                return False
            if obj.parent_id is None:
                return True
            obj = scene.objects[obj.parent_id]

    def copy_objects(
        self,
        from_scene: str,
        to_scene: str,
        object_ids: list[str],
        include_unused_materials: bool = False,
    ) -> dict[str, Any]:
        """Copy world-space subtrees and material ownership into another scene."""
        source, target = self.scene(from_scene), self.scene(to_scene)
        if source.fbx_documents or target.fbx_documents:
            raise ScissorsError(
                "UNSUPPORTED_OPERATION", "Retained FBX rigs cannot copy scene objects"
            )
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Copy selection must contain unique IDs")
        missing = [oid for oid in object_ids if oid not in source.objects]
        if missing:
            raise ScissorsError(
                "OBJECT_NOT_FOUND",
                "Copy selection contains unknown source objects",
                {"object_ids": missing},
            )
        selected = set(object_ids)
        for oid in object_ids:
            selected.update(obj.id for obj in self.descendants(source, oid))
        objects = [obj for obj in source.objects.values() if obj.id in selected]
        extra = sum(len(obj.vertices) * 384 + len(obj.faces) * 192 for obj in objects)
        used = sum(
            len(obj.vertices) * 384 + len(obj.faces) * 192
            for scene in self.scenes.values()
            for obj in scene.objects.values()
        )
        if used + extra > self.settings.max_memory_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Scene copy exceeds geometry memory budget")
        material_ids = (
            set(source.materials)
            if include_unused_materials
            else {obj.material_id for obj in objects if obj.material_id}
        )
        material_mapping = {
            mid: identifier("mat") for mid in source.materials if mid in material_ids
        }
        object_mapping = {obj.id: identifier("obj") for obj in objects}
        for mid, new_id in material_mapping.items():
            material = source.materials[mid].model_copy(deep=True)
            material.id = new_id
            target.materials[new_id] = material
        for obj in objects:
            duplicate = obj.model_copy(deep=True)
            duplicate.id = object_mapping[obj.id]
            duplicate.parent_id = object_mapping.get(obj.parent_id or "")
            duplicate.material_id = material_mapping.get(obj.material_id or "")
            instance_source = duplicate.metadata.get("instance_source")
            if instance_source:
                duplicate.metadata["instance_source"] = object_mapping.get(
                    instance_source, instance_source
                )
            target.objects[duplicate.id] = duplicate
        return {
            "from_scene": from_scene,
            "to_scene": to_scene,
            "object_mapping": object_mapping,
            "material_mapping": material_mapping,
            "copied_object_ids": list(object_mapping.values()),
            "texture_policy": "shared immutable workspace textures; independent material copies",
        }
