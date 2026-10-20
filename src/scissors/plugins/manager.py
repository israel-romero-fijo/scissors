import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from scissors.api.errors import ScissorsError
from scissors.api.registry import Operation, OperationRegistry
from scissors.core.engine import Engine
from scissors.core.mesh import MeshObject
from scissors.core.persistence import contained
from scissors.plugins.loader import files, fingerprint, read_manifest
from scissors.plugins.manifest import PluginManifest, PluginOperation
from scissors.plugins.runtime import limited_process, run_worker


class PluginManager:
    def __init__(self, engine: Engine, registry: OperationRegistry):
        self.engine = engine
        self.registry = registry
        configured = engine.settings.plugins_dir.expanduser().absolute()
        if configured.is_symlink():
            raise ScissorsError("PERMISSION_DENIED", "Plugin storage must not be a symlink")
        self.root = configured.resolve()
        if self.root.is_relative_to(engine.root) or engine.root.is_relative_to(self.root):
            raise ScissorsError(
                "INVALID_PARAMETER", "Plugin storage and workspace must be separate directories"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / ".plugins.lock"))
        self.errors: dict[str, dict[str, Any]] = {}
        with self._locked():
            self._load()

    def _load(self) -> None:
        engine = self.engine
        for value in engine.plugins.values():
            value.pop("authorized", None)
            value.pop("fingerprint", None)
        if self._scan():
            engine.commit()
        for pid, value in list(engine.plugins.items()):
            if value["enabled"]:
                try:
                    self._enable(pid)
                except Exception as exc:
                    self.errors[pid] = {"code": "PLUGIN_LOAD_FAILED", "message": str(exc)[:1000]}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize package access across processes without owning other workspaces."""
        try:
            self.lock.acquire(timeout=self.engine.settings.operation_timeout)
        except Timeout as exc:
            raise ScissorsError(
                "PLUGIN_STORE_BUSY", "Another process is using plugin storage"
            ) from exc
        try:
            yield
        finally:
            self.lock.release()

    def _scan(self) -> bool:
        """Register shared packages and retain this workspace's activation choices."""
        missing = [pid for pid in self.engine.plugins if not self._path(pid).is_dir()]
        for pid in missing:
            del self.engine.plugins[pid]
        added = bool(missing)
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            folder = manifest_path.parent
            if folder.name in self.engine.plugins:
                continue
            try:
                manifest = read_manifest(folder, self.engine.settings.max_file_bytes)
                if manifest.id != folder.name:
                    raise ScissorsError("PLUGIN_INVALID", "Manifest id must match the folder name")
            except Exception as exc:
                self.errors[folder.name] = {"code": "PLUGIN_INVALID", "message": str(exc)[:1000]}
                continue
            self.engine.plugins[manifest.id] = {
                "enabled": True,
                "version": manifest.version,
                "trust": "external",
            }
            added = True
        return added

    def list(self) -> dict[str, Any]:
        with self._locked():
            entries = []
            for pid, value in self.engine.plugins.items():
                installed = self._path(pid).is_dir()
                entries.append(
                    {
                        "id": pid,
                        **value,
                        "installed": installed,
                        "active": installed
                        and any(op.plugin == pid for op in self.registry.operations.values()),
                        "load_error": self.errors.get(pid)
                        if installed
                        else {"code": "PLUGIN_NOT_FOUND", "message": "Plugin is not installed"},
                    }
                )
            entries.extend(
                {
                    "id": pid,
                    "enabled": False,
                    "installed": False,
                    "active": False,
                    "load_error": error,
                }
                for pid, error in self.errors.items()
                if pid not in self.engine.plugins
            )
            return {"plugins": entries, "plugins_dir": str(self.root)}

    def describe(self, plugin_id: str) -> dict[str, Any]:
        with self._locked():
            return {
                "manifest": self._manifest(plugin_id).model_dump(),
                "state": self.engine.plugins[plugin_id],
                "load_error": self.errors.get(plugin_id),
            }

    def discover(self, path: str | Path) -> dict[str, Any]:
        folder = Path(path).resolve()
        found = []
        for manifest in folder.glob("*/manifest.json"):
            found.append({"path": str(manifest.parent), **self.validate(manifest.parent)})
        return {"plugins": found}

    def validate(self, path: str | Path) -> dict[str, Any]:
        try:
            folder = Path(path).resolve()
            manifest = read_manifest(folder, self.engine.settings.max_file_bytes)
            return {
                "valid": True,
                "manifest": manifest.model_dump(),
                "fingerprint": fingerprint(folder, self.engine.settings.max_file_bytes),
                "tests_present": (folder / "tests").is_dir(),
                "code_executed": False,
            }
        except Exception as exc:
            return {
                "valid": False,
                "error": {"code": "PLUGIN_INVALID", "message": str(exc)[:1000]},
                "code_executed": False,
            }

    def _path(self, plugin_id: str) -> Path:
        if not plugin_id or Path(plugin_id).name != plugin_id or plugin_id in {".", ".."}:
            raise ScissorsError("PERMISSION_DENIED", "Invalid plugin storage path")
        candidate = self.root / plugin_id
        if candidate.is_symlink():
            raise ScissorsError("PERMISSION_DENIED", "Plugin storage must not be a symlink")
        path = contained(self.root, plugin_id)
        if not path.is_relative_to(self.root) or path == self.root:
            raise ScissorsError("PERMISSION_DENIED", "Invalid plugin storage path")
        return path

    def _manifest(self, plugin_id: str) -> PluginManifest:
        """Read the current manifest of an installed package without executing code."""
        if plugin_id not in self.engine.plugins or not self._path(plugin_id).is_dir():
            raise ScissorsError("PLUGIN_NOT_FOUND", "Plugin is not installed")
        return read_manifest(self._path(plugin_id), self.engine.settings.max_file_bytes)

    def install(self, path: str | Path) -> dict[str, Any]:
        """Local operator API; never exposed through HTTP/MCP. Enables the package."""
        with self.engine.mutex, self._locked():
            if self.engine.closed:
                raise ScissorsError("WORKSPACE_CLOSED", "Scissors has been closed")
            source = Path(path).resolve()
            manifest = read_manifest(source, self.engine.settings.max_file_bytes)
            if manifest.id in self.engine.plugins or self._path(manifest.id).exists():
                raise ScissorsError("PLUGIN_COLLISION", "Plugin is already installed")
            target = self._path(manifest.id)
            try:
                target.mkdir()
                for item in files(source, self.engine.settings.max_file_bytes):
                    destination = target / item.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item, destination)
                self.engine.plugins[manifest.id] = {
                    "enabled": True,
                    "version": manifest.version,
                    "trust": "external",
                }
                self._enable(manifest.id)
                self.engine.commit()
            except Exception:
                self._unregister(manifest.id)
                self.engine.plugins.pop(manifest.id, None)
                shutil.rmtree(target)
                raise
            return self.describe(manifest.id)

    def _unregister(self, plugin_id: str) -> None:
        for name, op in list(self.registry.operations.items()):
            if op.plugin == plugin_id:
                self.registry.unregister(name)

    def _executor(self, manifest: PluginManifest, operation: PluginOperation) -> Any:
        def execute(**arguments: Any) -> dict[str, Any]:
            if manifest.id not in self.engine.plugins:
                raise ScissorsError("PLUGIN_NOT_FOUND", "Plugin is not installed")
            # Only identified objects/scenes are transferred to the child.
            ids = arguments.get("object_ids", []) + (
                [arguments["object_id"]] if "object_id" in arguments else []
            )
            meshes = {oid: self.engine.object(oid)[1].model_dump() for oid in ids}
            scene_ids = [arguments["scene_id"]] if "scene_id" in arguments else []
            if scene_ids:
                self.engine.scene(scene_ids[0])
            writable = operation.side_effects and "scene.write" in manifest.permissions
            with self._locked():
                self._manifest(manifest.id)
                response = run_worker(
                    self._path(manifest.id),
                    {
                        "operation": operation.name,
                        "arguments": arguments,
                        "meshes": meshes,
                        "scene_ids": scene_ids,
                        "writable": writable,
                    },
                    self.engine.settings,
                    self.engine.root,
                )
            created = []
            if response["changes"] and not writable:
                raise ScissorsError(
                    "PERMISSION_DENIED", "Read-only plugin attempted to modify geometry"
                )
            for change in response["changes"]:
                if change["kind"] == "replace" and change["object_id"] in meshes:
                    _, obj = self.engine.object(change["object_id"])
                    candidate = MeshObject(
                        id=obj.id,
                        name=obj.name,
                        vertices=change["vertices"],
                        faces=change["faces"],
                        uv=change.get("uv"),
                    )
                    self.engine.checked(candidate)
                    obj.replace(candidate.geometry(), candidate.uv)
                elif change["kind"] == "create" and change["scene_id"] in scene_ids:
                    candidate = MeshObject(
                        id="candidate",
                        name=change["name"],
                        vertices=change["vertices"],
                        faces=change["faces"],
                    )
                    self.engine.checked(candidate)
                    created.append(
                        self.engine.add_mesh(
                            change["scene_id"], candidate.geometry(), change["name"]
                        )
                    )
                else:
                    raise ScissorsError("PERMISSION_DENIED", "Invalid plugin change target")
            result = response["result"]
            if created:
                result["created_objects"] = created
            return result

        return execute

    def _enable(self, plugin_id: str) -> dict[str, Any]:
        with self._locked():
            manifest = self._manifest(plugin_id)
            run_worker(
                self._path(plugin_id), {"probe": True}, self.engine.settings, self.engine.root
            )
            existing = {
                name for name, op in self.registry.operations.items() if op.plugin != plugin_id
            }
            if any(op.name in existing for op in manifest.operations):
                raise ScissorsError(
                    "OPERATION_COLLISION", "Plugin conflicts with registered operations"
                )
            operations = [
                Operation(
                    op.name,
                    op.description,
                    self._executor(manifest, op),
                    op.input_schema,
                    version=manifest.version,
                    result_schema=op.output_schema,
                    examples=op.examples,
                    category=op.name.split(".")[0],
                    plugin=plugin_id,
                    side_effects=op.side_effects,
                    permissions=list(manifest.permissions),
                )
                for op in manifest.operations
            ]
            self._unregister(plugin_id)
            for op in operations:
                self.registry.register(op)
            self.engine.plugins[plugin_id]["enabled"] = True
            self.errors.pop(plugin_id, None)
            return {
                "plugin_id": plugin_id,
                "enabled": True,
                "operations": [op.name for op in operations],
            }

    def enable(self, plugin_id: str) -> dict[str, Any]:
        return self._invoke("enable", plugin_id)

    def disable(self, plugin_id: str) -> dict[str, Any]:
        return self._invoke("disable", plugin_id)

    def _disable(self, plugin_id: str) -> dict[str, Any]:
        if plugin_id not in self.engine.plugins:
            raise ScissorsError("PLUGIN_NOT_FOUND", "Plugin is not installed")
        self._unregister(plugin_id)
        self.engine.plugins[plugin_id]["enabled"] = False
        return {"plugin_id": plugin_id, "enabled": False}

    def reload(self, plugin_id: str) -> dict[str, Any]:
        return self._invoke("reload", plugin_id)

    def _invoke(self, action: str, plugin_id: str) -> dict[str, Any]:
        response = self.registry.execute(f"plugins.{action}", {"plugin_id": plugin_id})
        if not response["success"]:
            error = response["error"]
            raise ScissorsError(error["code"], error["message"], error["details"])
        return response["result"]

    def uninstall(self, plugin_id: str) -> dict[str, Any]:
        with self.engine.mutex, self._locked():
            if self.engine.closed:
                raise ScissorsError("WORKSPACE_CLOSED", "Scissors has been closed")
            if plugin_id not in self.engine.plugins or not self._path(plugin_id).is_dir():
                raise ScissorsError("PLUGIN_NOT_FOUND", "Plugin is not installed")
            self._disable(plugin_id)
            del self.engine.plugins[plugin_id]
            self.engine.commit()
            shutil.rmtree(self._path(plugin_id))
            return {"uninstalled": plugin_id}

    def test(self, plugin_id: str) -> dict[str, Any]:
        with self._locked():
            self._manifest(plugin_id)
            path = self._path(plugin_id)
            if not (path / "tests").is_dir():
                raise ScissorsError("PLUGIN_INVALID", "No tests directory")
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                code = limited_process(
                    [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                    path,
                    self.engine.settings,
                    out,
                    err,
                )
                out.seek(0)
                err.seek(0)
                return {
                    "passed": code == 0,
                    "exit_code": code,
                    "stdout": out.read(16000).decode("utf-8", errors="replace"),
                    "stderr": err.read(16000).decode("utf-8", errors="replace"),
                }
