import hashlib
import importlib.metadata
from pathlib import Path

from packaging.requirements import Requirement

from scissors.api.errors import ScissorsError
from scissors.core.persistence import read_json
from scissors.plugins.manifest import PluginManifest


def files(path: Path, limit: int) -> list[Path]:
    values = []
    size = 0
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ScissorsError("PLUGIN_INVALID", "Plugin symlinks are prohibited")
        if any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
            for part in item.relative_to(path).parts
        ):
            continue
        if item.is_file():
            size += item.stat().st_size
            values.append(item)
        if size > limit or len(values) > 1000:
            raise ScissorsError("RESOURCE_LIMIT", "Plugin package exceeds limits")
    return values


def fingerprint(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    for item in files(path, limit):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def read_manifest(path: Path, limit: int) -> PluginManifest:
    if not path.is_dir() or path.is_symlink():
        raise ScissorsError("PLUGIN_INVALID", "Plugin directory is missing or a symlink")
    files(path, limit)
    manifest = PluginManifest.model_validate(read_json(path / "manifest.json", limit))
    if not (path / manifest.entrypoint).is_file():
        raise ScissorsError("PLUGIN_INVALID", "Plugin entrypoint is missing")
    for dependency in manifest.dependencies:
        requirement = Requirement(dependency)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ScissorsError(
                "PLUGIN_DEPENDENCY",
                "Declared dependency is not installed",
                {"dependency": requirement.name},
            ) from exc
        if version not in requirement.specifier:
            raise ScissorsError(
                "PLUGIN_DEPENDENCY",
                "Dependency version is incompatible",
                {"dependency": requirement.name},
            )
    return manifest
