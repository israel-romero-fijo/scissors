"""Workspace format 2: atomic JSON manifest and immutable content-addressed NumPy arrays."""

import hashlib
import io
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.persistence import atomic_write, contained, json_bytes, read_json
from scissors.core.scene import Scene

FIELDS = {
    "vertices": ("<f8", 3),
    "faces": ("<i8", 3),
    "normals": ("<f8", 3),
    "uv": ("<f8", 2),
    "vertex_colors": ("|u1", 4),
}
FileCache = dict[str, tuple[int, int]]


def geometry_path(root: Path, relative: str) -> Path:
    path = contained(root, relative)
    if path.parent != root / ".geometry" or path.name != Path(relative).name:
        raise ScissorsError(
            "PERMISSION_DENIED", "Geometry files must stay in the private geometry directory"
        )
    return path


def signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_mtime_ns, info.st_size


def fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in FIELDS:
        if name not in arrays:
            digest.update(f"{name}:none;".encode())
        else:
            array = arrays[name]
            digest.update(
                json_bytes({"name": name, "dtype": array.dtype.str, "shape": array.shape})
            )
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def encoded_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            data = io.BytesIO()
            np.lib.format.write_array(data, array, version=(1, 0), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data.getvalue())
    return buffer.getvalue()


def descriptor(
    reference: Any, settings: Settings
) -> tuple[dict[str, tuple[str, tuple[int, int]]], int]:
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "array_sha256", "arrays"}
        or not isinstance(reference["array_sha256"], str)
        or re.fullmatch("[0-9a-f]{64}", reference["array_sha256"]) is None
        or reference["path"] != f".geometry/{reference['array_sha256']}.npz"
    ):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid geometry reference")
    table = reference["arrays"]
    if (
        not isinstance(table, dict)
        or not {"vertices", "faces", "normals"} <= set(table)
        or set(table) - set(FIELDS)
    ):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid geometry array table")
    result = {}
    for name, value in table.items():
        dtype, columns = FIELDS[name]
        if (
            not isinstance(value, dict)
            or set(value) != {"dtype", "shape"}
            or value["dtype"] != dtype
            or not isinstance(value["shape"], list)
            or len(value["shape"]) != 2
            or any(type(count) is not int or count < 0 for count in value["shape"])
            or value["shape"][1] != columns
        ):
            raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid geometry dtype or shape")
        result[name] = (dtype, (value["shape"][0], columns))
    count = result["vertices"][1][0]
    face_count = result["faces"][1][0]
    if count > settings.max_vertices or face_count > settings.max_faces:
        raise ScissorsError("RESOURCE_LIMIT", "Persisted geometry exceeds configured counts")
    for name in result:
        allowed = {0, count} if name == "normals" else {count}
        if name not in {"vertices", "faces"} and result[name][1][0] not in allowed:
            raise ScissorsError(
                "INCOMPATIBLE_FORMAT", "Persisted vertex attributes have mismatched counts"
            )
    estimate = count * 768 + face_count * 384
    if estimate > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Persisted geometry exceeds decode memory budget")
    return result, estimate


def decode(
    root: Path, reference: dict[str, Any], settings: Settings, known: FileCache
) -> dict[str, Any]:
    expected, _ = descriptor(reference, settings)
    path = geometry_path(root, reference["path"])
    if not path.is_file():
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Persisted geometry file is missing")
    if path.stat().st_size > settings.max_file_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Persisted geometry file exceeds byte limit")
    original_signature = signature(path)
    arrays = {}
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {f"{name}.npy" for name in expected}
            if len(entries) != len(names) or {entry.filename for entry in entries} != names:
                raise ValueError("Geometry archive has duplicate or unexpected members")
            expected_bytes = sum(
                int(np.prod(shape)) * np.dtype(dtype).itemsize for dtype, shape in expected.values()
            )
            if sum(entry.file_size for entry in entries) > expected_bytes + len(entries) * 1024:
                raise ValueError("Geometry archive expands beyond declared array sizes")
            for name, (dtype, shape) in expected.items():
                entry = archive.getinfo(f"{name}.npy")
                if entry.flag_bits & 1:
                    raise ValueError("Encrypted geometry arrays are unsupported")
                with archive.open(entry) as stream:
                    if np.lib.format.read_magic(stream) != (1, 0):
                        raise ValueError("Unsupported NumPy array version")
                    parsed_shape, fortran, parsed_dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=1024
                    )
                    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    if (
                        parsed_shape != shape
                        or fortran
                        or parsed_dtype.hasobject
                        or parsed_dtype.str != dtype
                        or entry.file_size != stream.tell() + size
                    ):
                        raise ValueError("Geometry array header differs from its manifest")
                    raw = stream.read(size + 1)
                    if len(raw) != size:
                        raise ValueError("Geometry array payload has the wrong size")
                    array = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
                    if array.dtype.kind == "f" and not np.isfinite(array).all():
                        raise ValueError("Geometry contains nonfinite numeric values")
                    arrays[name] = array
        if fingerprint(arrays) != reference["array_sha256"]:
            raise ValueError("Geometry array checksum mismatch")
        if signature(path) != original_signature:
            raise ValueError("Geometry file changed while being read")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ScissorsError(
            "INCOMPATIBLE_FORMAT", "Invalid binary geometry", {"reason": str(exc)[:300]}
        ) from exc
    known[reference["array_sha256"]] = original_signature
    return {name: arrays[name].tolist() if name in arrays else None for name in FIELDS}


def read_workspace(root: Path, settings: Settings, known: FileCache) -> dict[str, Any]:
    try:
        state = read_json(root / "state.json", settings.max_file_bytes)
    except ValueError as exc:
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid workspace JSON") from exc
    if (
        not isinstance(state, dict)
        or type(state.get("format_version")) is not int
        or state.get("format_version") not in {1, 2}
    ):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Unsupported workspace version")
    if state["format_version"] == 1:
        if not all(isinstance(state.get(key), dict) for key in ["scenes", "textures", "plugins"]):
            raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid legacy workspace tables")
        return state
    if set(state) != {"format_version", "scenes", "textures", "plugins"} or not all(
        isinstance(state[key], dict) for key in ["scenes", "textures", "plugins"]
    ):
        raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid workspace manifest")
    references = []
    estimate = 0
    for scene in state["scenes"].values():
        if not isinstance(scene, dict) or not isinstance(scene.get("objects"), dict):
            raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid scene manifest")
        for obj in scene["objects"].values():
            if not isinstance(obj, dict):
                raise ScissorsError("INCOMPATIBLE_FORMAT", "Invalid object manifest")
            if obj.get("kind", "mesh") == "mesh":
                if set(FIELDS) & set(obj) or "geometry" not in obj:
                    raise ScissorsError(
                        "INCOMPATIBLE_FORMAT", "Mesh arrays must use one binary reference"
                    )
                _, amount = descriptor(obj["geometry"], settings)
                estimate += amount
                references.append(obj)
            elif "geometry" in obj:
                raise ScissorsError(
                    "INCOMPATIBLE_FORMAT", "Empty node cannot reference binary geometry"
                )
    if estimate > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Workspace binary decode exceeds memory budget")
    for obj in references:
        reference = obj.pop("geometry")
        obj.update(decode(root, reference, settings, known))
    state["format_version"] = 1
    return state


def write_workspace(
    root: Path,
    settings: Settings,
    scenes: dict[str, Scene],
    textures: dict[str, Any],
    plugins: dict[str, Any],
    known: FileCache,
) -> None:
    manifest: dict[str, Any] = {
        "format_version": 2,
        "scenes": {},
        "textures": textures,
        "plugins": plugins,
    }
    estimate = sum(
        len(obj.vertices) * 768 + len(obj.faces) * 384
        for scene in scenes.values()
        for obj in scene.objects.values()
        if obj.kind == "mesh"
    )
    if estimate > settings.max_memory_bytes:
        raise ScissorsError("RESOURCE_LIMIT", "Workspace exceeds binary reload memory budget")
    for sid, scene in scenes.items():
        data = scene.model_dump(exclude={"objects"})
        data["objects"] = {}
        for oid, obj in scene.objects.items():
            if obj.kind != "mesh":
                data["objects"][oid] = obj.model_dump()
                continue
            arrays = {}
            for name, (dtype, columns) in FIELDS.items():
                values = getattr(obj, name)
                if values is not None:
                    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype))).reshape(
                        (-1, columns)
                    )
                    if array.dtype.kind == "f" and not np.isfinite(array).all():
                        raise ScissorsError(
                            "GEOMETRY_ERROR", "Cannot persist nonfinite geometry attributes"
                        )
                    arrays[name] = array
            digest = fingerprint(arrays)
            relative = f".geometry/{digest}.npz"
            path = geometry_path(root, relative)
            if not path.is_file() or known.get(digest) != signature(path):
                atomic_write(path, encoded_arrays(arrays), settings.max_file_bytes)
                known[digest] = signature(path)
            record = obj.model_dump(exclude=set(FIELDS))
            record["geometry"] = {
                "path": relative,
                "array_sha256": digest,
                "arrays": {
                    name: {"dtype": array.dtype.str, "shape": list(array.shape)}
                    for name, array in arrays.items()
                },
            }
            descriptor(record["geometry"], settings)
            data["objects"][oid] = record
        manifest["scenes"][sid] = data
    # Geometry files are fully published before the only mutable root pointer changes.
    atomic_write(root / "state.json", json_bytes(manifest), settings.max_file_bytes)
