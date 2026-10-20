import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scissors.api.errors import ScissorsError


def contained(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ScissorsError("PERMISSION_DENIED", "Path must be inside the workspace")
    return path


def atomic_write(path: Path, data: bytes, limit: int) -> None:
    if len(data) > limit:
        raise ScissorsError("RESOURCE_LIMIT", "File exceeds configured size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".scissors-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Directory sync strengthens durability where the filesystem supports it.
        # The rename is already committed, so unsupported directory sync is not a write failure.
        if os.name == "posix":
            directory = None
            try:
                directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                os.fsync(directory)
            except OSError:
                pass
            finally:
                if directory is not None:
                    try:
                        os.close(directory)
                    except OSError:
                        pass
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: Path, limit: int) -> Any:
    if path.stat().st_size > limit:
        raise ScissorsError("RESOURCE_LIMIT", "File exceeds configured size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
