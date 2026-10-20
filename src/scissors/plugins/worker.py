"""Process entry point, never imported to load external plugin code in the server."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from scissors.core.persistence import json_bytes, read_json
from scissors.plugins.context import PluginContext
from scissors.plugins.manifest import PluginManifest


def main() -> None:
    path, request, result = map(Path, sys.argv[1:4])
    memory_limit, file_limit = map(int, sys.argv[4:6])
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    response: dict[str, Any]
    try:
        manifest = PluginManifest.model_validate(read_json(path / "manifest.json", file_limit))
        payload = read_json(request, file_limit)
        spec = importlib.util.spec_from_file_location(
            "scissors_external_plugin", path / manifest.entrypoint
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load entrypoint")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operations = module.OPERATIONS
        if set(operations) != {op.name for op in manifest.operations} or not all(
            callable(fn) for fn in operations.values()
        ):
            raise ValueError("OPERATIONS must match the manifest exactly")
        if payload.get("probe"):
            response = {"success": True, "operations": list(operations)}
        else:
            context = PluginContext(payload["meshes"], payload["scene_ids"], payload["writable"])
            value = operations[payload["operation"]](context, **payload["arguments"])
            response = {"success": True, "result": value, "changes": context.changes}
    except BaseException as exc:
        response = {
            "success": False,
            "error": {"exception_type": type(exc).__name__, "message": str(exc)[:1000]},
        }
    data = json_bytes(response)
    if len(data) > file_limit:
        raise ValueError("Plugin result exceeds limit")
    result.write_bytes(data)


if __name__ == "__main__":
    main()
