import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from scissors.api.errors import ScissorsError
from scissors.config import Settings
from scissors.core.persistence import contained, json_bytes, read_json


def limited_process(
    command: list[str], cwd: Path, settings: Settings, stdout: Any, stderr: Any
) -> int:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "LANG"}
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        env=environment,
        creationflags=flags,
    )
    start = time.monotonic()
    try:
        while process.poll() is None:
            if time.monotonic() - start > settings.operation_timeout:
                raise ScissorsError("OPERATION_TIMEOUT", "Child operation timed out")
            if any(
                os.fstat(stream.fileno()).st_size > settings.max_file_bytes
                for stream in (stdout, stderr)
            ):
                raise ScissorsError("RESOURCE_LIMIT", "Child output exceeds size limit")
            time.sleep(0.02)
        return int(process.returncode)
    finally:
        if process.poll() is None:
            if os.name == "nt":
                # Windows venv Python can redirect to an interpreter child. Killing
                # only the redirector leaves that child and its file handles alive.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
            if process.poll() is None:
                process.kill()
        process.wait()


def run_worker(
    path: Path, payload: dict[str, Any], settings: Settings, root: Path
) -> dict[str, Any]:
    runtime = contained(root, ".runtime")
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        folder = Path(temporary)
        request, result = folder / "request.json", folder / "result.json"
        data = json_bytes(payload)
        if len(data) > settings.max_file_bytes:
            raise ScissorsError("RESOURCE_LIMIT", "Worker payload exceeds limit")
        request.write_bytes(data)
        with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
            code = limited_process(
                [
                    sys.executable,
                    "-m",
                    "scissors.plugins.worker",
                    str(path),
                    str(request),
                    str(result),
                    str(settings.max_memory_bytes),
                    str(settings.max_file_bytes),
                ],
                path,
                settings,
                out,
                err,
            )
        if code or not result.exists():
            raise ScissorsError(
                "PLUGIN_FAILED", "Plugin worker exited without a result", {"exit_code": code}
            )
        response = read_json(result, settings.max_file_bytes)
        if not response["success"]:
            raise ScissorsError("PLUGIN_FAILED", "Plugin raised an exception", response["error"])
        return response
