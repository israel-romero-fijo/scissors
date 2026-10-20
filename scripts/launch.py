"""Shared startup configuration for Windows BAT and Bash launchers."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def uv_command() -> list[str]:
    executable = shutil.which("uv")
    if executable:
        return [executable]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    # An activated venv can hide the uv module installed in its base Python.
    if os.name == "nt":
        executable_path = Path(sys.base_prefix) / "Scripts" / "uv.exe"
        if executable_path.is_file():
            return [str(executable_path)]
    raise RuntimeError("uv is missing. Install it with: python -m pip install uv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Scissors for AI agents")
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    parser.add_argument(
        "--workspace",
        default="workspaces/agents",
        help="Workspace path; relative paths are resolved from the repository",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=port_number, default=8765)
    args = parser.parse_args()
    # Launchers run in the system interpreter before uv selects the project's Python.
    if sys.version_info < (3, 12):  # noqa: UP036
        print("Scissors requires Python 3.12 or newer.", file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parents[1]
    try:
        command = [*uv_command(), "run", "--frozen"]
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    command += ["scissors", "--workspace", args.workspace]
    if args.transport == "stdio":
        command += ["mcp", "stdio"]
    else:
        command += ["serve", "--host", args.host, "--port", str(args.port)]
        print(f"Scissors MCP: http://{args.host}:{args.port}/mcp", file=sys.stderr)
        print(f"Scissors API docs: http://{args.host}:{args.port}/docs", file=sys.stderr)
    # Keep stdin/stdout/stderr inherited for stdio MCP. POSIX exec propagates signals
    # directly; on Windows use a child process with the original console handles.
    if os.name != "nt":
        os.chdir(root)
        os.execvp(command[0], command)
    try:
        return subprocess.call(command, cwd=root)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
