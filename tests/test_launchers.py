import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from mcp import Client, StdioServerParameters

from tests.test_mcp import result

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def launcher(kind):
    if kind == "bat":
        if os.name != "nt":
            pytest.skip("Windows BAT launcher")
        return ["cmd", "/d", "/c", "call", str(ROOT / "start_scissors.bat")]
    if os.name == "nt":
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            pytest.skip("Git Bash unavailable")
        return [str(bash), str(ROOT / "start_scissors.sh")]
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    return [bash, str(ROOT / "start_scissors.sh")]


@pytest.mark.parametrize("kind", ["bat", "bash"])
def test_launcher_help_and_invalid_port(kind, tmp_path):
    command = launcher(kind)
    help_result = subprocess.run(
        [*command, "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=15
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--transport" in help_result.stdout and "--port" in help_result.stdout
    assert set(re.findall(r"--[a-z][a-z-]*", help_result.stdout)) == {
        "--help",
        "--transport",
        "--workspace",
        "--host",
        "--port",
    }
    invalid = subprocess.run(
        [*command, "--port", "70000"], cwd=tmp_path, capture_output=True, text=True, timeout=15
    )
    assert invalid.returncode == 2
    assert "between 1 and 65535" in invalid.stderr


@pytest.mark.parametrize("kind", ["bat", "bash"])
async def test_launcher_real_stdio(kind, tmp_path):
    command = launcher(kind)
    workspace = tmp_path / "workspace with spaces"
    parameters = StdioServerParameters(
        command=command[0],
        args=[*command[1:], "--transport", "stdio", "--workspace", str(workspace)],
        cwd=str(tmp_path),
        # Preserve uv runner settings, including no-sync for an environment in active use.
        env={
            key: os.environ[key]
            for key in ("UV_NO_SYNC", "UV_PROJECT_ENVIRONMENT", "SCISSORS_PLUGINS_DIR")
            if key in os.environ
        },
    )
    async with Client(parameters) as client:
        assert (await client.list_tools()).tools
        scene = result(await client.call_tool("scene_create", {"name": "Launcher test"}))
        assert scene["name"] == "Launcher test"
        info = result(await client.call_tool("modeler_info"))
        assert Path(info["workspace"]) == workspace
        assert Path(info["plugins_dir"]) == Path(os.environ["SCISSORS_PLUGINS_DIR"])
        assert all(info["backends"].values())
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {
            "mesh__boolean",
            "mesh__simplify",
            "mesh__organic_union",
            "mesh__remesh",
            "uv__unwrap",
        } <= names
    assert (workspace / "state.json").exists()


@pytest.mark.parametrize("kind", ["bat", "bash"])
def test_launcher_http_custom_port(kind, tmp_path):
    command = launcher(kind)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workspace = tmp_path / "http workspace with spaces"
    with (tmp_path / "launcher.log").open("wb") as log:
        process = subprocess.Popen(
            [*command, "--port", str(port), "--workspace", str(workspace)],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 20
            with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
                while True:
                    assert process.poll() is None
                    try:
                        response = client.get("/api/v1/info")
                        if response.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if time.monotonic() > deadline:
                        pytest.fail("Launcher HTTP service did not become ready")
                    time.sleep(0.1)
                assert Path(response.json()["workspace"]) == workspace
                assert all(response.json()["backends"].values())
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
