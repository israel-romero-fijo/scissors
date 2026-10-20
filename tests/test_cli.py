import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_cli_help_info_catalog(tmp_path):
    base = [sys.executable, "-m", "scissors.cli", "--workspace", str(tmp_path / "workspace")]
    help_result = subprocess.run([*base, "--help"], capture_output=True, text=True, check=True)
    assert "mcp" in help_result.stdout and "plugins" in help_result.stdout
    info = subprocess.run([*base, "info"], capture_output=True, text=True, check=True)
    assert json.loads(info.stdout)["api_version"] == "1.0"
    description = subprocess.run(
        [*base, "operations", "describe", "mesh.create_cube"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scene_id" in json.loads(description.stdout)["parameters"]["required"]


def test_cli_rejects_removed_plugin_flags(tmp_path):
    base = [sys.executable, "-m", "scissors.cli", "--workspace", str(tmp_path / "workspace")]
    help_result = subprocess.run([*base, "--help"], capture_output=True, text=True, check=True)
    assert "--trusted-development" not in help_result.stdout
    removed = subprocess.run(
        [*base, "--trusted-development", "info"], capture_output=True, text=True
    )
    assert removed.returncode != 0
    install_help = subprocess.run(
        [*base, "plugins", "install", "--help"], capture_output=True, text=True, check=True
    )
    assert "--authorize" not in install_help.stdout
    assert "--enable" not in install_help.stdout


def test_cli_plugin_lifecycle(tmp_path, twist_path):
    base = [sys.executable, "-m", "scissors.cli", "--workspace", str(tmp_path / "plugins")]

    def run(*arguments):
        response = subprocess.run([*base, *arguments], capture_output=True, text=True, check=True)
        return json.loads(response.stdout)

    assert run("plugins", "validate", str(twist_path))["valid"]
    assert run("plugins", "install", str(twist_path))["state"]["enabled"] is True
    assert run("plugins", "test", "twist_mesh")["passed"]
    assert run("operations", "describe", "deform.twist_mesh")["plugin"] == "twist_mesh"
    assert run("plugins", "disable", "twist_mesh")["success"]
    shared = subprocess.run(
        [
            sys.executable,
            "-m",
            "scissors.cli",
            "--workspace",
            str(tmp_path / "other"),
            "operations",
            "describe",
            "deform.twist_mesh",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(shared.stdout)["plugin"] == "twist_mesh"
    assert run("plugins", "reload", "twist_mesh")["success"]
    assert run("plugins", "uninstall", "twist_mesh")["uninstalled"] == "twist_mesh"


def test_cli_plugin_store_override(tmp_path):
    store = tmp_path / "explicit_store"
    response = subprocess.run(
        [
            sys.executable,
            "-m",
            "scissors.cli",
            "--workspace",
            str(tmp_path / "workspace"),
            "--plugins-dir",
            str(store),
            "info",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(response.stdout)["plugins_dir"] == str(store.resolve())


def test_default_workspace_matches_launchers(tmp_path):
    from scissors.cli import parser
    from scissors.config import Settings

    expected = Path("workspaces/agents")
    assert Settings().workspace == expected
    assert parser().parse_args(["info"]).workspace == expected
    response = subprocess.run(
        [sys.executable, "-m", "scissors.cli", "info"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(json.loads(response.stdout)["workspace"]) == tmp_path / expected
    assert (tmp_path / expected).is_dir()
    assert not (tmp_path / "workspaces/default").exists()
