import json
import os
import sys
from pathlib import Path

import anyio
import pytest
import trimesh
from mcp import Client, StdioServerParameters

from scissors import Scissors

pytestmark = pytest.mark.integration


def test_legacy_http_initialization_advertises_tool_changes(app):
    from scissors.mcp.server import create_server

    options = create_server(app).create_initialization_options()
    assert options.capabilities.tools.list_changed is True


def parameters(workspace):
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "scissors.cli",
            "--workspace",
            str(workspace),
            "--plugins-dir",
            os.environ["SCISSORS_PLUGINS_DIR"],
            "mcp",
            "stdio",
        ],
    )


def result(response):
    assert not response.is_error, response.content
    value = response.structured_content or json.loads(response.content[0].text)
    assert value["success"]
    return value["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_real_stdio_client_export_and_resources(tmp_path, mode):
    async with Client(parameters(tmp_path / "workspace"), mode=mode) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        assert {
            "modeler_info",
            "operations_list",
            "operations_describe",
            "operations_invoke",
            "scene_create",
        } <= names
        capabilities = result(await client.call_tool("operations_list"))
        assert any(op["name"] == "mesh.create_cube" for op in capabilities["operations"])
        scene = result(await client.call_tool("scene_create", {"name": "MCP"}))
        cube = result(
            await client.call_tool(
                "operations_invoke",
                {"name": "mesh.create_cube", "arguments": {"scene_id": scene["id"]}},
            )
        )
        assert cube["face_count"] == 12
        sculpted = result(
            await client.call_tool(
                "operations_invoke",
                {
                    "name": "deform.local",
                    "arguments": {
                        "object_id": cube["id"],
                        "center": [0.5, 0.5, 0.5],
                        "radius": 0.4,
                        "displacement": [0.05, 0, 0],
                    },
                },
            )
        )
        assert sculpted["affected_vertices"] == 1
        inspection = result(await client.call_tool("scene_inspect", {"scene_id": scene["id"]}))
        assert inspection["objects"][0]["id"] == cube["id"]
        exported = result(
            await client.call_tool(
                "scene_export", {"scene_id": scene["id"], "output": "exports/test.glb"}
            )
        )
        assert Path(exported["path"]).stat().st_size > 0
        assert len(trimesh.load(exported["path"], force="scene").geometry) == 1
        resources = await client.list_resources()
        assert len(resources.resources) == 4
        assert await client.read_resource("scissors://info")


@pytest.mark.asyncio
async def test_dynamic_plugin_tools_notifications(tmp_path, twist_path):
    workspace = tmp_path / "workspace"
    with Scissors(workspace) as app:
        app.plugins.install(twist_path)
        app.plugins.disable("twist_mesh")
    async with Client(parameters(workspace)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert "deform__twist_mesh" not in names
        scene = result(await client.call_tool("scene_create", {"name": "Dynamic"}))
        obj = result(
            await client.call_tool(
                "operations_invoke",
                {"name": "mesh.create_cylinder", "arguments": {"scene_id": scene["id"]}},
            )
        )
        async with client.listen(tools_list_changed=True) as subscription:
            result(
                await client.call_tool(
                    "operations_invoke",
                    {"name": "plugins.enable", "arguments": {"plugin_id": "twist_mesh"}},
                )
            )
            with anyio.fail_after(10):
                event = await anext(subscription.__aiter__())
            assert "ToolsListChanged" in type(event).__name__
        names = {tool.name for tool in (await client.list_tools(cache_mode="refresh")).tools}
        assert "deform__twist_mesh" in names
        catalog = result(await client.call_tool("operations_list", {"category": "deform"}))
        operations = {operation["name"]: operation for operation in catalog["operations"]}
        assert operations["deform.twist_mesh"]["plugin"] == "twist_mesh"
        assert operations["deform.local"]["plugin"] is None
        twisted = result(
            await client.call_tool(
                "operations_invoke",
                {
                    "name": "deform.twist_mesh",
                    "arguments": {"object_id": obj["id"], "angle_degrees": 45},
                },
            )
        )
        assert twisted["vertex_count"] > 0
        result(await client.call_tool("deform__twist_mesh", {"object_id": obj["id"]}))
        result(
            await client.call_tool(
                "operations_invoke",
                {"name": "plugins.disable", "arguments": {"plugin_id": "twist_mesh"}},
            )
        )
        catalog = result(await client.call_tool("operations_list", {"category": "deform"}))
        assert {operation["name"] for operation in catalog["operations"]} == {
            "deform.local",
            "deform.warp",
        }
