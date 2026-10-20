import os
import socket
import subprocess
import sys

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from scissors.http.server import create_app
from tests.test_mcp import result


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "[::1]", "127.0.0.1:8765", "localhost:8765", "[::1]:8765"],
)
def test_mcp_loopback_initialization_with_and_without_port(app, host):
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "host-regression", "version": "1"},
        },
    }
    with TestClient(create_app(app, token="local-token"), base_url=f"http://{host}") as client:
        headers = {"Accept": "application/json, text/event-stream"}
        assert client.post("/mcp", headers=headers, json=body).status_code == 401
        headers["Authorization"] = "Bearer local-token"
        for origin in [None, f"http://{host}"]:
            request_headers = {**headers, **({"Origin": origin} if origin else {})}
            response = client.post("/mcp", headers=request_headers, json=body)
            assert response.status_code == 200, response.text
            assert '"serverInfo"' in response.text
        assert (
            client.post(
                "/mcp", headers={**headers, "Origin": "http://evil.test"}, json=body
            ).status_code
            == 403
        )
        for rejected_host in ["evil.test", "localhost.evil.test", "127.0.0.1.evil.test"]:
            assert (
                client.post(
                    "/mcp", headers={**headers, "Host": rejected_host}, json=body
                ).status_code
                == 403
            )


@pytest.mark.integration
async def test_real_streamable_http_shared_engine(tmp_path, twist_path):
    from scissors import Scissors

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workspace = tmp_path / "http-workspace"
    with Scissors(workspace) as app:
        app.plugins.install(twist_path)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (tmp_path / "server.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scissors.cli",
                "--workspace",
                str(workspace),
                "serve",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(base_url=url) as rest:
                with anyio.fail_after(20):
                    while True:
                        assert process.poll() is None
                        try:
                            if (await rest.get("/api/v1/info")).status_code == 200:
                                break
                        except httpx.ConnectError:
                            await anyio.sleep(0.1)
                async with Client(f"{url}/mcp") as client:
                    assert (await client.list_tools()).tools
                    scene = result(await client.call_tool("scene_create", {"name": "HTTP MCP"}))
                    assert (await rest.get(f"/api/v1/scenes/{scene['id']}")).json()[
                        "name"
                    ] == "HTTP MCP"
                    obj = (
                        await rest.post(
                            "/api/v1/operations/invoke",
                            json={
                                "name": "mesh.create_cube",
                                "arguments": {"scene_id": scene["id"]},
                            },
                        )
                    ).json()["result"]
                    inspected = result(
                        await client.call_tool("scene_inspect", {"scene_id": scene["id"]})
                    )
                    assert inspected["objects"][0]["id"] == obj["id"]
                    edited = result(
                        await client.call_tool(
                            "deform__local",
                            {
                                "object_id": obj["id"],
                                "center": [0.5, 0.5, 0.5],
                                "radius": 0.4,
                                "displacement": [0.05, 0, 0],
                            },
                        )
                    )
                    assert edited["affected_vertices"] == 1
                    rest_data = (
                        await rest.post(
                            "/api/v1/operations/invoke",
                            json={
                                "name": "mesh.inspect",
                                "arguments": {"object_id": obj["id"]},
                            },
                        )
                    ).json()["result"]
                    assert rest_data["bounds"][1][0] == 0.55
                    exported = result(
                        await client.call_tool(
                            "scene_export", {"scene_id": scene["id"], "output": "exports/http.glb"}
                        )
                    )
                    assert exported["size_bytes"] > 0
                    async with client.listen(tools_list_changed=True) as subscription:
                        enabled = await rest.post(
                            "/api/v1/operations/invoke",
                            json={
                                "name": "plugins.enable",
                                "arguments": {"plugin_id": "twist_mesh"},
                            },
                        )
                        assert enabled.json()["success"]
                        with anyio.fail_after(10):
                            event = await anext(subscription.__aiter__())
                        assert "ToolsListChanged" in type(event).__name__
                    tools = await client.list_tools(cache_mode="refresh")
                    assert "deform__twist_mesh" in {tool.name for tool in tools.tools}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
