import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "scissors.cli",
            "--workspace",
            str(repository / "workspaces/mcp_example"),
            "mcp",
            "stdio",
        ],
        cwd=str(repository),
    )
    async with Client(parameters) as client:
        tools = await client.list_tools()
        print(json.dumps({"tools": [tool.name for tool in tools.tools]}))
        scene = await client.call_tool("scene_create", {"name": "MCP example"})
        scene_id = scene.structured_content["result"]["id"]
        cube = await client.call_tool(
            "operations_invoke",
            {"name": "mesh.create_cube", "arguments": {"scene_id": scene_id, "size": 2}},
        )
        assert not cube.is_error
        export = await client.call_tool(
            "scene_export", {"scene_id": scene_id, "output": "exports/from_mcp.glb"}
        )
        assert Path(export.structured_content["result"]["path"]).stat().st_size > 0
        print(json.dumps(export.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
