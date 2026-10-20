# Python and MCP examples

These runnable examples show how to create, edit and export models with Scissors.
Start with `create_cube.py`, then try `full_workflow.py`.

Run commands from the repository root after installing the environment with `uv sync`.
Use `python -m uv` if `uv` is not on PATH.

## Core workflows

| Example | Command | What it demonstrates |
| --- | --- | --- |
| Basic model | `uv run python examples/create_cube.py` | Create a cube, translate it and export a GLB. |
| Complete workflow | `uv run python examples/full_workflow.py` | Add a material and checker texture, validate geometry, save the scene, generate a preview and reopen the exported files. |
| Plugin | `uv run python examples/create_with_plugin.py` | Install or enable `twist_mesh`, deform a cylinder, validate it and export a GLB and preview. |
| MCP client | `uv run python examples/mcp_client.py` | Start a stdio server, discover tools, create a cube and export it through MCP. |

## Specialized workflows

| Example | Command | What it demonstrates |
| --- | --- | --- |
| Organic surface | `uv run python examples/anatomical_workflow.py` | Combine head and muzzle volumes, deform and sculpt the surface, simplify it, apply UVs and texture, and export a preview and GLB. |
| Portable package | `uv run python examples/export_bundle.py` | Extrude a hollow frame and export a ZIP with a GLB, editable scene and camera preview. |
| Loft with preserved corners | `uv run python examples/preserved_loft.py` | Join profiles into a tapered body while preserving their corner vertices and report the intersection check. |

## Workspaces and outputs

Each persistent example uses its own workspace so its output stays separate from
`workspaces/agents`:

| Script | Workspace |
| --- | --- |
| `create_cube.py` | `workspaces/cube` |
| `full_workflow.py` | `workspaces/full_workflow` |
| `create_with_plugin.py` | `workspaces/twist_example` |
| `mcp_client.py` | `workspaces/mcp_example`, relative to the repository |
| `anatomical_workflow.py` | `workspaces/anatomy`; accepts `--workspace PATH` |
| `export_bundle.py` | `workspaces/bundle-example` |
| `preserved_loft.py` | A temporary workspace, removed when the script finishes |

The scripts print their results or output paths. Repeated runs add scenes to the
persistent workspaces. Only one process can own a workspace at a time.

The plugin example installs its package in the shared plugin store if needed and
activates it in its workspace. Set `SCISSORS_PLUGINS_DIR` to a separate directory
before running it if you want to use a separate plugin store.

See the [API reference](../docs/api.md), [plugin guide](../docs/plugins.md) and
[MCP guide](../docs/mcp.md) for the supported interfaces and configuration.
