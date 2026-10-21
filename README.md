# Scissors

Scissors lets your AI agent create and edit 3D models on your computer.
It can work with shapes, materials and textures, then export models to GLB, glTF,
OBJ or STL files.
It also imports FBX files directly, including static meshes, materials and embedded
textures. Binary FBX rigs and animation can be retained for complete FBX export.
FBX axes and units are converted automatically for the bind-pose view; see [importing assets](docs/api.md#imported-assets-and-image-planes).

Scissors has no graphical modeling window. You use it through a connected AI
agent or scripts. It runs locally on your CPU and does not require Blender,
a GPU or an API key of its own.

## Install

1. Install **Python 3.12 or newer** from [python.org](https://www.python.org/downloads/).
   Make sure Python is available in your terminal. On Windows, enable the installer
   option to add Python to PATH.
2. Download or clone this repository. Open a terminal in the Scissors folder.
3. Install the setup tool and the application:

   ```bash
   python -m pip install uv
   python -m uv sync
   ```

The first installation downloads the required packages. All built-in modeling
features are included.

To check that installation worked:

```bash
python -m uv run scissors info
```

## Start Scissors

On **Windows**, double-click `start_scissors.bat` in the Scissors folder.
You can also start it from PowerShell:

```powershell
.\start_scissors.bat
```

On **Linux or macOS**, run this command from the Scissors folder:

```bash
bash start_scissors.sh
```

Keep the terminal open while you use Scissors. Press **Ctrl+C** to stop it.

Once it is running:

- Connect your AI agent to `http://127.0.0.1:8765/mcp`.
- Open `http://127.0.0.1:8765/docs` in your browser to inspect the available tools.
  This page is an API reference.
- Your work is saved in `workspaces/agents`. A **workspace** is the folder that
  holds your scenes, textures and exported files.
- Plugin packages are shared by all workspaces in `~/.scissors/plugins`. Set
  `SCISSORS_PLUGINS_DIR` to choose another application plugin store. Activation
  settings belong to each workspace. See [plugins](docs/plugins.md).

For client setup, follow [Connect an AI agent](docs/mcp.md).
That guide also explains how to let the agent start Scissors automatically.

## Try an example

To create a textured cube without setting up an AI agent, run:

```bash
python -m uv run python examples/full_workflow.py
```

Find the model at `workspaces/full_workflow/exports/model.glb` and its preview at
`workspaces/full_workflow/previews/model.png`. Open the GLB file in a compatible
3D viewer. The preview shows shape and solid colors; it does not display textures.

See [runnable examples](examples/README.md) for Python, plugins, MCP and specialized modeling workflows.

## More help

- [Startup options and troubleshooting](docs/usage.md)
- [Connect an AI agent](docs/mcp.md)
- [Python and HTTP API](docs/api.md)
- [Add tools with plugins](docs/plugins.md)
- [Modeling features and limitations](docs/capabilities.md)
- [Advanced surface modeling](docs/anatomical-modeling.md)
- [Development and checks](docs/development.md)
- [Architecture](docs/architecture.md) and [security](docs/security.md)
