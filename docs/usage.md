# Startup options and troubleshooting

Install Scissors and start it using the [README](../README.md).
Run the commands below from the repository folder.
If `uv` is not on PATH, replace it with `python -m uv`.

## Choose a workspace or port

A **workspace** is the folder that holds scenes, textures, exported files and
plugin activation settings. Plugin packages live in a shared application folder:
`~/.scissors/plugins` by default. Set `SCISSORS_PLUGINS_DIR` before starting a launcher
to select another shared folder. Direct CLI calls also accept `--plugins-dir` before the
command. The launchers, CLI and Python API use `workspaces/agents` by default.
Direct CLI and Python calls resolve relative paths from the current directory;
the launchers resolve them from the repository. Use an absolute `--workspace`
path to select the same folder from another directory.
Existing workspaces are not moved or merged automatically.

Windows:

```powershell
.\start_scissors.bat --port 9000 --workspace "workspaces/my model"
```

Linux or macOS:

```bash
bash start_scissors.sh --port 9000 --workspace "workspaces/my model"
```

Connect your agent to `http://127.0.0.1:9000/mcp` when using port 9000.
Use the launcher's `--help` option to see all startup options.

The launchers require Python 3.12+ and uv. They prepare the environment from the
lockfile when needed, run in the foreground and return the application's exit status.

## Start without a launcher

```bash
uv run scissors --workspace workspaces/agents serve --host 127.0.0.1 --port 8765
```

This command uses `workspaces/agents`. It serves both the HTTP API and MCP.
`scissors mcp http` starts the same service.
The API reference is at `http://127.0.0.1:8765/docs`.
The MCP connection address is `http://127.0.0.1:8765/mcp`.
Press Ctrl+C to stop it.

For a client that starts its own process, the launchers also support
`--transport stdio`. See [MCP integration](mcp.md) for configuration.
Stdio launchers keep stdout reserved for the protocol.

## Common problems

- **Python is not found:** install Python 3.12 or newer and add it to PATH.
  Reopen your terminal after installation.
- **uv is not found:** run `python -m pip install uv`.
  You can use `python -m uv` for all uv commands.
- **The port is already in use:** choose another port with `--port 9000` and
  update your agent's connection address.
- **The workspace is busy:** another Scissors process owns that folder.
  Connect to its running service, stop that process, or choose another workspace.
  Never delete a running workspace's lock file.
- **A modeling backend is unavailable:** stop processes using the installation,
  run `uv sync`, then restart them. The standard installation includes all
  geometry backends. Native packages need a compatible platform and Python version.
  See [capabilities](capabilities.md) for backend details.

Move or back up the whole workspace to keep scenes and their textures together.
Only one process can own a workspace at a time. If the HTTP service owns it,
connect clients to that service instead of starting a stdio process for the same folder.

## Network access

The default address, `127.0.0.1`, accepts local connections.
Both launchers forward `SCISSORS_TOKEN` from the environment.
A non-loopback `--host` requires this token. Use a TLS proxy for network access.
See [security boundaries](security.md) before configuring remote access.
