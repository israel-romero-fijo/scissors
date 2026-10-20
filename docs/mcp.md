# MCP integration

Scissors uses the official MCP Python SDK 2.3.0. The server accepts
typed callbacks; input_schema/structured_content/is_error use SDK 2.x spelling. Protocol
negotiation supports modern discover/subscription and legacy initialize clients.

Run `uv run scissors --workspace workspaces/agents mcp stdio`; stdout is protocol-only.
For shared REST/MCP, run `scissors serve`, connect to http://127.0.0.1:8765/mcp, and do
not start another stdio owner of that workspace.

Plugin packages use the application plugin store shared by workspaces. To select a custom
store for stdio, pass `--plugins-dir /absolute/plugins` before `mcp stdio` in the server
arguments. Some clients filter inherited environment variables, so set the option explicitly
or configure `SCISSORS_PLUGINS_DIR` in the client's server environment. `modeler_info`
reports the resolved `plugins_dir`. See [plugins](plugins.md) for activation and storage.

## Codex

```bash
codex mcp add scissors -- uv --directory "/absolute/path/to/scissors" run scissors --workspace workspaces/agents mcp stdio
codex mcp list
```

TOML alternative in your own ~/.codex/config.toml:

```toml
[mcp_servers.scissors]
command = "uv"
args = ["--directory", "/absolute/path/to/scissors", "run", "scissors", "--workspace", "workspaces/agents", "mcp", "stdio"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

For HTTP replace command/args with `url="http://127.0.0.1:8765/mcp"`; optionally add
`bearer_token_env_var="SCISSORS_TOKEN"` when the server requires it. Windows TOML paths
can use literal strings with backslashes or forward slashes. If uv is only importable,
use command python and prepend `-m`, `uv` to args. Verify `/mcp` in Codex TUI.
No global configuration is modified automatically.
Source: [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

## Claude Code

### HTTP: connect to the running service

Start Scissors with the [startup launcher](usage.md). Run this in the project
where you use Claude Code:

```bash
claude mcp add --transport http scissors http://127.0.0.1:8765/mcp
claude mcp list
```

Open Claude Code and use `/mcp` to check the connection.
The default scope keeps this connection local to that project.
Add `--scope user` before `scissors` to use it across projects.

### Stdio: let Claude Code start Scissors

Use this instead of the HTTP configuration. Replace the repository path:

```bash
claude mcp add --transport stdio scissors -- uv --directory "/absolute/path/to/scissors" run scissors --workspace workspaces/agents mcp stdio
```

This uses `workspaces/agents`, the same workspace as the launchers.
Stop any process that owns it before starting stdio.
For an HTTP server with a token, add
`--header "Authorization: Bearer YOUR_TOKEN"` to the HTTP registration command.

Source: [official Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

## OpenCode

### HTTP: connect to the running service

Start Scissors with the [startup launcher](usage.md). Add this to `opencode.json`
in your project. Merge it with existing settings:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "scissors": {
      "type": "remote",
      "url": "http://127.0.0.1:8765/mcp",
      "oauth": false,
      "enabled": true
    }
  }
}
```

Scissors uses bearer tokens when configured; it does not provide OAuth.
If a token is required, add this field inside `scissors` and set the environment
variable before starting OpenCode:

```json
{"headers": {"Authorization": "Bearer {env:SCISSORS_TOKEN}"}}
```

### Stdio: let OpenCode start Scissors

Use this configuration instead of the HTTP entry. Replace the repository path:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "scissors": {
      "type": "local",
      "command": ["uv", "--directory", "/absolute/path/to/scissors", "run", "scissors", "--workspace", "workspaces/agents", "mcp", "stdio"],
      "enabled": true,
      "timeout": 30000
    }
  }
}
```

Restart OpenCode after editing its configuration. Run `opencode mcp list` from
that project to check the connection. The stdio workspace is `workspaces/agents`.
The timeout allows 30 seconds for startup and tool discovery.

Source: [official OpenCode MCP documentation](https://opencode.ai/docs/mcp-servers/).

## Connection notes for all clients

Choose one connection method for each client:

- **HTTP:** you start Scissors and the client connects to its `/mcp` address.
  Multiple clients can share the same running service.
- **Stdio:** the client starts an Scissors process and exchanges messages through
  its standard streams. Each process needs its own workspace.

The stdio configurations above are alternatives. They all use `workspaces/agents`.
Only one process can own that workspace. To share it across clients, connect them
to the same HTTP service. Do not start a stdio process against a workspace owned
by the HTTP service or another stdio client.
Windows repository paths can use forward slashes, such as
`D:/projects/scissors`. Keep paths with spaces quoted in shell commands.
If uv is only importable, use `python -m uv` in commands. In OpenCode's command
array, replace the first `"uv"` with `"python", "-m", "uv"`.

## Other clients and SDK use

Other compatible stdio clients commonly accept:

```json
{"mcpServers":{"scissors":{"command":"uv","args":["--directory","/absolute/path/to/scissors","run","scissors","--workspace","workspaces/agents","mcp","stdio"]}}}
```

examples/mcp_client.py is runnable: it launches stdio, lists tools, creates a scene/cube
and exports GLB. For HTTP use `async with Client("http://127.0.0.1:8765/mcp") as client`
with the same list_tools/call_tool methods. For bearer headers, supply the SDK Streamable
HTTP transport context manager configured with an authenticated HTTP client.

Required tools: modeler_info, operations_list, operations_describe, operations_invoke,
scene_create, scene_inspect, scene_export, plugins_list, plugins_describe, plugins_validate,
plugins_reload. Other registered operations have category__name tools.
Resources: scissors://info, scissors://operations (full descriptions), scissors://plugins,
scissors://scenes. They are read-only; modifications use tools.

Registry revision changes publish ToolsListChanged via InMemorySubscriptionBus/ListenHandler
for modern clients (`client.listen(tools_list_changed=True)`); known legacy sessions receive
notifications/tools/list_changed. Refresh SDK tools with list_tools(cache_mode="refresh").
operations_invoke works even with cached tool lists. Tests cover real transports, shared
REST/MCP state, dynamic plugin tools/events, and exported assets.
Primary SDK source: [official Python SDK](https://github.com/modelcontextprotocol/python-sdk).

## Use the startup launchers

For HTTP, start the launcher described in the [README](../README.md), then
connect to `http://127.0.0.1:8765/mcp`.
For stdio, let the client launch the script with `--transport stdio`.

On Windows, stdio clients can use `command="cmd"` with
`args=["/d", "/c", "call", "D:/path/to/scissors/start_scissors.bat",
"--transport", "stdio"]`.
For Bash, use `command="bash"` with the script's absolute path and
`--transport stdio` in the argument list.
See [startup options](usage.md) for workspace and port settings.

## Give your agent the operating skill

Copy `skill/scissors/` into your agent's skills directory, then load `$scissors`.
This skill guides modeling and plugin use through public tools.
Engine developers follow the repository's [AGENTS.md](../AGENTS.md).

## First model through MCP

1. Call `modeler_info` and `operations_list` to discover the available tools.
2. Call `operations_describe` with `{"name":"mesh.create_cube"}`.
3. Call `scene_create` with `{"name":"First model"}` and keep the returned scene ID.
4. Call `operations_invoke` with
   `{"name":"mesh.create_cube","arguments":{"scene_id":"RETURNED_ID","size":2}}`.
5. Call `scene_export` with that scene ID and `output="exports/first.glb"`.

The runnable [SDK example](../examples/mcp_client.py) checks a real connection.
