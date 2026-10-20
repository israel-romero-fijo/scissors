# Plugins: public contract 1.0

Each independent package has manifest.json, a local .py entrypoint, README and tests.
The formal schema is docs/plugin-manifest.schema.json. See plugins/examples/twist_mesh
for a complete package, and examples/create_with_plugin.py for its working GLB workflow.

The module exports OPERATIONS mapping every manifest name to a callable receiving
PluginContext plus validated keyword arguments. Dependencies are installed PEP 508
requirements; Scissors checks versions but never installs dependencies automatically.

| Public context method | Behavior |
| --- | --- |
| get_mesh(object_id) | Copied plain geometry/UV/normals/material/transform/metadata |
| replace_mesh(object_id,vertices,faces,uv=None) | Stage replacement, preserving UV by default |
| create_mesh(scene_id,vertices,faces,name) | Stage new object, parent returns IDs in created_objects |
| statistics(object_id) | Staged counts and bounds |

Only targets in object_id/object_ids/scene_id arguments are supplied. Coordinates are
already scene-space; do not reapply transforms. Parent checks scope/limits/geometry/schema,
recomputes normals and commits together. Read-only operations cannot stage writes.

Agents search operations_list/plugins_list before writing a new plugin. Prepare under
<workspace>/plugin_staging/ID, validate with plugins_validate using a workspace-relative
path and fix the reported static errors. Static validation never imports code or runs tests;
it only describes the manifest and its fingerprint.

Local workflow from repo root:

```bash
uv run scissors --workspace workspaces/dev plugins validate workspaces/dev/plugin_staging/my_plugin
uv run scissors --workspace workspaces/dev plugins install workspaces/dev/plugin_staging/my_plugin
uv run scissors --workspace workspaces/dev plugins test my_plugin
uv run scissors --workspace workspaces/dev operations describe namespace.operation
uv run scissors --workspace workspaces/dev plugins disable my_plugin
uv run scissors --workspace workspaces/dev plugins reload my_plugin
uv run scissors --workspace workspaces/dev plugins uninstall my_plugin
```

CLI paths are shell-relative; MCP validation paths are workspace-relative. Installing runs
Python with the operator's OS permissions, so install only packages you trust. Python has
the equivalent app.plugins.install(path). Installation, tests and uninstallation are not
public tools.

Installed packages live under `<plugins_dir>/ID` in the application plugin store.
The default is `~/.scissors/plugins`, shared by all workspaces for the current OS user.
Set `SCISSORS_PLUGINS_DIR`, pass `scissors --plugins-dir /absolute/plugins ...`, or use
`Settings(plugins_dir=Path("/absolute/plugins"))` to select another shared store. An explicit
CLI or Python setting overrides the environment. The store and workspace must be separate,
non-overlapping directories. `modeler.info` and `plugins.list` report `plugins_dir`.

Install copies a package once and enables it in the current workspace. Every new session
discovers valid packages from the shared store and enables them unless that workspace has
explicitly disabled them. Activation is per workspace; installation and uninstallation affect
all workspaces using the store. Running sessions keep their catalog until reopened; restart
them after adding or removing shared packages. A removed package cannot execute in an older
session. A shared file lock serializes package changes, probes, invocations and tests;
`PLUGIN_STORE_BUSY` means another process held it beyond `operation_timeout`.

Code changes take effect on the next invocation, which imports the
current files in a fresh child. Name collisions reject activation. Load failures appear in
plugins list, and a folder whose manifest is invalid is listed with installed false and its
error. An explicit disable is kept in state.json, so reopening the workspace does not
re-enable that package. Stop the HTTP service for local install or uninstallation;
public enable/disable/reload works while the service runs.

Each invocation has a fresh child, sanitized environment, timeout and bounded payload,
result and log sizes. POSIX workers also have address-space/file limits; Windows lacks
a hard child memory ceiling. Child exceptions or invalid staged changes preserve the
parent's scene. Tests execute Python with the operator's permissions.

Native core is trusted; all installed example/local packages are external. Repository
plugins/builtin and plugins/local organize source; they are not automatically imported.
**Subprocesses are not a security sandbox.** Package code can use OS APIs directly or spawn
descendants, and it runs as soon as the workspace opens. Context scope protects the normal
extension contract, not hostile OS activity. Keep untrusted folders out of
the application plugin store and use OS containers/accounts for genuinely untrusted sources.
