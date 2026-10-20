# Development

```text
src/scissors/
  core/       scene/mesh/PBR models, geometry, UV, textures, export, persistence
  api/        facade, operations, registry, structured errors
  http/       REST + shared Streamable HTTP MCP application
  mcp/        official SDK server, resources and dynamic discovery
  plugins/    manifest, context, loader, manager, subprocess worker/runtime
  cli.py      local operator commands
  config.py   validated workspace/resource settings
plugins/examples/twist_mesh/    independent example with tests
plugins/builtin/, plugins/local/, plugin_staging/    source organization
skill/scissors/                distributable Agent Skills package for consumers
examples/, docs/, tests/        workflows, technical docs and integration checks
```

Read [AGENTS.md](../AGENTS.md) before coding. It defines architecture, stable contracts,
plugin and core responsibilities, and completion checks.

For feedback after each change, run the quick suite. It checks geometry, schemas,
errors, transactions, persistence and the in-process HTTP API. It deselects tests
marked `integration`, including real CLI/MCP/plugin processes and geometry backends.
The selection is reported in pytest's summary.

```bash
uv run --no-sync pytest --quick -q
```

Use `--no-sync` after installing dependencies with `uv sync`; it avoids updating
the environment while an agent or server may be using it. Run affected integration
tests when changing their modules, and the full suite before finishing a change:

```bash
uv run ruff check src tests examples plugins scripts
uv run ruff format --check src tests examples plugins scripts
uv run mypy src
uv run pytest -q
```

To run only integration tests, use `uv run --no-sync pytest -m integration -q`.
Add `--durations=20` to inspect the slowest tests. The default suite remains complete.

Tests use temporary workspaces and real MCP transports.
