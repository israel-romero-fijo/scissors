import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from scissors.api.errors import ScissorsError


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="scissors", description="Local agent-first 3D modeling")
    cli.add_argument("--workspace", type=Path, default=Path("workspaces/agents"))
    cli.add_argument(
        "--plugins-dir", type=Path, help="Application plugin store shared by workspaces"
    )
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    serve = commands.add_parser("serve", help="REST and Streamable HTTP MCP, sharing one engine")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    mcp = commands.add_parser("mcp").add_subparsers(dest="transport", required=True)
    mcp.add_parser("stdio")
    http = mcp.add_parser("http")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8765)
    operations = commands.add_parser("operations").add_subparsers(dest="action", required=True)
    listing = operations.add_parser("list")
    listing.add_argument("--category")
    operations.add_parser("describe").add_argument("name")
    invoke = operations.add_parser("invoke")
    invoke.add_argument("name")
    invoke.add_argument("--arguments", default="{}", help="JSON argument object")
    plugins = commands.add_parser("plugins").add_subparsers(dest="action", required=True)
    plugins.add_parser("list")
    plugins.add_parser("discover").add_argument("path", type=Path)
    plugins.add_parser("validate").add_argument("path", type=Path)
    install = plugins.add_parser("install", help="Install and enable a local plugin package")
    install.add_argument("path", type=Path)
    for action in ("describe", "enable", "disable", "reload", "uninstall", "test"):
        plugins.add_parser(action).add_argument("plugin_id")
    return cli


def main() -> None:
    args = parser().parse_args()
    # Help and argument errors do not need geometry libraries or a workspace.
    from scissors.api.service import Scissors
    from scissors.config import Settings

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        settings = Settings(workspace=args.workspace)
        if args.plugins_dir is not None:
            settings.plugins_dir = args.plugins_dir
        with Scissors(settings=settings) as app:
            result: Any
            if args.command == "serve" or (args.command == "mcp" and args.transport == "http"):
                import uvicorn

                from scissors.http.server import create_app

                uvicorn.run(
                    create_app(app, os.environ.get("SCISSORS_TOKEN"), args.host),
                    host=args.host,
                    port=args.port,
                    log_level="warning",
                )
                return
            if args.command == "mcp":
                from scissors.mcp.server import run_stdio

                asyncio.run(run_stdio(app))
                return
            if args.command == "info":
                result = app.call("modeler.info")
            elif args.command == "operations":
                if args.action == "list":
                    result = app.registry.list_operations(args.category)
                elif args.action == "describe":
                    result = app.registry.describe_operation(args.name)
                else:
                    result = app.registry.execute(args.name, json.loads(args.arguments))
            elif args.action == "list":
                result = app.plugins.list()
            elif args.action in {"validate", "discover"}:
                result = getattr(app.plugins, args.action)(args.path)
            elif args.action == "install":
                result = app.plugins.install(args.path)
            elif args.action in {"uninstall", "test", "describe"}:
                result = getattr(app.plugins, args.action)(args.plugin_id)
            else:
                result = app.registry.execute(
                    f"plugins.{args.action}", {"plugin_id": args.plugin_id}
                )
            print(json.dumps(result, indent=2))
            if isinstance(result, dict) and (
                result.get("success") is False
                or result.get("valid") is False
                or result.get("passed") is False
            ):
                raise SystemExit(1)
    except ScissorsError as exc:
        print(json.dumps({"success": False, "error": exc.as_dict()}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
