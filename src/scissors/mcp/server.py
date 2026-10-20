import json
from contextlib import asynccontextmanager
from typing import Any

import anyio
import mcp.types as types
from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ToolsListChanged

from scissors.api.errors import ScissorsError
from scissors.api.service import Scissors


class DynamicServer(Server[Any]):
    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
        extensions: dict[str, dict[str, Any]] | None = None,
    ) -> InitializationOptions:
        # The HTTP session manager uses default initialization options too.
        return super().create_initialization_options(
            notification_options or NotificationOptions(tools_changed=True),
            experimental_capabilities,
            extensions,
        )


def create_server(app: Scissors) -> Server[Any]:
    registry = app.registry
    bus = InMemorySubscriptionBus()
    sessions: dict[int, tuple[Any, str]] = {}
    aliases = {
        "modeler_info": "modeler.info",
        "scene_create": "scene.create",
        "scene_inspect": "scene.inspect",
        "scene_export": "scene.export",
        "plugins_list": "plugins.list",
        "plugins_describe": "plugins.describe",
        "plugins_validate": "plugins.validate",
        "plugins_reload": "plugins.reload",
    }

    async def watch() -> None:
        revision = registry.revision
        while True:
            await anyio.sleep(0.1)
            if revision != registry.revision:
                revision = registry.revision
                await bus.publish(ToolsListChanged())
                for key, (session, version) in list(sessions.items()):
                    if version < "2026-07-28":
                        try:
                            await session.send_tool_list_changed()
                        except Exception:
                            sessions.pop(key, None)

    @asynccontextmanager
    async def lifespan(server: Server[Any]) -> Any:
        async with anyio.create_task_group() as group:
            group.start_soon(watch)
            try:
                yield None
            finally:
                group.cancel_scope.cancel()

    def remember(ctx: ServerRequestContext[Any, Any]) -> None:
        sessions[id(ctx.session)] = (ctx.session, ctx.protocol_version)

    def tool(
        name: str, description: str, schema: dict[str, Any], write: bool = False
    ) -> types.Tool:
        return types.Tool(
            name=name,
            description=description,
            input_schema=schema,
            annotations=types.ToolAnnotations(
                read_only_hint=not write, destructive_hint=write, open_world_hint=False
            ),
        )

    catalog_schema = {
        "type": "object",
        "properties": {"category": {"type": ["string", "null"]}},
        "additionalProperties": False,
    }
    describe_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    invoke_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    async def list_tools(ctx: ServerRequestContext[Any, Any], params: Any) -> types.ListToolsResult:
        remember(ctx)
        result = [
            tool("operations_list", "Discover available operations by category.", catalog_schema),
            tool(
                "operations_describe",
                "Get operation input/output schemas and examples.",
                describe_schema,
            ),
            tool(
                "operations_invoke",
                "Invoke any registered operation, including newly enabled plugins.",
                invoke_schema,
                True,
            ),
        ]
        with app.engine.mutex:
            for name, op in registry.operations.items():
                alias = next(
                    (key for key, value in aliases.items() if value == name),
                    name.replace(".", "__"),
                )
                result.append(tool(alias, op.description, op.parameters, op.side_effects))
        return types.ListToolsResult(tools=result)

    async def call_tool(
        ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        remember(ctx)
        arguments = params.arguments or {}
        try:
            if params.name in {"operations_list", "operations_describe", "operations_invoke"}:
                from jsonschema import Draft202012Validator

                schema = {
                    "operations_list": catalog_schema,
                    "operations_describe": describe_schema,
                    "operations_invoke": invoke_schema,
                }[params.name]
                Draft202012Validator(schema).validate(arguments)
                if params.name == "operations_list":
                    response = {
                        "success": True,
                        "result": {
                            "operations": registry.list_operations(arguments.get("category"))
                        },
                    }
                elif params.name == "operations_describe":
                    response = {
                        "success": True,
                        "result": registry.describe_operation(arguments["name"]),
                    }
                else:
                    response = await anyio.to_thread.run_sync(
                        lambda: registry.execute(arguments["name"], arguments.get("arguments", {}))
                    )
            else:
                name = aliases.get(params.name, params.name.replace("__", "."))
                response = await anyio.to_thread.run_sync(lambda: registry.execute(name, arguments))
        except ScissorsError as exc:
            response = {"success": False, "error": exc.as_dict()}
        except Exception as exc:
            response = {
                "success": False,
                "error": {
                    "code": "INVALID_PARAMETER",
                    "message": "Invalid discovery arguments",
                    "details": {"exception_type": type(exc).__name__},
                },
            }
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(response))],
            structured_content=response,
            is_error=not response["success"],
        )

    async def list_resources(
        ctx: ServerRequestContext[Any, Any], params: Any
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri="scissors://info", name="Scissors runtime", mime_type="application/json"
                ),
                types.Resource(
                    uri="scissors://operations",
                    name="Operation catalog",
                    mime_type="application/json",
                ),
                types.Resource(
                    uri="scissors://plugins",
                    name="Installed plugins",
                    mime_type="application/json",
                ),
                types.Resource(
                    uri="scissors://scenes", name="Scene summaries", mime_type="application/json"
                ),
            ]
        )

    async def read_resource(
        ctx: ServerRequestContext[Any, Any], params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        if uri == "scissors://operations":
            value = {
                "operations": [
                    registry.describe_operation(op["name"]) for op in registry.list_operations()
                ]
            }
        else:
            operation = {
                "scissors://info": "modeler.info",
                "scissors://plugins": "plugins.list",
                "scissors://scenes": "scene.list",
            }.get(uri)
            if not operation:
                raise ValueError("Unknown resource")
            value = app.call(operation)
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri, mime_type="application/json", text=json.dumps(value)
                )
            ]
        )

    return DynamicServer(
        "Scissors",
        version="0.1.0",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
        on_subscriptions_listen=ListenHandler(bus),
    )


async def run_stdio(app: Scissors) -> None:
    server = create_server(app)
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            server.create_initialization_options(
                notification_options=NotificationOptions(tools_changed=True)
            ),
        )
