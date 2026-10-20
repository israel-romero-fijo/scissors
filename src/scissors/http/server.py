from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field

from scissors.api.errors import ScissorsError
from scissors.api.service import Scissors
from scissors.http.security import SecurityMiddleware
from scissors.mcp.server import create_server


class Invocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def create_app(service: Scissors, token: str | None = None, host: str = "127.0.0.1") -> FastAPI:
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not token:
        raise ScissorsError("PERMISSION_DENIED", "Non-loopback HTTP requires SCISSORS_TOKEN")
    mcp = create_server(service).streamable_http_app(
        host=host,
        max_request_body_size=service.engine.settings.max_file_bytes,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=loopback,
            # SDK port wildcards do not match headers that omit the default port.
            allowed_hosts=[
                "127.0.0.1",
                "localhost",
                "[::1]",
                "127.0.0.1:*",
                "localhost:*",
                "[::1]:*",
            ],
            allowed_origins=[
                "http://127.0.0.1",
                "http://localhost",
                "http://[::1]",
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
            ],
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        async with mcp.router.lifespan_context(mcp):
            yield

    api = FastAPI(title="Scissors", version="0.1.0", lifespan=lifespan)
    api.state.service = service

    api.add_middleware(
        SecurityMiddleware,
        token=token,
        loopback=loopback,
        max_body=service.engine.settings.max_file_bytes,
    )

    @api.exception_handler(ScissorsError)
    async def scissors_error(request: Request, exc: ScissorsError) -> JSONResponse:
        return JSONResponse({"success": False, "error": exc.as_dict()}, status_code=404)

    @api.get("/api/v1/info")
    def info() -> dict[str, Any]:
        return service.call("modeler.info")

    @api.get("/api/v1/operations")
    def operations(category: str | None = None) -> dict[str, Any]:
        return {"operations": service.registry.list_operations(category)}

    @api.get("/api/v1/operations/{name}")
    def describe(name: str) -> dict[str, Any]:
        return service.registry.describe_operation(name)

    @api.post("/api/v1/operations/invoke")
    def invoke(body: Invocation) -> dict[str, Any]:
        return service.registry.execute(body.name, body.arguments)

    @api.get("/api/v1/plugins")
    def plugins() -> dict[str, Any]:
        return service.call("plugins.list")

    @api.get("/api/v1/scenes")
    def scenes() -> dict[str, Any]:
        return service.call("scene.list")

    @api.get("/api/v1/scenes/{scene_id}")
    def scene(scene_id: str) -> dict[str, Any]:
        return service.call("scene.inspect", scene_id=scene_id)

    api.mount("/", mcp)
    return api
