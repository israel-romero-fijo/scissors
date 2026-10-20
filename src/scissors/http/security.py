"""ASGI request guards that preserve MCP streaming and disconnect messages."""

import hmac

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, *, token: str | None, loopback: bool, max_body: int):
        self.app, self.token, self.loopback, self.max_body = app, token, loopback, max_body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path", "").rstrip("/") == "/mcp":
            original_send = send

            async def protocol_send(message: Message) -> None:
                if message["type"] == "http.response.start":
                    # Let the HTTP server frame both JSON replies and SSE streams.
                    # Fixed lengths must not survive a streaming transport wrapper.
                    message = {
                        **message,
                        "headers": [
                            (key, value)
                            for key, value in message.get("headers", [])
                            if key.lower() != b"content-length"
                        ],
                    }
                await original_send(message)

            send = protocol_send
        request = Request(scope)
        response = None
        if self.token and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {self.token}"
        ):
            response = JSONResponse(
                {
                    "success": False,
                    "error": {
                        "code": "PERMISSION_DENIED",
                        "message": "Bearer authentication required",
                    },
                },
                status_code=401,
            )
        host = request.headers.get("host", "").split(":")[0]
        if (
            response is None
            and self.loopback
            and host not in {"127.0.0.1", "localhost", "[", "testserver"}
        ):
            response = JSONResponse({"error": "Invalid Host header"}, status_code=403)
        origin = request.headers.get("origin")
        if (
            response is None
            and origin
            and origin.rstrip("/") != f"{request.url.scheme}://{request.headers.get('host')}"
        ):
            response = JSONResponse({"error": "Cross-origin requests forbidden"}, status_code=403)
        length = request.headers.get("content-length")
        if response is None and length and (not length.isdigit() or int(length) > self.max_body):
            response = JSONResponse({"error": "Request body exceeds limit"}, status_code=413)
        if response is not None:
            await response(scope, receive, send)
            return
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body:
                await JSONResponse({"error": "Request body exceeds limit"}, status_code=413)(
                    scope, receive, send
                )
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
