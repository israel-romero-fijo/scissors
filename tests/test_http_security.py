"""ASGI regression cases for body replay and streaming disconnect delivery."""

import anyio

from scissors.http.security import SecurityMiddleware


def scope():
    return {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1")],
        "server": ("127.0.0.1", 80),
    }


def test_security_replays_body_and_preserves_stream_disconnect():
    async def run():
        incoming = iter(
            [
                {"type": "http.request", "body": b"first", "more_body": True},
                {"type": "http.request", "body": b"second", "more_body": False},
                {"type": "http.disconnect"},
            ]
        )
        outgoing = []

        async def receive():
            return next(incoming)

        async def send(message):
            outgoing.append(message)

        async def application(scope, receive, send):
            assert await receive() == {
                "type": "http.request",
                "body": b"firstsecond",
                "more_body": False,
            }
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {"type": "http.response.body", "body": b"data: ready\n\n", "more_body": True}
            )
            assert await receive() == {"type": "http.disconnect"}
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        await SecurityMiddleware(application, token=None, loopback=True, max_body=1024)(
            scope(), receive, send
        )
        assert outgoing[1]["more_body"]
        assert not outgoing[2]["more_body"]

    anyio.run(run)


def test_security_rejects_oversize_chunked_body_before_dispatch():
    async def run():
        incoming = iter(
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"456", "more_body": False},
            ]
        )
        outgoing = []

        async def receive():
            return next(incoming)

        async def send(message):
            outgoing.append(message)

        async def application(scope, receive, send):
            raise AssertionError("Oversize request reached the shared application")

        await SecurityMiddleware(application, token=None, loopback=True, max_body=5)(
            scope(), receive, send
        )
        assert outgoing[0]["status"] == 413

    anyio.run(run)


def test_mcp_reply_framing_preserves_body_and_leaves_rest_lengths_unchanged():
    async def run():
        for path, keeps_length in [("/mcp", False), ("/mcp/", False), ("/api/v1/info", True)]:
            outgoing = []
            incoming = iter([{"type": "http.request", "body": b"", "more_body": False}])

            async def receive(events=incoming):
                return next(events)

            async def send(message, events=outgoing):
                events.append(message)

            async def application(request_scope, read, write):
                assert await read() == {"type": "http.request", "body": b"", "more_body": False}
                await write(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-length", b"2"),
                            (b"content-type", b"application/json"),
                        ],
                    }
                )
                await write({"type": "http.response.body", "body": b"{}", "more_body": False})

            request_scope = scope()
            request_scope["path"] = path
            await SecurityMiddleware(application, token=None, loopback=True, max_body=1024)(
                request_scope, receive, send
            )
            assert ((b"content-length", b"2") in outgoing[0]["headers"]) is keeps_length
            assert outgoing[1] == {"type": "http.response.body", "body": b"{}", "more_body": False}

    anyio.run(run)
