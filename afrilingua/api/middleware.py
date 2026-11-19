"""Pure-ASGI middleware (safe for WebSockets and streaming responses).

* :class:`ServerTimingMiddleware` adds ``Server-Timing: app;dur=<ms>``.
* :class:`APIKeyMiddleware` enforces ``API_KEYS`` (only when configured).
* :class:`BodySizeLimitMiddleware` rejects bodies above ``MAX_UPLOAD_MB`` with 413.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qs

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from afrilingua.api.errors import error_body

PROTECTED_PREFIXES = ("/api", "/v1", "/mcp")
# Endpoints that never need an API key.
PUBLIC_PATHS = {"/api/v1/health"}
# Room audience endpoints are public by design (access is controlled by the room PIN):
# the join URL is handed to people who do not have the server's API key.
_PUBLIC_ROOM_RE = re.compile(r"^/api/v1/rooms/[A-Za-z0-9_-]+(/listen)?/?$")


def _headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


def extract_api_key(scope: Scope) -> str | None:
    headers = _headers(scope)
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    if headers.get("x-api-key"):
        return headers["x-api-key"].strip()
    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    for name in ("key", "api_key"):
        if qs.get(name):
            return qs[name][0]
    return None


def key_is_valid(key: str | None, valid: Iterable[str]) -> bool:
    if not key:
        return False
    ok = False
    for candidate in valid:  # constant-time compare against every key
        ok |= secrets.compare_digest(key.encode(), candidate.encode())
    return ok


def path_requires_auth(path: str, method: str, scope_type: str) -> bool:
    if path in PUBLIC_PATHS:
        return False
    if not any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES):
        return False  # SPA / static files / docs
    m = _PUBLIC_ROOM_RE.match(path)
    if m:
        is_listen = bool(m.group(1))
        if scope_type == "websocket" and is_listen:
            return False
        if scope_type == "http" and not is_listen and method in {"GET", "HEAD"}:
            return False
    return not (scope_type == "http" and method == "OPTIONS")


async def send_json(
    send: Send, status: int, body: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    headers += extra_headers or []
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


async def reject_websocket(receive: Receive, send: Send, code: str, message: str, close_code: int) -> None:
    """Accept, send an ``error`` event (so browsers can read it) and close."""
    msg = await receive()
    if msg["type"] != "websocket.connect":
        return
    await send({"type": "websocket.accept"})
    await send(
        {"type": "websocket.send", "text": json.dumps({"type": "error", "code": code, "message": message})}
    )
    await send({"type": "websocket.close", "code": close_code, "reason": code})


class APIKeyMiddleware:
    def __init__(self, app: ASGIApp, keys: Iterable[str]) -> None:
        self.app = app
        self.keys = frozenset(keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.keys or scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path_requires_auth(path, scope.get("method", "GET"), scope["type"]):
            await self.app(scope, receive, send)
            return
        if key_is_valid(extract_api_key(scope), self.keys):
            await self.app(scope, receive, send)
            return
        message = (
            "Missing or invalid API key. Send 'Authorization: Bearer <key>' (or ?key=<key> for WebSockets)."
        )
        if scope["type"] == "websocket":
            await reject_websocket(receive, send, "unauthorized", message, 4401)
            return
        openai = path.startswith("/v1")
        await send_json(
            send, 401, error_body("unauthorized", message, openai=openai), [(b"www-authenticate", b"Bearer")]
        )


class _TooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        limit_mb = round(self.max_bytes / (1024 * 1024), 1)
        message = f"Request body exceeds the {limit_mb:g} MB limit (MAX_UPLOAD_MB)."
        openai = scope.get("path", "").startswith("/v1")
        length = _headers(scope).get("content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            await send_json(send, 413, error_body("payload_too_large", message, openai=openai))
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await send_json(send, 413, error_body("payload_too_large", message, openai=openai))
        except Exception as exc:  # python-multipart / starlette may wrap our exception
            if isinstance(exc.__cause__, _TooLarge) or isinstance(exc.__context__, _TooLarge):
                if not started:
                    await send_json(send, 413, error_body("payload_too_large", message, openai=openai))
                return
            raise


class ServerTimingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()

        async def timed_send(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                dur = (time.perf_counter() - t0) * 1000
                headers = list(msg.get("headers", []))
                headers.append((b"server-timing", f"app;dur={dur:.1f}".encode()))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, timed_send)
