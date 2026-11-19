"""Shared FastAPI dependencies."""
from __future__ import annotations

from typing import Any

from fastapi import Request

from afrilingua.api.errors import APIError
from afrilingua.engine import AfriLinguaEngine


def get_engine(request: Request) -> AfriLinguaEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise APIError(503, "unavailable", "The engine is still starting up; retry in a moment.")
    return engine


def get_state(request: Request, name: str) -> Any:
    return getattr(request.app.state, name, None)


def public_base_url(request: Request) -> str:
    """Base URL the *browser* used to reach us (handles Vite dev proxy / reverse proxies).

    Priority: ``X-Forwarded-Host``/``X-Forwarded-Proto`` → ``Origin`` header → request base URL.
    """
    headers = request.headers
    fwd_host = headers.get("x-forwarded-host")
    if fwd_host:
        proto = headers.get("x-forwarded-proto", request.url.scheme)
        return f"{proto.split(',')[0].strip()}://{fwd_host.split(',')[0].strip()}"
    origin = headers.get("origin")
    if origin and origin != "null":
        return origin.rstrip("/")
    return str(request.base_url).rstrip("/")
