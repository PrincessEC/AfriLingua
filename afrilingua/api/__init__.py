"""AfriLingua HTTP API: FastAPI app, REST routers, live rooms, OpenAI-compatible /v1, SPA hosting.

``create_app`` is imported lazily so that ``afrilingua.api.routers.*`` modules can be
imported without pulling in the whole application (and to avoid import cycles).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from afrilingua.api.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from afrilingua.api.app import create_app

        return create_app
    raise AttributeError(name)
