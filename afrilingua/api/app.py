"""
FastAPI application factory.

    from afrilingua.api import create_app
    app = create_app()                 # builds the real engine in the lifespan
    app = create_app(fake_engine)      # tests: inject an engine

Surfaces: REST ``/api/v1`` · live-room WebSockets · OpenAI-compatible ``/v1`` ·
MCP Streamable HTTP ``/mcp`` · React SPA at ``/`` (``web/dist``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from afrilingua import __version__
from afrilingua.api.errors import error_response, install_exception_handlers
from afrilingua.api.jobs import JobManager
from afrilingua.api.middleware import APIKeyMiddleware, BodySizeLimitMiddleware, ServerTimingMiddleware
from afrilingua.api.rooms import RoomManager, SessionFactory
from afrilingua.core.settings import PROJECT_ROOT, Settings, apply_runtime_environment, get_settings
from afrilingua.engine import MT_KEY, AfriLinguaEngine

logger = logging.getLogger(__name__)

WEB_DIST = PROJECT_ROOT / "web" / "dist"
SPA_EXCLUDED_PREFIXES = ("/api", "/v1", "/mcp", "/docs", "/redoc", "/openapi.json")
IDLE_UNLOAD_S = float(os.environ.get("AFRILINGUA_IDLE_UNLOAD_S", "1800"))
MAINTENANCE_INTERVAL_S = 60.0

TAGS = [
    {"name": "system", "description": "Health, runtime status and supported languages."},
    {
        "name": "translation",
        "description": "Text translation (cache → translation memory → MT) and language detection.",
    },
    {"name": "speech", "description": "Speech recognition and synthesis."},
    {"name": "documents", "description": "Format-preserving document translation as background jobs."},
    {"name": "rooms", "description": "Live conference interpretation rooms (REST + WebSockets)."},
    {"name": "translation-memory", "description": "Fast fuzzy search over vetted translations."},
    {"name": "contributions", "description": "Community contributions, review and export."},
    {
        "name": "openai-compatible",
        "description": "Drop-in OpenAI audio API (`base_url=http://host:8000/v1`).",
    },
]

FALLBACK_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AfriLingua</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:46rem;margin:4rem auto;padding:0 1.25rem;
 line-height:1.6;color:#1c1917;background:#fafaf9}
 code{background:#e7e5e4;padding:.15rem .4rem;border-radius:.3rem} a{color:#b45309}
 h1{font-size:1.9rem;margin-bottom:.25rem}
</style></head><body>
<h1>AfriLingua API is running</h1>
<p>Version __VERSION__. The web app has not been built yet. Build it with:</p>
<p><code>pnpm --dir web install &amp;&amp; pnpm --dir web build</code></p>
<p>then reload this page. Meanwhile you can use the
<a href="/docs">interactive API docs</a>, the OpenAI-compatible API at <code>/v1</code>
and the MCP endpoint at <code>/mcp</code>.</p>
</body></html>"""


# ── stores ──────────────────────────────────────────────────────────────────
def _open_tm(engine: AfriLinguaEngine) -> Any:
    if engine.tm is not None:
        return engine.tm
    try:
        from afrilingua.tm import open_translation_memory
    except ImportError:
        logger.info("afrilingua.tm not available; translation memory disabled")
        return None
    try:
        return open_translation_memory()
    except Exception:
        logger.exception("Could not open the translation memory")
        return None


def _open_contrib(tm: Any) -> Any:
    try:
        from afrilingua.contrib import open_contribution_store
    except ImportError:
        logger.info("afrilingua.contrib not available; contributions disabled")
        return None
    try:
        return open_contribution_store(tm)
    except Exception:
        logger.exception("Could not open the contribution store")
        return None


# ── warm-up / maintenance ───────────────────────────────────────────────────
def _warmup_keys(engine: AfriLinguaEngine, mode: str) -> list[str]:
    keys = engine.manager.keys()
    if mode == "all":
        return keys
    if mode == "adaptive":
        return [MT_KEY] if MT_KEY in keys else []
    return []


def _start_warmup(app: FastAPI, engine: AfriLinguaEngine, mode: str) -> None:
    keys = _warmup_keys(engine, mode)
    state: dict[str, Any] = {
        "mode": mode,
        "state": "idle" if not keys else "loading",
        "keys": keys,
        "error": None,
    }
    app.state.warmup = state
    if not keys:
        return

    def run() -> None:
        t0 = time.perf_counter()
        try:
            engine.manager.preload(keys)
            state.update(state="ready", seconds=round(time.perf_counter() - t0, 2))
            logger.info("Warm-up complete (%s) in %.1fs", ", ".join(keys), time.perf_counter() - t0)
        except Exception as exc:  # the server keeps running; models load lazily on first use
            state.update(state="error", error=f"{type(exc).__name__}: {exc}")
            logger.warning("Warm-up failed: %s", exc)

    threading.Thread(target=run, name="afrilingua-warmup", daemon=True).start()


async def _maintenance(app: FastAPI, engine: AfriLinguaEngine, mode: str) -> None:
    while True:
        await asyncio.sleep(MAINTENANCE_INTERVAL_S)
        try:
            if IDLE_UNLOAD_S > 0 and mode != "all":
                evicted = await anyio.to_thread.run_sync(engine.manager.unload_idle, IDLE_UNLOAD_S)
                if evicted:
                    logger.info("Unloaded idle models: %s", ", ".join(evicted))
            jobs: JobManager | None = getattr(app.state, "jobs", None)
            if jobs is not None:
                await anyio.to_thread.run_sync(jobs.cleanup)
        except Exception:
            logger.exception("Maintenance task failed")


# ── SPA ─────────────────────────────────────────────────────────────────────
def _resolve_spa_path(web_dist: Path, full_path: str) -> tuple[str, Path | None]:
    """Return ("fallback"|"file"|"index"|"missing", path) for a client path (blocking FS calls)."""
    index = web_dist / "index.html"
    if not index.is_file():
        return "fallback", None
    if full_path:
        root = web_dist.resolve()
        candidate = (web_dist / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(root):
            return "file", candidate
        if "." in Path(full_path).name and not full_path.endswith(".html"):
            return "missing", None
    return "index", index


def _install_spa(app: FastAPI, web_dist: Path) -> None:
    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa(full_path: str, request: Request) -> Response:
        path = "/" + full_path
        if any(path == p or path.startswith(p + "/") for p in SPA_EXCLUDED_PREFIXES):
            return error_response(404, "not_found", f"No route for {path}.", request)
        kind, target = await run_in_threadpool(_resolve_spa_path, web_dist, full_path)
        if kind == "fallback":
            return HTMLResponse(FALLBACK_HTML.replace("__VERSION__", __version__))
        if kind == "missing" or target is None:
            return error_response(404, "not_found", f"Static file {path} not found.", request)
        if kind == "file":
            cache = "public, max-age=31536000, immutable" if "/assets/" in path else "public, max-age=3600"
            return FileResponse(target, headers={"Cache-Control": cache})
        return FileResponse(target, headers={"Cache-Control": "no-cache"})


# ── MCP ─────────────────────────────────────────────────────────────────────
def _mount_mcp(app: FastAPI, settings: Settings) -> Any:
    try:
        from mcp.server.transport_security import TransportSecuritySettings

        from afrilingua.mcp.server import build_mcp
    except ImportError as exc:
        logger.warning("MCP over HTTP disabled (mcp SDK missing): %s", exc)
        return None

    def state(name: str) -> Callable[[], Any]:
        return lambda: getattr(app.state, name, None)

    server = build_mcp(
        state("engine"),
        tm_provider=state("tm"),
        contrib_provider=state("contrib"),
        rooms_provider=state("rooms"),
        settings=settings,
    )
    loopback = settings.api_host in {"127.0.0.1", "localhost", "::1"}
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=loopback,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "127.0.0.1", "localhost", "[::1]"],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "http://127.0.0.1",
            "http://localhost",
            *settings.cors_origin_list,
        ],
    )
    http_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=security,
        max_request_body_size=settings.max_upload_mb * 1024 * 1024,
        host=settings.api_host,
    )
    for route in http_app.routes:
        app.router.routes.append(route)
    endpoint = next(
        (getattr(r, "endpoint", None) for r in http_app.routes if getattr(r, "path", "") == "/mcp"), None
    )
    if endpoint is not None:
        from starlette.routing import Route

        app.router.routes.append(Route("/mcp/", endpoint=endpoint))
    return server


# ── factory ─────────────────────────────────────────────────────────────────
def _include_community_routers(app: FastAPI) -> None:
    for module_name in ("tm", "contributions"):
        try:
            module = __import__(f"afrilingua.api.routers.{module_name}", fromlist=["router"])
            app.include_router(module.router)
        except ImportError as exc:
            logger.warning("Router afrilingua.api.routers.%s not available yet: %s", module_name, exc)
        except Exception:
            logger.exception("Failed to include router afrilingua.api.routers.%s", module_name)


def create_app(
    engine: AfriLinguaEngine | None = None,
    *,
    serve_web: bool = True,
    settings: Settings | None = None,
    session_factory: SessionFactory | None = None,
    mount_mcp: bool = True,
    web_dist: Path | None = None,
) -> FastAPI:
    """Create the AfriLingua FastAPI app.

    Args:
        engine: Inject an engine (tests / embedding). ``None`` → ``build_engine()`` in the lifespan.
        serve_web: Serve the React SPA (``web/dist``) at ``/``.
        settings: Override settings (default: :func:`get_settings`).
        session_factory: Override the live-room StreamingSession factory (tests).
        mount_mcp: Mount the MCP Streamable HTTP endpoint at ``/mcp``.
        web_dist: Override the SPA build directory.
    """
    s = settings or get_settings()
    mcp_holder: dict[str, Any] = {"server": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        apply_runtime_environment(s)
        eng = engine if engine is not None else await anyio.to_thread.run_sync(_build_default_engine)
        app.state.engine = eng
        app.state.tm = _open_tm(eng)
        app.state.contrib = _open_contrib(app.state.tm)
        app.state.rooms = RoomManager(eng, session_factory=session_factory)
        await app.state.rooms.start()
        app.state.jobs = JobManager(eng, s.data_dir / "jobs")
        _start_warmup(app, eng, s.preload_mode)
        maintenance = asyncio.create_task(
            _maintenance(app, eng, s.preload_mode), name="afrilingua-maintenance"
        )
        try:
            async with contextlib.AsyncExitStack() as stack:
                server = mcp_holder["server"]
                if server is not None:
                    await stack.enter_async_context(server.session_manager.run())
                logger.info("AfriLingua %s ready", __version__)
                yield
        finally:
            maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await maintenance
            await app.state.rooms.shutdown()
            app.state.jobs.shutdown()

    app = FastAPI(
        title="AfriLingua",
        version=__version__,
        summary="Offline-first speech & translation engine for African languages.",
        description=(
            "REST API (`/api/v1`), live interpretation rooms (WebSockets), OpenAI-compatible audio "
            "endpoints (`/v1`) and an MCP server (`/mcp`). See `docs/api-contract.md`."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = s
    app.state.warmup = {"state": "idle"}

    install_exception_handlers(app)

    # Middleware: last added = outermost. Order (outer → inner): timing → CORS → auth → size limit.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=s.max_upload_mb * 1024 * 1024)
    app.add_middleware(APIKeyMiddleware, keys=s.api_key_set)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Server-Timing", "Content-Disposition", "X-Sample-Rate", "X-Audio-Duration"],
    )
    app.add_middleware(ServerTimingMiddleware)

    from afrilingua.api.routers import documents, openai_compat, rooms, speech, system, translate

    for module in (system, translate, speech, documents, rooms, openai_compat):
        app.include_router(module.router)
    _include_community_routers(app)

    if mount_mcp:
        mcp_holder["server"] = _mount_mcp(app, s)
    app.state.mcp = mcp_holder["server"]

    if serve_web:
        _install_spa(app, web_dist or WEB_DIST)
    return app


def _build_default_engine() -> AfriLinguaEngine:
    from afrilingua.engine import build_engine

    return build_engine()
