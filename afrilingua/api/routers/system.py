"""System endpoints: health, status, languages."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from afrilingua import __version__
from afrilingua.api.deps import get_engine
from afrilingua.api.schemas import HealthResponse
from afrilingua.engine import AfriLinguaEngine

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe (never requires auth)")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/status", summary="Runtime, hardware, model and language status")
async def status(request: Request, engine: AfriLinguaEngine = Depends(get_engine)) -> dict[str, Any]:
    data: dict[str, Any] = await run_in_threadpool(engine.status)
    rooms = getattr(request.app.state, "rooms", None)
    jobs = getattr(request.app.state, "jobs", None)
    data["version"] = __version__
    data["warmup"] = getattr(request.app.state, "warmup", {"state": "idle"})
    data["contributions"] = getattr(request.app.state, "contrib", None) is not None
    if rooms is not None:
        data["rooms"] = rooms.summary()
    if jobs is not None:
        data["jobs"] = jobs.summary()
    return data


@router.get("/languages", summary="Supported languages")
async def languages(engine: AfriLinguaEngine = Depends(get_engine)) -> dict[str, list[dict[str, Any]]]:
    return {"languages": [spec.to_public_dict() for spec in engine.registry]}
