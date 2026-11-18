"""FastAPI application factory."""
from __future__ import annotations
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from afrilingua.core.errors import AfriLinguaError, error_payload
from afrilingua.engine import AfriLinguaEngine, build_engine
from afrilingua.tm.tm import open_translation_memory
from afrilingua.contrib.contrib import open_contribution_store

from afrilingua.api.routers import system, translate, rooms, documents, contributions
from afrilingua.api.rooms import RoomManager

def create_app(engine: AfriLinguaEngine | None = None, *, serve_web: bool = True) -> FastAPI:
    app = FastAPI(title="AfriLingua v2 API")
    
    # Lifespan: Init services
    @app.on_event("startup")
    async def startup():
        app.state.engine = engine or build_engine()
        app.state.tm = app.state.engine.tm or open_translation_memory()
        app.state.contrib = open_contribution_store(app.state.tm)
        app.state.rooms = RoomManager()
        
    # Global exception handler
    @app.exception_handler(AfriLinguaError)
    async def afri_exception_handler(request: Request, exc: AfriLinguaError):
        return JSONResponse(
            status_code=exc.status,
            content=error_payload(exc),
        )

    # Health Check
    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "version": "2.0.0.dev0"}
        
    app.include_router(system.router)
    app.include_router(translate.router)
    app.include_router(rooms.router)
    app.include_router(documents.router)
    app.include_router(contributions.router)
        
    return app
