"""System routers."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1", tags=["system"])

@router.get("/status")
async def get_status(request: Request):
    return request.app.state.engine.status()

@router.get("/languages")
async def get_languages(request: Request):
    return {"languages": [s.to_public_dict() for s in request.app.state.engine.registry]}
