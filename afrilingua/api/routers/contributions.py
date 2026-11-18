"""Community contributions router."""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/v1/contributions", tags=["contributions"])

@router.post("")
async def submit_contribution(payload: dict, request: Request):
    return request.app.state.contrib.submit(payload)

@router.get("/stats")
async def get_stats(request: Request):
    return {"total": 0} # Placeholder
