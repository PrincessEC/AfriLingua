"""Rooms router."""
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["rooms"])

class RoomRequest(BaseModel):
    title: str
    src_lang: str
    targets: list[str]

@router.post("/rooms")
async def create_room(req: RoomRequest, request: Request):
    room = request.app.state.rooms.create_room(req.title, req.src_lang, req.targets)
    return {"id": room.id, "host_token": room.host_token, "join_url": f"/room/{room.id}"}
