"""Live room management."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Room:
    id: str
    title: str
    src_lang: str
    targets: list[str]
    host_token: str
    
class RoomManager:
    def __init__(self) -> None:
        self.rooms: Dict[str, Room] = {}
        
    def create_room(self, title: str, src_lang: str, targets: list[str]) -> Room:
        room_id = str(uuid.uuid4())[:8]
        token = str(uuid.uuid4())
        room = Room(room_id, title, src_lang, targets, token)
        self.rooms[room_id] = room
        return room
