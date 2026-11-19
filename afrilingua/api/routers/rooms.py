"""Live rooms: REST (create/info) + host and listener WebSockets (contract §6)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect

from afrilingua.api.deps import public_base_url
from afrilingua.api.errors import APIError, not_found
from afrilingua.api.rooms import Room, RoomError, RoomManager, Subscriber
from afrilingua.api.schemas import ERROR_RESPONSES, RoomCreate, RoomCreated, RoomInfo
from afrilingua.core.errors import AfriLinguaError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["rooms"])

_TRUE = {"1", "true", "yes", "on"}


def get_rooms(request: Request) -> RoomManager:
    rooms = getattr(request.app.state, "rooms", None)
    if rooms is None:
        raise APIError(503, "unavailable", "Live rooms are not available yet.")
    return rooms


RoomsDep = Annotated[RoomManager, Depends(get_rooms)]


def lan_ip() -> str | None:
    """Best-effort LAN address of this machine (no packets are sent)."""
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    return None


def room_links(room: Room, base_url: str, *, lan_port: int | None = None) -> dict[str, Any]:
    links: dict[str, Any] = {
        "join_url": f"{base_url}/room/{room.id}",
        "host_ws": f"/api/v1/rooms/{room.id}/host",
        "listen_ws": f"/api/v1/rooms/{room.id}/listen",
        "lan_join_url": None,
    }
    if lan_port:
        ip = lan_ip()
        if ip:
            links["lan_join_url"] = f"http://{ip}:{lan_port}/room/{room.id}"
    return links


@router.post(
    "/rooms",
    response_model=RoomCreated,
    status_code=201,
    responses=ERROR_RESPONSES,
    summary="Create a live interpretation room",
    description="Returns a secret `host_token` (keep it private) and a `join_url` to share with the audience.",
)
async def create_room(body: RoomCreate, request: Request, rooms: RoomsDep) -> dict[str, Any]:
    room = rooms.create(title=body.title, src_lang=body.src_lang, targets=body.targets, pin=body.pin)
    lan_port = getattr(request.app.state, "lan_port", None)
    return {
        "id": room.id,
        "host_token": room.host_token,
        **room_links(room, public_base_url(request), lan_port=lan_port),
    }


@router.get(
    "/rooms/{room_id}", response_model=RoomInfo, responses=ERROR_RESPONSES, summary="Room info (public)"
)
async def room_info(room_id: str, rooms: RoomsDep) -> dict[str, Any]:
    room = rooms.get(room_id)
    if room is None:
        raise not_found("Room")
    return room.info()


@router.delete("/rooms/{room_id}", status_code=204, summary="Close a room (requires the host token)")
async def close_room(
    room_id: str, token: Annotated[str, Query(description="host_token")], rooms: RoomsDep
) -> None:
    room = rooms.get(room_id)
    if room is None:
        raise not_found("Room")
    if not room.check_token(token):
        raise APIError(401, "unauthorized", "Invalid host token.")
    await rooms.close_room(room, reason="closed_by_host")


# ── WebSockets ──────────────────────────────────────────────────────────────
async def _reject(ws: WebSocket, code: str, message: str, close_code: int) -> None:
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"type": "error", "code": code, "message": message}))
        await ws.close(code=close_code, reason=code)


def _error_event(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}


def _ws_rooms(ws: WebSocket) -> RoomManager | None:
    return getattr(ws.app.state, "rooms", None)


@router.websocket("/rooms/{room_id}/host")
async def host_socket(ws: WebSocket, room_id: str, token: str = "") -> None:
    """Host stream: first `{"type":"start",...}`, then binary PCM16 LE mono frames, `{"type":"stop"}` to end."""
    await ws.accept()
    rooms = _ws_rooms(ws)
    room = rooms.get(room_id) if rooms else None
    if rooms is None or room is None:
        await _reject(ws, "not_found", "Room not found.", 4404)
        return
    if not room.check_token(token):
        await _reject(ws, "unauthorized", "Invalid or missing host token.", 4401)
        return
    try:
        sub = rooms.attach_host(room, ws)
    except RoomError as exc:
        await _reject(ws, exc.code, str(exc), exc.close_code)
        return
    sender = asyncio.create_task(sub.run_sender())
    warned_not_started = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if room.session is None:
                    if not warned_not_started:
                        sub.push(_error_event("not_started", 'Send {"type":"start"} before audio frames.'))
                        warned_not_started = True
                    continue
                try:
                    rooms.feed(room, data)
                except Exception as exc:  # never kill the socket for one bad frame
                    logger.warning("feed_pcm16 failed in room %s: %s", room.id, exc)
                continue
            text = msg.get("text")
            if not text:
                continue
            await _handle_host_command(rooms, room, sub, text)
            warned_not_started = False
    except WebSocketDisconnect:
        pass
    finally:
        await rooms.detach_host(room, sub)
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await sender


async def _handle_host_command(rooms: RoomManager, room: Room, sub: Subscriber, text: str) -> None:
    try:
        cmd = json.loads(text)
    except json.JSONDecodeError:
        sub.push(_error_event("invalid_input", "Text frames must be JSON."))
        return
    ctype = cmd.get("type") if isinstance(cmd, dict) else None
    try:
        if ctype == "start":
            await rooms.start_session(
                room,
                sample_rate=int(cmd.get("sample_rate", 16_000)),
                encoding=str(cmd.get("encoding", "pcm_s16le")),
            )
        elif ctype == "stop":
            await rooms.stop_session(room)
        elif ctype == "ping":
            sub.push({"type": "pong"})
        else:
            sub.push(_error_event("invalid_input", f"Unknown command {ctype!r}. Use start | stop | ping."))
    except AfriLinguaError as exc:
        sub.push(_error_event(exc.code, str(exc)))
    except Exception as exc:
        logger.exception("Host command %r failed in room %s", ctype, room.id)
        sub.push(_error_event("internal_error", f"{type(exc).__name__}: {exc}"))


@router.websocket("/rooms/{room_id}/listen")
async def listen_socket(
    ws: WebSocket, room_id: str, lang: str = "english", audio: str = "0", pin: str | None = None
) -> None:
    """Listener stream. Query: `lang`, `audio=1` for synthesized speech, `pin` if the room has one.

    Clients may send `{"type":"subscribe","lang":"igbo","audio":true}` to switch without reconnecting.
    """
    await ws.accept()
    rooms = _ws_rooms(ws)
    room = rooms.get(room_id) if rooms else None
    if rooms is None or room is None:
        await _reject(ws, "not_found", "Room not found.", 4404)
        return
    if not room.check_pin(pin):
        await _reject(ws, "unauthorized", "Invalid or missing room PIN.", 4403)
        return
    try:
        sub = await rooms.add_listener(room, ws, lang=lang, audio=audio.lower() in _TRUE)
    except RoomError as exc:
        await _reject(ws, exc.code, str(exc), exc.close_code)
        return
    except AfriLinguaError as exc:
        await _reject(ws, exc.code, str(exc), 4400)
        return
    sender = asyncio.create_task(sub.run_sender())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            text = msg.get("text")
            if not text:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(cmd, dict):
                continue
            if cmd.get("type") == "subscribe":
                try:
                    audio_flag = cmd.get("audio")
                    await rooms.update_listener(
                        room,
                        sub,
                        lang=cmd.get("lang"),
                        audio=None if audio_flag is None else str(audio_flag).lower() in _TRUE | {"true"},
                    )
                except AfriLinguaError as exc:
                    sub.push(_error_event(exc.code, str(exc)))
            elif cmd.get("type") == "ping":
                sub.push({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await rooms.remove_listener(room, sub)
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await sender
