"""
Live interpretation rooms (docs/api-contract.md §6).

One host streams PCM16 audio into a :class:`StreamingSessionProtocol`; any number
of listeners subscribe with a language (and optionally audio). Events produced on
the session's worker thread are marshalled onto the event loop with
``loop.call_soon_threadsafe`` and fanned out through per-connection bounded
queues, so a slow client can never block the session or other listeners.

Fan-out rules
-------------
* ``partial`` / ``final``      → everyone (host + all listeners)
* ``translation``              → host + listeners of that language
* ``audio``                    → listeners of that language who asked for audio
* ``metrics``                  → host only
* ``error`` / ``status``       → everyone
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from afrilingua.core.errors import CapabilityUnavailableError, InvalidInputError
from afrilingua.pipeline import Event, EventCallback, StreamingSessionProtocol

logger = logging.getLogger(__name__)

ROOM_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o/1/l/i
DEFAULT_IDLE_TTL_S = 600.0
DEFAULT_HISTORY = 50


class _WebSocketLike(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


SessionFactory = Callable[..., StreamingSessionProtocol]


def default_session_factory(engine: Any, **kwargs: Any) -> StreamingSessionProtocol:
    try:
        from afrilingua.pipeline.streaming import StreamingSession
    except ImportError as exc:  # engine workstream not merged yet
        raise CapabilityUnavailableError(f"Live streaming pipeline is not available: {exc}") from exc
    return StreamingSession(engine, **kwargs)


def new_room_code() -> str:
    raw = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(6))
    return f"{raw[:3]}-{raw[3:]}"


def normalize_room_id(room_id: str) -> str:
    return room_id.strip().lower()


# ── connections ─────────────────────────────────────────────────────────────
class Subscriber:
    """A connected WebSocket (host or listener) with a bounded, lossy send queue."""

    def __init__(
        self,
        ws: _WebSocketLike,
        *,
        role: str,
        lang: str | None = None,
        audio: bool = False,
        max_queue: int = 256,
        max_audio: int = 8,
    ) -> None:
        self.id = secrets.token_hex(4)
        self.ws = ws
        self.role = role
        self.lang = lang
        self.audio = audio
        self.max_queue = max_queue
        self.max_audio = max_audio
        self.dropped = 0
        self.closed = False
        self._q: deque[Event] = deque()
        self._wake = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()

    def push(self, event: Event) -> None:
        """Enqueue without blocking (must be called on the event loop)."""
        if self.closed:
            return
        etype = event.get("type")
        if etype == "partial":  # only the newest partial matters
            self._remove_first(lambda e: e.get("type") == "partial", count_drop=False)
        elif etype == "audio" and sum(1 for e in self._q if e.get("type") == "audio") >= self.max_audio:
            self._remove_first(lambda e: e.get("type") == "audio")
        while len(self._q) >= self.max_queue:
            if not (
                self._remove_first(lambda e: e.get("type") == "audio")
                or self._remove_first(lambda e: e.get("type") in {"partial", "metrics"})
            ):
                self._q.popleft()
                self.dropped += 1
        self._q.append(event)
        self.drained.clear()
        self._wake.set()

    def _remove_first(self, pred: Callable[[Event], bool], *, count_drop: bool = True) -> bool:
        for i, e in enumerate(self._q):
            if pred(e):
                del self._q[i]
                if count_drop:
                    self.dropped += 1
                return True
        return False

    @property
    def pending(self) -> int:
        return len(self._q)

    async def run_sender(self) -> None:
        try:
            while not self.closed:
                await self._wake.wait()
                self._wake.clear()
                while self._q and not self.closed:
                    event = self._q.popleft()
                    await self.ws.send_text(json.dumps(event, ensure_ascii=False))
                if not self._q:
                    self.drained.set()
        except asyncio.CancelledError:
            raise
        except Exception:  # client went away mid-send
            self.closed = True
            self.drained.set()

    def close(self) -> None:
        self.closed = True
        self._wake.set()
        self.drained.set()


# ── room ────────────────────────────────────────────────────────────────────
@dataclass
class Room:
    id: str
    title: str
    src_lang: str
    targets: list[str]
    host_token: str
    pin: str | None = None
    created_at: float = field(default_factory=time.time)
    host: Subscriber | None = None
    listeners: dict[str, Subscriber] = field(default_factory=dict)
    session: StreamingSessionProtocol | None = None
    sample_rate: int = 16_000
    idle_since: float = field(default_factory=time.monotonic)
    history: dict[str, deque[tuple[int, Event]]] = field(default_factory=dict)
    closed: bool = False
    _counter: int = 0

    @property
    def live(self) -> bool:
        return self.session is not None

    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sub in self.listeners.values():
            if sub.lang:
                counts[sub.lang] = counts.get(sub.lang, 0) + 1
        return counts

    def session_targets(self) -> list[str]:
        langs = set(self.targets) | {s.lang for s in self.listeners.values() if s.lang}
        langs.discard(self.src_lang)
        return sorted(langs)

    def audio_targets(self) -> list[str]:
        return sorted(
            {s.lang for s in self.listeners.values() if s.audio and s.lang and s.lang != self.src_lang}
        )

    def check_pin(self, pin: str | None) -> bool:
        if not self.pin:
            return True
        return bool(pin) and secrets.compare_digest(str(pin).encode(), self.pin.encode())

    def check_token(self, token: str | None) -> bool:
        return bool(token) and secrets.compare_digest(str(token).encode(), self.host_token.encode())

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "src_lang": self.src_lang,
            "targets": list(self.targets),
            "listeners": len(self.listeners),
            "live": self.live,
            "pin_required": bool(self.pin),
            "host_connected": self.host is not None,
            "languages": self.language_counts(),
            "created_at": self.created_at,
        }

    def status_event(self) -> Event:
        return {
            "type": "status",
            "live": self.live,
            "listeners": len(self.listeners),
            "host": self.host is not None,
            "languages": self.language_counts(),
        }


class RoomError(Exception):
    def __init__(self, code: str, message: str, close_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.close_code = close_code


class RoomManager:
    def __init__(
        self,
        engine: Any,
        *,
        session_factory: SessionFactory | None = None,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        history_size: int = DEFAULT_HISTORY,
        reap_interval_s: float = 30.0,
        max_rooms: int = 200,
        max_listeners: int = 1000,
    ) -> None:
        self.engine = engine
        self.session_factory: SessionFactory = session_factory or default_session_factory
        self.idle_ttl_s = idle_ttl_s
        self.history_size = history_size
        self.reap_interval_s = reap_interval_s
        self.max_rooms = max_rooms
        self.max_listeners = max_listeners
        self.rooms: dict[str, Room] = {}
        self._reaper: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_forever(), name="afrilingua-room-reaper")

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reaper
            self._reaper = None
        for room in list(self.rooms.values()):
            await self.close_room(room, reason="server_shutdown")

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self.reap_interval_s)
            try:
                await self.reap()
            except Exception:
                logger.exception("Room reaper failed")

    async def reap(self, now: float | None = None) -> list[str]:
        now = now if now is not None else time.monotonic()
        closed = []
        for room in list(self.rooms.values()):
            if room.host is None and not room.listeners and now - room.idle_since >= self.idle_ttl_s:
                await self.close_room(room, reason="idle")
                closed.append(room.id)
        return closed

    def summary(self) -> dict[str, int]:
        return {
            "rooms": len(self.rooms),
            "live": sum(1 for r in self.rooms.values() if r.live),
            "listeners": sum(len(r.listeners) for r in self.rooms.values()),
        }

    # ── rooms ───────────────────────────────────────────────────────────────
    def create(self, *, title: str, src_lang: str, targets: list[str], pin: str | None = None) -> Room:
        if len(self.rooms) >= self.max_rooms:
            raise InvalidInputError(f"Too many open rooms (max {self.max_rooms}).")
        registry = self.engine.registry
        src = registry.normalize(src_lang)
        tgts: list[str] = []
        for t in targets:
            name = registry.normalize(t)
            if name != src and name not in tgts:
                tgts.append(name)
        pin = (pin or "").strip() or None
        room_id = new_room_code()
        while room_id in self.rooms:
            room_id = new_room_code()
        room = Room(
            id=room_id,
            title=(title or "Live session").strip() or "Live session",
            src_lang=src,
            targets=tgts,
            host_token=secrets.token_urlsafe(24),
            pin=pin,
        )
        self.rooms[room_id] = room
        logger.info("Room %s created (%s → %s)", room_id, src, ", ".join(tgts) or "listener languages")
        return room

    def get(self, room_id: str) -> Room | None:
        return self.rooms.get(normalize_room_id(room_id))

    async def close_room(self, room: Room, *, reason: str = "closed") -> None:
        room.closed = True
        self.rooms.pop(room.id, None)
        await self._close_session(room)
        bye = {"type": "status", "live": False, "listeners": 0, "closed": True, "reason": reason}
        subs = [*room.listeners.values(), *([room.host] if room.host else [])]
        for sub in subs:
            sub.push(bye)
        deadline = 0.5
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*(s.drained.wait() for s in subs if not s.closed)), deadline
            )
        for sub in subs:
            sub.close()
            with contextlib.suppress(Exception):
                await sub.ws.close(code=4410, reason="room_closed")
        room.listeners.clear()
        room.host = None
        logger.info("Room %s closed (%s)", room.id, reason)

    # ── host ────────────────────────────────────────────────────────────────
    def attach_host(self, room: Room, ws: _WebSocketLike) -> Subscriber:
        if room.closed:
            raise RoomError("not_found", "Room is closed.", 4404)
        if room.host is not None and not room.host.closed:
            raise RoomError(
                "host_already_connected", "Another host connection is active for this room.", 4409
            )
        sub = Subscriber(ws, role="host")
        room.host = sub
        self._replay(room, sub, room.src_lang)
        self.broadcast_status(room)
        return sub

    async def detach_host(self, room: Room, sub: Subscriber) -> None:
        sub.close()
        if room.host is sub:
            room.host = None
            await self._close_session(room)
            room.idle_since = time.monotonic()
            self.broadcast_status(room)

    async def start_session(
        self, room: Room, *, sample_rate: int = 16_000, encoding: str = "pcm_s16le"
    ) -> None:
        if encoding.lower() not in {"pcm_s16le", "pcm16", "s16le", "linear16"}:
            raise InvalidInputError(f"Unsupported encoding {encoding!r}; send pcm_s16le.")
        if not 8_000 <= int(sample_rate) <= 192_000:
            raise InvalidInputError(f"Unsupported sample_rate {sample_rate}.")
        if room.session is not None:
            room.sample_rate = int(sample_rate)
            return
        loop = asyncio.get_running_loop()
        self._loop = loop

        def on_event(event: Event) -> None:  # called on the session worker thread
            with contextlib.suppress(RuntimeError):  # loop closed during shutdown
                loop.call_soon_threadsafe(self.dispatch, room, event)

        callback: EventCallback = on_event
        session = await asyncio.to_thread(
            self.session_factory,
            self.engine,
            src_lang=room.src_lang,
            targets=room.session_targets(),
            synthesize_audio=bool(room.audio_targets()),
            on_event=callback,
        )
        await asyncio.to_thread(session.start)
        room.session = session
        room.sample_rate = int(sample_rate)
        self._apply_audio_targets(room)
        self.broadcast_status(room)

    def feed(self, room: Room, data: bytes) -> None:
        if room.session is None:
            raise RoomError("not_started", 'Send {"type":"start"} before audio frames.', 0)
        room.session.feed_pcm16(data, sample_rate=room.sample_rate)

    async def stop_session(self, room: Room) -> None:
        await self._close_session(room)
        self.broadcast_status(room)

    async def _close_session(self, room: Room) -> None:
        session, room.session = room.session, None
        if session is not None:
            try:
                await asyncio.to_thread(session.close)
            except Exception:
                logger.exception("Error closing streaming session for room %s", room.id)

    # ── listeners ───────────────────────────────────────────────────────────
    async def add_listener(self, room: Room, ws: _WebSocketLike, *, lang: str, audio: bool) -> Subscriber:
        if room.closed:
            raise RoomError("not_found", "Room is closed.", 4404)
        if len(room.listeners) >= self.max_listeners:
            raise RoomError("room_full", "This room is full.", 4429)
        sub = Subscriber(ws, role="listener", lang=self.engine.registry.normalize(lang), audio=audio)
        room.listeners[sub.id] = sub
        self._replay(room, sub, sub.lang)
        await self._targets_changed(room)
        self.broadcast_status(room)
        return sub

    async def update_listener(
        self, room: Room, sub: Subscriber, *, lang: str | None, audio: bool | None
    ) -> None:
        if lang is not None:
            new_lang = self.engine.registry.normalize(lang)
            if new_lang != sub.lang:
                sub.lang = new_lang
                self._replay(room, sub, new_lang, translations_only=True)
        if audio is not None:
            sub.audio = audio
        await self._targets_changed(room)
        self.broadcast_status(room)

    async def remove_listener(self, room: Room, sub: Subscriber) -> None:
        sub.close()
        if room.listeners.pop(sub.id, None) is None:
            return
        if not room.listeners and room.host is None:
            room.idle_since = time.monotonic()
        if not room.closed:
            await self._targets_changed(room)
            self.broadcast_status(room)

    async def _targets_changed(self, room: Room) -> None:
        session = room.session
        if session is None:
            return
        try:
            await asyncio.to_thread(session.set_targets, room.session_targets())
        except Exception:
            logger.exception("set_targets failed for room %s", room.id)
        self._apply_audio_targets(room)

    @staticmethod
    def _apply_audio_targets(room: Room) -> None:
        """Only synthesize speech when someone asked for it (optional session APIs)."""
        session = room.session
        if session is None:
            return
        langs = room.audio_targets()
        with contextlib.suppress(Exception):
            if hasattr(session, "set_audio_targets"):
                session.set_audio_targets(langs)
            elif hasattr(session, "set_synthesize_audio"):
                session.set_synthesize_audio(bool(langs))
            elif hasattr(session, "synthesize_audio"):
                session.synthesize_audio = bool(langs)

    # ── fan-out ─────────────────────────────────────────────────────────────
    def broadcast_status(self, room: Room) -> None:
        self._send_all(room, room.status_event())

    def _send_all(self, room: Room, event: Event) -> None:
        if room.host is not None:
            room.host.push(event)
        for sub in room.listeners.values():
            sub.push(event)

    def _remember(self, room: Room, lang: str, event: Event) -> None:
        room._counter += 1
        buf = room.history.get(lang)
        if buf is None:
            buf = room.history[lang] = deque(maxlen=self.history_size)
        buf.append((room._counter, event))

    def _replay(
        self, room: Room, sub: Subscriber, lang: str | None, *, translations_only: bool = False
    ) -> None:
        items: list[tuple[int, Event]] = []
        if not translations_only:
            items.extend(room.history.get(room.src_lang, ()))
        if lang and lang != room.src_lang:
            items.extend(room.history.get(lang, ()))
        for _, event in sorted(items, key=lambda x: x[0]):
            sub.push({**event, "replay": True})

    def dispatch(self, room: Room, event: Event) -> None:
        """Route one pipeline event (runs on the event loop)."""
        if room.closed:
            return
        etype = event.get("type")
        lang = event.get("lang")
        if etype in {"partial", "final"}:
            if etype == "final":
                self._remember(room, lang or room.src_lang, event)
            self._send_all(room, event)
        elif etype == "translation":
            if lang:
                self._remember(room, lang, event)
            if room.host is not None:
                room.host.push(event)
            for sub in room.listeners.values():
                if sub.lang == lang:
                    sub.push(event)
        elif etype == "audio":
            for sub in room.listeners.values():
                if sub.audio and sub.lang == lang:
                    sub.push(event)
        elif etype == "metrics":
            if room.host is not None:
                room.host.push(event)
        else:  # error / status / unknown
            self._send_all(room, event)
