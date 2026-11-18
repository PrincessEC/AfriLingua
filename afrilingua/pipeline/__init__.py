"""
Streaming interpretation pipeline contract.

A ``StreamingSession`` turns a live PCM stream (one speaker) into a sequence of
events for many target languages. It is transport-agnostic: the rooms
WebSocket, the desktop shell and the MCP live tool all drive it the same way.

Contract (implemented by ``afrilingua.pipeline.streaming.StreamingSession``)::

    session = StreamingSession(engine, src_lang="english", targets=["yoruba", "igbo"],
                               synthesize_audio=True, on_event=callback)
    session.start()
    session.feed_pcm16(chunk_bytes, sample_rate=16000)   # thread-safe, never blocks on models
    session.set_targets([...])                           # listeners joined/left
    session.close()                                      # flushes, stops workers

Events delivered to ``on_event`` (called from a worker thread; callers must
marshal to their own loop, e.g. ``loop.call_soon_threadsafe``) are plain dicts
matching docs/api-contract.md §6::

    {"type":"partial","seq":7,"lang":"english","text":"…"}
    {"type":"final","seq":7,"lang":"english","text":"…","ts":…}
    {"type":"translation","seq":7,"lang":"yoruba","text":"…","origin":"mt","latency_ms":412}
    {"type":"audio","seq":7,"lang":"yoruba","format":"wav","data":"<base64>"}
    {"type":"metrics","seq":7,"asr_ms":…,"mt_ms":…,"tts_ms":…}
    {"type":"error","code":"…","message":"…"}
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

Event = dict[str, Any]
EventCallback = Callable[[Event], None]


@runtime_checkable
class StreamingSessionProtocol(Protocol):
    src_lang: str

    def start(self) -> None: ...

    def feed_pcm16(self, data: bytes, sample_rate: int = 16_000) -> None: ...

    def feed_float32(self, samples: Any, sample_rate: int = 16_000) -> None: ...

    def set_targets(self, targets: list[str]) -> None: ...

    def close(self) -> None: ...


__all__ = ["Event", "EventCallback", "StreamingSessionProtocol"]
