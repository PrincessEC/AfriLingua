"""Test doubles for the API / rooms / MCP tests (not a conftest: import explicitly)."""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, ClassVar

import numpy as np
import soundfile as sf

from afrilingua.engine import TMMatch


def wav_bytes(seconds: float = 0.5, sr: int = 16_000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_b64(seconds: float = 0.5) -> str:
    return base64.b64encode(wav_bytes(seconds)).decode()


class FakeDetector:
    def detect(self, text: str) -> tuple[str, float]:
        return ("yoruba", 0.91) if "ẹ" in text.lower() else ("english", 0.88)


class FakeTM:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, str, str]] = []

    def lookup(self, text: str, src_lang: str, tgt_lang: str, *, k: int = 3) -> list[TMMatch]:
        out = [
            TMMatch(
                id=str(i),
                source=s,
                target=t,
                score=1.0 if s == text else 0.5,
                approved=True,
                origin="community",
            )
            for i, (s, t, sl, tl) in enumerate(self.entries)
            if sl == src_lang and tl == tgt_lang
        ]
        return sorted(out, key=lambda m: -m.score)[:k]

    def add(self, source, target, src_lang, tgt_lang, *, origin="mt", approved=False) -> str:
        self.entries.append((source, target, src_lang, tgt_lang))
        return str(len(self.entries) - 1)

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self.entries), "approved": len(self.entries), "by_pair": {}}


class FakeContrib:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.submitted.append(payload)
        return {
            **payload,
            "id": f"c{len(self.submitted)}",
            "status": "pending",
            "votes_up": 0,
            "votes_down": 0,
        }


class FakeStreamingSession:
    """Emits partial → final → translation(per target) → audio(per target) for every fed chunk.

    Events are emitted from a worker thread, like the real pipeline.
    """

    instances: ClassVar[list[FakeStreamingSession]] = []

    def __init__(
        self, engine: Any, *, src_lang: str, targets: list[str], synthesize_audio: bool, on_event
    ) -> None:
        self.engine = engine
        self.src_lang = src_lang
        self.targets = list(targets)
        self.synthesize_audio = synthesize_audio
        self.on_event = on_event
        self.seq = 0
        self.fed = 0
        self.started = False
        self.closed = False
        self.target_history: list[list[str]] = [list(targets)]
        self.audio_history: list[list[str]] = []
        FakeStreamingSession.instances.append(self)

    def start(self) -> None:
        self.started = True

    def feed_pcm16(self, data: bytes, sample_rate: int = 16_000) -> None:
        self.fed += len(data)
        self.seq += 1
        seq, targets = self.seq, list(self.targets)
        threading.Thread(target=self._emit, args=(seq, targets), daemon=True).start()

    def _emit(self, seq: int, targets: list[str]) -> None:
        self.on_event({"type": "partial", "seq": seq, "lang": self.src_lang, "text": f"utterance {seq}"})
        self.on_event(
            {
                "type": "final",
                "seq": seq,
                "lang": self.src_lang,
                "text": f"Utterance {seq}.",
                "ts": time.time(),
            }
        )
        for lang in targets:
            self.on_event(
                {
                    "type": "translation",
                    "seq": seq,
                    "lang": lang,
                    "text": f"[{lang}] {seq}",
                    "origin": "mt",
                    "latency_ms": 5,
                }
            )
        self.on_event({"type": "metrics", "seq": seq, "asr_ms": 1, "mt_ms": 1, "tts_ms": 1})
        for lang in targets:
            self.on_event({"type": "audio", "seq": seq, "lang": lang, "format": "wav", "data": "UklGRg=="})

    def feed_float32(self, samples: Any, sample_rate: int = 16_000) -> None:
        self.feed_pcm16(b"\x00\x00", sample_rate)

    def set_targets(self, targets: list[str]) -> None:
        self.targets = list(targets)
        self.target_history.append(list(targets))

    def set_audio_targets(self, langs: list[str]) -> None:
        self.audio_history.append(list(langs))

    def close(self) -> None:
        self.closed = True
