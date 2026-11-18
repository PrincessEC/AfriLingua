"""
Backend contracts.

Every ML engine (CTranslate2, Transformers, faster-whisper, MMS, ONNX VITS …)
implements one of these Protocols. Higher layers (engine facade, pipeline,
API, MCP) depend only on these contracts, never on a concrete library.

Conventions
-----------
* Audio in: ``np.ndarray`` float32, mono, 16 kHz, values in [-1, 1].
* Audio out: ``(waveform float32 mono, sample_rate)``.
* Languages are passed as *canonical registry names* ("yoruba"), never raw
  model codes; backends map names to their own codes via the registry.
* ``load()`` is idempotent and may be slow (download + init). Inference
  methods are **not** required to be thread-safe — the ModelManager
  serializes access per backend instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    segments: tuple[Segment, ...] = field(default_factory=tuple)
    confidence: float | None = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class SpeechAudio:
    waveform: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return float(len(self.waveform)) / float(self.sample_rate or 1)


@runtime_checkable
class Backend(Protocol):
    """Lifecycle shared by every backend."""

    name: str

    def load(self) -> None: ...

    def is_loaded(self) -> bool: ...

    def unload(self) -> None: ...


@runtime_checkable
class MTBackend(Backend, Protocol):
    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        beam_size: int = 1,
        max_length: int = 256,
    ) -> list[str]: ...


@runtime_checkable
class ASRBackend(Backend, Protocol):
    def languages(self) -> list[str]: ...

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript: ...


@runtime_checkable
class TTSBackend(Backend, Protocol):
    def languages(self) -> list[str]: ...

    def synthesize(self, text: str, lang: str) -> SpeechAudio: ...


@runtime_checkable
class LanguageIdentifier(Protocol):
    def detect(self, text: str) -> tuple[str, float]:
        """Return (canonical language name | "unknown", confidence 0..1)."""
        ...
