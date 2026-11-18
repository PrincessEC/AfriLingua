"""Backend contracts and concrete implementations."""

from afrilingua.backends.base import (
    ASRBackend,
    Backend,
    LanguageIdentifier,
    MTBackend,
    Segment,
    SpeechAudio,
    Transcript,
    TTSBackend,
)

__all__ = [
    "ASRBackend",
    "Backend",
    "LanguageIdentifier",
    "MTBackend",
    "Segment",
    "SpeechAudio",
    "TTSBackend",
    "Transcript",
]
