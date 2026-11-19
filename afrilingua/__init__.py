"""
AfriLingua: fast, lightweight AI translation and speech for African languages.

Quick use (models download on first call, then everything runs offline)::

    import afrilingua

    afrilingua.translate("Good morning, everyone.", to="yoruba")
    # -> e.g. "Ẹ káàárọ̀, gbogbo yín." (illustrative)

    afrilingua.translate(["Thank you", "Welcome"], to="hausa", source="english")
    text = afrilingua.transcribe(audio_16k_float32, lang="igbo")
    speech = afrilingua.speak("Ndewo", lang="igbo")   # SpeechAudio(waveform, sample_rate)

For full control (custom backends, translation memory, streaming sessions) use
:func:`afrilingua.engine.build_engine` and :class:`afrilingua.engine.AfriLinguaEngine`.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from afrilingua.backends.base import SpeechAudio
    from afrilingua.engine import AfriLinguaEngine

__all__ = ["__version__", "get_engine", "languages", "speak", "transcribe", "translate"]

__version__ = "2.0.0.dev0"

_engine: AfriLinguaEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> AfriLinguaEngine:
    """Return the shared, lazily built engine (models load on first use, not here)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from afrilingua.core.settings import apply_runtime_environment
                from afrilingua.engine import build_engine

                apply_runtime_environment()
                _engine = build_engine()
    return _engine


@overload
def translate(text: str, to: str, *, source: str = "auto", mode: str = "live") -> str: ...
@overload
def translate(text: Sequence[str], to: str, *, source: str = "auto", mode: str = "live") -> list[str]: ...
def translate(
    text: str | Sequence[str], to: str, *, source: str = "auto", mode: str = "live"
) -> str | list[str]:
    """Translate one string or a batch.

    Args:
        text: A string or a list of strings.
        to: Target language name or code ("yoruba", "yo", "yor_Latn").
        source: Source language, or "auto" to detect it.
        mode: "live" (greedy, lowest latency) or "document" (beam search, higher quality).
    """
    engine = get_engine()
    if isinstance(text, str):
        return engine.translate_one(text, source, to, mode=mode).text  # type: ignore[arg-type]
    return [r.text for r in engine.translate(list(text), source, to, mode=mode)]  # type: ignore[arg-type]


def transcribe(audio: Any, lang: str) -> str:
    """Speech to text. ``audio`` is mono float32 at 16 kHz (numpy array)."""
    return get_engine().transcribe(audio, lang).text


def speak(text: str, lang: str) -> SpeechAudio:
    """Text to speech. Returns ``SpeechAudio(waveform, sample_rate)``."""
    return get_engine().synthesize(text, lang)


def languages() -> list[dict[str, Any]]:
    """Supported languages and their capabilities (translation, speech in, speech out)."""
    from afrilingua.core.languages import get_registry

    return [spec.to_public_dict() for spec in get_registry()]
