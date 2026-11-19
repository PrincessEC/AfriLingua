"""
Default backend wiring.

:func:`register_default_backends` registers lazy factories in a
:class:`~afrilingua.models.manager.ModelManager` under the keys used by the
engine facade (``mt``, ``asr.en``, ``asr.mms``, ``tts``) plus ``vad``.
Choosing ``auto`` prefers the fast CPU backend (CTranslate2 / faster-whisper)
when its package imports and transparently falls back to the PyTorch
implementation — at registration time if the package is missing, or at load
time if the fast backend fails to load.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from afrilingua.backends._common import module_available
from afrilingua.backends.base import Backend, LanguageIdentifier
from afrilingua.core.settings import Settings, get_settings
from afrilingua.engine import ASR_EN_KEY, ASR_MMS_KEY, MT_KEY, TTS_KEY
from afrilingua.models.manager import ModelManager

logger = logging.getLogger(__name__)

VAD_KEY = "vad"


class FallbackBackend:
    """Try ``primary`` on load; if it fails, use ``fallback`` instead.

    Attribute access is delegated to whichever backend is active, so callers
    see the normal backend API (``translate_batch``, ``transcribe`` …).
    """

    def __init__(
        self, primary: Callable[[], Backend], fallback: Callable[[], Backend], *, label: str
    ) -> None:
        self._primary_factory = primary
        self._fallback_factory = fallback
        self._label = label
        self._active: Any = None
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return getattr(self._active, "name", f"{self._label} (not loaded)")

    def load(self) -> None:
        if self._active is not None and self._active.is_loaded():
            return
        if self._active is None:
            try:
                backend = self._primary_factory()
                backend.load()
                self._active = backend
                return
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s: fast backend failed to load (%s); falling back.", self._label, exc, exc_info=True
                )
                self._active = self._fallback_factory()
        self._active.load()

    def is_loaded(self) -> bool:
        return self._active is not None and self._active.is_loaded()

    def unload(self) -> None:
        if self._active is not None:
            self._active.unload()

    def __getattr__(self, item: str) -> Any:
        active = self.__dict__.get("_active")
        if active is None:
            raise AttributeError(item)
        return getattr(active, item)


def _mt_factory(settings: Settings) -> tuple[Callable[[], Backend], str]:
    from afrilingua.backends.mt_ct2 import CT2NLLBBackend
    from afrilingua.backends.mt_hf import HFNLLBBackend

    choice = settings.mt_backend
    if choice == "transformers":
        return HFNLLBBackend, "NLLB (Transformers)"
    if choice == "ctranslate2":
        return CT2NLLBBackend, "NLLB (CTranslate2)"
    if not module_available("ctranslate2"):
        logger.warning("ctranslate2 not installed; using the slower Transformers NLLB backend.")
        return HFNLLBBackend, "NLLB (Transformers fallback)"
    return lambda: FallbackBackend(CT2NLLBBackend, HFNLLBBackend, label="mt"), "NLLB (CTranslate2, auto)"


def _asr_en_factory(settings: Settings) -> tuple[Callable[[], Backend], str]:
    from afrilingua.backends.asr_whisper import FasterWhisperBackend, OpenAIWhisperBackend

    choice = settings.asr_en_backend
    if choice == "openai-whisper":
        return OpenAIWhisperBackend, "Whisper (openai-whisper)"
    if choice == "faster-whisper":
        return FasterWhisperBackend, "Whisper (faster-whisper)"
    if not module_available("faster_whisper"):
        logger.warning("faster-whisper not installed; using openai-whisper for English ASR.")
        return OpenAIWhisperBackend, "Whisper (openai-whisper fallback)"
    return (
        lambda: FallbackBackend(FasterWhisperBackend, OpenAIWhisperBackend, label="asr.en"),
        "Whisper (faster-whisper, auto)",
    )


def register_default_backends(
    manager: ModelManager, *, settings: Settings | None = None, replace: bool = False
) -> None:
    """Register lazy factories for ``mt``, ``asr.en``, ``asr.mms``, ``tts`` and ``vad``.

    Nothing is loaded here; models load on first use (or via ``manager.preload``).
    """
    from afrilingua.backends.asr_mms import MMSASRBackend
    from afrilingua.backends.tts_mms import MMSTTSBackend
    from afrilingua.backends.vad import SileroVADBackend

    s = settings or get_settings()
    mt, mt_desc = _mt_factory(s)
    asr_en, asr_desc = _asr_en_factory(s)
    manager.register(MT_KEY, mt, description=mt_desc, replace=replace)
    manager.register(ASR_EN_KEY, asr_en, description=asr_desc, replace=replace)
    manager.register(ASR_MMS_KEY, MMSASRBackend, description="MMS ASR (wav2vec2 + adapters)", replace=replace)
    manager.register(TTS_KEY, MMSTTSBackend, description="MMS TTS (VITS per language)", replace=replace)
    manager.register(VAD_KEY, SileroVADBackend, description="Silero VAD", replace=replace)


def build_detector() -> LanguageIdentifier:
    """The default text language identifier."""
    from afrilingua.backends.lid import HeuristicLanguageIdentifier

    return HeuristicLanguageIdentifier()
