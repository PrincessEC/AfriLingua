"""
Speech synthesis for every registry language with one backend.

* Per-language MMS VITS voices (``LanguageSpec.mms_tts_model``; English uses
  ``settings.model_mms_tts_english``), loaded lazily on first use and kept in
  a small LRU (3 voices on ``cpu-lite``, 6 otherwise).
* Text front-end: NFC, number expansion (English), unsupported-character
  stripping against the voice vocabulary, sentence splitting with short
  silences between sentences.
* English fallback to the OS voice via pyttsx3 when the MMS English voice is
  unavailable (or ``TTS_EN_BACKEND=pyttsx3``). pyttsx3 is not thread-safe, so
  it runs on one dedicated thread.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from afrilingua.backends._common import (
    configure_torch_threads,
    dtype_kwargs,
    empty_cuda_cache,
    hf_kwargs,
    is_oom,
    module_available,
)
from afrilingua.backends.base import SpeechAudio
from afrilingua.backends.text_frontend import clean_tts_text, filter_to_vocab, split_sentences
from afrilingua.core.errors import CapabilityUnavailableError, ModelLoadError
from afrilingua.core.hardware import RuntimePlan, resolve_runtime
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_SENTENCE_GAP_S = 0.18


@dataclass
class _Voice:
    model_id: str
    tokenizer: Any
    model: Any
    vocab: frozenset[str]
    lowercase: bool
    sample_rate: int
    device: str


class Pyttsx3Voice:
    """OS text-to-speech (SAPI5 / NSSpeech / eSpeak) on a single dedicated thread."""

    def __init__(self, rate: int = 160) -> None:
        self._rate = rate
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyttsx3")
        self._engine: Any = None

    def _init_engine(self) -> None:
        if self._engine is not None:
            return
        try:  # COM must be initialised on the thread that owns SAPI5.
            import pythoncom  # type: ignore[import-not-found]

            pythoncom.CoInitialize()
        except Exception:
            pass
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", self._rate)

    def _synth(self, text: str) -> SpeechAudio:
        import soundfile as sf

        self._init_engine()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._engine.save_to_file(text, path)
            self._engine.runAndWait()
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return SpeechAudio(data.mean(axis=1).astype(np.float32), int(sr))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def synthesize(self, text: str) -> SpeechAudio:
        """Synthesize ``text`` (blocking) on the dedicated pyttsx3 thread."""
        return self._executor.submit(self._synth, text).result(timeout=120)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class MMSTTSBackend:
    """:class:`~afrilingua.backends.base.TTSBackend` for all registry languages."""

    name = "mms-tts"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        plan: RuntimePlan | None = None,
        registry: LanguageRegistry | None = None,
        max_voices: int | None = None,
        english_backend: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.plan = plan or resolve_runtime()
        self.registry = registry or get_registry()
        self.max_voices = max_voices or (3 if self.plan.profile == "cpu-lite" else 6)
        self.english_backend = english_backend or self.settings.tts_en_backend
        self._voices: OrderedDict[str, _Voice] = OrderedDict()
        self._failed: dict[str, str] = {}
        self._pyttsx3: Pyttsx3Voice | None = None
        self._loaded = False
        self._lock = threading.RLock()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Voices load lazily per language; this only marks the backend ready."""
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        with self._lock:
            self._voices.clear()
            if self._pyttsx3 is not None:
                self._pyttsx3.close()
                self._pyttsx3 = None
            self._loaded = False
        empty_cuda_cache()

    def languages(self) -> list[str]:
        """Languages with a configured voice (English always has a fallback)."""
        return self.registry.tts_languages()

    def loaded_voices(self) -> list[str]:
        return list(self._voices)

    # ── voices ──────────────────────────────────────────────────────────────
    def _model_id(self, lang: str) -> str | None:
        if lang == "english":
            return self.settings.model_mms_tts_english or self.registry.get(lang).mms_tts_model
        return self.registry.get(lang).mms_tts_model

    def _load_voice(self, lang: str, model_id: str) -> _Voice:
        import torch
        from transformers import AutoTokenizer, VitsModel

        kwargs = hf_kwargs(self.settings)
        logger.info("Loading MMS TTS voice for %s from %s ...", lang, model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        dtype_name = (
            "float16" if self.plan.device == "cuda" and self.plan.torch_dtype == "float16" else "float32"
        )
        model = VitsModel.from_pretrained(model_id, **kwargs, **dtype_kwargs(dtype_name))
        model.eval()
        device = self.plan.device
        if device == "cuda":
            try:
                model.to("cuda")
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                empty_cuda_cache()
                model.to(device="cpu", dtype=torch.float32)
                device = "cpu"
        if device == "cpu":
            configure_torch_threads(self.plan.threads)
        vocab = frozenset(tokenizer.get_vocab())
        letters = [t for t in vocab if len(t) == 1 and t.isalpha()]
        lowercase = bool(letters) and not any(t.isupper() for t in letters)
        return _Voice(model_id, tokenizer, model, vocab, lowercase, int(model.config.sampling_rate), device)

    def _voice(self, lang: str) -> _Voice | None:
        with self._lock:
            voice = self._voices.get(lang)
            if voice is not None:
                self._voices.move_to_end(lang)
                return voice
            model_id = self._model_id(lang)
            if not model_id:
                return None
            if lang in self._failed:
                if lang == "english":
                    return None
                raise ModelLoadError(f"TTS voice for {lang} failed to load earlier: {self._failed[lang]}")
            try:
                voice = self._load_voice(lang, model_id)
            except Exception as exc:
                self._failed[lang] = f"{type(exc).__name__}: {exc}"
                if lang == "english" and self._english_fallback_allowed():
                    logger.warning("MMS English voice unavailable (%s); using pyttsx3.", exc)
                    return None
                raise ModelLoadError(f"Could not load TTS voice {model_id!r} for {lang}: {exc}") from exc
            self._voices[lang] = voice
            while len(self._voices) > self.max_voices:
                evicted, _ = self._voices.popitem(last=False)
                logger.info("Evicted TTS voice %s (LRU, max %d)", evicted, self.max_voices)
            return voice

    def _english_fallback_allowed(self) -> bool:
        return self.english_backend in {"auto", "pyttsx3"} and module_available("pyttsx3")

    def _english_os_voice(self) -> Pyttsx3Voice:
        if self._pyttsx3 is None:
            self._pyttsx3 = Pyttsx3Voice()
        return self._pyttsx3

    # ── synthesis ───────────────────────────────────────────────────────────
    def _synth_sentence(self, voice: _Voice, sentence: str) -> np.ndarray:
        import torch

        text = sentence.lower() if voice.lowercase else sentence
        text = filter_to_vocab(text, voice.vocab)
        if not any(ch.isalnum() for ch in text):
            return np.zeros(0, dtype=np.float32)
        inputs = voice.tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            return np.zeros(0, dtype=np.float32)
        inputs = {k: v.to(voice.device) for k, v in inputs.items()}
        with torch.inference_mode():
            wave = voice.model(**inputs).waveform
        return wave.squeeze().float().cpu().numpy().astype(np.float32).reshape(-1)

    def synthesize(self, text: str, lang: str) -> SpeechAudio:
        """Synthesize ``text`` in ``lang`` (canonical name) → mono float32 waveform."""
        spec = self.registry.get(lang)
        if not spec.supports_tts:
            raise CapabilityUnavailableError(f"No speech synthesis configured for {spec.name!r}.")
        cleaned = clean_tts_text(text, spec.name)
        if not cleaned:
            return SpeechAudio(np.zeros(0, dtype=np.float32), 16_000)

        use_os_voice = spec.name == "english" and self.english_backend == "pyttsx3"
        voice = None if use_os_voice else self._voice(spec.name)
        if voice is None:
            if spec.name != "english" or not self._english_fallback_allowed():
                raise CapabilityUnavailableError(f"No usable speech synthesis voice for {spec.name!r}.")
            return self._english_os_voice().synthesize(cleaned)

        gap = np.zeros(int(_SENTENCE_GAP_S * voice.sample_rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for sentence in split_sentences(cleaned, max_chars=220):
            wave = self._synth_sentence(voice, sentence)
            if wave.size:
                if pieces:
                    pieces.append(gap)
                pieces.append(wave)
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return SpeechAudio(audio, voice.sample_rate)
