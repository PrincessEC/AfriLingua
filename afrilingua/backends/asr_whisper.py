"""
English (and other Whisper-language) speech recognition.

* :class:`FasterWhisperBackend` — CTranslate2 Whisper, int8 on CPU (default).
* :class:`OpenAIWhisperBackend` — the reference PyTorch implementation (fallback).

Both decode greedily by default, never condition on previous text (prevents
hallucination loops in live use), skip timestamp tokens for speed, and leave
voice-activity detection to the pipeline (``vad_filter=False``).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from afrilingua.backends._common import (
    configure_torch_threads,
    ct2_compute_type,
    ct2_device_for,
    empty_cuda_cache,
    module_available,
)
from afrilingua.backends.base import Segment, Transcript
from afrilingua.core.errors import CapabilityUnavailableError, ModelLoadError, OfflineModelMissingError
from afrilingua.core.hardware import RuntimePlan, resolve_runtime
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def faster_whisper_repo(size: str) -> str:
    """Hugging Face repo id that faster-whisper uses for a model size."""
    if "/" in size:
        return size
    if size.startswith("distil-"):
        return f"Systran/faster-distil-whisper-{size.removeprefix('distil-')}"
    return f"Systran/faster-whisper-{size}"


class _WhisperBase:
    name = "whisper"

    def __init__(
        self,
        model_size: str | None = None,
        *,
        settings: Settings | None = None,
        plan: RuntimePlan | None = None,
        registry: LanguageRegistry | None = None,
        beam_size: int = 1,
    ) -> None:
        self.settings = settings or get_settings()
        self.plan = plan or resolve_runtime()
        self.registry = registry or get_registry()
        self.model_size = model_size or self.plan.model_whisper
        self.beam_size = max(1, beam_size)
        self._model: Any = None

    def languages(self) -> list[str]:
        """Canonical names of registry languages that map to a Whisper code."""
        return [s.name for s in self.registry if s.whisper_lang]

    def _whisper_code(self, lang: str) -> str:
        spec = self.registry.get(lang)
        if not spec.whisper_lang:
            raise CapabilityUnavailableError(f"Whisper is not configured for {spec.name!r}.")
        return spec.whisper_lang

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        empty_cuda_cache()


class FasterWhisperBackend(_WhisperBase):
    """:class:`~afrilingua.backends.base.ASRBackend` using faster-whisper."""

    name = "faster-whisper"

    def load(self) -> None:
        """Load (downloading into ``HF_CACHE`` on first use) the CT2 Whisper model."""
        if self._model is not None:
            return
        if not module_available("faster_whisper"):
            raise ModelLoadError("faster-whisper is not installed (pip install 'afrilingua[fast]').")
        from faster_whisper import WhisperModel

        device = ct2_device_for(self.plan.device)
        compute_type = ct2_compute_type(device, self.plan.compute_type)
        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=self.plan.threads if device == "cpu" else 0,
                num_workers=1,
                download_root=str(self.settings.hf_cache),
                local_files_only=bool(self.settings.offline),
            )
        except Exception as exc:
            if self.settings.offline:
                raise OfflineModelMissingError(faster_whisper_repo(self.model_size)) from exc
            raise
        self.device, self.compute_type = device, compute_type
        logger.info("faster-whisper %s ready (%s, %s)", self.model_size, device, compute_type)

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        """Transcribe 16 kHz mono float32 audio."""
        self.load()
        code = self._whisper_code(lang)
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / SAMPLE_RATE
        segments_iter, _info = self._model.transcribe(
            audio,
            language=code,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        segments: list[Segment] = []
        logprobs: list[float] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            if getattr(seg, "no_speech_prob", 0.0) > 0.8 and getattr(seg, "avg_logprob", 0.0) < -1.0:
                continue  # Whisper's classic "hallucination on silence" signature
            segments.append(Segment(float(seg.start), float(min(seg.end, duration)), text))
            logprobs.append(float(seg.avg_logprob))
        text = " ".join(s.text for s in segments).strip()
        conf = float(np.mean([math.exp(lp) for lp in logprobs])) if logprobs else None
        spec = self.registry.get(lang)
        return Transcript(text=text, language=spec.name, segments=tuple(segments), confidence=conf, duration_s=duration)


class OpenAIWhisperBackend(_WhisperBase):
    """:class:`~afrilingua.backends.base.ASRBackend` using openai-whisper (PyTorch)."""

    name = "openai-whisper"

    def load(self) -> None:
        """Load the reference Whisper model (``HF_CACHE/whisper`` download root)."""
        if self._model is not None:
            return
        if not module_available("whisper"):
            raise ModelLoadError("openai-whisper is not installed (pip install 'afrilingua[legacy]').")
        import whisper

        root = self.settings.hf_cache / "whisper"
        local = root / f"{self.model_size}.pt"
        if self.settings.offline and not local.is_file():
            raise OfflineModelMissingError(f"whisper/{self.model_size}")
        self._model = whisper.load_model(self.model_size, device=self.plan.device, download_root=str(root))
        self.device = self.plan.device
        if self.device == "cpu":
            configure_torch_threads(self.plan.threads)
        logger.info("openai-whisper %s ready on %s", self.model_size, self.device)

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        """Transcribe 16 kHz mono float32 audio."""
        self.load()
        import torch

        code = self._whisper_code(lang)
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / SAMPLE_RATE
        with torch.inference_mode():
            result = self._model.transcribe(
                audio,
                language=code,
                task="transcribe",
                fp16=self.device == "cuda",
                temperature=0.0,
                beam_size=self.beam_size if self.beam_size > 1 else None,
                condition_on_previous_text=False,
                without_timestamps=True,
                verbose=None,
            )
        segments = tuple(
            Segment(float(s["start"]), float(min(s["end"], duration)), s["text"].strip())
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        )
        logprobs = [float(s["avg_logprob"]) for s in result.get("segments", []) if "avg_logprob" in s]
        conf = float(np.mean([math.exp(lp) for lp in logprobs])) if logprobs else None
        spec = self.registry.get(lang)
        return Transcript(
            text=str(result.get("text", "")).strip(),
            language=spec.name,
            segments=segments,
            confidence=conf,
            duration_s=duration,
        )
