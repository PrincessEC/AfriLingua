"""faster-whisper ASR implementation."""
from __future__ import annotations

import logging
import numpy as np
from faster_whisper import WhisperModel

from afrilingua.backends.base import ASRBackend, Segment, Transcript
from afrilingua.core.hardware import resolve_runtime

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    def __init__(self, model_size: str) -> None:
        self.name = "faster-whisper"
        self.model_size = model_size
        self.plan = resolve_runtime()
        self._model: WhisperModel | None = None

    def load(self) -> None:
        if self.is_loaded():
            return
        
        device = self.plan.device
        compute_type = self.plan.compute_type
        
        logger.info("Loading faster-whisper %s (device=%s, compute=%s)", self.model_size, device, compute_type)
        
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.plan.threads,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None

    def languages(self) -> list[str]:
        # Faster-whisper supports many, but we restrict to registry ones.
        return ["english"] # Simplified for now, will be dynamic.

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        if not self.is_loaded():
            self.load()
        
        segments, info = self._model.transcribe(
            audio, 
            beam_size=1, 
            language=lang if lang != "english" else "en"
        )
        
        full_segments = []
        full_text = []
        for s in segments:
            full_segments.append(Segment(start=s.start, end=s.end, text=s.text))
            full_text.append(s.text)
            
        return Transcript(
            text=" ".join(full_text),
            language=lang,
            segments=tuple(full_segments),
            confidence=info.language_probability,
            duration_s=info.duration
        )
