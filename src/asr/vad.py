"""
src/asr/vad.py
Voice Activity Detection using Silero VAD.
Accumulates raw audio frames and yields complete utterances.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Optional

import numpy as np
import torch

from src.config import CHUNK_DURATION_S, SAMPLE_RATE, VAD_THRESHOLD

logger = logging.getLogger(__name__)

# Silero VAD requires 512-sample windows at 16 kHz (32ms)
_VAD_WINDOW_SAMPLES = 512
_SILENCE_TIMEOUT_S  = 0.8   # end utterance after 0.8 s of silence


class VADProcessor:
    """
    Wraps Silero VAD to detect speech boundaries in a raw audio stream.

    Call `process_frame(chunk)` with small chunks of float32 PCM audio.
    When a complete utterance is detected, `on_utterance(audio_np)` is called.

    Usage::
        def handle(audio): print("Got", len(audio), "samples")
        vad = VADProcessor(on_utterance=handle)
        while True:
            chunk = read_mic_chunk()
            vad.process_frame(chunk)
    """

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        sample_rate: int = SAMPLE_RATE,
        threshold: float = VAD_THRESHOLD,
        silence_timeout_s: float = _SILENCE_TIMEOUT_S,
    ) -> None:
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._silence_frames_needed = int(silence_timeout_s * sample_rate / _VAD_WINDOW_SAMPLES)

        self._model: Optional[torch.nn.Module] = None
        self._buffer: list[float] = []         # accumulates current utterance
        self._pending: list[float] = []        # leftover samples < window size
        self._silence_count = 0
        self._in_speech = False

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Silero VAD …")
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self._model = model
        self._model.eval()
        logger.info("Silero VAD ready.")

    def _is_speech(self, window: np.ndarray) -> bool:
        tensor = torch.from_numpy(window).unsqueeze(0)  # (1, 512)
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return prob >= self.threshold

    def process_frame(self, chunk: np.ndarray) -> None:
        """
        Feed a chunk of float32 mono PCM audio.
        Internally batches into 512-sample windows for VAD.
        Calls on_utterance when an utterance boundary is detected.
        """
        self._load()

        samples = np.concatenate([np.array(self._pending, dtype=np.float32), chunk.astype(np.float32)])
        offset = 0

        while offset + _VAD_WINDOW_SAMPLES <= len(samples):
            window = samples[offset: offset + _VAD_WINDOW_SAMPLES]
            offset += _VAD_WINDOW_SAMPLES
            speech = self._is_speech(window)

            if speech:
                self._in_speech = True
                self._silence_count = 0
                self._buffer.extend(window.tolist())
            else:
                if self._in_speech:
                    self._silence_count += 1
                    self._buffer.extend(window.tolist())
                    if self._silence_count >= self._silence_frames_needed:
                        # Utterance ended — fire callback
                        utterance = np.array(self._buffer, dtype=np.float32)
                        self.on_utterance(utterance)
                        self._buffer.clear()
                        self._silence_count = 0
                        self._in_speech = False

        self._pending = samples[offset:].tolist()

    def flush(self) -> None:
        """Emit any buffered audio as a final utterance (call when stopping)."""
        if self._buffer:
            utterance = np.array(self._buffer, dtype=np.float32)
            self.on_utterance(utterance)
            self._buffer.clear()
            self._in_speech = False
            self._silence_count = 0
