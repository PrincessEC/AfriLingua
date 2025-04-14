"""
src/asr/vad.py
Voice Activity Detection using Silero VAD.

Accumulates raw audio frames and yields complete utterances.

Two firing conditions:
  1. Silence boundary  — speech followed by >= 0.8 s of silence.
  2. Max utterance     — continuous speech flushed every MAX_UTTERANCE_S seconds
                         so simultaneous interpretation can start while the
                         speaker is still talking.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch

from src.config import CHUNK_DURATION_S, SAMPLE_RATE, VAD_THRESHOLD

logger = logging.getLogger(__name__)

# Silero VAD requires 512-sample windows at 16 kHz (32 ms per window)
_VAD_WINDOW_SAMPLES  = 512
_SILENCE_TIMEOUT_S   = 0.8   # end utterance after 0.8 s of silence
_MAX_UTTERANCE_S     = 4.0   # flush mid-speech every 4 s for simultaneous interpretation


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
        max_utterance_s: float = _MAX_UTTERANCE_S,
    ) -> None:
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._silence_frames_needed = int(silence_timeout_s * sample_rate / _VAD_WINDOW_SAMPLES)
        self._max_utterance_samples = int(max_utterance_s * sample_rate)

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

    def _fire_utterance(self, keep_in_speech: bool = False) -> None:
        """Emit buffered audio as an utterance.

        Args:
            keep_in_speech: When True the speaker is still talking (max-duration
                            flush). Resets the buffer but stays in speech mode so
                            the next chunk is captured immediately without
                            requiring a fresh speech onset.
        """
        if not self._buffer:
            return
        audio = np.array(self._buffer, dtype=np.float32)
        self._buffer.clear()
        self._silence_count = 0
        if not keep_in_speech:
            self._in_speech = False
        try:
            self.on_utterance(audio)
        except Exception:
            logger.exception("on_utterance callback raised an exception")

    def process_frame(self, chunk: np.ndarray) -> None:
        """
        Feed a chunk of float32 mono PCM audio.
        Internally batches into 512-sample windows for VAD.

        Fires on_utterance when:
          - A silence boundary is detected (speaker paused), OR
          - The accumulated buffer reaches MAX_UTTERANCE_S (simultaneous mode).
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

                # ── Simultaneous interpretation flush ────────────────────────
                # Fire mid-speech so TTS can start playing while speaker
                # continues talking (4-second rolling chunks).
                if len(self._buffer) >= self._max_utterance_samples:
                    logger.debug("VAD: max-utterance flush (%d samples)", len(self._buffer))
                    self._fire_utterance(keep_in_speech=True)

            else:
                if self._in_speech:
                    self._silence_count += 1
                    self._buffer.extend(window.tolist())
                    if self._silence_count >= self._silence_frames_needed:
                        # ── Silence boundary — utterance ended ───────────────
                        logger.debug("VAD: silence boundary (%d samples)", len(self._buffer))
                        self._fire_utterance(keep_in_speech=False)

        self._pending = samples[offset:].tolist()

    def flush(self) -> None:
        """Emit any buffered audio as a final utterance (call when stopping)."""
        self._fire_utterance(keep_in_speech=False)
