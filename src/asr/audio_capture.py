"""
src/asr/audio_capture.py
Microphone capture using sounddevice.
Feeds PCM frames into a VADProcessor and triggers ASR on complete utterances.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from src.config import CHUNK_DURATION_S, SAMPLE_RATE
from src.asr.vad import VADProcessor

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Opens the default microphone, streams 16 kHz mono PCM frames into
    a VADProcessor, and fires `on_utterance(audio_np)` for every
    detected speech segment.

    Usage::
        def handle(audio):
            text = asr.transcribe(audio, lang="english")
            print(text)

        cap = AudioCapture(on_utterance=handle)
        cap.start()
        # ... do other work ...
        cap.stop()
    """

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_s: float = CHUNK_DURATION_S,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_s)
        self._vad = VADProcessor(on_utterance=on_utterance, sample_rate=sample_rate)
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
        if status:
            logger.warning("Audio stream status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self._vad.process_frame(mono.copy())

    def start(self) -> None:
        """Open the microphone stream and start processing."""
        with self._lock:
            if self._stream is not None:
                return
            logger.info("Opening mic stream at %d Hz …", self.sample_rate)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Microphone capture started.")

    def stop(self) -> None:
        """Stop the stream and flush any remaining buffered audio."""
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._vad.flush()
            logger.info("Microphone capture stopped.")

    def is_running(self) -> bool:
        return self._stream is not None
