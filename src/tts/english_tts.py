"""
src/tts/english_tts.py
Offline English TTS using pyttsx3 (no internet required).
Returns PCM audio as bytes via a temp WAV file.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np
import pyttsx3

logger = logging.getLogger(__name__)


class EnglishTTS:
    """
    English TTS using pyttsx3 (system voices, offline).

    Returns (waveform_float32_np, sample_rate) to match the MMSTTS interface.

    Usage::
        tts = EnglishTTS()
        waveform, sr = tts.synthesize("Welcome to the meeting.")
    """

    def __init__(self, rate: int = 150, volume: float = 1.0) -> None:
        self._rate = rate
        self._volume = volume
        self._engine: Optional[pyttsx3.Engine] = None

    def _load(self) -> None:
        if self._engine is not None:
            return
        logger.info("Initialising pyttsx3 TTS engine …")
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate",   self._rate)
        self._engine.setProperty("volume", self._volume)
        logger.info("pyttsx3 TTS ready.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesise English speech.

        Returns:
            (waveform_float32_np, sample_rate_hz)
        """
        self._load()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()

            with wave.open(tmp_path, "rb") as wf:
                n_channels  = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate  = wf.getframerate()
                raw_bytes   = wf.readframes(wf.getnframes())

            # Convert raw PCM bytes to float32 [-1, 1]
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sample_width, np.int16)
            samples = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
            samples /= np.iinfo(dtype).max

            # Downmix to mono if needed
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            return samples, frame_rate

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
