"""
src/asr/whisper_asr.py
Whisper-based ASR for English.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import whisper

from src.config import DEVICE, HF_CACHE, MODEL_WHISPER, SAMPLE_RATE

logger = logging.getLogger(__name__)

# whisper.load_model() takes a size string, not a HF model ID
# Map our config value to a Whisper size identifier
_MODEL_SIZE_MAP = {
    "openai/whisper-tiny":   "tiny",
    "openai/whisper-base":   "base",
    "openai/whisper-small":  "small",
    "openai/whisper-medium": "medium",
    "openai/whisper-large":  "large",
}


class WhisperASR:
    """
    English speech to text using OpenAI Whisper.

    Usage::
        asr = WhisperASR()
        text = asr.transcribe(audio_np_array, sample_rate=16000)
    """

    def __init__(self) -> None:
        self._model: Optional[whisper.Whisper] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        size = _MODEL_SIZE_MAP.get(MODEL_WHISPER, "small")
        device = "cuda" if DEVICE == "cuda" else "cpu"
        logger.info("Loading Whisper '%s' on %s ...", size, device)
        self._model = whisper.load_model(
            size,
            device=device,
            download_root=str(HF_CACHE / "whisper"),
        )
        logger.info("Whisper ready.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str = "en",
    ) -> str:
        """
        Transcribe a numpy float32 audio array.

        Args:
            audio:       Float32 array, shape (N,), values in [-1, 1].
            sample_rate: Must match the recorded audio; Whisper expects 16000 Hz.
            language:    ISO 639-1 language hint (default "en").

        Returns:
            Transcribed text string.
        """
        self._load()

        if audio is None or len(audio) == 0:
            return ""

        # Ensure float32, mono, correct sample rate
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (whisper requires 16 kHz)
        if sample_rate != 16000:
            import torchaudio.functional as F
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = F.resample(audio_t, sample_rate, 16000)
            audio = audio_t.squeeze(0).numpy()

        result = self._model.transcribe(audio, language=language, fp16=DEVICE == "cuda")
        return result.get("text", "").strip()

    def is_loaded(self) -> bool:
        return self._model is not None
