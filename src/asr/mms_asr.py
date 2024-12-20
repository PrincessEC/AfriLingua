"""
src/asr/mms_asr.py
Meta MMS-300M ASR for Igbo, Yoruba, and Hausa (CPU).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from src.config import HF_CACHE, MODEL_MMS_ASR, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Lingua / internal name → MMS language tag
MMS_LANG_MAP: dict[str, str] = {
    "igbo":   "ibo",
    "yoruba": "yor",
    "hausa":  "hau",
}


class MMSASR:
    """
    Meta MMS-300M speech → text for African languages (CPU).

    The processor must be updated with the target language before inference.
    Models are lazy-loaded on first call.

    Usage::
        asr = MMSASR()
        text = asr.transcribe(audio_np, lang="igbo")
    """

    def __init__(self) -> None:
        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading MMS-300M processor + model on CPU …")
        self._processor = AutoProcessor.from_pretrained(
            MODEL_MMS_ASR,
            cache_dir=str(HF_CACHE),
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_MMS_ASR,
            torch_dtype=torch.float32,
            cache_dir=str(HF_CACHE),
        )
        self._model.eval()
        logger.info("MMS-300M ready.")

    def transcribe(
        self,
        audio: np.ndarray,
        lang: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """
        Transcribe audio using MMS.

        Args:
            audio:       Float32 numpy array, shape (N,), values in [-1, 1].
            lang:        Language name: "igbo", "yoruba", or "hausa".
            sample_rate: Source sample rate; must be 16000 Hz (auto-resampled).

        Returns:
            Transcribed text string.
        """
        self._load()

        if audio is None or len(audio) == 0:
            return ""

        mms_lang = MMS_LANG_MAP.get(lang.lower())
        if mms_lang is None:
            raise ValueError(
                f"Unsupported language for MMS ASR: '{lang}'. "
                f"Supported: {list(MMS_LANG_MAP.keys())}"
            )

        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            import torchaudio.functional as F
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = F.resample(audio_t, sample_rate, 16000)
            audio = audio_t.squeeze(0).numpy()

        # Set target language on the processor
        self._processor.tokenizer.set_target_lang(mms_lang)
        self._model.load_adapter(mms_lang)

        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def is_loaded(self) -> bool:
        return self._model is not None
