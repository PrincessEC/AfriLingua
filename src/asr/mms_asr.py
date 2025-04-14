"""
src/asr/mms_asr.py
Meta MMS-300M ASR for Igbo, Yoruba, and Hausa (CPU).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.config import HF_CACHE, MODEL_MMS_ASR, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Lingua / internal name → MMS language tag (also used as subfolder name in the HF repo)
MMS_LANG_MAP: dict[str, str] = {
    "igbo":   "ibo",
    "yoruba": "yor",
    "hausa":  "hau",
}


class MMSASR:
    """
    Meta MMS-300M speech → text for African languages (CPU).

    facebook/mms-300m stores per-language vocab files in subdirectories
    (ibo/, yor/, hau/) — AutoProcessor cannot resolve them in transformers 5.x
    so we build the processor manually from its components.

    Usage::
        asr = MMSASR()
        text = asr.transcribe(audio_np, lang="igbo")
    """

    def __init__(self) -> None:
        self._feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None
        self._current_lang: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading MMS-300M feature extractor + model on CPU …")
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_MMS_ASR,
            cache_dir=str(HF_CACHE),
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_MMS_ASR,
            torch_dtype=torch.float32,
            cache_dir=str(HF_CACHE),
        )
        self._model.eval()
        # Bootstrap with Igbo so the processor is ready for first use
        self._switch_lang("ibo")
        logger.info("MMS-300M ready.")

    def _switch_lang(self, mms_lang: str) -> None:
        """Load the per-language tokenizer from its subfolder and update the adapter."""
        if mms_lang == self._current_lang:
            return
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            MODEL_MMS_ASR,
            subfolder=mms_lang,  # each language lives in its own subdirectory
            cache_dir=str(HF_CACHE),
        )
        self._processor = Wav2Vec2Processor(
            feature_extractor=self._feature_extractor,
            tokenizer=tokenizer,
        )
        if self._current_lang is not None:
            # Swap the language adapter weights in the already-loaded model
            assert self._model is not None
            self._model.load_adapter(mms_lang)
        self._current_lang = mms_lang

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

        self._switch_lang(mms_lang)

        assert self._processor is not None
        assert self._model is not None

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
