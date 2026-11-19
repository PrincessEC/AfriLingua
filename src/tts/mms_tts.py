"""
src/tts/mms_tts.py
Meta MMS Text-to-Speech for Igbo, Yoruba, and Hausa.
Models are lazy-loaded per language on first call.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from src.config import (
    DEVICE,
    HF_CACHE,
    TORCH_DTYPE,
)
from src.hf_compat import dtype_name_kwargs
from src.languages import TTS_MODEL_IDS, normalize_language

logger = logging.getLogger(__name__)

_LANG_TO_MODEL_ID: dict[str, str] = dict(TTS_MODEL_IDS)

SUPPORTED_LANGUAGES = list(_LANG_TO_MODEL_ID.keys())


class MMSTTS:
    """
    Meta MMS VITS TTS — synthesises audio for African languages.

    Models are downloaded on first use and cached to HF_CACHE.
    Each language model is loaded separately and kept in memory.

    Usage::
        tts = MMSTTS()
        waveform, sample_rate = tts.synthesize("Ụtụtụ ọma", "igbo")
        # waveform: float32 numpy array at sample_rate Hz
    """

    def __init__(self) -> None:
        self._models:     dict[str, VitsModel]     = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}
        self._device: str = DEVICE

    def _load_lang(self, lang: str) -> None:
        if lang in self._models:
            return
        model_id = _LANG_TO_MODEL_ID.get(lang)
        if model_id is None:
            raise ValueError(
                f"Unsupported language for MMS TTS: '{lang}'. "
                f"Supported: {SUPPORTED_LANGUAGES}"
            )
        logger.info(
            "Loading MMS TTS for '%s' on %s from %s ...",
            lang,
            DEVICE,
            model_id,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(HF_CACHE),
            )
            model = VitsModel.from_pretrained(
                model_id,
                cache_dir=str(HF_CACHE),
                **dtype_name_kwargs(TORCH_DTYPE),
            )
        except OSError as exc:
            raise ValueError(
                f"Could not load MMS TTS model {model_id!r} for {lang!r}. "
                f"Check MODEL_MMS_TTS_{lang.upper()} in your .env file."
            ) from exc
        model.eval()
        try:
            model.to(DEVICE)
            self._device = DEVICE
        except RuntimeError as exc:
            if DEVICE == "cuda" and "out of memory" in str(exc).lower():
                logger.warning("CUDA ran out of memory for MMS TTS %s; falling back to CPU.", lang)
                torch.cuda.empty_cache()
                model.to(device="cpu", dtype=torch.float32)
                self._device = "cpu"
            else:
                raise
        self._tokenizers[lang] = tokenizer
        self._models[lang] = model
        logger.info("MMS TTS '%s' ready.", lang)

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """
        Synthesise speech for the given text.

        Args:
            text: Input text in the target language.
            lang: One of "igbo", "yoruba", "hausa".

        Returns:
            Tuple of (waveform_float32_np, sample_rate_hz).
        """
        lang = normalize_language(lang)
        self._load_lang(lang)

        tokenizer = self._tokenizers[lang]
        model     = self._models[lang]

        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)

        waveform = (
            output.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        )
        sample_rate = model.config.sampling_rate
        return waveform, sample_rate

    def loaded_languages(self) -> list[str]:
        return list(self._models.keys())
