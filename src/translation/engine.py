"""
src/translation/engine.py
Core translation engine using Meta NLLB-200 (CPU-only, distilled-600M).

Supported language pairs (bidirectional):
    English  ↔  Igbo
    English  ↔  Yoruba
    English  ↔  Hausa
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from src.config import DEVICE, FINETUNED_MODEL_DIR, HF_CACHE, MODEL_NLLB

logger = logging.getLogger(__name__)

# NLLB language codes for the four supported languages
LANG_CODES: dict[str, str] = {
    "english": "eng_Latn",
    "igbo":    "ibo_Latn",
    "yoruba":  "yor_Latn",
    "hausa":   "hau_Latn",
}

# Inverse lookup: NLLB code → friendly name
CODE_TO_LANG: dict[str, str] = {v: k for k, v in LANG_CODES.items()}

SUPPORTED_LANGUAGES = list(LANG_CODES.keys())


def _resolve_model_path() -> str:
    """Return fine-tuned checkpoint path if it exists, otherwise base model ID."""
    if FINETUNED_MODEL_DIR.exists() and any(FINETUNED_MODEL_DIR.iterdir()):
        logger.info("Loading fine-tuned NLLB from %s", FINETUNED_MODEL_DIR)
        return str(FINETUNED_MODEL_DIR)
    logger.info("Fine-tuned model not found — loading base model: %s", MODEL_NLLB)
    return MODEL_NLLB


class TranslationEngine:
    """
    Lazy-loaded NLLB translation engine.
    The model is only downloaded/loaded on first call to translate().

    Usage::
        engine = TranslationEngine()
        result = engine.translate("Good morning", "english", "igbo")
        # → "Ụtụtụ ọma"
    """

    def __init__(self) -> None:
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._model_path: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Download / load model + tokenizer (called once on first translate)."""
        if self._model is not None:
            return

        path = _resolve_model_path()
        self._model_path = path

        logger.info("Loading tokenizer …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            path,
            cache_dir=str(HF_CACHE),
        )

        logger.info("Loading NLLB model onto %s …", DEVICE)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            cache_dir=str(HF_CACHE),
        )
        self._model.eval()
        logger.info("NLLB model ready.")

    def _get_lang_code(self, lang: str) -> str:
        """Normalise language name and return the NLLB language code."""
        lang_lower = lang.strip().lower()
        if lang_lower in LANG_CODES:
            return LANG_CODES[lang_lower]
        # Accept NLLB codes passed directly (e.g. "eng_Latn")
        if lang_lower in CODE_TO_LANG:
            return lang_lower
        raise ValueError(
            f"Unsupported language: '{lang}'. "
            f"Choose from: {SUPPORTED_LANGUAGES} or a valid NLLB code."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate a single string.

        Args:
            text:     Input text.
            src_lang: Source language name or NLLB code.
            tgt_lang: Target language name or NLLB code.

        Returns:
            Translated string.
        """
        if not text.strip():
            return ""

        results = self.translate_batch([text], src_lang, tgt_lang)
        return results[0]

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Translate a list of strings in a single forward pass.

        Args:
            texts:      List of input strings.
            src_lang:   Source language name or NLLB code.
            tgt_lang:   Target language name or NLLB code.
            max_length: Maximum token length for generated output.
            num_beams:  Beam search width (4 is a good CPU trade-off).

        Returns:
            List of translated strings (same order as input).
        """
        self._load()

        src_code = self._get_lang_code(src_lang)
        tgt_code = self._get_lang_code(tgt_lang)

        if src_code == tgt_code:
            return list(texts)

        # Tokenise with source language tag
        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Force target language BOS token
        forced_bos_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=max_length,
                num_beams=num_beams,
            )

        translations = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return translations

    def is_loaded(self) -> bool:
        return self._model is not None
