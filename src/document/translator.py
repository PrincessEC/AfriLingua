"""
src/document/translator.py
Translates an entire document (TXT / PDF / DOCX) sentence-by-sentence
using the NLLB TranslationEngine with auto language detection.

Output is written to <original_dir>/translated_<filename>.<ext>
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.detection.lang_detector import LanguageDetector
from src.document.reader import DocumentReader
from src.translation.engine import TranslationEngine, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

_BATCH_SIZE = 16   # sentences per NLLB forward pass


class DocumentTranslator:
    """
    End-to-end document translation.

    Usage::
        dt = DocumentTranslator()
        out_path = dt.translate_file("speech.txt", target_lang="igbo")
        print(f"Translated saved to: {out_path}")
    """

    def __init__(
        self,
        *,
        reader: Optional[DocumentReader] = None,
        detector: Optional[LanguageDetector] = None,
        engine: Optional[TranslationEngine] = None,
    ) -> None:
        self._reader   = reader if reader is not None else DocumentReader()
        self._detector = detector if detector is not None else LanguageDetector()
        self._engine   = engine if engine is not None else TranslationEngine()

    def translate_file(
        self,
        file_path: str | Path,
        target_lang: str,
        src_lang: Optional[str] = None,
    ) -> Path:
        """
        Translate a document file and write the result next to the original.

        Args:
            file_path:   Path to .txt, .pdf, or .docx file.
            target_lang: Target language (e.g. "igbo", "yoruba", "hausa", "english").
            src_lang:    Optional source language override. Auto-detected if None.

        Returns:
            Path to the translated output file.
        """
        path = Path(file_path)
        sentences = self._reader.read(path)
        if not sentences:
            raise ValueError(f"No text could be extracted from {path}")

        # Detect source language from the first 5 sentences if not supplied
        if src_lang is None:
            sample_text = " ".join(sentences[:5])
            src_lang = self._detector.detect_or_default(sample_text, default="english")
            logger.info("Auto-detected source language: %s", src_lang)

        logger.info("Translating %d sentences: %s → %s …", len(sentences), src_lang, target_lang)

        # Batch translate
        translated: List[str] = []
        for i in range(0, len(sentences), _BATCH_SIZE):
            batch = sentences[i: i + _BATCH_SIZE]
            translated.extend(self._engine.translate_batch(batch, src_lang, target_lang))
            logger.info("  %d / %d sentences translated", min(i + _BATCH_SIZE, len(sentences)), len(sentences))

        # Write output
        out_path = path.parent / f"translated_{path.name}"
        out_path.write_text("\n".join(translated), encoding="utf-8")
        logger.info("Translation saved → %s", out_path)
        return out_path

    def translate_text(
        self,
        text: str,
        target_lang: str,
        src_lang: Optional[str] = None,
    ) -> str:
        """
        Translate a raw string (no file I/O).

        Args:
            text:        Source text.
            target_lang: Target language name.
            src_lang:    Source language (auto-detected if None).

        Returns:
            Translated text.
        """
        if src_lang is None:
            src_lang = self._detector.detect_or_default(text, default="english")

        return self._engine.translate(text, src_lang, target_lang)
