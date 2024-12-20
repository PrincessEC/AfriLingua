"""
src/detection/lang_detector.py
Auto-detects source language from text using lingua-language-detector.
Detects: English, Igbo, Yoruba, Hausa.
Falls back to "unknown" when confidence is below threshold.
"""
from __future__ import annotations

import logging
from typing import Optional

from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

# Only load the four languages we care about — faster and more accurate
_SUPPORTED = [Language.ENGLISH, Language.IGBO, Language.YORUBA, Language.HAUSA]

# Lingua language → our normalised name
_LINGUA_TO_NAME: dict[Language, str] = {
    Language.ENGLISH: "english",
    Language.IGBO:    "igbo",
    Language.YORUBA:  "yoruba",
    Language.HAUSA:   "hausa",
}

_DEFAULT_CONFIDENCE_THRESHOLD = 0.70


class LanguageDetector:
    """
    Detects whether a text string is English, Igbo, Yoruba, or Hausa.

    Usage::
        detector = LanguageDetector()
        lang, conf = detector.detect_with_confidence("Kedu ka ọ dị?")
        # → ("igbo", 0.94)
        lang = detector.detect("How are you?")
        # → "english"
    """

    def __init__(self, confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold
        self._detector = (
            LanguageDetectorBuilder
            .from_languages(*_SUPPORTED)
            .with_preloaded_language_models()
            .build()
        )

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Return (language_name, confidence_score).
        language_name is "unknown" when no language exceeds the threshold.
        """
        if not text.strip():
            return "unknown", 0.0

        result = self._detector.detect_language_of(text)
        if result is None:
            return "unknown", 0.0

        # Get confidence values for all detected languages
        confidence_values = self._detector.compute_language_confidence_values(text)
        confidence = 0.0
        for cv in confidence_values:
            if cv.language == result:
                confidence = cv.value
                break

        if confidence < self.confidence_threshold:
            return "unknown", confidence

        lang_name = _LINGUA_TO_NAME.get(result, "unknown")
        return lang_name, confidence

    def detect(self, text: str) -> str:
        """
        Return language name, or "unknown" if confidence is below threshold.
        """
        lang, _ = self.detect_with_confidence(text)
        return lang

    def detect_or_default(self, text: str, default: str = "english") -> str:
        """
        Return detected language, falling back to `default` on low confidence.
        """
        lang = self.detect(text)
        return lang if lang != "unknown" else default
