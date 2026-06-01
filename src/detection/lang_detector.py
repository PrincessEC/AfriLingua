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

from src.languages import LANGUAGE_REGISTRY

logger = logging.getLogger(__name__)

# Lingua only supports ENGLISH and YORUBA from our target set.
# Igbo and Hausa are handled via a lightweight word-list fallback.
_SUPPORTED = [Language.ENGLISH, Language.YORUBA]

# Lingua language → our normalised name
_LINGUA_TO_NAME: dict[Language, str] = {
    Language.ENGLISH: "english",
    Language.YORUBA:  "yoruba",
}

# Minimal word-list heuristics for languages lingua doesn't cover
_IGBO_WORDS  = {"kedu", "nne", "nna", "ọ", "gị", "anyị", "daalụ", "nnoo", "biko", "ụlọ", "aha", "nọ"}
_HAUSA_WORDS = {"yaya", "kai", "ina", "gida", "kana", "tafi", "yaro", "wannan", "alhaji", "nagode", "sannu"}
_HEURISTIC_WORDS: dict[str, set[str]] = {
    name: {word.lower() for word in spec.detection_words}
    for name, spec in LANGUAGE_REGISTRY.items()
    if spec.detection_words
}


def _wordlist_detect(text: str) -> Optional[str]:
    """Return a configured heuristic language if enough known words are found."""
    tokens = set(text.lower().split())
    scores = {lang: len(tokens & words) for lang, words in _HEURISTIC_WORDS.items()}
    if not scores:
        return None
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return None
    return best_lang

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
        Igbo/Hausa use a word-list heuristic (confidence reported as 0.80).
        """
        if not text.strip():
            return "unknown", 0.0

        # Word-list heuristic first — catches Igbo/Hausa before lingua
        # misclassifies them as English or another Latin-script language.
        wl = _wordlist_detect(text)
        if wl:
            return wl, 0.80

        # Lingua handles English and Yoruba reliably
        result = self._detector.detect_language_of(text)
        if result is not None:
            confidence_values = self._detector.compute_language_confidence_values(text)
            confidence = 0.0
            for cv in confidence_values:
                if cv.language == result:
                    confidence = cv.value
                    break
            if confidence >= self.confidence_threshold:
                lang_name = _LINGUA_TO_NAME.get(result, "unknown")
                return lang_name, confidence

        return "unknown", 0.0

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
