"""Fast, offline heuristic quality estimation for contributions.

The score is a product of penalty factors in ``[0, 1]`` (1.0 = no issues found).
Each applied factor adds a human-readable *flag* so reviewers see why a score is
low. It is a triage signal for the review queue, never an automatic rejection.

Checks
------
* ``identical_to_source``: target equals source (after normalisation) → ×0.2
* ``length_ratio``: target/source character ratio outside the expected band for
  the language pair → ×0.4 … ×0.8 (graded)
* ``non_latin_script``: <90% of letters are Latin for Latin-script languages → ×0.3
* ``missing_tone_marks``: Yoruba target with no tone marks / dots below → ×0.85
  (``tone_marks`` positive flag otherwise); Igbo dotted vowels give a small bonus
* ``pii``: e-mail / URL / phone number found (they are scrubbed on store) → ×0.6
* ``duplicate``: near-identical approved pair already in the TM → ×0.5
* ``lid_mismatch``: language identifier confidently says another language → ×0.6
  (only when ``afrilingua.backends.factory.build_detector`` is available)
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from afrilingua.tm.text import tm_key

logger = logging.getLogger(__name__)

__all__ = ["QualityReport", "estimate_quality"]

# Expected target/source character-length ratio bands for English → X (and inverses).
_RATIO_BANDS: dict[str, tuple[float, float]] = {
    "yoruba": (0.6, 2.2),
    "igbo": (0.6, 2.0),
    "hausa": (0.6, 2.0),
}
_DEFAULT_BAND = (0.45, 2.4)
_LATIN_LANGS = {"english", "igbo", "yoruba", "hausa"}
_YORUBA_MARKS = {"\u0300", "\u0301", "\u0323"}  # grave, acute, dot below
_IGBO_DOTTED = set("ịọụỊỌỤṅṄ")


@dataclass
class QualityReport:
    score: float
    flags: list[str] = field(default_factory=list)


def _latin_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    return latin / len(letters)


def _ratio_band(src: str, tgt: str) -> tuple[float, float]:
    if src == "english" and tgt in _RATIO_BANDS:
        return _RATIO_BANDS[tgt]
    if tgt == "english" and src in _RATIO_BANDS:
        lo, hi = _RATIO_BANDS[src]
        return 1.0 / hi, 1.0 / lo
    return _DEFAULT_BAND


def estimate_quality(
    *,
    kind: str,
    src_lang: str,
    tgt_lang: str,
    source_text: str,
    target_text: str,
    pii_found: list[str] | None = None,
    tm: Any | None = None,
    detector: Callable[[str], tuple[str, float]] | None = None,
) -> QualityReport:
    """Score a (validated) contribution; see module docstring."""
    score = 1.0
    flags: list[str] = []

    def penalise(factor: float, flag: str) -> None:
        nonlocal score
        score *= factor
        flags.append(flag)

    src_key, tgt_key = tm_key(source_text), tm_key(target_text)
    if kind != "transcription":
        if src_key == tgt_key:
            penalise(0.2, "identical_to_source")
        # Short glossary terms / greetings have noisy ratios: only judge longer texts.
        if len(source_text) >= 12 and kind != "glossary":
            ratio = len(target_text) / max(1, len(source_text))
            lo, hi = _ratio_band(src_lang, tgt_lang)
            if ratio < lo or ratio > hi:
                off = ratio / lo if ratio < lo else hi / ratio
                penalise(0.4 if off < 0.5 else 0.8, "length_ratio")

    out_lang, out_text = tgt_lang, target_text
    if out_lang in _LATIN_LANGS and _latin_fraction(out_text) < 0.9:
        penalise(0.3, "non_latin_script")

    letters = sum(ch.isalpha() for ch in out_text)
    if out_lang == "yoruba" and letters >= 4:
        decomposed = unicodedata.normalize("NFD", out_text)
        if any(ch in _YORUBA_MARKS for ch in decomposed):
            flags.append("tone_marks")
        else:
            penalise(0.85, "missing_tone_marks")
    if out_lang == "igbo" and any(ch in _IGBO_DOTTED for ch in out_text):
        flags.append("igbo_dotted_vowels")

    if pii_found:
        penalise(0.6, "pii")

    if tm is not None and kind in {"translation", "correction", "glossary"}:
        try:
            for match in tm.lookup(source_text, src_lang, tgt_lang, k=3):
                if match.approved and match.score >= 0.97 and tm_key(match.target) == tgt_key:
                    penalise(0.5, "duplicate")
                    break
        except Exception:  # TM is optional for scoring
            logger.debug("TM duplicate check failed", exc_info=True)

    if detector is not None and letters >= 12:
        try:
            lang, conf = detector(out_text)
            if lang not in {"unknown", out_lang} and conf >= 0.8:
                penalise(0.6, "lid_mismatch")
        except Exception:
            logger.debug("Language identification failed", exc_info=True)

    return QualityReport(round(max(0.0, min(1.0, score)), 3), flags)
