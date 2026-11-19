"""Text normalisation and string similarity helpers for the translation memory.

* :func:`tm_key` is the *matching key* of a source sentence: NFC, lower-cased,
  punctuation stripped, whitespace collapsed. Diacritics (Yoruba tone marks, Igbo
  dot-below vowels) are **kept**: they change meaning, so ``ọjà`` ≠ ``oja``.
* :func:`base_form` additionally strips combining marks; it is only used to build
  the char n-gram *candidate* vectors so that a query typed without tone marks still
  finds its tone-marked neighbours. Final scores are computed on :func:`tm_key`.
* :func:`similarity` is the reranking score (rapidfuzz, with a difflib fallback).
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

from afrilingua.engine import normalize_text

__all__ = ["base_form", "similarity", "tm_key"]

#: Weight of ``fuzz.ratio`` vs ``fuzz.token_set_ratio`` in :func:`similarity`.
RATIO_WEIGHT = 0.7


@lru_cache(maxsize=1)
def _punct_table() -> dict[int, str]:
    """``str.translate`` table mapping every BMP punctuation char (category P*) to a space."""
    return {cp: " " for cp in range(0x10000) if unicodedata.category(chr(cp)).startswith("P")}


def tm_key(text: str) -> str:
    """Return the normalised matching key of ``text`` (see module docstring)."""
    text = normalize_text(text).lower().translate(_punct_table())
    return unicodedata.normalize("NFC", " ".join(text.split()))


def base_form(key: str) -> str:
    """Strip combining marks (tone marks, dots below) from an already-normalised key."""
    if key.isascii():
        return key
    decomposed = unicodedata.normalize("NFD", key)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


@lru_cache(maxsize=1)
def _rapidfuzz():  # pragma: no cover - trivial import shim
    try:
        from rapidfuzz import fuzz

        return fuzz
    except Exception:
        return None


def _token_set_ratio_fallback(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    inter = " ".join(sorted(ta & tb))
    da = " ".join(sorted(ta - tb))
    db = " ".join(sorted(tb - ta))
    s1 = inter
    s2 = f"{inter} {da}".strip()
    s3 = f"{inter} {db}".strip()

    def r(x: str, y: str) -> float:
        return SequenceMatcher(None, x, y).ratio() if (x or y) else 1.0

    if not inter:
        return r(s2, s3)
    return max(r(s1, s2), r(s1, s3), r(s2, s3))


def similarity(a: str, b: str) -> float:
    """Blend of character ratio and token-set ratio, in ``[0, 1]``.

    ``a`` and ``b`` should already be :func:`tm_key` normalised.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    fuzz = _rapidfuzz()
    if fuzz is not None:
        ratio = fuzz.ratio(a, b) / 100.0
        tset = fuzz.token_set_ratio(a, b) / 100.0
    else:
        ratio = SequenceMatcher(None, a, b).ratio()
        tset = _token_set_ratio_fallback(a, b)
    return RATIO_WEIGHT * ratio + (1.0 - RATIO_WEIGHT) * tset
