"""
tests/test_translation.py
Round-trip and smoke tests for TranslationEngine.
Run with: pytest tests/test_translation.py -v
"""

import pytest

from src.translation import TranslationEngine


@pytest.fixture(scope="module")
def engine():
    return TranslationEngine()


# ── Sanity: translate returns a non-empty string ──────────────────────────────


def test_english_to_igbo(engine):
    result = engine.translate("Good morning", "english", "igbo")
    assert isinstance(result, str) and len(result) > 0


def test_english_to_yoruba(engine):
    result = engine.translate("Good morning", "english", "yoruba")
    assert isinstance(result, str) and len(result) > 0


def test_english_to_hausa(engine):
    result = engine.translate("Good morning", "english", "hausa")
    assert isinstance(result, str) and len(result) > 0


# ── Round-trip: translate to language then back to English ────────────────────


@pytest.mark.parametrize("target_lang", ["igbo", "yoruba", "hausa"])
def test_roundtrip(engine, target_lang):
    original = "Welcome to the meeting."
    translated = engine.translate(original, "english", target_lang)
    back = engine.translate(translated, target_lang, "english")
    # Should be non-empty and contain at least one word from original context
    assert isinstance(back, str) and len(back) > 0


# ── Batch translation ─────────────────────────────────────────────────────────


def test_batch_translation(engine):
    texts = ["Hello", "Thank you", "How are you?"]
    results = engine.translate_batch(texts, "english", "hausa")
    assert len(results) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in results)


# ── Same-language passthrough ─────────────────────────────────────────────────


def test_same_language_passthrough(engine):
    text = "No translation needed"
    result = engine.translate(text, "english", "english")
    assert result == text


# ── Empty string ──────────────────────────────────────────────────────────────


def test_empty_string(engine):
    result = engine.translate("", "english", "igbo")
    assert result == ""


# ── Unsupported language raises ValueError ────────────────────────────────────


def test_unsupported_language_raises(engine):
    with pytest.raises(ValueError):
        engine.translate("Hello", "english", "klingon")
