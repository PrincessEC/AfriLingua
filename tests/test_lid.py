"""Language identification (heuristic lexical + orthographic scoring, lingua tie-break)."""

from __future__ import annotations

import pytest

from afrilingua.backends.base import LanguageIdentifier
from afrilingua.backends.lid import HeuristicLanguageIdentifier, tokenize


@pytest.fixture(params=[False, True], ids=["heuristic", "with-lingua"])
def detector(request):
    if request.param:
        pytest.importorskip("lingua")
    return HeuristicLanguageIdentifier(use_lingua=request.param)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Aha, I see", "english"),  # v1 bug: "aha" made this Igbo
        ("Good morning everyone, welcome to the meeting.", "english"),
        ("Thank you very much for coming today.", "english"),
        ("The meeting starts at noon in Abuja.", "english"),
        ("Kedu ka ị mere?", "igbo"),
        ("kedu ka i mere", "igbo"),  # no diacritics
        ("Nnọọ, kedu ka ị mere taa?", "igbo"),
        ("Daalụ nke ukwuu", "igbo"),
        ("Ẹ kú àárọ̀, ṣé dáadáa ni?", "yoruba"),
        ("Bawo ni", "yoruba"),
        ("Mo fẹ́ lọ sí ilé", "yoruba"),
        ("E ku aaro, bawo ni o se wa?", "yoruba"),
        ("Sannu, yaya kake? Nagode.", "hausa"),
        ("Barka da safe", "hausa"),
        ("Ina son abinci sosai", "hausa"),
        ("Ƙasar Najeriya tana da ɗimbin jama'a", "hausa"),
    ],
)
def test_detects_languages(detector, text, expected):
    lang, conf = detector.detect(text)
    assert lang == expected, (text, lang, conf)
    assert 0.0 < conf <= 1.0


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!!", "ni", "da", "xyzzy"])
def test_unknown_when_no_evidence(detector, text):
    assert detector.detect(text) == ("unknown", 0.0)


def test_single_african_word_does_not_decide_long_english(detector):
    lang, _ = detector.detect("I told my friend that Ina will come to the office on Monday")
    assert lang == "english"


def test_single_weak_hit_is_unknown_without_other_evidence():
    d = HeuristicLanguageIdentifier(use_lingua=False)
    assert d.detect("Ina Garten cooks pasta")[0] == "unknown"


def test_scores_are_length_normalised_and_capped():
    d = HeuristicLanguageIdentifier(use_lingua=False)
    scores, n = d.scores("ẹ̀kọ́")
    assert n == 1
    assert scores["yoruba"] <= 2.0


def test_tokenize_keeps_combining_marks_and_nfc():
    decomposed = "ọ\u0300rẹ\u0301 E\u0301"
    toks = tokenize(decomposed)
    assert toks == ["ọ̀rẹ́", "é"]


def test_registry_extension_participates():
    from afrilingua.core.languages import get_registry

    reg = get_registry()
    reg.register(
        {
            "name": "swahili",
            "nllb_code": "swh_Latn",
            "detection_words": ["habari", "asante", "sana", "karibu"],
        },
        persist=False,
    )
    d = HeuristicLanguageIdentifier(reg, use_lingua=False)
    assert d.detect("Habari, asante sana")[0] == "swahili"


def test_satisfies_protocol():
    assert isinstance(HeuristicLanguageIdentifier(use_lingua=False), LanguageIdentifier)


def test_engine_auto_detect_uses_detector(fake_engine):
    fake_engine.detector = HeuristicLanguageIdentifier(use_lingua=False)
    r = fake_engine.translate_one("Kedu ka ị mere?", "auto", "english")
    assert r.src_lang == "igbo" and r.detected
    r = fake_engine.translate_one("Aha, I see", None, "yoruba")
    assert r.src_lang == "english"
