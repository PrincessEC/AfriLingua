"""Latency features: sentence batching, exact-only memory serving, cache coherence, vocab map."""

from __future__ import annotations

import pytest

from afrilingua.core.errors import InvalidInputError
from afrilingua.engine import TMMatch


class _TM:
    """Minimal TM double with a generation counter like TranslationMemory."""

    def __init__(self, entries: dict[str, tuple[str, float]] | None = None) -> None:
        self.entries = entries or {}
        self.generation = 0
        self.lookups: list[str] = []

    def lookup(self, text, src, tgt, *, k=3):
        self.lookups.append(text)
        if text in self.entries:
            target, score = self.entries[text]
            return [TMMatch("1", text, target, score, True, "community")]
        return []

    def add(self, *a, **kw):
        return "x"


def _mt(engine):
    return engine.manager.get("mt")  # FakeMT registered by the fixture


def test_multi_sentence_text_is_one_batched_call(fake_engine):
    r = fake_engine.translate_one("Good morning. Where is the clinic? Thank you.", "english", "yoruba")
    calls = _mt(fake_engine).calls
    assert len(calls) == 1
    assert calls[0][0] == ["Good morning.", "Where is the clinic?", "Thank you."]
    assert r.text == "[yoruba] Good morning. [yoruba] Where is the clinic? [yoruba] Thank you."
    assert r.origin == "mt"


def test_repeated_sentences_are_translated_once(fake_engine):
    fake_engine.translate(["Hello there. Hello there.", "Hello there."], "english", "igbo")
    assert _mt(fake_engine).calls[0][0] == ["Hello there."]


def test_known_sentence_comes_from_memory_rest_from_model(fake_engine):
    fake_engine.tm = _TM({"Good morning.": ("Ẹ káàárọ̀.", 1.0)})
    r = fake_engine.translate_one("Good morning. Where is the clinic?", "english", "yoruba")
    assert _mt(fake_engine).calls[0][0] == ["Where is the clinic?"]
    assert r.text.startswith("Ẹ káàárọ̀.") and r.origin == "mt_tm_assisted"


def test_similar_but_not_exact_memory_is_not_served(fake_engine):
    # A fuzzy score just below exact must not be returned verbatim ("two" vs "ten" tablets).
    fake_engine.tm = _TM({"Take ten tablets daily.": ("Mu wàláà méjì lójoojúmọ́.", 0.98)})
    r = fake_engine.translate_one("Take ten tablets daily.", "english", "yoruba")
    assert r.origin == "mt_tm_assisted"
    assert r.text == "[yoruba] Take ten tablets daily."


def test_cache_is_dropped_when_memory_changes(fake_engine):
    tm = _TM({"Good morning": ("POISON", 1.0)})
    fake_engine.tm = tm
    assert fake_engine.translate_one("Good morning", "english", "yoruba").text == "POISON"
    del tm.entries["Good morning"]
    tm.generation += 1  # what TranslationMemory.delete() does
    r = fake_engine.translate_one("Good morning", "english", "yoruba")
    assert r.text == "[yoruba] Good morning" and r.origin == "mt"


def test_use_tm_false_ignores_memory_results_in_cache(fake_engine):
    fake_engine.tm = _TM({"Good morning": ("FROM-TM", 1.0)})
    assert fake_engine.translate_one("Good morning", "english", "yoruba").origin == "tm_exact"
    r = fake_engine.translate(["Good morning"], "english", "yoruba", use_tm=False)[0]
    assert r.text == "[yoruba] Good morning"


def test_over_long_text_is_rejected(fake_engine):
    fake_engine.settings.max_text_chars = 50
    with pytest.raises(InvalidInputError):
        fake_engine.translate_one("x" * 51, "english", "yoruba")


def test_tm_generation_bumps_on_writes(tmp_path):
    from afrilingua.tm import open_translation_memory

    with open_translation_memory(tmp_path) as tm:
        g0 = tm.generation
        entry_id = tm.add("Good morning", "Ẹ káàárọ̀", "english", "yoruba")
        assert tm.generation > g0
        g1 = tm.generation
        tm.approve(entry_id)
        assert tm.generation > g1
        g2 = tm.generation
        tm.delete(entry_id)
        assert tm.generation > g2


class _Tok:
    all_special_tokens = ("<s>", "</s>", "<unk>", "<pad>")

    def __init__(self, vocab):
        self._vocab = vocab

    def __len__(self):
        return len(self._vocab)

    def convert_ids_to_tokens(self, ids):
        return [self._vocab[i] for i in ids]


def test_vocab_map_keeps_registered_scripts_only(tmp_path):
    from afrilingua.backends.mt_ct2 import VMAP_FILE, ensure_vocab_map

    vocab = ["<s>", "</s>", "<unk>", "<pad>", "yor_Latn", "▁ọ̀kọ̀", "▁ƙasa", "▁中国", "▁Привет", "▁hello"]
    assert ensure_vocab_map(tmp_path, _Tok(vocab), {"Latn"})
    allowed = (tmp_path / VMAP_FILE).read_text(encoding="utf-8").strip("\t\n").split(" ")
    assert "▁ọ̀kọ̀" in allowed and "▁ƙasa" in allowed and "</s>" in allowed
    assert "▁中国" not in allowed and "▁Привет" not in allowed
    # cached: second call does not rewrite
    mtime = (tmp_path / VMAP_FILE).stat().st_mtime_ns
    assert ensure_vocab_map(tmp_path, _Tok(vocab), {"Latn"})
    assert (tmp_path / VMAP_FILE).stat().st_mtime_ns == mtime


def test_vocab_map_skipped_for_unknown_script(tmp_path):
    from afrilingua.backends.mt_ct2 import ensure_vocab_map

    assert not ensure_vocab_map(tmp_path, _Tok(["<s>"]), {"Latn", "Zzzz"})
