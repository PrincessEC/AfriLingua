"""Meaning-based TM search: labelled matches, off the translation path, background indexing."""

from __future__ import annotations

import numpy as np

from afrilingua.tm.memory import TranslationMemory

# A tiny "embedder": sentences sharing a concept id get the same vector.
CONCEPTS = {
    "where is the nearest clinic": 0,
    "where can i find the closest hospital": 0,
    "ibo ni ilé ìwòsàn wà": 0,
    "the meeting starts at ten": 1,
    "we begin at 10 am": 1,
    "the road is closed": 2,
}


class FakeEmbedder:
    model_id = "fake/embedder"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for row, text in enumerate(texts):
            key = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace() or ord(ch) > 127).strip()
            concept = CONCEPTS.get(" ".join(key.split()), 7)
            out[row, concept] = 1.0
        return out


def _tm(tmp_path) -> tuple[TranslationMemory, FakeEmbedder]:
    emb = FakeEmbedder()
    tm = TranslationMemory(tmp_path, embedder=emb)
    tm.add_many(
        [
            ("Where is the nearest clinic?", "Ibo ni ilé ìwòsàn tó sún mọ́ jù lọ wà?"),
            ("The meeting starts at ten.", "Ìpàdé náà bẹ̀rẹ̀ ní aago mẹ́wàá."),
            ("The road is closed.", "Ọ̀nà náà ti dí."),
        ],
        "english",
        "yoruba",
        approved=True,
    )
    assert tm.wait_for_semantic_index(timeout=10)
    return tm, emb


def test_meaning_search_finds_paraphrase_and_labels_it(tmp_path):
    tm, _ = _tm(tmp_path)
    try:
        matches, _ = tm.search(
            "Where can I find the closest hospital?", "english", "yoruba", 3, mode="meaning"
        )
        assert matches[0].source == "Where is the nearest clinic?"
        assert matches[0].kind == "meaning"
        assert matches[0].score < 1.0  # never presented as an exact translation
    finally:
        tm.close()


def test_meaning_search_works_across_languages(tmp_path):
    tm, _ = _tm(tmp_path)
    try:
        matches, _ = tm.search("Ibo ni ilé ìwòsàn wà?", "english", "yoruba", 3, mode="meaning")
        assert matches and matches[0].source == "Where is the nearest clinic?"
    finally:
        tm.close()


def test_exact_match_stays_first_and_labelled_exact(tmp_path):
    tm, _ = _tm(tmp_path)
    try:
        matches, _ = tm.search("the meeting starts at ten", "english", "yoruba", 3, mode="meaning")
        assert matches[0].kind == "exact" and matches[0].score == 1.0
    finally:
        tm.close()


def test_translation_path_never_calls_the_embedder(tmp_path):
    tm, emb = _tm(tmp_path)
    try:
        before = emb.calls
        tm.lookup("Where can I find the closest hospital?", "english", "yoruba")
        tm.search("We begin at 10 am", "english", "yoruba", 3)  # default mode is fast
        assert emb.calls == before
    finally:
        tm.close()


def test_fast_mode_does_not_return_meaning_matches(tmp_path):
    tm, _ = _tm(tmp_path)
    try:
        matches, _ = tm.search("We begin at 10 am", "english", "yoruba", 3, mode="fast")
        assert all(m.kind != "meaning" for m in matches)
    finally:
        tm.close()


def test_failing_embedder_falls_back_to_wording(tmp_path):
    class Broken:
        model_id = "broken"

        def __call__(self, texts):
            raise RuntimeError("model not downloaded")

    tm = TranslationMemory(tmp_path, embedder=Broken())
    try:
        tm.add("The road is closed.", "Ọ̀nà náà ti dí.", "english", "yoruba", approved=True)
        tm.wait_for_semantic_index(timeout=5)
        matches, _ = tm.search("The road is closed now", "english", "yoruba", 3, mode="meaning")
        assert matches and matches[0].kind == "wording"
        assert tm.semantic_status()["enabled"] is False
    finally:
        tm.close()


def test_semantic_index_survives_reopen(tmp_path):
    tm, _ = _tm(tmp_path)
    tm.close()
    emb2 = FakeEmbedder()
    tm2 = TranslationMemory(tmp_path, embedder=emb2)
    try:
        matches, _ = tm2.search("We begin at 10 am", "english", "yoruba", 3, mode="meaning")
        assert matches[0].source == "The meeting starts at ten."
        # vectors were loaded from disk: only the query was embedded
        assert emb2.calls == 1
    finally:
        tm2.close()
