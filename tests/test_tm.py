"""Translation memory: storage, exact/fuzzy search, dedupe, import/export, persistence."""

from __future__ import annotations

import itertools
import json
import os
import random
import threading
import time

import numpy as np
import pytest

from afrilingua.core.errors import InvalidInputError, NotFoundError, UnsupportedLanguageError
from afrilingua.engine import TMMatch, TranslationMemoryProtocol
from afrilingua.tm import TranslationMemory, open_translation_memory
from afrilingua.tm.index import CharNgramIndex, IndexCorruptError
from afrilingua.tm.text import base_form, similarity, tm_key
from afrilingua.tm.vectorizer import vectorize_many, vectorize_one


@pytest.fixture
def tm(tmp_path):
    mem = open_translation_memory(tmp_path / "tm")
    yield mem
    mem.close()


# ── text / vectors ──────────────────────────────────────────────────────────
def test_tm_key_normalises_case_punctuation_and_nfc():
    decomposed = "E\u0323 ka\u0301a\u0300a\u0301ro\u0323\u0300!"  # NFD Yoruba
    assert tm_key(decomposed) == tm_key("Ẹ káàárọ̀")
    assert tm_key("  Good   morning!! ") == "good morning"
    assert base_form("ẹ káàárọ̀") == "e kaaaro"


def test_similarity_bounds():
    assert similarity("good morning", "good morning") == 1.0
    assert 0.8 < similarity("good morning", "good mornings") < 1.0
    assert similarity("good morning", "xyz") < 0.4


def test_vectorizer_is_stable_and_normalised():
    indptr, dims, vals = vectorize_many(["hello world", "", "ẹ káàárọ̀"])
    assert indptr.shape == (4,)
    for i in range(3):
        seg = vals[indptr[i] : indptr[i + 1]]
        assert abs(float(np.sum(seg.astype(np.float64) ** 2)) - 1.0) < 1e-5
    d1, v1 = vectorize_one("hello world")
    d2, v2 = vectorize_one("hello world")
    assert np.array_equal(d1, d2) and np.allclose(v1, v2)
    # Hash values must not depend on PYTHONHASHSEED: pin a couple of buckets.
    assert np.array_equal(d1, dims[indptr[0] : indptr[1]])


# ── protocol & basic behaviour ──────────────────────────────────────────────
def test_implements_engine_protocol(tm):
    assert isinstance(tm, TranslationMemoryProtocol)


def test_add_and_exact_lookup(tm):
    entry_id = tm.add("Good morning!", "Ẹ káàárọ̀", "english", "yo", approved=True, origin="community")
    matches = tm.lookup("good morning", "en", "yoruba")
    assert matches and isinstance(matches[0], TMMatch)
    m = matches[0]
    assert (m.id, m.target, m.score, m.approved, m.origin) == (entry_id, "Ẹ káàárọ̀", 1.0, True, "community")


def test_fuzzy_lookup_ranks_closest_first(tm):
    tm.add("Good morning everyone", "Ẹ káàárọ̀ gbogbo yín", "english", "yoruba")
    tm.add("Good evening", "Ẹ káalẹ́", "english", "yoruba")
    tm.add("Where is the hospital?", "Ibo ni ilé ìwòsàn wà?", "english", "yoruba")
    matches, took = tm.search("good morning every one", "english", "yoruba", k=3)
    assert took >= 0
    assert matches[0].source == "Good morning everyone"
    assert 0.8 <= matches[0].score < 1.0
    assert all(a.score >= b.score for a, b in itertools.pairwise(matches))


def test_fuzzy_finds_tone_marked_source_without_marks(tm):
    tm.add("Ẹ káàárọ̀ gbogbo yín", "Good morning everyone", "yoruba", "english")
    matches = tm.lookup("e kaaaro gbogbo yin", "yoruba", "english")
    assert matches and matches[0].target == "Good morning everyone"
    assert matches[0].score < 0.97  # never treated as an exact hit


def test_non_exact_never_reaches_exact_threshold(tm):
    tm.add("Good morning", "Ẹ káàárọ̀", "english", "yoruba", approved=True)
    for q in ("Good mornings", "Good morning to you", "good mornin"):
        m = tm.lookup(q, "english", "yoruba")
        assert m and m[0].score < 0.97


def test_pairs_are_isolated(tm):
    tm.add("Thank you", "Ẹ ṣé", "english", "yoruba")
    assert tm.lookup("Thank you", "english", "hausa") == []
    assert tm.lookup("Thank you", "yoruba", "english") == []


def test_dedupe_bumps_uses_and_upgrades_approval(tm):
    a = tm.add("Thank you", "Na gode", "english", "hausa")
    b = tm.add("thank you!", "Na gode", "english", "hausa", approved=True, origin="community")
    assert a == b
    entry = tm.get(a)
    assert entry["uses"] == 2 and entry["approved"] is True and entry["origin"] == "community"
    # approved never downgrades
    tm.add("Thank you", "Na gode", "english", "hausa", approved=False, origin="mt")
    assert tm.get(a)["approved"] is True and tm.get(a)["origin"] == "community"
    # a different target is a separate entry
    c = tm.add("Thank you", "Nagode", "english", "hausa")
    assert c != a
    assert tm.stats()["entries"] == 2


def test_exact_prefers_approved(tm):
    tm.add("Welcome", "Barka da zuwa (mt)", "english", "hausa")
    tm.add("Welcome", "Barka da zuwa", "english", "hausa", approved=True)
    assert tm.lookup("welcome", "english", "hausa")[0].target == "Barka da zuwa"


def test_approve_delete_get(tm):
    entry_id = tm.add("Please", "Biko", "english", "igbo")
    assert tm.get(entry_id)["approved"] is False
    assert tm.approve(entry_id)["approved"] is True
    assert tm.delete(entry_id) is True
    assert tm.get(entry_id) is None
    assert tm.lookup("Please", "english", "igbo") == []
    with pytest.raises(NotFoundError):
        tm.approve(entry_id)
    with pytest.raises(NotFoundError):
        tm.delete(entry_id)


def test_validation(tm):
    with pytest.raises(InvalidInputError):
        tm.add("", "x", "english", "igbo")
    with pytest.raises(InvalidInputError):
        tm.add("hello", "   ", "english", "igbo")
    with pytest.raises(InvalidInputError):
        tm.add("x" * 5000, "y", "english", "igbo")
    with pytest.raises(UnsupportedLanguageError):
        tm.add("hello", "x", "english", "klingon")
    assert tm.search("", "english", "igbo")[0] == []


def test_stats_shape(tm):
    tm.add("a b c", "x", "english", "yoruba", approved=True)
    tm.add("d e f", "y", "english", "yoruba")
    tm.add("g h i", "z", "english", "hausa")
    assert tm.stats() == {"entries": 3, "approved": 1, "by_pair": {"english→yoruba": 2, "english→hausa": 1}}


# ── import / export ─────────────────────────────────────────────────────────
def test_import_tsv_csv_jsonl(tm, tmp_path):
    tsv = tmp_path / "a.tsv"
    tsv.write_text("source\ttarget\nGood morning\tẸ káàárọ̀\nThank you\tẸ ṣé\n\nbad line\n", encoding="utf-8")
    assert tm.import_file(tsv, "english", "yoruba") == 2
    csv_file = tmp_path / "b.csv"
    csv_file.write_text('english,yoruba\n"Hello, friend",Báwo ni ọ̀rẹ́\n', encoding="utf-8")
    assert tm.import_file(csv_file, "english", "yoruba") == 1
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text(
        json.dumps({"source": "Welcome", "target": "Ẹ kú àbọ̀"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"english": "Goodbye", "yoruba": "Ó dàbọ̀"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert tm.import_file(jsonl, "english", "yoruba") == 2
    assert tm.lookup("Hello, friend", "english", "yoruba")[0].target == "Báwo ni ọ̀rẹ́"
    assert tm.lookup("goodbye", "english", "yoruba")[0].origin == "import"
    assert tm.stats()["entries"] == 5


def test_import_tmx(tm, tmp_path):
    tmx = tmp_path / "mem.tmx"
    tmx.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<tmx version="1.4"><header srclang="en" datatype="plaintext" segtype="sentence" adminlang="en"
 creationtool="test" creationtoolversion="1" o-tmf="none"/>
<body>
 <tu><tuv xml:lang="en-US"><seg>Good night</seg></tuv><tuv xml:lang="ha-NG"><seg>Sai da safe</seg></tuv></tu>
 <tu><tuv xml:lang="ha"><seg>Na gode</seg></tuv><tuv xml:lang="en"><seg>Thank <bpt i="1">x</bpt>you</seg></tuv></tu>
 <tu><tuv xml:lang="fr"><seg>Bonjour</seg></tuv><tuv xml:lang="ha"><seg>Sannu</seg></tuv></tu>
</body></tmx>""",
        encoding="utf-8",
    )
    assert tm.import_file(tmx, "english", "hausa") == 2
    assert tm.lookup("thank xyou", "english", "hausa")[0].target == "Na gode"


def test_import_bad_tmx_raises(tm, tmp_path):
    bad = tmp_path / "bad.tmx"
    bad.write_text("<tmx><body><tu>", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        tm.import_file(bad, "english", "hausa")


def test_export_jsonl(tm, tmp_path):
    tm.add("One", "Ọ̀kan", "english", "yoruba", approved=True)
    tm.add("Two", "Èjì", "english", "yoruba")
    out = tmp_path / "out" / "tm.jsonl"
    assert tm.export_jsonl(out) == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["target"] == "Ọ̀kan" and rows[0]["approved"] is True
    assert tm.export_jsonl(out, approved_only=False) == 2


# ── persistence & robustness ────────────────────────────────────────────────
def test_persistence_and_reload(tmp_path):
    path = tmp_path / "tm"
    with open_translation_memory(path) as tm1:
        tm1.add("Good morning", "Ina kwana", "english", "hausa", approved=True)
        tm1.add("Good evening", "Barka da yamma", "english", "hausa")
    assert (path / "index" / "english__hausa" / "meta.json").exists()
    with open_translation_memory(path) as tm2:
        assert tm2.lookup("good morning", "english", "hausa")[0].target == "Ina kwana"
        assert tm2.lookup("good evenin", "english", "hausa")[0].target == "Barka da yamma"


def test_index_reconciles_with_sqlite_after_unflushed_writes(tmp_path):
    path = tmp_path / "tm"
    with open_translation_memory(path) as tm1:
        tm1.add("Good morning", "Ina kwana", "english", "hausa")
    tm2 = open_translation_memory(path)
    tm2.add("Market day", "Ranar kasuwa", "english", "hausa")
    gone = tm2.lookup("good morning", "english", "hausa")[0].id
    tm2.delete(gone)
    # simulate a crash: close SQLite connections without flushing the index
    tm2._closed = True
    for c in tm2._conns:
        c.close()
    with open_translation_memory(path) as tm3:
        assert tm3.lookup("market dya", "english", "hausa")[0].target == "Ranar kasuwa"
        assert all(m.id != gone for m in tm3.lookup("good morning", "english", "hausa"))


@pytest.mark.parametrize("victim", ["rows.npy", "meta.json", "ids.json", "colptr.npy"])
def test_corrupt_index_is_rebuilt(tmp_path, victim):
    path = tmp_path / "tm"
    with open_translation_memory(path) as tm1:
        for i in range(20):
            tm1.add(f"sentence number {i} about the market", f"t{i}", "english", "igbo")
    target = path / "index" / "english__igbo" / victim
    target.write_bytes(b"garbage")
    with open_translation_memory(path) as tm2:
        m = tm2.lookup("sentence number 7 about the markt", "english", "igbo")
        assert m and m[0].target == "t7"


def test_rebuild_index(tm):
    tm.add("Hello there", "Sannu", "english", "hausa")
    assert tm.rebuild_index() == 1
    assert tm.lookup("hello ther", "english", "hausa")[0].target == "Sannu"


def test_char_index_load_rejects_inconsistent_arrays(tmp_path):
    idx = CharNgramIndex()
    idx.add(["a", "b"], ["hello", "world"])
    idx.save(tmp_path / "i")
    np.save(tmp_path / "i" / "rows.npy", np.array([0, 5], dtype=np.int32))
    with pytest.raises(IndexCorruptError):
        CharNgramIndex.load(tmp_path / "i")


def test_char_index_delta_compaction_and_removal():
    idx = CharNgramIndex(delta_capacity=8)
    texts = [f"the quick brown fox number {i}" for i in range(50)]
    for i, t in enumerate(texts):
        idx.add([str(i)], [t])
    assert len(idx) == 50
    for i in range(0, 50, 2):
        assert idx.remove(str(i))
    assert not idx.remove("0")
    assert len(idx) == 25
    res = idx.search("the quick brown fox number 13", 3)
    assert res[0][0] == "13"
    assert all(int(r) % 2 == 1 for r, _ in idx.search("quick brown fox", 30))
    idx.add(["13"], ["completely different words"])  # replace
    assert idx.search("completely different words", 1)[0][0] == "13"


def test_thread_safety(tm):
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(30):
                tm.add(f"writer {n} sentence {i}", f"w{n}-{i}", "english", "yoruba")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def reader() -> None:
        try:
            for i in range(30):
                tm.search(f"writer 1 sentence {i}", "english", "yoruba")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert tm.stats()["entries"] == 90
    assert tm.lookup("writer 2 sentence 29", "english", "yoruba")[0].target == "w2-29"


# ── latency ─────────────────────────────────────────────────────────────────
def test_lookup_latency_micro_benchmark(tmp_path):
    """Exact + fuzzy latency on a synthetic 20k-entry TM (100k: see docs/benchmarks-tm.md)."""
    if os.environ.get("AFRILINGUA_SKIP_BENCH"):
        pytest.skip("benchmarks disabled")
    rng = random.Random(7)
    vocab = [
        "market",
        "price",
        "doctor",
        "school",
        "church",
        "meeting",
        "today",
        "tomorrow",
        "please",
        "thank",
        "water",
        "food",
        "road",
        "money",
        "family",
        "children",
        "teacher",
        "village",
        "city",
        "health",
        "morning",
        "evening",
        "hospital",
        "bank",
        "government",
        "office",
        "farm",
        "rain",
        "night",
        "week",
        "month",
        "year",
        "welcome",
        "everyone",
        "speaker",
        "question",
        "answer",
        "agenda",
        "minutes",
        "break",
        "lunch",
    ]
    pairs = list({" ".join(rng.choices(vocab, k=rng.randint(4, 12))): None for _ in range(20_000)})
    with TranslationMemory(tmp_path / "tm") as tm:
        t0 = time.perf_counter()
        tm.add_many([(s, f"t{i}") for i, s in enumerate(pairs)], "english", "hausa")
        build_s = time.perf_counter() - t0
        if build_s > 60:
            pytest.skip(f"machine too slow for the micro-benchmark ({build_s:.0f}s to build)")
        sample = rng.sample(pairs, 100)
        tm.search(sample[0] + " x", "english", "hausa")  # warm up
        exact, fuzzy = [], []
        for s in sample:
            exact.append(tm.search(s, "english", "hausa")[1])
            fuzzy.append(tm.search(s + " extra", "english", "hausa")[1])
        exact.sort()
        fuzzy.sort()
        p50_exact, p50_fuzzy = exact[50], fuzzy[50]
        print(f"\nTM 20k: build {build_s:.1f}s, exact p50 {p50_exact:.3f} ms, fuzzy p50 {p50_fuzzy:.2f} ms")
        assert p50_exact < 5
        assert p50_fuzzy < 50  # generous: shared CI runners are noisy
