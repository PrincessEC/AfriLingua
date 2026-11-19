"""Community contribution store: validation, quality, review workflow, prompts, stats, export, audio."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from afrilingua.contrib import ContributionStore, ReviewPolicy, open_contribution_store
from afrilingua.contrib.prompts import load_prompt_file, seed_prompts
from afrilingua.contrib.quality import estimate_quality
from afrilingua.contrib.validation import find_pii, scrub_pii
from afrilingua.core.errors import (
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedLanguageError,
)
from afrilingua.tm import open_translation_memory


@pytest.fixture
def tm(tmp_path):
    mem = open_translation_memory(tmp_path / "tm")
    yield mem
    mem.close()


@pytest.fixture
def store(tmp_path, tm):
    s = ContributionStore(tmp_path / "contrib", tm=tm, use_lid=False)
    yield s
    s.close()


def payload(**over):
    base = {
        "kind": "translation",
        "src_lang": "english",
        "tgt_lang": "yoruba",
        "source_text": "Good morning, everyone.",
        "target_text": "Ẹ káàárọ̀, gbogbo yín.",
        "contributor": "Ada",
        "license": "CC-BY-4.0",
        "consent": True,
    }
    base.update(over)
    return base


def wav_bytes(seconds=1.0, amp=0.3, sr=48_000, fmt="WAV", channels=1):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt)
    return buf.getvalue()


# ── validation ──────────────────────────────────────────────────────────────
def test_submit_returns_contract_shape(store):
    c = store.submit(payload(tgt_lang="yo", domain="Education"))
    for key in (
        "id",
        "kind",
        "src_lang",
        "tgt_lang",
        "source_text",
        "target_text",
        "license",
        "consent",
        "status",
        "votes_up",
        "votes_down",
        "quality_score",
        "created_at",
        "has_audio",
        "contributor",
        "domain",
    ):
        assert key in c
    assert c["status"] == "pending" and c["tgt_lang"] == "yoruba" and c["domain"] == "education"
    assert c["has_audio"] is False and 0 <= c["quality_score"] <= 1


@pytest.mark.parametrize(
    "bad",
    [
        {"consent": False},
        {"consent": "true"},
        {"license": "MIT"},
        {"kind": "poem"},
        {"tgt_lang": "english"},
        {"source_text": "  "},
        {"target_text": ""},
        {"source_text": "x" * 1001},
        {"domain": "not a slug!"},
        {"dialect": "d" * 101},
    ],
)
def test_validation_errors(store, bad):
    with pytest.raises(InvalidInputError):
        store.submit(payload(**bad))


def test_unknown_language(store):
    with pytest.raises(UnsupportedLanguageError):
        store.submit(payload(tgt_lang="klingon"))


def test_text_is_nfc_normalised(store):
    c = store.submit(payload(target_text="E\u0323 ka\u0301a\u0300a\u0301ro\u0323\u0300"))
    assert c["target_text"] == "Ẹ káàárọ̀"


def test_pii_scrub_and_flag(store):
    c = store.submit(
        payload(source_text="Email me at ada@example.com or call +234 803 123 4567, see www.example.com")
    )
    assert "ada@example.com" not in c["source_text"]
    assert "[email]" in c["source_text"] and "[phone]" in c["source_text"] and "[url]" in c["source_text"]
    assert "pii" in c["quality_flags"]
    assert find_pii("Meet at 10:30 on 12/05") == []
    assert scrub_pii("I have 3 children") == "I have 3 children"


def test_contributor_email_is_pseudonymised(store):
    c = store.submit(payload(contributor="ada@example.com"))
    assert c["contributor"].startswith("anon-")


# ── quality ─────────────────────────────────────────────────────────────────
def test_quality_heuristics():
    good = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="yoruba",
        source_text="Good morning, everyone.",
        target_text="Ẹ káàárọ̀, gbogbo yín.",
    )
    assert good.score == 1.0 and "tone_marks" in good.flags
    same = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="hausa",
        source_text="Good morning everyone",
        target_text="Good morning everyone",
    )
    assert "identical_to_source" in same.flags and same.score <= 0.2
    no_tones = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="yoruba",
        source_text="Good morning everyone",
        target_text="E kaaaro gbogbo yin",
    )
    assert "missing_tone_marks" in no_tones.flags and no_tones.score < good.score
    ratio = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="hausa",
        source_text="Please take this medicine twice a day after food.",
        target_text="Ee",
    )
    assert "length_ratio" in ratio.flags
    script = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="igbo",
        source_text="Thank you very much",
        target_text="Спасибо большое",
    )
    assert "non_latin_script" in script.flags
    lid = estimate_quality(
        kind="translation",
        src_lang="english",
        tgt_lang="hausa",
        source_text="Thank you very much my friend",
        target_text="Merci beaucoup mon ami",
        detector=lambda text: ("french", 0.99),
    )
    assert "lid_mismatch" in lid.flags


def test_quality_flags_duplicate_of_approved_tm_pair(store, tm):
    tm.add("Good morning, everyone.", "Ẹ káàárọ̀, gbogbo yín.", "english", "yoruba", approved=True)
    c = store.submit(payload())
    assert "duplicate" in c["quality_flags"] and c["quality_score"] <= 0.5


# ── review workflow ─────────────────────────────────────────────────────────
def test_two_approvals_approve_and_write_tm(store, tm):
    c = store.submit(payload())
    assert tm.lookup("Good morning, everyone", "english", "yoruba") == []
    c = store.review(c["id"], "Bola", "approve")
    assert c["status"] == "pending" and c["votes_up"] == 1
    c = store.review(c["id"], "Chidi", "approve", comment="great")
    assert c["status"] == "approved" and c["votes_up"] == 2
    m = tm.lookup("good morning everyone", "english", "yoruba")[0]
    assert m.approved and m.origin == "community" and m.target == "Ẹ káàárọ̀, gbogbo yín."
    assert [r["reviewer"] for r in store.reviews(c["id"])] == ["Bola", "Chidi"]


def test_split_votes_need_a_majority(store, tm):
    c = store.submit(payload())
    store.review(c["id"], "Bola", "reject")
    c = store.review(c["id"], "Chidi", "approve")
    assert c["status"] == "pending"  # 1 vs 1
    c = store.review(c["id"], "Dayo", "approve")
    assert c["status"] == "approved"  # 2 approvals > 1 rejection
    d = store.submit(payload(source_text="Thank you", target_text="Ẹ ṣé"))
    store.review(d["id"], "Bola", "approve")
    store.review(d["id"], "Chidi", "reject")
    d = store.review(d["id"], "Dayo", "reject")
    assert d["status"] == "rejected"  # 2 rejections > 1 approval


def test_rejected_when_rejections_dominate(store, tm):
    c = store.submit(payload())
    store.review(c["id"], "Bola", "reject")
    c = store.review(c["id"], "Chidi", "reject")
    assert c["status"] == "rejected"
    assert tm.lookup("Good morning, everyone", "english", "yoruba") == []
    with pytest.raises(InvalidInputError):
        store.review(c["id"], "Dayo", "approve")


def test_self_review_blocked_and_one_vote_per_reviewer(store):
    c = store.submit(payload(contributor="Ada"))
    with pytest.raises(InvalidInputError):
        store.review(c["id"], "ada", "approve")
    store.review(c["id"], "Bola", "approve")
    c = store.review(c["id"], "Bola", "approve")  # re-vote replaces, does not double count
    assert c["votes_up"] == 1 and c["status"] == "pending"
    c = store.review(c["id"], "Bola", "reject")
    assert (c["votes_up"], c["votes_down"]) == (0, 1)
    with pytest.raises(InvalidInputError):
        store.review(c["id"], "", "approve")
    with pytest.raises(InvalidInputError):
        store.review(c["id"], "Chidi", "maybe")
    with pytest.raises(NotFoundError):
        store.review("nope", "Chidi", "approve")


def test_reputation_and_trusted_weight(tmp_path, tm):
    policy = ReviewPolicy(trusted_reputation=2, trusted_weight=2)
    with ContributionStore(tmp_path / "c2", tm=tm, policy=policy, use_lid=False) as s:
        for i in range(2):
            c = s.submit(
                payload(source_text=f"Sentence number {i}", target_text=f"Gbólóhùn {i}", contributor="Bola")
            )
            s.review(c["id"], "X", "approve")
            s.review(c["id"], "Y", "approve")
        leader = s.stats()["leaderboard"][0]
        assert leader["contributor"] == "Bola" and leader["reputation"] == 2 and leader["approved"] == 2
        c = s.submit(payload(contributor="Ada"))
        c = s.review(c["id"], "bola", "approve")  # trusted: weight 2 -> approved by a single vote
        assert c["votes_up"] == 2 and c["status"] == "approved"
        c = s.submit(payload(source_text="Bad one", target_text="Búburú", contributor="Ada"))
        s.review(c["id"], "X", "reject")
        s.review(c["id"], "Y", "reject")
        ada = next(x for x in s.stats()["leaderboard"] if x["contributor"] == "Ada")
        assert ada["reputation"] == 0  # +1 approved, -1 rejected


def test_review_queue_excludes_own_and_voted(store):
    a = store.submit(payload(contributor="Ada"))
    b = store.submit(
        payload(contributor="Bola", source_text="Thank you", target_text="Ẹ ṣé", tgt_lang="yoruba")
    )
    h = store.submit(payload(contributor="Chidi", tgt_lang="hausa", target_text="Barka da safiya, ku duka."))
    queue = {c["id"] for c in store.review_queue("Ada")}
    assert a["id"] not in queue and {b["id"], h["id"]} <= queue
    store.review(b["id"], "Ada", "approve")
    assert b["id"] not in {c["id"] for c in store.review_queue("Ada")}
    assert {c["id"] for c in store.review_queue("Ada", lang="hausa")} == {h["id"]}


def test_list_filters_and_pagination(store):
    for i in range(5):
        store.submit(payload(source_text=f"Sentence {i}", target_text=f"Gbólóhùn {i}"))
    store.submit(payload(tgt_lang="igbo", target_text="Ụtụtụ ọma, unu niile."))
    items, total = store.list(limit=2)
    assert total == 6 and len(items) == 2
    items, total = store.list(lang="igbo")
    assert total == 1 and items[0]["tgt_lang"] == "igbo"
    assert store.list(status="approved") == ([], 0)
    with pytest.raises(InvalidInputError):
        store.list(status="weird")


# ── prompts ─────────────────────────────────────────────────────────────────
def test_bundled_prompt_files():
    en = load_prompt_file("prompts_en.txt", "english", "en")
    assert len(en) >= 250
    domains = {p.domain for p in en}
    assert {
        "greetings",
        "conference",
        "health",
        "government",
        "education",
        "faith",
        "market",
        "numbers",
    } <= domains
    for iso, lang in (("yo", "yoruba"), ("ig", "igbo"), ("ha", "hausa")):
        seeds = seed_prompts(lang, iso)
        assert seeds and all(p.domain == "seed" and p.lang == lang for p in seeds)
    assert len({p.id for p in en}) == len(en)


def test_translate_prompts_prioritise_untranslated(store):
    first = store.get_prompts("yoruba", "translate", n=100)
    assert len(first) == 100
    assert all(p["lang"] == "english" and p["text"] for p in first)
    done = first[0]
    store.submit(payload(source_text=done["text"]))
    again = store.get_prompts("yoruba", "translate", n=100)  # pool ~280 > 100: translated one sorts last
    assert done["text"] not in {p["text"] for p in again}
    assert len(store.get_prompts("hausa", "translate", n=5)) == 5


def test_record_prompts_use_seeds_then_approved(store):
    seeds = store.get_prompts("yoruba", "record", n=50)
    assert seeds and all(p["lang"] == "yoruba" for p in seeds)
    assert any(p["domain"] == "seed" for p in seeds)
    c = store.submit(payload(target_text="Ẹ kú iṣẹ́ àṣekára."))
    store.review(c["id"], "B", "approve")
    store.review(c["id"], "C", "approve")
    texts = {p["text"] for p in store.get_prompts("yoruba", "record", n=100)}
    assert "Ẹ kú iṣẹ́ àṣekára." in texts
    with pytest.raises(InvalidInputError):
        store.get_prompts("yoruba", "sing")


# ── audio ───────────────────────────────────────────────────────────────────
def test_audio_upload_stored_as_16k_flac(store):
    c = store.add_audio(
        wav_bytes(1.5, channels=2),
        text="Ẹ káàárọ̀",
        lang="yo",
        filename="r.wav",
        consent=True,
        license="CC0-1.0",
        contributor="Ada",
        accent="Ọ̀yọ́",
        gender="female",
        age_range="25-34",
    )
    assert c["kind"] == "transcription" and c["has_audio"] and c["lang"] == "yoruba"
    assert abs(c["audio_duration_s"] - 1.5) < 0.01 and c["accent"] == "Ọ̀yọ́"
    path, media = store.audio_file(c["id"])
    assert media == "audio/flac" and path.parent.name == "yoruba" and path.suffix == ".flac"
    info = sf.info(str(path))
    assert info.samplerate == 16_000 and info.channels == 1
    assert store.stats()["by_lang"]["yoruba"]["audio"] == 1
    assert store.stats()["hours_audio"] > 0


def test_audio_rejects_bad_recordings(store):
    kw = {"text": "Sannu", "lang": "hausa", "consent": True, "license": "CC-BY-4.0"}
    with pytest.raises(InvalidInputError):
        store.add_audio(wav_bytes(1.0, amp=0.0), **kw)  # silence
    with pytest.raises(InvalidInputError):
        store.add_audio(wav_bytes(1.0, amp=3.0), **kw)  # clipped
    with pytest.raises(InvalidInputError):
        store.add_audio(wav_bytes(31.5, sr=16_000), **kw)  # too long
    with pytest.raises(InvalidInputError):
        store.add_audio(wav_bytes(1.0), **{**kw, "consent": False})
    with pytest.raises(InvalidInputError):
        store.add_audio(b"", **kw)
    with pytest.raises(NotFoundError):
        store.audio_file(store.submit(payload())["id"])


def test_audio_upload_limit(tmp_path):
    with (
        ContributionStore(tmp_path / "c", use_lid=False, max_upload_mb=1) as s,
        pytest.raises(PayloadTooLargeError),
    ):
        s.add_audio(b"0" * (2 * 1024 * 1024), text="Sannu", lang="hausa", consent=True)


def test_undecodable_audio_is_kept_for_transcoding(store):
    c = store.add_audio(
        b"\x1aE\xdf\xa3 definitely not audio", text="Sannu", lang="hausa", filename="x.webm", consent=True
    )
    assert c["needs_transcode"] is True and c["has_audio"]
    path, media = store.audio_file(c["id"])
    assert path.suffix == ".webm" and media == "audio/webm"


# ── stats / export / withdraw ───────────────────────────────────────────────
def _approve(store, cid):
    store.review(cid, "Rev1", "approve")
    return store.review(cid, "Rev2", "approve")


def test_stats_shape(store):
    a = store.submit(payload())
    store.submit(payload(tgt_lang="hausa", target_text="Barka da safiya, ku duka.", contributor="Bola"))
    _approve(store, a["id"])
    s = store.stats()
    assert (s["total"], s["approved"], s["pending"]) == (2, 1, 1)
    assert s["by_lang"]["yoruba"] == {"text": 1, "audio": 0, "approved": 1}
    assert s["by_lang"]["hausa"]["text"] == 1
    assert s["contributors"] == 2 and s["hours_audio"] == 0
    assert s["leaderboard"][0]["contributor"] == "Ada"


def test_export_jsonl_and_hf(store, tmp_path):
    a = store.submit(payload())
    store.submit(payload(source_text="Pending one", target_text="Ọ̀kan"))
    r = store.add_audio(wav_bytes(1.0), text="Ẹ káàárọ̀", lang="yoruba", consent=True, contributor="Bola")
    _approve(store, a["id"])
    _approve(store, r["id"])
    path = store.export(fmt="jsonl", out_dir=tmp_path / "ex")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(
        row["license"] in {"CC-BY-4.0", "CC0-1.0"} and row["contributor"] for row in rows
    )
    only_text = store.export(kind="translation", out_dir=tmp_path / "ex")
    assert len(only_text.read_text(encoding="utf-8").splitlines()) == 1
    folder = store.export(lang="yoruba", fmt="hf", out_dir=tmp_path / "ex")
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: cc-by-4.0") and "- yo" in card and "Bola" in card
    data = [json.loads(line) for line in (folder / "data.jsonl").read_text(encoding="utf-8").splitlines()]
    audio_rows = [d for d in data if d["audio"]]
    assert len(audio_rows) == 1 and (folder / audio_rows[0]["audio"]).is_file()
    with pytest.raises(InvalidInputError):
        store.export(fmt="csv")


def test_export_loads_with_datasets(store, tmp_path):
    datasets = pytest.importorskip("datasets")
    _approve(store, store.submit(payload())["id"])
    folder = store.export(fmt="hf", out_dir=tmp_path / "ex")
    ds = datasets.load_dataset(
        "json", data_files=str(folder / "data.jsonl"), split="train", cache_dir=str(tmp_path / "hf")
    )
    assert ds[0]["target_text"] == "Ẹ káàárọ̀, gbogbo yín."


def test_withdraw_deletes_pending_and_anonymises_approved(store):
    approved = store.submit(payload(contributor="Ada"))
    _approve(store, approved["id"])
    pending = store.submit(payload(contributor="Ada", source_text="Thank you", target_text="Ẹ ṣé"))
    rec = store.add_audio(
        wav_bytes(1.0), text="Ẹ ṣé", lang="yoruba", consent=True, contributor="Ada", gender="f"
    )
    _approve(store, rec["id"])
    rec_path, _ = store.audio_file(rec["id"])
    other = store.submit(payload(contributor="Bola", source_text="Welcome", target_text="Ẹ kú àbọ̀"))
    store.review(other["id"], "Ada", "approve")
    result = store.withdraw("ada")
    assert result == {"deleted": 1, "anonymized": 2, "audio_deleted": 1}
    with pytest.raises(NotFoundError):
        store.get(pending["id"])
    anon = store.get(approved["id"])
    assert anon["contributor"] == "anonymous"
    r = store.get(rec["id"])
    assert r["has_audio"] is False and r["gender"] is None and not rec_path.exists()
    assert store.reviews(other["id"])[0]["reviewer"] == "anonymous"
    assert all(x["contributor"] != "Ada" for x in store.stats()["leaderboard"])
    with pytest.raises(InvalidInputError):
        store.withdraw("")


def test_open_contribution_store_default_path(tmp_path):
    s = open_contribution_store()
    try:
        assert s.path.name == "contributions" and s.path.is_dir()
    finally:
        s.close()
