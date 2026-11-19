"""Tests for Bible/subtitle/document corpus alignment, filtering, license & domain caps, and splits."""

from __future__ import annotations

import pytest

from afrilingua.core.errors import InvalidInputError
from afrilingua.tm.text import tm_key
from afrilingua.training.corpora import open_corpus_store
from afrilingua.training.datasets import build_training_splits
from afrilingua.training.evaluate import chrf_score, render_model_card, should_promote, wer_cer
from afrilingua.training.mt_lora import MTLoRAConfig, run_mt_lora


def test_ebible_and_usfm_verse_alignment(tmp_path):
    store = open_corpus_store(path=tmp_path / "corpora", use_lid=False)

    # 1. eBible line-aligned with vref + <range> placeholder skipped
    vref = tmp_path / "vref.txt"
    en = tmp_path / "eng.txt"
    ig = tmp_path / "ibo.txt"
    vref.write_text("GEN 1:1\nGEN 1:2\nGEN 1:3\n", encoding="utf-8")
    en.write_text(
        "In the beginning God created the heaven and the earth.\n"
        "<range>\n"
        "And God said, Let there be light: and there was light.\n",
        encoding="utf-8",
    )
    ig.write_text(
        "Na mbụ Chineke kere eluigwe na ụwa.\n"
        "<range>\n"
        "Chineke wee sị, Ka ìhè dị: ìhè wee dị.\n",
        encoding="utf-8",
    )
    rec = store.import_corpus(
        fmt="ebible",
        files=[en, ig, vref],
        src_lang="english",
        tgt_lang="igbo",
        source_name="Igbo Open Bible Sample",
        license="public-domain",
        domain="religious",
    )
    assert rec.accepted_pairs == 2 and rec.commercial_safe is True

    # 2. USFM with footnotes stripped and merged verse ranges (\v 2-3) skipped safely
    usfm_en = tmp_path / "jhn_en.usfm"
    usfm_yo = tmp_path / "jhn_yo.usfm"
    usfm_en.write_text(
        "\\id JHN\n\\c 1\n"
        "\\v 1 In the beginning was the Word.\\f + \\ft note\\f*\n"
        "\\v 2-3 Merged span in one edition.\n"
        "\\v 4 In him was life, and that life was the light of all mankind.\n",
        encoding="utf-8",
    )
    usfm_yo.write_text(
        "\\id JHN\n\\c 1\n"
        "\\v 1 Ní ìpilẹ̀ṣẹ̀ ni Ọ̀rọ̀ wà.\n"
        "\\v 2 Ó wà ní ìpilẹ̀ṣẹ̀ pẹ̀lú Ọlọ́run.\n"
        "\\v 4 Nínú rẹ̀ ni ìyè wà, ìyè náà sì ni ìmọ́lẹ̀ aráyé.\n",
        encoding="utf-8",
    )
    rec_usfm = store.import_corpus(
        fmt="usfm",
        files=[usfm_en, usfm_yo],
        src_lang="english",
        tgt_lang="yoruba",
        source_name="Yoruba Bibeli Mimo Sample",
        license="public-domain",
        domain="religious",
    )
    assert rec_usfm.accepted_pairs == 2


def test_subtitle_merge_and_unaligned_document_alignment(tmp_path):
    store = open_corpus_store(path=tmp_path / "corpora", use_lid=False)

    # 1:2 cue merge (English has 1 cue spanning 1.0..5.0s; Hausa splits across 1.0..2.8s and 3.0..5.0s)
    srt_en = tmp_path / "movie_en.srt"
    srt_ha = tmp_path / "movie_ha.srt"
    srt_en.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\n<i>Good evening my friends, welcome to Kano city.</i>\n\n"
        "2\n00:00:06,000 --> 00:00:07,000\n[MUSIC]\n",
        encoding="utf-8",
    )
    srt_ha.write_text(
        "1\n00:00:01,000 --> 00:00:02,800\n- Barka da yamma abokaina,\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nbarka da zuwa birnin Kano.\n",
        encoding="utf-8",
    )
    rec = store.import_corpus(
        fmt="srt",
        files=[srt_en, srt_ha],
        src_lang="english",
        tgt_lang="hausa",
        source_name="Kano Documentary Subtitles",
        license="CC-BY-4.0",
        domain="media",
    )
    assert rec.accepted_pairs == 1
    pair = next(store.iter_pairs(corpus_ids={rec.id}))
    assert "birnin Kano" in pair["target"]

    # Unaligned document alignment (Gale-Church)
    doc_en = tmp_path / "doc_en.txt"
    doc_yo = tmp_path / "doc_yo.txt"
    doc_en.write_text(
        "Clean drinking water protects children from illness. "
        "Every clinic in the district received new solar refrigerators this month.",
        encoding="utf-8",
    )
    doc_yo.write_text(
        "Omi mímu tó mọ́ ń dáàbò bo àwọn ọmọdé lọ́wọ́ àìsàn. "
        "Gbogbo ilé ìwòsàn ní agbègbè náà gba fíríìjì oòrùn tuntun ní oṣù yìí.",
        encoding="utf-8",
    )
    rec_doc = store.import_corpus(
        fmt="unaligned",
        files=[doc_en, doc_yo],
        src_lang="english",
        tgt_lang="yoruba",
        source_name="Health Bulletin",
        license="CC0-1.0",
        domain="medical",
    )
    assert rec_doc.accepted_pairs == 2


def test_training_splits_domain_cap_license_gate_and_zero_leakage(tmp_path):
    store = open_corpus_store(path=tmp_path / "corpora", use_lid=False)

    # 30 religious pairs (research-only vs public-domain) + 10 general pairs
    rel_en = tmp_path / "rel_en.txt"
    rel_yo = tmp_path / "rel_yo.txt"
    rel_en.write_text(
        "\n".join(f"Blessed are the peacemakers in chapter {i} of the assembly." for i in range(30)),
        encoding="utf-8",
    )
    rel_yo.write_text(
        "\n".join(f"Alábùkún ni fún àwọn onílàjà ní orí {i} ti ìjọ." for i in range(30)),
        encoding="utf-8",
    )
    store.import_corpus(
        fmt="lines",
        files=[rel_en, rel_yo],
        src_lang="en",
        tgt_lang="yo",
        source_name="Public Domain Scripture",
        license="public-domain",
        domain="religious",
    )

    gen_en = tmp_path / "gen_en.txt"
    gen_yo = tmp_path / "gen_yo.txt"
    gen_en.write_text(
        "\n".join(f"The morning bus number {i} leaves the market station at eight." for i in range(10)),
        encoding="utf-8",
    )
    gen_yo.write_text(
        "\n".join(f"Ọkọ̀ àárọ̀ nọ́ńbà {i} ń kúrò ní ibùdókọ̀ ọjà ní agogo mẹ́jọ." for i in range(10)),
        encoding="utf-8",
    )
    store.import_corpus(
        fmt="lines",
        files=[gen_en, gen_yo],
        src_lang="en",
        tgt_lang="yo",
        source_name="Research Dialogue Corpus",
        license="research-only",
        domain="general",
    )

    # 1. Domain cap (religious capped to <=40% -> 10 general + ~7 religious = 17 base pairs, x2 bidirectional = 34)
    out1 = tmp_path / "splits_all"
    s1 = build_training_splits(out1, corpora=store, max_domain_share=0.40, val_ratio=0.1, test_ratio=0.1)
    assert "religious" in s1.capped_domains
    assert s1.by_domain["religious"] <= s1.by_domain["general"]

    # Verify zero train/test group leakage across forward and reverse directions
    import json

    def group_keys(split_file):
        rows = [json.loads(ln) for ln in (out1 / split_file).read_text(encoding="utf-8").splitlines() if ln]
        return {tuple(sorted((tm_key(r["source"]), tm_key(r["target"])))) for r in rows}

    assert group_keys("train.jsonl").isdisjoint(group_keys("test.jsonl"))
    assert group_keys("train.jsonl").isdisjoint(group_keys("val.jsonl"))

    # 2. commercial_safe=True drops the research-only corpus completely
    out2 = tmp_path / "splits_commercial"
    s2 = build_training_splits(out2, corpora=store, commercial_safe=True)
    assert s2.licenses == ["public-domain"]

    # 3. FLORES contamination guard
    flores_en = tmp_path / "f_en.txt"
    flores_yo = tmp_path / "f_yo.txt"
    flores_en.write_text("This is a benchmark test sentence today.\n", encoding="utf-8")
    flores_yo.write_text("Èyí jẹ́ gbolóhùn ìdánwò òní.\n", encoding="utf-8")
    store.import_corpus(
        fmt="lines",
        files=[flores_en, flores_yo],
        src_lang="en",
        tgt_lang="yo",
        source_name="FLORES-200 devtest",
        license="CC-BY-SA-4.0",
        domain="general",
    )
    with pytest.raises(InvalidInputError, match="FLORES"):
        build_training_splits(tmp_path / "splits_bad", corpora=store)

    # 4. Dry-run MT LoRA smoke check on generated splits
    dry = run_mt_lora(MTLoRAConfig(data_dir=out1, dry_run=True))
    assert dry["dry_run"] is True and dry["train_rows"] == s1.train


def test_evaluate_metrics_gate_and_card():
    scores = chrf_score(["Ẹ káàárọ̀ gbogbo yín"], ["Ẹ káàárọ̀ gbogbo yín"])
    assert scores["chrf"] > 99.0
    errs = wer_cer(["bawo ni o"], ["bawo ni"])
    assert errs["wer"] == 0.5 and errs["cer"] > 0.0
    assert should_promote(48.5, 47.0, min_improvement=0.5).promote is True
    assert should_promote(47.2, 47.0, min_improvement=0.5).promote is False
    card = render_model_card(
        model_name="AfriLingua-NLLB-Yoruba",
        base_model="facebook/nllb-200-distilled-600M",
        task="translation",
        languages=["en", "yo"],
        metrics={"chrf": 48.5},
        data_licenses=["CC-BY-4.0", "public-domain"],
        sources=["Yoruba Bibeli Mimo", "community-contributions"],
        contributors=["Ada", "Tunde"],
    )
    assert "Ada, Tunde" in card and "CC-BY-NC-4.0" in card
