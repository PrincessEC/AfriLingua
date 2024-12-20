"""
data/collect_data.py
Downloads open parallel corpora for English ↔ Igbo / Yoruba / Hausa from
HuggingFace datasets (FLORES-200, OPUS-100, Helsinki-NLP/tatoeba) and writes
clean JSONL files into data/processed/.

Usage:
    python data/collect_data.py

Each output line is a JSON object:
    {"english": "...", "translation": "...", "lang": "igbo"}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running as a script directly from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, Dataset, concatenate_datasets
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# NLLB / FLORES language codes used in HuggingFace datasets
_FLORES_CODES = {
    "igbo":   "ibo_Latn",
    "yoruba": "yor_Latn",
    "hausa":  "hau_Latn",
}

# Helsinki-NLP OPUS-100 language codes
_OPUS_CODES = {
    "igbo":   "ig",
    "yoruba": "yo",
    "hausa":  "ha",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(records), path)


def _clean(text: str) -> str:
    return text.strip()


# ── Source 1: FLORES-200 (official Meta evaluation set) ──────────────────────

def collect_flores(lang_name: str, flores_code: str) -> list[dict]:
    """Return English↔lang pairs from the FLORES-200 dev/devtest splits."""
    records: list[dict] = []
    for split in ("dev", "devtest"):
        try:
            logger.info("FLORES-200 %s  split=%s …", lang_name, split)
            eng_ds = load_dataset("facebook/flores", "eng_Latn", split=split, trust_remote_code=True)
            tgt_ds = load_dataset("facebook/flores", flores_code,  split=split, trust_remote_code=True)
            for eng_row, tgt_row in zip(eng_ds, tgt_ds):
                eng_text = _clean(eng_row["sentence"])
                tgt_text = _clean(tgt_row["sentence"])
                if eng_text and tgt_text:
                    records.append({"english": eng_text, "translation": tgt_text, "lang": lang_name})
        except Exception as exc:
            logger.warning("FLORES-200 %s/%s failed: %s", lang_name, split, exc)
    return records


# ── Source 2: Helsinki-NLP OPUS-100 ──────────────────────────────────────────

def collect_opus100(lang_name: str, opus_code: str) -> list[dict]:
    """Return English↔lang pairs from OPUS-100 (train split, up to 50k)."""
    records: list[dict] = []
    try:
        logger.info("OPUS-100 en-%s …", opus_code)
        ds = load_dataset(
            "Helsinki-NLP/opus-100",
            f"en-{opus_code}",
            split="train",
            trust_remote_code=True,
        )
        for row in ds:
            pair = row.get("translation", {})
            eng_text = _clean(pair.get("en", ""))
            tgt_text = _clean(pair.get(opus_code, ""))
            if eng_text and tgt_text:
                records.append({"english": eng_text, "translation": tgt_text, "lang": lang_name})
            if len(records) >= 50_000:
                break
    except Exception as exc:
        logger.warning("OPUS-100 en-%s failed: %s", opus_code, exc)
    return records


# ── Source 3: Helsinki-NLP Tatoeba ────────────────────────────────────────────

def collect_tatoeba(lang_name: str, opus_code: str) -> list[dict]:
    """Return English↔lang pairs from the Tatoeba challenge set."""
    records: list[dict] = []
    try:
        logger.info("Tatoeba en-%s …", opus_code)
        ds = load_dataset(
            "Helsinki-NLP/tatoeba_mt",
            f"eng-{opus_code}",
            split="test",
            trust_remote_code=True,
        )
        for row in ds:
            eng_text = _clean(row.get("sourceString", "") or row.get("src", ""))
            tgt_text = _clean(row.get("targetString", "") or row.get("tgt", ""))
            if eng_text and tgt_text:
                records.append({"english": eng_text, "translation": tgt_text, "lang": lang_name})
    except Exception as exc:
        logger.warning("Tatoeba en-%s failed: %s", opus_code, exc)
    return records


# ── Main collection loop ──────────────────────────────────────────────────────

def collect_all() -> None:
    for lang_name, flores_code in _FLORES_CODES.items():
        opus_code = _OPUS_CODES[lang_name]
        all_records: list[dict] = []

        all_records.extend(collect_flores(lang_name, flores_code))
        all_records.extend(collect_opus100(lang_name, opus_code))
        all_records.extend(collect_tatoeba(lang_name, opus_code))

        # Add reverse direction (native → English)
        reverse = [
            {"english": r["translation"], "translation": r["english"], "lang": "english_from_" + lang_name}
            for r in all_records
        ]
        all_records.extend(reverse)

        out_path = DATA_PROCESSED_DIR / f"en_{lang_name}_pairs.jsonl"
        _write_jsonl(all_records, out_path)
        logger.info("Total pairs for %s: %d", lang_name, len(all_records) // 2)

    logger.info("Data collection complete.")


if __name__ == "__main__":
    collect_all()
