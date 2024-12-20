"""
data/preprocess.py
Reads the raw JSONL files produced by collect_data.py, deduplicates, filters
noisy pairs, tokenises with the NLLB tokenizer, and writes train/val/test
splits ready for fine-tuning.

Output layout (in data/processed/):
    train.jsonl   — 90%
    val.jsonl     — 5%
    test.jsonl    — 5%

Each record:
    {"src_lang": "eng_Latn", "tgt_lang": "ibo_Latn",
     "english": "...", "translation": "..."}

Usage:
    python data/preprocess.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src.config import DATA_PROCESSED_DIR, HF_CACHE, MODEL_NLLB
from src.translation.engine import LANG_CODES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MIN_TOKENS = 3
MAX_TOKENS = 200

_LANG_TO_NLLB = {
    "igbo":   LANG_CODES["igbo"],
    "yoruba": LANG_CODES["yoruba"],
    "hausa":  LANG_CODES["hausa"],
    # reverse direction entries produced by collect_data
    "english_from_igbo":   LANG_CODES["igbo"],
    "english_from_yoruba": LANG_CODES["yoruba"],
    "english_from_hausa":  LANG_CODES["hausa"],
}


def _is_reverse(lang: str) -> bool:
    return lang.startswith("english_from_")


def load_all_pairs() -> list[dict]:
    pairs: list[dict] = []
    for jsonl_file in DATA_PROCESSED_DIR.glob("en_*_pairs.jsonl"):
        logger.info("Reading %s …", jsonl_file.name)
        with jsonl_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for p in pairs:
        key = (p["english"], p["translation"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    logger.info("After dedup: %d pairs", len(out))
    return out


def filter_pairs(pairs: list[dict], tokenizer: AutoTokenizer) -> list[dict]:
    out: list[dict] = []
    for p in pairs:
        eng_ids  = tokenizer(p["english"],     add_special_tokens=False).input_ids
        tgt_ids  = tokenizer(p["translation"], add_special_tokens=False).input_ids
        if (
            MIN_TOKENS <= len(eng_ids) <= MAX_TOKENS
            and MIN_TOKENS <= len(tgt_ids) <= MAX_TOKENS
        ):
            out.append(p)
    logger.info("After length filter: %d pairs", len(out))
    return out


def enrich_with_nllb_codes(pairs: list[dict]) -> list[dict]:
    """Add src_lang / tgt_lang NLLB codes to every record."""
    out: list[dict] = []
    for p in pairs:
        lang = p["lang"]
        if lang not in _LANG_TO_NLLB:
            continue
        native_code = _LANG_TO_NLLB[lang]
        if _is_reverse(lang):
            p["src_lang"] = native_code
            p["tgt_lang"] = LANG_CODES["english"]
        else:
            p["src_lang"] = LANG_CODES["english"]
            p["tgt_lang"] = native_code
        out.append(p)
    return out


def split_and_write(pairs: list[dict]) -> None:
    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_val  = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))

    test_set  = pairs[:n_test]
    val_set   = pairs[n_test: n_test + n_val]
    train_set = pairs[n_test + n_val:]

    for name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        out_path = DATA_PROCESSED_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in subset:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %d records → %s", len(subset), out_path)


def main() -> None:
    logger.info("Loading NLLB tokenizer for filtering …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NLLB, cache_dir=str(HF_CACHE))

    pairs = load_all_pairs()
    if not pairs:
        logger.warning("No JSONL files found in %s — run collect_data.py first.", DATA_PROCESSED_DIR)
        return

    pairs = deduplicate(pairs)
    pairs = filter_pairs(pairs, tokenizer)
    pairs = enrich_with_nllb_codes(pairs)
    split_and_write(pairs)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
