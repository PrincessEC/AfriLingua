"""Build balanced, leak-free train/val/test splits from community & imported corpora.

Key guarantees:

1. **FLORES-200 exclusion**: ``flores`` / ``flores200`` / ``flores-200`` sources are
   rejected from training so standard evaluation benchmarks remain uncontaminated.
2. **License gate**: ``commercial_safe=True`` excludes ``research-only`` and copyleft
   (``CC-BY-SA-4.0``) corpora, keeping only ``public-domain``, ``CC0-1.0``,
   ``CC-BY-4.0`` and ``proprietary-permission``.
3. **Domain cap**: no single domain (e.g. ``religious`` from a full 31k-verse Bible)
   exceeds ``max_domain_share`` (default 0.40) when non-dominant domain data exists,
   preventing archaic or domain-specific style from dominating general speech.
4. **Group-aware split**: each pair's canonical group key is derived from the sorted
   normalised text pair ``sorted((tm_key(src), tm_key(tgt)))``, so forward and reverse
   directions (``en→yo`` and ``yo→en``) always land in the *same* split (zero leakage).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from afrilingua.core.errors import InvalidInputError
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import get_settings
from afrilingua.engine import normalize_text
from afrilingua.tm.text import tm_key
from afrilingua.training.corpora import COMMERCIAL_SAFE_LICENSES

FORBIDDEN_EVAL_SOURCES = ("flores", "flores-200", "flores200", "fleurs-test")


@dataclass
class SplitSummary:
    out_dir: str
    train: int = 0
    val: int = 0
    test: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, int] = field(default_factory=dict)
    licenses: list[str] = field(default_factory=list)
    contributors: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    capped_domains: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_forbidden_eval_source(name: str) -> bool:
    low = (name or "").lower()
    return any(tok in low for tok in FORBIDDEN_EVAL_SOURCES)


def _group_bucket(src: str, tgt: str) -> float:
    a, b = sorted((tm_key(src), tm_key(tgt)))
    digest = hashlib.blake2b(f"{a}\0{b}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _enforce_domain_cap(
    records: list[dict[str, Any]], max_domain_share: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Down-sample any domain exceeding ``max_domain_share`` when other domains exist."""
    if not records or not (0.0 < max_domain_share < 1.0):
        return records, {}
    by_dom: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_dom.setdefault(rec.get("domain") or "general", []).append(rec)
    if len(by_dom) <= 1:
        return records, {}

    rng = random.Random(seed)
    total = len(records)
    capped_counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for dom, items in by_dom.items():
        other = total - len(items)
        if other > 0 and (len(items) / total) > max_domain_share:
            # Solve cap / (cap + other) = max_domain_share  =>  cap = other * s / (1 - s)
            cap = max(1, round(other * max_domain_share / (1.0 - max_domain_share)))
            if cap < len(items):
                shuffled = list(items)
                rng.shuffle(shuffled)
                capped_counts[dom] = len(items) - cap
                kept.extend(shuffled[:cap])
                continue
        kept.extend(items)
    return kept, capped_counts


def build_training_splits(
    out_dir: str | Path | None = None,
    *,
    contrib: Any | None = None,
    corpora: Any | None = None,
    tm: Any | None = None,
    lang: str | None = None,
    commercial_safe: bool = False,
    include_tm: bool = False,
    bidirectional: bool = True,
    max_domain_share: float = 0.40,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> SplitSummary:
    """Collect vetted parallel pairs, apply license & domain rules, and write JSONL splits."""
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 0.5:
        raise InvalidInputError("val_ratio + test_ratio must be in [0, 0.5).")

    settings = get_settings()
    reg = get_registry()
    target_filter = reg.normalize(lang) if lang else None
    dest = Path(out_dir) if out_dir is not None else settings.data_dir / "processed"
    dest.mkdir(parents=True, exist_ok=True)

    raw_items: list[dict[str, Any]] = []
    contributors: set[str] = set()
    sources: set[str] = set()
    licenses: set[str] = set()

    if contrib is not None:
        for item in contrib.iter_approved(lang=target_filter):
            if item.get("kind") == "transcription":
                continue
            lic = item.get("license") or "CC-BY-4.0"
            if commercial_safe and lic not in COMMERCIAL_SAFE_LICENSES:
                continue
            raw_items.append(
                {
                    "src_lang": item["src_lang"],
                    "tgt_lang": item["tgt_lang"],
                    "source": item["source_text"],
                    "target": item["target_text"],
                    "domain": item.get("domain") or "general",
                    "license": lic,
                    "origin": "community",
                }
            )
            licenses.add(lic)
            sources.add("community-contributions")
            who = item.get("contributor")
            if who and who != "anonymous":
                contributors.add(who)

    if corpora is not None:
        for rec in corpora.iter_pairs(lang=target_filter, commercial_safe_only=commercial_safe):
            src_name = rec.get("source_name") or ""
            if _is_forbidden_eval_source(src_name):
                raise InvalidInputError(
                    f"Corpus {src_name!r} looks like an evaluation benchmark (FLORES/FLEURS) and cannot be used for training."
                )
            lic = rec.get("license") or "research-only"
            raw_items.append(
                {
                    "src_lang": rec["src_lang"],
                    "tgt_lang": rec["tgt_lang"],
                    "source": rec["source"],
                    "target": rec["target"],
                    "domain": rec.get("domain") or "general",
                    "license": lic,
                    "origin": f"corpus:{rec.get('corpus_id', '')}",
                }
            )
            licenses.add(lic)
            if src_name:
                sources.add(src_name)

    if include_tm and tm is not None and hasattr(tm, "iter_entries"):
        for entry in tm.iter_entries(approved_only=True):
            if target_filter and target_filter not in {entry["src_lang"], entry["tgt_lang"]}:
                continue
            if str(entry.get("origin", "")).startswith("corpus:") or entry.get("origin") == "community":
                continue
            if commercial_safe:
                continue  # ad-hoc TM entries carry no explicit license
            raw_items.append(
                {
                    "src_lang": entry["src_lang"],
                    "tgt_lang": entry["tgt_lang"],
                    "source": entry["source"],
                    "target": entry["target"],
                    "domain": "general",
                    "license": "unspecified",
                    "origin": "tm",
                }
            )
            sources.add("translation-memory")

    # Deduplicate canonical (src_lang, tgt_lang, src_key, tgt_key)
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in raw_items:
        s_txt = normalize_text(item["source"])
        t_txt = normalize_text(item["target"])
        s_lang = reg.normalize(item["src_lang"])
        t_lang = reg.normalize(item["tgt_lang"])
        if not s_txt or not t_txt or s_lang == t_lang:
            continue
        k = (s_lang, t_lang, tm_key(s_txt), tm_key(t_txt))
        if not k[2] or not k[3] or k in seen:
            continue
        seen.add(k)
        deduped.append({**item, "src_lang": s_lang, "tgt_lang": t_lang, "source": s_txt, "target": t_txt})

    balanced, capped = _enforce_domain_cap(deduped, max_domain_share=max_domain_share, seed=seed)

    # Expand reverse direction if requested, keeping both directions in the SAME group split!
    expanded: list[dict[str, Any]] = []
    expanded_seen: set[tuple[str, str, str, str]] = set()
    for item in balanced:
        candidates = [item]
        if bidirectional:
            candidates.append(
                {
                    **item,
                    "src_lang": item["tgt_lang"],
                    "tgt_lang": item["src_lang"],
                    "source": item["target"],
                    "target": item["source"],
                }
            )
        for cand in candidates:
            ck = (cand["src_lang"], cand["tgt_lang"], tm_key(cand["source"]), tm_key(cand["target"]))
            if ck in expanded_seen:
                continue
            expanded_seen.add(ck)
            expanded.append(
                {
                    **cand,
                    "src_code": reg.nllb_code(cand["src_lang"]),
                    "tgt_code": reg.nllb_code(cand["tgt_lang"]),
                }
            )

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    # Group by canonical text pair so forward+reverse never straddle splits
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in expanded:
        gk = tuple(sorted((tm_key(row["source"]), tm_key(row["target"]))))
        groups.setdefault((gk[0], gk[1]), []).append(row)

    ordered_groups = sorted(groups.items(), key=lambda kv: _group_bucket(kv[0][0], kv[0][1]))
    n_groups = len(ordered_groups)
    n_test = round(n_groups * test_ratio) if n_groups >= 10 and test_ratio > 0 else 0
    n_val = round(n_groups * val_ratio) if n_groups >= 10 and val_ratio > 0 else 0

    for idx, (_gk, rows) in enumerate(ordered_groups):
        if idx < n_test:
            test_rows.extend(rows)
        elif idx < n_test + n_val:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    for split_name, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        p = dest / f"{split_name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_dom: dict[str, int] = {}
    by_dir: dict[str, int] = {}
    for row in expanded:
        dom = row.get("domain") or "general"
        by_dom[dom] = by_dom.get(dom, 0) + 1
        direction = f"{row['src_lang']}→{row['tgt_lang']}"
        by_dir[direction] = by_dir.get(direction, 0) + 1

    summary = SplitSummary(
        out_dir=str(dest),
        train=len(train_rows),
        val=len(val_rows),
        test=len(test_rows),
        by_domain=by_dom,
        by_direction=by_dir,
        licenses=sorted(licenses),
        contributors=sorted(contributors),
        sources=sorted(sources),
        capped_domains=capped,
    )
    (dest / "manifest.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
