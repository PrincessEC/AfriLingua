"""Bulk parallel-corpus ingestion, alignment, filtering and provenance tracking.

Supported parallel formats:

* ``ebible`` — eBible.org verse-aligned files (``src`` + ``tgt`` line-per-verse files,
  with optional ``vref.txt`` reference index; skips ``<range>`` / blank verses).
* ``usfm`` — Unified Standard Format Markers Bible files (``\\id``, ``\\c``, ``\\v``),
  aligned by exact ``BOOK.CHAPTER.VERSE`` reference, skipping mismatched verse spans.
* ``srt`` / ``vtt`` — SubRip and WebVTT subtitle pairs aligned by timestamp IoU ≥ 0.5,
  considering 1:1, 1:2 and 2:1 cue merges and stripping markup / speaker labels.
* ``lines`` — Equal-length line-aligned plain-text pair.
* ``tsv`` / ``csv`` / ``jsonl`` / ``tmx`` — Tabular / CAT parallel formats.
* ``unaligned`` — Sentence-splits both documents and runs a Gale–Church length aligner,
  keeping pairs above ``min_align_score``.

Every imported corpus requires explicit ``source_name``, ``license`` and ``domain``
metadata so downstream training jobs can enforce ``--commercial-safe`` filtering and
cap any single domain's share (e.g. preventing scripture style from dominating speech).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afrilingua.core.errors import InvalidInputError, NotFoundError, UnsupportedLanguageError
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import get_settings
from afrilingua.engine import normalize_text
from afrilingua.tm.io import iter_pairs as iter_tabular_tm_pairs
from afrilingua.tm.text import tm_key

logger = logging.getLogger(__name__)

CORPUS_LICENSES: frozenset[str] = frozenset(
    {
        "public-domain",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "research-only",
        "proprietary-permission",
    }
)
COMMERCIAL_SAFE_LICENSES: frozenset[str] = frozenset(
    {"public-domain", "CC0-1.0", "CC-BY-4.0", "proprietary-permission"}
)
CORPUS_DOMAINS: frozenset[str] = frozenset(
    {"general", "religious", "media", "medical", "legal", "education", "government"}
)
CORPUS_FORMATS: frozenset[str] = frozenset(
    {"ebible", "usfm", "srt", "vtt", "lines", "tsv", "csv", "jsonl", "tmx", "unaligned"}
)

_URL_EMAIL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b)", re.IGNORECASE)
_USFM_ID_RE = re.compile(r"\\id\s+([A-Z0-9]{3})\b")
_USFM_NOTE_RE = re.compile(r"\\(?:f|fe|x)\b.*?\\(?:f|fe|x)\*", re.DOTALL)
_USFM_ATTR_RE = re.compile(r"\\(\+?[a-z0-9]+)\s+([^\\|]+?)\|[^\\]*?\\\1\*")
_USFM_TAG_RE = re.compile(r"\\\+?[a-z0-9]+\*?")
_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]+\}")
_BRACKET_NOISE_RE = re.compile(r"^[\[\(][^\]\)]{1,60}[\]\)]\s*$")
_SPEAKER_DASH_RE = re.compile(r"^\s*[-–—]\s*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[^\s])")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


# ── 1. Format-specific parallel extractors ──────────────────────────────────


def iter_ebible_pairs(
    src_path: Path, tgt_path: Path, vref_path: Path | None = None
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(src_verse, tgt_verse, ref)`` from eBible.org line-per-verse files."""
    src_lines = _read_text(src_path).splitlines()
    tgt_lines = _read_text(tgt_path).splitlines()
    refs = _read_text(vref_path).splitlines() if vref_path is not None else None

    if len(src_lines) != len(tgt_lines):
        raise InvalidInputError(
            f"eBible files must have equal line counts (got {len(src_lines)} vs {len(tgt_lines)})."
        )
    if refs is not None and len(refs) != len(src_lines):
        raise InvalidInputError(
            f"vref file line count ({len(refs)}) does not match text files ({len(src_lines)})."
        )

    skip_markers = {"<range>", "...", "—", "-"}
    for idx, (s, t) in enumerate(zip(src_lines, tgt_lines, strict=True)):
        s_clean = s.strip()
        t_clean = t.strip()
        if not s_clean or not t_clean:
            continue
        if s_clean.lower() in skip_markers or t_clean.lower() in skip_markers:
            continue
        ref = refs[idx].strip() if refs is not None else f"line:{idx + 1}"
        yield s_clean, t_clean, ref


def _clean_usfm_text(raw: str) -> str:
    text = _USFM_NOTE_RE.sub(" ", raw)
    text = _USFM_ATTR_RE.sub(r"\2", text)
    text = _USFM_TAG_RE.sub(" ", text)
    return normalize_text(text)


def parse_usfm_verses(paths: Iterable[Path]) -> dict[str, str]:
    """Parse one or more USFM files into ``{"GEN.1.1": "In the beginning ..."}``.

    Verse ranges such as ``\\v 1-2`` are skipped so misaligned verse numbering
    across translations never produces mismatched pairs.
    """
    verses: dict[str, str] = {}
    for path in paths:
        raw = _read_text(path)
        m_id = _USFM_ID_RE.search(raw)
        book = m_id.group(1).upper() if m_id else path.stem.upper()[:6]
        chapter = "0"
        current_ref: str | None = None
        buf: list[str] = []

        def flush() -> None:
            nonlocal current_ref, buf
            if current_ref is not None and buf:
                cleaned = _clean_usfm_text(" ".join(buf))
                if cleaned:
                    verses[current_ref] = cleaned
            current_ref = None
            buf = []

        for token in re.split(r"(?=\\(?:c|v)\s)", raw):
            if token.startswith("\\c "):
                flush()
                m_c = re.match(r"\\c\s+(\d+)", token)
                if m_c:
                    chapter = m_c.group(1)
            elif token.startswith("\\v "):
                flush()
                m_v = re.match(r"\\v\s+(\S+)\s*(.*)", token, re.DOTALL)
                if not m_v:
                    continue
                vnum, rest = m_v.group(1), m_v.group(2)
                if not vnum.isdigit():
                    # Skip merged verse ranges like "1-2" or "3a"
                    continue
                current_ref = f"{book}.{chapter}.{vnum}"
                buf.append(rest)
            elif current_ref is not None:
                buf.append(token)
        flush()
    return verses


def iter_usfm_pairs(src_paths: Iterable[Path], tgt_paths: Iterable[Path]) -> Iterator[tuple[str, str, str]]:
    """Align two USFM collections by exact ``BOOK.CHAPTER.VERSE`` reference."""
    src_map = parse_usfm_verses(src_paths)
    tgt_map = parse_usfm_verses(tgt_paths)
    if not src_map or not tgt_map:
        raise InvalidInputError("No USFM verses found; check \\id, \\c, and \\v markers.")
    for ref, s_text in src_map.items():
        t_text = tgt_map.get(ref)
        if t_text:
            yield s_text, t_text, ref


@dataclass(frozen=True)
class SubtitleCue:
    start_s: float
    end_s: float
    text: str


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    frac = int(ms.ljust(3, "0")[:3]) / 1000.0
    return int(h) * 3600 + int(m) * 60 + int(s) + frac


def parse_subtitles(path: Path) -> list[SubtitleCue]:
    """Parse SRT or WebVTT subtitles into cleaned, timestamped cues."""
    raw = _read_text(path)
    blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        ts_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), -1)
        if ts_idx < 0:
            continue
        m = _SRT_TIME_RE.search(lines[ts_idx])
        if not m:
            continue
        start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        if end <= start:
            continue
        cleaned_lines: list[str] = []
        for ln in lines[ts_idx + 1 :]:
            ln = _HTML_TAG_RE.sub("", ln).strip()
            if not ln or _BRACKET_NOISE_RE.match(ln):
                continue
            ln = _SPEAKER_DASH_RE.sub("", ln).strip()
            if ln:
                cleaned_lines.append(ln)
        text = normalize_text(" ".join(cleaned_lines))
        if text:
            cues.append(SubtitleCue(start, end, text))
    return cues


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def iter_subtitle_pairs(
    src_path: Path, tgt_path: Path, *, min_iou: float = 0.5
) -> Iterator[tuple[str, str, str]]:
    """Align SRT/VTT cues by time overlap (IoU ≥ ``min_iou``), supporting 1:1, 1:2 and 2:1 merges."""
    src_cues = parse_subtitles(src_path)
    tgt_cues = parse_subtitles(tgt_path)
    if not src_cues or not tgt_cues:
        raise InvalidInputError("Subtitle file contained no valid timed cues.")

    i = 0
    j = 0
    n_src = len(src_cues)
    n_tgt = len(tgt_cues)
    while i < n_src and j < n_tgt:
        s1 = src_cues[i]
        t1 = tgt_cues[j]
        if s1.end_s <= t1.start_s:
            i += 1
            continue
        if t1.end_s <= s1.start_s:
            j += 1
            continue

        candidates: list[tuple[float, int, int, str, str]] = [
            (_iou(s1.start_s, s1.end_s, t1.start_s, t1.end_s), 1, 1, s1.text, t1.text)
        ]
        if j + 1 < n_tgt:
            t2 = tgt_cues[j + 1]
            if t2.start_s - t1.end_s <= 1.2:
                candidates.append(
                    (
                        _iou(s1.start_s, s1.end_s, t1.start_s, t2.end_s),
                        1,
                        2,
                        s1.text,
                        f"{t1.text} {t2.text}",
                    )
                )
        if i + 1 < n_src:
            s2 = src_cues[i + 1]
            if s2.start_s - s1.end_s <= 1.2:
                candidates.append(
                    (
                        _iou(s1.start_s, s2.end_s, t1.start_s, t1.end_s),
                        2,
                        1,
                        f"{s1.text} {s2.text}",
                        t1.text,
                    )
                )
        best_iou, di, dj, s_txt, t_txt = max(candidates, key=lambda c: c[0])
        if best_iou >= min_iou:
            yield s_txt, t_txt, f"t={s1.start_s:.2f}s"
            i += di
            j += dj
        elif s1.end_s < t1.end_s:
            i += 1
        else:
            j += 1


def iter_line_pairs(src_path: Path, tgt_path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield pairs from two line-aligned text files."""
    src_lines = _read_text(src_path).splitlines()
    tgt_lines = _read_text(tgt_path).splitlines()
    if len(src_lines) != len(tgt_lines):
        raise InvalidInputError(
            f"Line-aligned files must have equal line counts (got {len(src_lines)} vs {len(tgt_lines)})."
        )
    for idx, (s, t) in enumerate(zip(src_lines, tgt_lines, strict=True), start=1):
        if s.strip() and t.strip():
            yield s.strip(), t.strip(), f"line:{idx}"


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\r?\n\s*\r?\n", text):
        para_clean = normalize_text(para)
        if not para_clean:
            continue
        for sent in _SENT_SPLIT_RE.split(para_clean):
            s = sent.strip()
            if s:
                out.append(s)
    return out


def _gc_match_cost(len_s: int, len_t: int, c: float = 1.0, s2: float = 6.8) -> float:
    if len_s == 0 and len_t == 0:
        return 0.0
    mean = (len_s + len_t / c) / 2.0
    if mean <= 0:
        return 100.0
    delta = abs(len_s - len_t / c) / math.sqrt(mean * s2)
    tail = math.erfc(delta / math.sqrt(2.0))
    return -math.log(max(tail, 1e-12))


def iter_unaligned_pairs(
    src_path: Path,
    tgt_path: Path,
    *,
    min_score: float = 0.55,
    window: int = 12,
) -> Iterator[tuple[str, str, str]]:
    """Sentence-split two unaligned documents and align via banded Gale–Church DP."""
    src_sents = _split_sentences(_read_text(src_path))
    tgt_sents = _split_sentences(_read_text(tgt_path))
    n, m = len(src_sents), len(tgt_sents)
    if not n or not m:
        raise InvalidInputError("Document contained no extractable sentences.")

    total_s = sum(len(s) for s in src_sents)
    total_t = sum(len(t) for t in tgt_sents)
    ratio = total_t / max(total_s, 1)

    beads: tuple[tuple[int, int, float], ...] = (
        (1, 1, 0.0),
        (1, 0, 4.5),
        (0, 1, 4.5),
        (2, 1, 2.3),
        (1, 2, 2.3),
    )
    dp: dict[tuple[int, int], tuple[float, int, int]] = {(0, 0): (0.0, 0, 0)}

    for i in range(n + 1):
        center = round(i * m / max(n, 1))
        j_lo = max(0, center - window)
        j_hi = min(m, center + window)
        if i == n:
            j_hi = m
        for j in range(j_lo, j_hi + 1):
            cur = dp.get((i, j))
            if cur is None:
                continue
            base_cost = cur[0]
            for di, dj, penalty in beads:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                ncenter = round(ni * m / max(n, 1))
                if ni < n and abs(nj - ncenter) > window + 2:
                    continue
                ls = sum(len(src_sents[k]) for k in range(i, ni))
                lt = sum(len(tgt_sents[k]) for k in range(j, nj))
                step_cost = penalty + (_gc_match_cost(ls, lt, c=ratio) if (di and dj) else 3.0)
                cand = base_cost + step_cost
                prev = dp.get((ni, nj))
                if prev is None or cand < prev[0]:
                    dp[(ni, nj)] = (cand, di, dj)

    if (n, m) not in dp:
        raise InvalidInputError("Documents differ too drastically in length to align safely.")

    aligned: list[tuple[str, str, float, int]] = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        cost_here, di, dj = dp[(ci, cj)]
        pi, pj = ci - di, cj - dj
        prev_cost = dp[(pi, pj)][0]
        if di > 0 and dj > 0:
            s_txt = " ".join(src_sents[pi:ci])
            t_txt = " ".join(tgt_sents[pj:cj])
            step = cost_here - prev_cost
            score = math.exp(-max(0.0, step) / 4.0)
            aligned.append((s_txt, t_txt, score, pi + 1))
        ci, cj = pi, pj

    for s_txt, t_txt, score, idx in reversed(aligned):
        if score >= min_score:
            yield s_txt, t_txt, f"sent:{idx}"


# ── 2. Quality filtering ─────────────────────────────────────────────────────


@dataclass
class FilterStats:
    raw_pairs: int = 0
    accepted: int = 0
    dropped_empty: int = 0
    dropped_length_tokens: int = 0
    dropped_length_ratio: int = 0
    dropped_identical: int = 0
    dropped_junk_or_url: int = 0
    dropped_duplicate: int = 0
    dropped_wrong_language: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _is_junk(text: str) -> bool:
    if _URL_EMAIL_RE.search(text):
        return True
    letters = sum(ch.isalpha() for ch in text)
    non_space = sum(not ch.isspace() for ch in text)
    if non_space == 0:
        return True
    return (letters / non_space) < 0.40


def filter_parallel_pairs(
    raw_pairs: Iterable[tuple[str, str, str]],
    src_lang: str,
    tgt_lang: str,
    *,
    min_tokens: int = 2,
    max_tokens: int = 200,
    min_ratio: float = 0.20,
    max_ratio: float = 4.8,
    detector: Callable[[str], tuple[str, float]] | None = None,
) -> tuple[list[tuple[str, str, str]], FilterStats]:
    """Clean, deduplicate and quality-filter ``(source, target, ref)`` pairs."""
    stats = FilterStats()
    seen: set[tuple[str, str]] = set()
    accepted: list[tuple[str, str, str]] = []

    for raw_src, raw_tgt, ref in raw_pairs:
        stats.raw_pairs += 1
        src = normalize_text(raw_src)
        tgt = normalize_text(raw_tgt)
        if not src or not tgt:
            stats.dropped_empty += 1
            continue
        src_toks = len(src.split())
        tgt_toks = len(tgt.split())
        if not (min_tokens <= src_toks <= max_tokens and min_tokens <= tgt_toks <= max_tokens):
            stats.dropped_length_tokens += 1
            continue
        ratio = len(tgt) / max(len(src), 1)
        if not (min_ratio <= ratio <= max_ratio):
            stats.dropped_length_ratio += 1
            continue
        if len(src) > 4 and src.casefold() == tgt.casefold():
            stats.dropped_identical += 1
            continue
        if _is_junk(src) or _is_junk(tgt):
            stats.dropped_junk_or_url += 1
            continue
        key = (tm_key(src), tm_key(tgt))
        if not key[0] or not key[1]:
            stats.dropped_empty += 1
            continue
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        if detector is not None and tgt_toks >= 5:
            pred_lang, conf = detector(tgt)
            if pred_lang not in {"unknown", tgt_lang} and conf >= 0.85:
                stats.dropped_wrong_language += 1
                continue
        seen.add(key)
        accepted.append((src, tgt, ref))

    stats.accepted = len(accepted)
    return accepted, stats


# ── 3. Persistent CorpusStore ────────────────────────────────────────────────

_CORPUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    id              TEXT PRIMARY KEY,
    source_name     TEXT NOT NULL,
    fmt             TEXT NOT NULL,
    src_lang        TEXT NOT NULL,
    tgt_lang        TEXT NOT NULL,
    license         TEXT NOT NULL,
    domain          TEXT NOT NULL,
    raw_pairs       INTEGER NOT NULL,
    accepted_pairs  INTEGER NOT NULL,
    filter_stats    TEXT NOT NULL,
    added_to_tm     INTEGER NOT NULL DEFAULT 0,
    tm_imported     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_corpora_langs ON corpora (src_lang, tgt_lang);
"""


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    source_name: str
    fmt: str
    src_lang: str
    tgt_lang: str
    license: str
    domain: str
    commercial_safe: bool
    raw_pairs: int
    accepted_pairs: int
    filter_stats: dict[str, int] = field(default_factory=dict)
    added_to_tm: bool = False
    tm_imported: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorpusStore:
    """Manages imported parallel corpora on disk + optional TM ingestion."""

    def __init__(self, path: Path, *, tm: Any | None = None, use_lid: bool = True) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "corpora.sqlite"
        self.tm = tm
        self._use_lid = use_lid
        self._detector: Callable[[str], tuple[str, float]] | None = None
        self._detector_loaded = False
        self._lock = threading.RLock()
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_CORPUS_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _lid(self) -> Callable[[str], tuple[str, float]] | None:
        if not self._use_lid:
            return None
        if not self._detector_loaded:
            self._detector_loaded = True
            with contextlib.suppress(Exception):
                from afrilingua.backends.factory import build_detector

                det = build_detector()
                self._detector = det.detect if det is not None else None
        return self._detector

    def _pairs_path(self, corpus_id: str) -> Path:
        return self.path / f"{corpus_id}.jsonl"

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CorpusRecord:
        lic = str(row["license"])
        return CorpusRecord(
            id=row["id"],
            source_name=row["source_name"],
            fmt=row["fmt"],
            src_lang=row["src_lang"],
            tgt_lang=row["tgt_lang"],
            license=lic,
            domain=row["domain"],
            commercial_safe=lic in COMMERCIAL_SAFE_LICENSES,
            raw_pairs=int(row["raw_pairs"]),
            accepted_pairs=int(row["accepted_pairs"]),
            filter_stats=json.loads(row["filter_stats"] or "{}"),
            added_to_tm=bool(row["added_to_tm"]),
            tm_imported=int(row["tm_imported"]),
            created_at=row["created_at"],
        )

    def import_corpus(
        self,
        *,
        fmt: str,
        files: list[Path],
        src_lang: str,
        tgt_lang: str,
        source_name: str,
        license: str,
        domain: str = "general",
        add_to_tm: bool = False,
    ) -> CorpusRecord:
        """Parse, align, filter and store a parallel corpus (and optionally add it to the TM)."""
        fmt_norm = fmt.strip().lower()
        if fmt_norm not in CORPUS_FORMATS:
            raise InvalidInputError(
                f"Unsupported corpus format {fmt!r}. Choose from: {sorted(CORPUS_FORMATS)}"
            )
        source_name = normalize_text(source_name)
        if not source_name:
            raise InvalidInputError("source_name is required for provenance tracking.")
        if license not in CORPUS_LICENSES:
            raise InvalidInputError(
                f"Unsupported license {license!r}. Choose from: {sorted(CORPUS_LICENSES)}"
            )
        domain_norm = domain.strip().lower() or "general"
        if domain_norm not in CORPUS_DOMAINS:
            raise InvalidInputError(f"Unsupported domain {domain!r}. Choose from: {sorted(CORPUS_DOMAINS)}")

        reg = get_registry()
        src = reg.normalize(src_lang)
        tgt = reg.normalize(tgt_lang)
        if src == tgt:
            raise InvalidInputError("src_lang and tgt_lang must be different.")

        raw_iter = self._extract_raw_pairs(fmt_norm, files, src, tgt)
        accepted, stats = filter_parallel_pairs(raw_iter, src, tgt, detector=self._lid())
        if not accepted:
            raise InvalidInputError(
                f"No valid parallel pairs remained after filtering ({stats.raw_pairs} raw pairs inspected)."
            )

        corpus_id = uuid.uuid4().hex[:12]
        jsonl_path = self._pairs_path(corpus_id)
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
            for s_txt, t_txt, ref in accepted:
                rec = {
                    "corpus_id": corpus_id,
                    "source_name": source_name,
                    "license": license,
                    "domain": domain_norm,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "source": s_txt,
                    "target": t_txt,
                    "ref": ref,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tm_count = 0
        if add_to_tm and self.tm is not None:
            tm_origin = f"corpus:{corpus_id}"
            tm_pairs = [
                (s_txt, t_txt) for s_txt, t_txt, _ in accepted if len(s_txt) <= 2000 and len(t_txt) <= 2000
            ]
            if hasattr(self.tm, "add_many"):
                tm_count = int(self.tm.add_many(tm_pairs, src, tgt, origin=tm_origin, approved=True))
                if hasattr(self.tm, "flush"):
                    self.tm.flush()
            else:
                for s_txt, t_txt in tm_pairs:
                    self.tm.add(s_txt, t_txt, src, tgt, origin=tm_origin, approved=True)
                    tm_count += 1

        created = _now()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO corpora (id, source_name, fmt, src_lang, tgt_lang, license, domain, "
                "raw_pairs, accepted_pairs, filter_stats, added_to_tm, tm_imported, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    corpus_id,
                    source_name,
                    fmt_norm,
                    src,
                    tgt,
                    license,
                    domain_norm,
                    stats.raw_pairs,
                    stats.accepted,
                    json.dumps(stats.to_dict()),
                    int(bool(add_to_tm and self.tm is not None)),
                    tm_count,
                    created,
                ),
            )
        return self.get(corpus_id)

    @staticmethod
    def _extract_raw_pairs(fmt: str, files: list[Path], src: str, tgt: str) -> Iterator[tuple[str, str, str]]:
        if fmt == "ebible":
            if len(files) == 2:
                return iter_ebible_pairs(files[0], files[1])
            if len(files) == 3:
                return iter_ebible_pairs(files[0], files[1], vref_path=files[2])
            raise InvalidInputError("ebible format expects 2 files (src, tgt) or 3 files (src, tgt, vref).")
        if fmt == "usfm":
            if len(files) < 2 or len(files) % 2 != 0:
                raise InvalidInputError(
                    "usfm format expects an even number of files (src file(s), then tgt file(s))."
                )
            mid = len(files) // 2
            return iter_usfm_pairs(files[:mid], files[mid:])
        if fmt in {"srt", "vtt"}:
            if len(files) != 2:
                raise InvalidInputError(f"{fmt} format expects exactly 2 subtitle files (source, target).")
            return iter_subtitle_pairs(files[0], files[1])
        if fmt == "lines":
            if len(files) != 2:
                raise InvalidInputError("lines format expects exactly 2 line-aligned text files.")
            return iter_line_pairs(files[0], files[1])
        if fmt == "unaligned":
            if len(files) != 2:
                raise InvalidInputError("unaligned format expects exactly 2 document files (source, target).")
            return iter_unaligned_pairs(files[0], files[1])
        if fmt in {"tsv", "csv", "jsonl", "tmx"}:
            if len(files) != 1:
                raise InvalidInputError(f"{fmt} format expects 1 parallel file.")
            reg = get_registry()

            def resolve(code: str) -> str | None:
                for candidate in (code, code.replace("-", "_"), code.split("-")[0].split("_")[0]):
                    with contextlib.suppress(UnsupportedLanguageError):
                        return reg.normalize(candidate)
                return None

            aliases = {
                name: {a.lower() for a in (*reg.get(name).aliases, reg.get(name).nllb_code)}
                for name in (src, tgt)
            }
            return (
                (s, t, f"{fmt}:{idx}")
                for idx, (s, t) in enumerate(
                    iter_tabular_tm_pairs(
                        files[0], src, tgt, fmt=fmt, resolve_lang=resolve, lang_aliases=aliases
                    ),
                    start=1,
                )
            )
        raise InvalidInputError(f"Unhandled format {fmt!r}")

    def list(self) -> list[CorpusRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM corpora ORDER BY created_at DESC, id").fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, corpus_id: str) -> CorpusRecord:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM corpora WHERE id=?", (corpus_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Corpus {corpus_id!r} not found.")
        return self._row_to_record(row)

    def iter_pairs(
        self,
        *,
        lang: str | None = None,
        commercial_safe_only: bool = False,
        corpus_ids: set[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield stored parallel records across matching corpora."""
        target_lang = get_registry().normalize(lang) if lang else None
        for rec in self.list():
            if corpus_ids is not None and rec.id not in corpus_ids:
                continue
            if commercial_safe_only and not rec.commercial_safe:
                continue
            if target_lang and target_lang not in {rec.src_lang, rec.tgt_lang}:
                continue
            p = self._pairs_path(rec.id)
            if not p.is_file():
                continue
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def delete(self, corpus_id: str) -> dict[str, Any]:
        rec = self.get(corpus_id)
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM corpora WHERE id=?", (corpus_id,))
        self._pairs_path(corpus_id).unlink(missing_ok=True)
        tm_removed = 0
        if rec.added_to_tm and self.tm is not None and hasattr(self.tm, "delete_by_origin"):
            tm_removed = int(self.tm.delete_by_origin(f"corpus:{corpus_id}"))
        return {"id": corpus_id, "deleted": True, "tm_entries_removed": tm_removed}


def open_corpus_store(
    tm: Any | None = None, path: str | Path | None = None, *, use_lid: bool = True
) -> CorpusStore:
    directory = Path(path) if path is not None else get_settings().data_dir / "corpora"
    return CorpusStore(directory, tm=tm, use_lid=use_lid)
