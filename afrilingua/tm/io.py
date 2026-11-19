"""Parsers for translation-memory import files: TSV, CSV, JSONL and TMX.

Each parser yields ``(source, target)`` string pairs. TMX is parsed incrementally
with the stdlib ``xml.etree.ElementTree.iterparse`` (expat ≥ 2.4.1 protects against
entity-expansion attacks; external entities are never resolved).
"""

from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from afrilingua.core.errors import InvalidInputError

__all__ = ["ImportFormat", "detect_format", "iter_pairs"]

ImportFormat = Literal["auto", "tsv", "csv", "jsonl", "tmx"]

_HEADER_WORDS = {
    "source",
    "target",
    "src",
    "tgt",
    "en",
    "english",
    "translation",
    "source_text",
    "target_text",
}
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def detect_format(path: Path, head: bytes | None = None) -> str:
    """Guess the import format from the file extension, then from the content."""
    ext = path.suffix.lower()
    by_ext = {
        ".tsv": "tsv",
        ".tab": "tsv",
        ".txt": "tsv",
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".json": "jsonl",
        ".tmx": "tmx",
        ".xml": "tmx",
    }
    if ext in by_ext:
        return by_ext[ext]
    if head is None:
        with path.open("rb") as fh:
            head = fh.read(4096)
    text = head.decode("utf-8", errors="ignore").lstrip("\ufeff \r\n\t")
    if text.startswith("<"):
        return "tmx"
    if text.startswith("{"):
        return "jsonl"
    first = text.splitlines()[0] if text else ""
    return "tsv" if "\t" in first else "csv"


def _is_header(cols: list[str], names: set[str]) -> bool:
    low = {c.strip().lower() for c in cols[:2]}
    return bool(low) and low <= (_HEADER_WORDS | names)


def _iter_delimited(path: Path, delimiter: str, names: set[str]) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        if delimiter == "\t":
            rows: Iterator[list[str]] = (line.rstrip("\r\n").split("\t") for line in fh)
        else:
            rows = csv.reader(fh)
        for lineno, cols in enumerate(rows):
            if len(cols) < 2:
                continue
            if lineno == 0 and _is_header(cols, names):
                continue
            yield cols[0], cols[1]


def _iter_jsonl(
    path: Path, src_keys: tuple[str, ...], tgt_keys: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line in {"[", "]"}:
                continue
            try:
                obj = json.loads(line.rstrip(","))
            except json.JSONDecodeError as exc:
                raise InvalidInputError(f"Invalid JSON on line {lineno}: {exc.msg}") from exc
            if not isinstance(obj, dict):
                continue
            src = next((obj[k] for k in src_keys if isinstance(obj.get(k), str)), None)
            tgt = next((obj[k] for k in tgt_keys if isinstance(obj.get(k), str)), None)
            if src is not None and tgt is not None:
                yield src, tgt


def _iter_tmx(
    path: Path, src_lang: str, tgt_lang: str, resolve: Callable[[str], str | None]
) -> Iterator[tuple[str, str]]:
    try:
        for _event, elem in ET.iterparse(str(path), events=("end",)):
            if elem.tag != "tu":
                continue
            segs: list[tuple[str | None, str]] = []
            for tuv in elem.iter("tuv"):
                lang = tuv.get(_XML_LANG) or tuv.get("lang") or ""
                seg = tuv.find("seg")
                text = "".join(seg.itertext()) if seg is not None else ""
                segs.append((resolve(lang) if lang else None, text))
            elem.clear()
            src = next((t for lang, t in segs if lang == src_lang), None)
            tgt = next((t for lang, t in segs if lang == tgt_lang), None)
            if (src is None or tgt is None) and len(segs) == 2 and all(lang is None for lang, _ in segs):
                src, tgt = segs[0][1], segs[1][1]
            if src is not None and tgt is not None:
                yield src, tgt
    except ET.ParseError as exc:
        raise InvalidInputError(f"Invalid TMX file: {exc}") from exc


def iter_pairs(
    path: Path,
    src_lang: str,
    tgt_lang: str,
    *,
    fmt: str = "auto",
    resolve_lang: Callable[[str], str | None] | None = None,
    lang_aliases: dict[str, set[str]] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(source, target)`` pairs from ``path``.

    ``resolve_lang`` maps TMX ``xml:lang`` codes (``en``, ``yo-NG``) to canonical
    language names. ``lang_aliases`` maps canonical names to accepted alias keys so
    JSONL files keyed by language (``{"english": ..., "yoruba": ...}``) also work.
    """
    fmt = detect_format(path) if fmt == "auto" else fmt
    aliases = lang_aliases or {}
    names = aliases.get(src_lang, set()) | aliases.get(tgt_lang, set()) | {src_lang, tgt_lang}
    if fmt == "tsv":
        return _iter_delimited(path, "\t", names)
    if fmt == "csv":
        return _iter_delimited(path, ",", names)
    if fmt == "jsonl":
        src_keys = ("source", "src", "source_text", src_lang, *sorted(aliases.get(src_lang, ())))
        tgt_keys = (
            "target",
            "tgt",
            "target_text",
            "translation",
            tgt_lang,
            *sorted(aliases.get(tgt_lang, ())),
        )
        return _iter_jsonl(path, src_keys, tgt_keys)
    if fmt == "tmx":
        return _iter_tmx(path, src_lang, tgt_lang, resolve_lang or (lambda _code: None))
    raise InvalidInputError(f"Unsupported import format {fmt!r}; use tsv, csv, jsonl or tmx.")


def sniff_bytes_format(filename: str, data: bytes) -> str:
    """Format detection for in-memory uploads (used by the API router)."""
    return detect_format(Path(filename or "upload"), io.BytesIO(data).read(4096))
