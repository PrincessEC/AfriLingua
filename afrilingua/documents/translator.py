"""
Format-preserving document translation.

* **DOCX → DOCX**: body paragraphs, tables (incl. nested), headers and
  footers are translated paragraph by paragraph. Paragraph styles and the
  first run's character formatting are kept: the translation goes into the
  first run and the remaining runs are emptied (simple and robust).
* **PDF → DOCX**: text blocks are extracted per page with PyMuPDF and
  written as one paragraph per block with a page break between pages.
* **TXT → TXT**: blank-line paragraph structure is preserved.

Sentences are split with a regex splitter and translated in batches of
``batch_size`` through ``engine.translate(..., mode="document")``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from afrilingua.backends.text_frontend import split_sentences
from afrilingua.core.errors import InvalidInputError

if TYPE_CHECKING:
    from afrilingua.engine import AfriLinguaEngine

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]
SUPPORTED_SUFFIXES = (".docx", ".pdf", ".txt")
_HAS_LETTER = re.compile(r"[^\W\d_]")


@dataclass
class _Unit:
    """One translatable paragraph: its sentences and where the result goes."""

    text: str
    sentences: list[str] = field(default_factory=list)
    result: str = ""


def output_path(src_path: Path, out_dir: Path, tgt_lang: str) -> Path:
    """``<out_dir>/<stem>.<tgt>.<ext>`` (PDF input → .docx)."""
    suffix = ".docx" if src_path.suffix.lower() == ".pdf" else src_path.suffix.lower()
    return out_dir / f"{src_path.stem}.{tgt_lang}{suffix}"


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _translate_units(
    engine: AfriLinguaEngine,
    units: Sequence[_Unit],
    src_lang: str,
    tgt_lang: str,
    *,
    batch_size: int,
    on_progress: ProgressCallback | None,
) -> None:
    """Split units into sentences, translate in batches, write ``unit.result``."""
    flat: list[tuple[int, int]] = []  # (unit index, sentence index) of translatable sentences
    for ui, unit in enumerate(units):
        unit.sentences = split_sentences(unit.text) or ([unit.text] if unit.text.strip() else [])
        for si, sent in enumerate(unit.sentences):
            if _HAS_LETTER.search(sent):
                flat.append((ui, si))
    outputs: dict[tuple[int, int], str] = {}
    total = len(flat)
    _progress(on_progress, 0.0)
    for start in range(0, total, batch_size):
        chunk = flat[start : start + batch_size]
        texts = [units[ui].sentences[si] for ui, si in chunk]
        results = engine.translate(texts, src_lang, tgt_lang, mode="document")
        for key, res in zip(chunk, results, strict=True):
            outputs[key] = res.text
        _progress(on_progress, min(0.99, (start + len(chunk)) / total))
    for ui, unit in enumerate(units):
        unit.result = " ".join(outputs.get((ui, si), s) for si, s in enumerate(unit.sentences)).strip()


def _progress(cb: ProgressCallback | None, value: float) -> None:
    if cb is None:
        return
    try:
        cb(float(value))
    except Exception:
        logger.exception("on_progress callback raised")


# ── DOCX ────────────────────────────────────────────────────────────────────


def _iter_docx_paragraphs(doc: Any) -> list[Any]:
    """All paragraphs in body, tables (nested) and section headers/footers, deduplicated."""
    seen: set[int] = set()
    out: list[Any] = []

    def add(paragraphs: Sequence[Any]) -> None:
        for p in paragraphs:
            key = id(p._p)
            if key not in seen:
                seen.add(key)
                out.append(p)

    def walk_tables(tables: Sequence[Any]) -> None:
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    add(cell.paragraphs)
                    walk_tables(cell.tables)

    add(doc.paragraphs)
    walk_tables(doc.tables)
    for section in doc.sections:
        for part_name in (
            "header",
            "footer",
            "first_page_header",
            "first_page_footer",
            "even_page_header",
            "even_page_footer",
        ):
            part = getattr(section, part_name, None)
            if part is None or getattr(part, "is_linked_to_previous", False):
                continue
            add(part.paragraphs)
            walk_tables(part.tables)
    return out


def _paragraph_runs(paragraph: Any) -> list[Any]:
    """Runs that make up ``paragraph.text`` (direct, hyperlink, tracked-insert, smart-tag)."""
    from docx.text.run import Run

    elements = paragraph._p.xpath("./w:r | ./w:hyperlink/w:r | ./w:ins/w:r | ./w:smartTag/w:r")
    return [Run(r, paragraph) for r in elements]


def _set_paragraph_text(paragraph: Any, text: str) -> None:
    runs = _paragraph_runs(paragraph)
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run.text = ""


def _translate_docx(
    engine: AfriLinguaEngine, src: Path, dst: Path, src_lang: str | None, tgt: str, **kw: Any
) -> str:
    from docx import Document

    doc = Document(str(src))
    paragraphs = [p for p in _iter_docx_paragraphs(doc) if _clean(p.text)]
    units = [_Unit(_clean(p.text)) for p in paragraphs]
    src_lang = _resolve_source(engine, units, src_lang)
    _translate_units(engine, units, src_lang, tgt, **kw)
    for p, unit in zip(paragraphs, units, strict=True):
        _set_paragraph_text(p, unit.result)
    doc.save(str(dst))
    return src_lang


# ── PDF ─────────────────────────────────────────────────────────────────────


def _translate_pdf(
    engine: AfriLinguaEngine, src: Path, dst: Path, src_lang: str | None, tgt: str, **kw: Any
) -> str:
    import fitz  # PyMuPDF
    from docx import Document

    pages: list[list[_Unit]] = []
    with fitz.open(str(src)) as pdf:
        for page in pdf:
            blocks = page.get_text("blocks", sort=True)
            units = [_Unit(_clean(b[4])) for b in blocks if len(b) > 6 and b[6] == 0 and _clean(b[4])]
            pages.append(units)
    all_units = [u for page in pages for u in page]
    if not all_units:
        raise InvalidInputError("No extractable text found in the PDF (scanned PDFs need OCR first).")
    src_lang = _resolve_source(engine, all_units, src_lang)
    _translate_units(engine, all_units, src_lang, tgt, **kw)
    doc = Document()
    for i, page_units in enumerate(pages):
        if i > 0:
            doc.add_page_break()
        for unit in page_units:
            doc.add_paragraph(unit.result)
    doc.save(str(dst))
    return src_lang


# ── TXT ─────────────────────────────────────────────────────────────────────


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _translate_txt(
    engine: AfriLinguaEngine, src: Path, dst: Path, src_lang: str | None, tgt: str, **kw: Any
) -> str:
    text = _read_text(src).replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(\n[ \t]*\n\s*)", text)  # keep blank-line separators
    units: list[_Unit] = []
    slots: list[int | None] = []
    for i, part in enumerate(parts):
        if i % 2 == 0 and _clean(part):
            slots.append(len(units))
            units.append(_Unit(_clean(part)))
        else:
            slots.append(None)
    if not units:
        raise InvalidInputError("The text file is empty.")
    src_lang = _resolve_source(engine, units, src_lang)
    _translate_units(engine, units, src_lang, tgt, **kw)
    out: list[str] = []
    for part, slot in zip(parts, slots, strict=True):
        if slot is None:
            out.append(part)
        else:
            leading = part[: len(part) - len(part.lstrip())]
            trailing = part[len(part.rstrip()) :]
            out.append(f"{leading}{units[slot].result}{trailing}")
    dst.write_text("".join(out), encoding="utf-8", newline="\n")
    return src_lang


# ── entry point ─────────────────────────────────────────────────────────────


def _resolve_source(engine: AfriLinguaEngine, units: Sequence[_Unit], src_lang: str | None) -> str:
    if src_lang and src_lang.lower() not in {"auto", "auto-detect"}:
        return engine.registry.normalize(src_lang)
    sample = " ".join(u.text for u in units)[:2000]
    lang, conf = engine.detect_language(sample)
    resolved = lang if lang != "unknown" else "english"
    logger.info("Document source language auto-detected as %s (confidence %.2f)", resolved, conf)
    return resolved


def translate_document(
    engine: AfriLinguaEngine,
    src_path: str | Path,
    out_dir: str | Path,
    tgt_lang: str,
    src_lang: str | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    batch_size: int = 32,
) -> Path:
    """Translate a .docx / .pdf / .txt file into ``out_dir`` and return the output path.

    ``on_progress`` receives floats from 0.0 to 1.0. The source language is
    auto-detected when ``src_lang`` is ``None`` or ``"auto"``.
    """
    src = Path(src_path)
    if not src.is_file():
        raise InvalidInputError(f"File not found: {src}")
    suffix = src.suffix.lower()
    handlers = {".docx": _translate_docx, ".pdf": _translate_pdf, ".txt": _translate_txt}
    if suffix not in handlers:
        raise InvalidInputError(
            f"Unsupported document type {suffix!r}. Use one of: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    tgt = engine.registry.normalize(tgt_lang)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = output_path(src, out, tgt)
    used_src = handlers[suffix](
        engine, src, dst, src_lang, tgt, batch_size=max(1, batch_size), on_progress=on_progress
    )
    logger.info("Translated %s (%s → %s) → %s", src.name, used_src, tgt, dst)
    _progress(on_progress, 1.0)
    return dst
