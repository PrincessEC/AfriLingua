"""
src/document/reader.py
Reads TXT, PDF (via PyMuPDF) and DOCX (via python-docx) files into a list
of sentence strings, ready for batch translation.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter using punctuation boundaries.
    Falls back gracefully when nltk is unavailable.
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if s.strip()]


def read_txt(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _split_sentences(text)


def read_pdf(path: Path) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF support. Install with: pip install pymupdf")

    doc = fitz.open(str(path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return _split_sentences(full_text)


def read_docx(path: Path) -> List[str]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")

    doc = Document(str(path))
    sentences: List[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            sentences.extend(_split_sentences(para.text))
    return sentences


class DocumentReader:
    """
    Unified document reader that dispatches to the correct format handler.

    Usage::
        reader = DocumentReader()
        sentences = reader.read("report.pdf")
    """

    _HANDLERS = {
        ".txt":  read_txt,
        ".pdf":  read_pdf,
        ".docx": read_docx,
    }

    def read(self, file_path: str | Path) -> List[str]:
        """
        Read a document and return its content as a list of sentences.

        Args:
            file_path: Path to .txt, .pdf, or .docx file.

        Returns:
            List of sentence strings.

        Raises:
            ValueError:  Unsupported file format.
            FileNotFoundError: File does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        handler = self._HANDLERS.get(suffix)
        if handler is None:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported: {list(self._HANDLERS.keys())}"
            )

        logger.info("Reading %s …", path.name)
        sentences = handler(path)
        logger.info("Extracted %d sentences from %s", len(sentences), path.name)
        return sentences
