"""Format-preserving document translation (DOCX → DOCX, PDF → DOCX, TXT → TXT)."""

from afrilingua.documents.translator import SUPPORTED_SUFFIXES, output_path, translate_document

__all__ = ["SUPPORTED_SUFFIXES", "output_path", "translate_document"]
