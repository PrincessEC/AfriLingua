"""Translation memory: SQLite storage + fast char n-gram vector search.

Use :func:`open_translation_memory` to get a :class:`TranslationMemory`, which
implements :class:`afrilingua.engine.TranslationMemoryProtocol`.
"""
from __future__ import annotations

from afrilingua.tm.memory import TranslationMemory, open_translation_memory

__all__ = ["TranslationMemory", "open_translation_memory"]
