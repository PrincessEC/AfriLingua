"""Translation Memory implementation."""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

from afrilingua.core.settings import get_settings
from afrilingua.engine import TMMatch, TranslationMemoryProtocol


class TranslationMemory(TranslationMemoryProtocol):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path / "tm.sqlite", check_same_thread=False)
        self._init_db()
        self._lock = threading.RLock()

    def _init_db(self) -> None:
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    src_lang TEXT,
                    tgt_lang TEXT,
                    source TEXT,
                    source_norm TEXT,
                    target TEXT,
                    approved INTEGER,
                    origin TEXT,
                    created_at TEXT
                )
            """)
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_source ON entries (src_lang, tgt_lang, source_norm)")

    def lookup(self, text: str, src_lang: str, tgt_lang: str, *, k: int = 3) -> list[TMMatch]:
        # Simplified lookup for now, vector index to follow.
        with self._lock:
            # Using LIKE for fuzzy, ideally replace with vector index
            cursor = self.db.execute(
                "SELECT id, source, target, approved, origin FROM entries WHERE src_lang=? AND tgt_lang=? AND source LIKE ?",
                (src_lang, tgt_lang, f"%{text}%")
            )
            rows = cursor.fetchall()

        matches = []
        for r in rows:
            # Simple score based on fuzz ratio
            score = fuzz.ratio(text, r[1]) / 100.0
            matches.append(TMMatch(id=r[0], source=r[1], target=r[2], score=score, approved=bool(r[3]), origin=r[4]))

        return sorted(matches, key=lambda x: x.score, reverse=True)[:k]

    def add(
        self,
        source: str,
        target: str,
        src_lang: str,
        tgt_lang: str,
        *,
        origin: str = "mt",
        approved: bool = False,
    ) -> str:
        entry_id = str(uuid.uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, src_lang, tgt_lang, source, source.lower().strip(), target, int(approved), origin, datetime.utcnow().isoformat())
            )
        return entry_id

def open_translation_memory() -> TranslationMemory:
    return TranslationMemory(get_settings().data_dir / "tm")
