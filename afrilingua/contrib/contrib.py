"""Contribution Store."""
from __future__ import annotations
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from afrilingua.tm.tm import TranslationMemory
from afrilingua.core.settings import get_settings

class ContributionStore:
    def __init__(self, tm: TranslationMemory | None = None) -> None:
        self.tm = tm
        path = get_settings().contributions_dir
        path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path / "contrib.sqlite", check_same_thread=False)
        self._init_db()
        self._lock = threading.RLock()
        
    def _init_db(self) -> None:
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS contributions (
                    id TEXT PRIMARY KEY,
                    kind TEXT,
                    src_lang TEXT,
                    tgt_lang TEXT,
                    source_text TEXT,
                    target_text TEXT,
                    status TEXT,
                    license TEXT,
                    created_at TEXT
                )
            """)

    def submit(self, payload: dict) -> dict:
        # Validate payload per contract §5
        if not payload.get("consent"):
            raise ValueError("Consent required")
        
        contrib_id = str(uuid.uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO contributions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (contrib_id, payload["kind"], payload["src_lang"], payload["tgt_lang"], 
                 payload["source_text"], payload["target_text"], "pending", payload["license"], datetime.utcnow().isoformat())
            )
        return {"id": contrib_id, "status": "pending"}

def open_contribution_store(tm: TranslationMemory | None = None) -> ContributionStore:
    return ContributionStore(tm)
