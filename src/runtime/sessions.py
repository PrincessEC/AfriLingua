"""Persistent session storage for the local AfriLingua workspace."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_DIR = _ROOT / "data" / "sessions"


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:48] or "session"


def save_session(title: str, entries: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> str:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().replace(microsecond=0)
    clean_title = title.strip() or f"Session {now.strftime('%Y-%m-%d %H:%M')}"
    session_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{_slug(clean_title)}"
    path = SESSION_DIR / f"{session_id}.json"
    payload = {
        "id": session_id,
        "title": clean_title,
        "created_at": now.isoformat(),
        "metadata": metadata or {},
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def list_sessions() -> list[str]:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    for path in sorted(SESSION_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = payload.get("title") or path.stem
        created_at = str(payload.get("created_at") or "").replace("T", " ")[:16]
        labels.append(f"{title} - {created_at} [{path.stem}]")
    return labels


def parse_session_label(label: str | None) -> str | None:
    if not label:
        return None
    match = re.search(r"\[([^\]]+)\]\s*$", label)
    return match.group(1) if match else label


def load_session(label: str) -> dict[str, Any]:
    session_id = parse_session_label(label)
    if not session_id:
        raise ValueError("No session selected.")
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))

