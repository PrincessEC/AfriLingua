"""Community contribution store: submissions, peer review, prompts, stats, export.

Storage (``settings.contributions_dir`` by default)::

    <dir>/contrib.sqlite            contributions, reviews, contributors (WAL)
    <dir>/audio/<lang>/<id>.flac     16 kHz mono recordings
    <dir>/exports/                   generated JSONL / Hugging Face exports

Review policy (:class:`ReviewPolicy`, configurable):

* one vote per reviewer per item (re-voting replaces the earlier vote); a
  contributor can never review their own items; only *pending* items are reviewed;
* reviewers whose reputation ≥ ``trusted_reputation`` (default 10) cast a vote of
  weight ``trusted_weight`` (default 2), everyone else weight 1;
  ``votes_up`` / ``votes_down`` hold these *weighted* sums;
* ``approved`` when weighted approvals ≥ ``min_approvals`` (2) and approvals >
  rejections; ``rejected`` when rejections ≥ ``min_rejections`` (2) and
  rejections > approvals;
* contributor reputation: +1 per approved item, −1 per rejected item;
* on approval, translation / correction / glossary pairs are written into the
  translation memory with ``approved=True, origin="community"``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afrilingua.contrib.audio import process_audio, write_flac
from afrilingua.contrib.prompts import Prompt, load_prompt_file, seed_prompts
from afrilingua.contrib.quality import estimate_quality
from afrilingua.contrib.validation import (
    ANONYMOUS,
    TM_KINDS,
    anonymous_id,
    contributor_key,
    validate_contribution,
)
from afrilingua.core.errors import InvalidInputError, NotFoundError, PayloadTooLargeError
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import get_settings
from afrilingua.tm.text import tm_key

logger = logging.getLogger(__name__)

__all__ = ["ContributionStore", "ReviewPolicy", "open_contribution_store"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contributions (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    src_lang         TEXT NOT NULL,
    tgt_lang         TEXT NOT NULL,
    source_text      TEXT NOT NULL,
    source_key       TEXT NOT NULL,
    target_text      TEXT NOT NULL,
    original_mt      TEXT,
    domain           TEXT NOT NULL DEFAULT 'general',
    dialect          TEXT,
    region           TEXT,
    accent           TEXT,
    gender           TEXT,
    age_range        TEXT,
    contributor      TEXT NOT NULL,
    contributor_key  TEXT NOT NULL,
    license          TEXT NOT NULL,
    consent          INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    votes_up         INTEGER NOT NULL DEFAULT 0,
    votes_down       INTEGER NOT NULL DEFAULT 0,
    quality_score    REAL NOT NULL DEFAULT 0,
    quality_flags    TEXT NOT NULL DEFAULT '[]',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    audio_path       TEXT,
    audio_duration_s REAL,
    needs_transcode  INTEGER NOT NULL DEFAULT 0,
    tm_entry_id      TEXT
);
CREATE INDEX IF NOT EXISTS ix_contrib_status ON contributions (status, created_at);
CREATE INDEX IF NOT EXISTS ix_contrib_langs ON contributions (src_lang, tgt_lang, source_key);
CREATE INDEX IF NOT EXISTS ix_contrib_contributor ON contributions (contributor_key);
CREATE TABLE IF NOT EXISTS reviews (
    contribution_id TEXT NOT NULL REFERENCES contributions(id) ON DELETE CASCADE,
    reviewer        TEXT NOT NULL,
    reviewer_key    TEXT NOT NULL,
    vote            TEXT NOT NULL CHECK (vote IN ('approve', 'reject')),
    weight          INTEGER NOT NULL DEFAULT 1,
    comment         TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (contribution_id, reviewer_key)
);
CREATE TABLE IF NOT EXISTS contributors (
    id            TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    contributions INTEGER NOT NULL DEFAULT 0,
    approved      INTEGER NOT NULL DEFAULT 0,
    rejected      INTEGER NOT NULL DEFAULT 0,
    reputation    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
"""

_PUBLIC_FIELDS = (
    "id",
    "kind",
    "src_lang",
    "tgt_lang",
    "source_text",
    "target_text",
    "original_mt",
    "domain",
    "dialect",
    "region",
    "accent",
    "gender",
    "age_range",
    "contributor",
    "license",
    "consent",
    "status",
    "votes_up",
    "votes_down",
    "quality_score",
    "quality_flags",
    "created_at",
    "updated_at",
    "audio_duration_s",
    "needs_transcode",
)
_EXPORT_FIELDS = (
    "id",
    "kind",
    "src_lang",
    "tgt_lang",
    "source_text",
    "target_text",
    "original_mt",
    "domain",
    "dialect",
    "region",
    "accent",
    "gender",
    "age_range",
    "contributor",
    "license",
    "created_at",
    "audio_duration_s",
)
_MEDIA_TYPES = {
    "flac": "audio/flac",
    "wav": "audio/wav",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ReviewPolicy:
    """Thresholds of the peer-review workflow (see module docstring)."""

    min_approvals: int = 2
    min_rejections: int = 2
    trusted_reputation: int = 10
    trusted_weight: int = 2

    def decide(self, up: int, down: int) -> str:
        if up >= self.min_approvals and up > down:
            return "approved"
        if down >= self.min_rejections and down > up:
            return "rejected"
        return "pending"


class ContributionStore:
    """SQLite-backed community contribution hub (see module docstring)."""

    def __init__(
        self,
        path: Path,
        *,
        tm: Any | None = None,
        policy: ReviewPolicy | None = None,
        use_lid: bool = True,
        max_upload_mb: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "contrib.sqlite"
        self.audio_dir = self.path / "audio"
        self.tm = tm
        self.policy = policy or ReviewPolicy()
        self.max_upload_bytes = int((max_upload_mb or get_settings().max_upload_mb) * 1024 * 1024)
        self._use_lid = use_lid
        self._detector: Callable[[str], tuple[str, float]] | None = None
        self._detector_loaded = False
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._closed = False
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── connections ─────────────────────────────────────────────────────────
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise RuntimeError("ContributionStore is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            with self._conns_lock:
                self._conns.append(conn)
            self._local.conn = conn
        yield conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._conns_lock:
            for conn in self._conns:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._conns.clear()

    def __enter__(self) -> ContributionStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── helpers ─────────────────────────────────────────────────────────────
    def _lid(self) -> Callable[[str], tuple[str, float]] | None:
        if not self._use_lid:
            return None
        if not self._detector_loaded:
            self._detector_loaded = True
            try:
                from afrilingua.backends.factory import build_detector

                detector = build_detector()
                self._detector = detector.detect if detector is not None else None
            except Exception:
                logger.debug("Language identifier unavailable for quality checks", exc_info=True)
                self._detector = None
        return self._detector

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = {k: row[k] for k in _PUBLIC_FIELDS}
        d["consent"] = bool(d["consent"])
        d["needs_transcode"] = bool(d["needs_transcode"])
        d["quality_flags"] = json.loads(d["quality_flags"] or "[]")
        d["has_audio"] = bool(row["audio_path"])
        if d["kind"] == "transcription":
            d["lang"] = d["src_lang"]
            d["text"] = d["source_text"]
        return d

    def _row(self, contribution_id: str) -> sqlite3.Row:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM contributions WHERE id=?", (contribution_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Contribution {contribution_id!r} not found")
        return row

    def _bump_contributor(self, conn: sqlite3.Connection, key: str, name: str, **deltas: int) -> None:
        conn.execute(
            "INSERT INTO contributors (id, display_name, created_at) VALUES (?,?,?) ON CONFLICT(id) DO NOTHING",
            (key, name, _now()),
        )
        if deltas:
            sets = ", ".join(f"{col} = {col} + ?" for col in deltas)
            conn.execute(f"UPDATE contributors SET {sets} WHERE id=?", (*deltas.values(), key))

    def _reputation(self, conn: sqlite3.Connection, key: str) -> int:
        row = conn.execute("SELECT reputation FROM contributors WHERE id=?", (key,)).fetchone()
        return int(row[0]) if row else 0

    # ── submissions ─────────────────────────────────────────────────────────
    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate, score and store a ``ContributionIn`` payload; returns the ``Contribution``."""
        return self._insert(payload, audio=None)

    def add_audio(
        self,
        audio: bytes,
        *,
        text: str,
        lang: str,
        filename: str | None = None,
        content_type: str | None = None,
        dialect: str | None = None,
        accent: str | None = None,
        gender: str | None = None,
        age_range: str | None = None,
        region: str | None = None,
        consent: bool = False,
        license: str = "CC-BY-4.0",
        contributor: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Store a voice recording of ``text`` in ``lang`` (kind ``transcription``)."""
        if len(audio) > self.max_upload_bytes:
            raise PayloadTooLargeError(f"audio upload exceeds {self.max_upload_bytes // (1024 * 1024)} MB")
        payload = {
            "kind": "transcription",
            "src_lang": lang,
            "tgt_lang": lang,
            "source_text": text,
            "target_text": text,
            "dialect": dialect,
            "accent": accent,
            "gender": gender,
            "age_range": age_range,
            "region": region,
            "consent": consent,
            "license": license,
            "contributor": contributor,
            "domain": domain,
        }
        validate_contribution(payload)  # fail fast before decoding audio
        processed = process_audio(audio, filename, content_type)
        return self._insert(payload, audio=processed)

    def _insert(self, payload: dict[str, Any], audio: Any | None) -> dict[str, Any]:
        v = validate_contribution(payload)
        report = estimate_quality(
            kind=v.kind,
            src_lang=v.src_lang,
            tgt_lang=v.tgt_lang,
            source_text=v.source_text,
            target_text=v.target_text,
            pii_found=v.pii_found,
            tm=self.tm,
            detector=self._lid(),
        )
        cid = uuid.uuid4().hex
        audio_rel: str | None = None
        duration: float | None = None
        needs_transcode = False
        if audio is not None:
            lang_dir = self.audio_dir / v.src_lang
            if audio.needs_transcode:
                target = lang_dir / f"{cid}.{audio.original_ext}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(audio.original)
                needs_transcode = True
            else:
                target = lang_dir / f"{cid}.flac"
                write_flac(target, audio.waveform)
                duration = audio.duration_s
            audio_rel = target.relative_to(self.path).as_posix()
        now = _now()
        ckey = contributor_key(v.contributor)
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO contributions (id, kind, src_lang, tgt_lang, source_text, source_key, target_text, "
                    "original_mt, domain, dialect, region, accent, gender, age_range, contributor, contributor_key, "
                    "license, consent, status, quality_score, quality_flags, created_at, updated_at, audio_path, "
                    "audio_duration_s, needs_transcode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        v.kind,
                        v.src_lang,
                        v.tgt_lang,
                        v.source_text,
                        tm_key(v.source_text),
                        v.target_text,
                        v.original_mt,
                        v.domain,
                        v.dialect,
                        v.region,
                        v.accent,
                        v.gender,
                        v.age_range,
                        v.contributor,
                        ckey,
                        v.license,
                        1,
                        "pending",
                        report.score,
                        json.dumps(report.flags),
                        now,
                        now,
                        audio_rel,
                        duration,
                        int(needs_transcode),
                    ),
                )
                self._bump_contributor(conn, ckey, v.contributor, contributions=1)
        except BaseException:
            if audio_rel:
                (self.path / audio_rel).unlink(missing_ok=True)
            raise
        return self.get(cid)

    # ── queries ─────────────────────────────────────────────────────────────
    def get(self, contribution_id: str) -> dict[str, Any]:
        return self._to_dict(self._row(contribution_id))

    def list(
        self,
        *,
        status: str | None = None,
        lang: str | None = None,
        kind: str | None = None,
        contributor: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Filtered, newest-first page of contributions plus the total count."""
        where, args = ["1=1"], []
        if status:
            if status not in {"pending", "approved", "rejected"}:
                raise InvalidInputError("status must be pending, approved or rejected")
            where.append("status=?")
            args.append(status)
        if lang:
            code = get_registry().normalize(lang)
            where.append("(src_lang=? OR tgt_lang=?)")
            args += [code, code]
        if kind:
            where.append("kind=?")
            args.append(kind)
        if contributor:
            where.append("contributor_key=?")
            args.append(contributor_key(contributor))
        clause = " AND ".join(where)
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with self._conn() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM contributions WHERE {clause}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM contributions WHERE {clause} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [self._to_dict(r) for r in rows], int(total)

    def review_queue(
        self, reviewer: str, *, lang: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Pending items the reviewer did not submit and has not voted on yet (lowest votes first)."""
        rkey = contributor_key(reviewer)
        where, args = ["c.status='pending'", "c.contributor_key != ?"], [rkey]
        if lang:
            code = get_registry().normalize(lang)
            where.append("(c.src_lang=? OR c.tgt_lang=?)")
            args += [code, code]
        sql = (
            f"SELECT c.* FROM contributions c WHERE {' AND '.join(where)} AND NOT EXISTS "
            "(SELECT 1 FROM reviews r WHERE r.contribution_id=c.id AND r.reviewer_key=?) "
            "ORDER BY (c.votes_up + c.votes_down) ASC, c.quality_score DESC, c.created_at ASC LIMIT ?"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, [*args, rkey, max(1, min(int(limit), 200))]).fetchall()
        return [self._to_dict(r) for r in rows]

    def reviews(self, contribution_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT reviewer, vote, weight, comment, created_at FROM reviews WHERE contribution_id=? "
                "ORDER BY created_at",
                (contribution_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── review workflow ─────────────────────────────────────────────────────
    def review(
        self, contribution_id: str, reviewer: str, vote: str, comment: str | None = None
    ) -> dict[str, Any]:
        """Record a vote and apply the approval rule; returns the updated ``Contribution``."""
        if vote not in {"approve", "reject"}:
            raise InvalidInputError("vote must be 'approve' or 'reject'")
        reviewer = " ".join((reviewer or "").split())
        rkey = contributor_key(reviewer)
        if not reviewer or rkey == ANONYMOUS:
            raise InvalidInputError("reviewer name is required")
        if len(reviewer) > 100 or (comment and len(comment) > 1000):
            raise InvalidInputError("reviewer/comment too long")
        approved_pair: tuple[str, str, str, str] | None = None
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM contributions WHERE id=?", (contribution_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Contribution {contribution_id!r} not found")
            if row["contributor_key"] == rkey:
                raise InvalidInputError("contributors cannot review their own contributions")
            if row["status"] != "pending":
                raise InvalidInputError(f"contribution is already {row['status']}")
            weight = (
                self.policy.trusted_weight
                if self._reputation(conn, rkey) >= self.policy.trusted_reputation
                else 1
            )
            conn.execute(
                "INSERT INTO reviews (contribution_id, reviewer, reviewer_key, vote, weight, comment, created_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(contribution_id, reviewer_key) DO UPDATE SET "
                "vote=excluded.vote, weight=excluded.weight, comment=excluded.comment, created_at=excluded.created_at",
                (contribution_id, reviewer, rkey, vote, weight, comment, _now()),
            )
            up, down = conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN vote='approve' THEN weight END), 0), "
                "COALESCE(SUM(CASE WHEN vote='reject' THEN weight END), 0) FROM reviews WHERE contribution_id=?",
                (contribution_id,),
            ).fetchone()
            status = self.policy.decide(int(up), int(down))
            conn.execute(
                "UPDATE contributions SET votes_up=?, votes_down=?, status=?, updated_at=? WHERE id=?",
                (up, down, status, _now(), contribution_id),
            )
            ckey, cname = row["contributor_key"], row["contributor"]
            if status == "approved":
                self._bump_contributor(conn, ckey, cname, approved=1, reputation=1)
                if row["kind"] in TM_KINDS:
                    approved_pair = (row["source_text"], row["target_text"], row["src_lang"], row["tgt_lang"])
            elif status == "rejected":
                self._bump_contributor(conn, ckey, cname, rejected=1, reputation=-1)
        if approved_pair is not None and self.tm is not None:
            try:
                entry_id = self.tm.add(*approved_pair, origin="community", approved=True)
                with self._tx() as conn:
                    conn.execute(
                        "UPDATE contributions SET tm_entry_id=? WHERE id=?", (entry_id, contribution_id)
                    )
            except Exception:
                logger.exception("Could not write approved contribution %s into the TM", contribution_id)
        return self.get(contribution_id)

    # ── audio ───────────────────────────────────────────────────────────────
    def audio_file(self, contribution_id: str) -> tuple[Path, str]:
        """Return ``(path, media_type)`` of a contribution's recording."""
        row = self._row(contribution_id)
        if not row["audio_path"]:
            raise NotFoundError(f"Contribution {contribution_id!r} has no audio")
        path = (self.path / row["audio_path"]).resolve()
        if not path.is_file() or self.path.resolve() not in path.parents:
            raise NotFoundError(f"Audio for contribution {contribution_id!r} is missing")
        return path, _MEDIA_TYPES.get(path.suffix.lstrip(".").lower(), "application/octet-stream")

    # ── prompts ─────────────────────────────────────────────────────────────
    def get_prompts(self, lang: str, kind: str = "translate", n: int = 10) -> list[dict[str, str]]:
        """Prompts to translate into ``lang`` (``translate``) or to read aloud in ``lang`` (``record``).

        ``translate``: English sentences from the bundled corpus (native seed sentences
        when ``lang`` is English), least-translated first. ``record``: sentences *in*
        ``lang`` from approved contributions and the TM, then bundled seeds, least-recorded
        first. Ties are shuffled so concurrent contributors see different prompts.
        """
        reg = get_registry()
        code = reg.normalize(lang)
        n = max(1, min(int(n), 100))
        if kind not in {"translate", "record"}:
            raise InvalidInputError("kind must be 'translate' or 'record'")
        if kind == "translate":
            if code == "english":
                pool: list[Prompt] = []
                for spec in reg:
                    if spec.name != "english":
                        pool.extend(seed_prompts(spec.name, spec.iso639_1))
                with self._conn() as conn:
                    counts = self._key_counts(
                        conn,
                        "src_lang IN (SELECT DISTINCT src_lang FROM contributions) "
                        "AND tgt_lang='english' AND kind != 'transcription'",
                        (),
                    )
            else:
                pool = list(load_prompt_file("prompts_en.txt", "english", "en"))
                with self._conn() as conn:
                    counts = self._key_counts(
                        conn, "src_lang='english' AND tgt_lang=? AND kind != 'transcription'", (code,)
                    )
        else:
            pool = self._native_sentences(code, limit=max(200, 4 * n))
            pool.extend(seed_prompts(code, reg.get(code).iso639_1))
            with self._conn() as conn:
                counts = self._key_counts(
                    conn, "src_lang=? AND kind='transcription' AND audio_path IS NOT NULL", (code,)
                )
        seen: set[str] = set()
        unique: list[Prompt] = []
        for p in pool:
            key = tm_key(p.text)
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        rng = random.Random(time.time_ns())
        rng.shuffle(unique)
        unique.sort(key=lambda p: counts.get(tm_key(p.text), 0))
        return [p.to_dict() for p in unique[:n]]

    @staticmethod
    def _key_counts(conn: sqlite3.Connection, where: str, args: tuple[Any, ...]) -> dict[str, int]:
        rows = conn.execute(
            f"SELECT source_key, COUNT(*) FROM contributions WHERE {where} GROUP BY source_key", args
        )
        return {r[0]: int(r[1]) for r in rows}

    def _native_sentences(self, lang: str, limit: int) -> list[Prompt]:
        out: list[Prompt] = []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, target_text, domain FROM contributions WHERE status='approved' AND tgt_lang=? "
                "AND kind IN ('translation', 'correction') ORDER BY created_at DESC LIMIT ?",
                (lang, limit),
            ).fetchall()
        out.extend(Prompt(f"c-{r[0]}", r[1], lang, r[2] or "general") for r in rows)
        if self.tm is not None and hasattr(self.tm, "iter_entries"):
            try:
                for i, entry in enumerate(self.tm.iter_entries(approved_only=True, tgt_lang=lang)):
                    if i >= limit:
                        break
                    if len(entry["target"]) <= 200:
                        out.append(Prompt(f"tm-{entry['id']}", entry["target"], lang, "tm"))
            except Exception:
                logger.debug("TM prompt lookup failed", exc_info=True)
        return out

    # ── stats ───────────────────────────────────────────────────────────────
    def stats(self, *, top: int = 10) -> dict[str, Any]:
        """Totals, per-language breakdown, hours of audio and a contributor leaderboard (contract §5)."""
        with self._conn() as conn:
            total, approved, pending, rejected = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(status='approved'), 0), COALESCE(SUM(status='pending'), 0), "
                "COALESCE(SUM(status='rejected'), 0) FROM contributions"
            ).fetchone()
            by_lang: dict[str, dict[str, int]] = {}
            for r in conn.execute(
                "SELECT CASE WHEN tgt_lang='english' THEN src_lang ELSE tgt_lang END AS lang, "
                "SUM(audio_path IS NULL), SUM(audio_path IS NOT NULL), SUM(status='approved') "
                "FROM contributions GROUP BY lang"
            ):
                by_lang[r[0]] = {"text": int(r[1]), "audio": int(r[2]), "approved": int(r[3])}
            seconds = conn.execute("SELECT COALESCE(SUM(audio_duration_s), 0) FROM contributions").fetchone()[
                0
            ]
            contributors = conn.execute(
                "SELECT COUNT(*) FROM contributors WHERE contributions > 0 AND id != ?", (ANONYMOUS,)
            ).fetchone()[0]
            leaders = [
                {
                    "contributor": r[0],
                    "contributions": int(r[1]),
                    "count": int(r[1]),
                    "approved": int(r[2]),
                    "reputation": int(r[3]),
                }
                for r in conn.execute(
                    "SELECT display_name, contributions, approved, reputation FROM contributors "
                    "WHERE id != ? AND contributions > 0 ORDER BY approved DESC, contributions DESC, display_name "
                    "LIMIT ?",
                    (ANONYMOUS, top),
                )
            ]
        return {
            "total": int(total),
            "approved": int(approved),
            "pending": int(pending),
            "rejected": int(rejected),
            "by_lang": by_lang,
            "contributors": int(contributors),
            "hours_audio": round(float(seconds) / 3600, 4),
            "leaderboard": leaders,
        }

    # ── export ──────────────────────────────────────────────────────────────
    def iter_approved(self, *, lang: str | None = None, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Approved contributions with license + attribution fields (export / training)."""
        where, args = ["status='approved'"], []
        if lang:
            code = get_registry().normalize(lang)
            where.append("(src_lang=? OR tgt_lang=?)")
            args += [code, code]
        if kind:
            where.append("kind=?")
            args.append(kind)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM contributions WHERE {' AND '.join(where)} ORDER BY created_at, id", args
            ).fetchall()
        for r in rows:
            rec = {k: r[k] for k in _EXPORT_FIELDS}
            rec["audio_path"] = r["audio_path"]
            yield rec

    def export(
        self,
        lang: str | None = None,
        kind: str | None = None,
        fmt: str = "jsonl",
        out_dir: Path | None = None,
    ) -> Path:
        """Export approved items. ``jsonl`` → one file; ``hf`` → dataset folder (data.jsonl, audio/, README.md)."""
        if fmt not in {"jsonl", "hf"}:
            raise InvalidInputError("format must be 'jsonl' or 'hf'")
        out_dir = Path(out_dir) if out_dir else self.path / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
        suffix = "-".join(x for x in (lang and get_registry().normalize(lang), kind) if x)
        name = f"afrilingua-community-{suffix + '-' if suffix else ''}{stamp}"
        records = list(self.iter_approved(lang=lang, kind=kind))
        if fmt == "jsonl":
            path = out_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                for rec in records:
                    rec["audio"] = rec.pop("audio_path")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return path
        folder = out_dir / name
        (folder / "audio").mkdir(parents=True, exist_ok=True)
        with (folder / "data.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for rec in records:
                rel = rec.pop("audio_path")
                rec["audio"] = None
                if rel and (self.path / rel).is_file():
                    dst = folder / rel  # audio/<lang>/<id>.<ext>
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.path / rel, dst)
                    rec["audio"] = Path(rel).as_posix()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (folder / "README.md").write_text(_dataset_card(records), encoding="utf-8")
        return folder

    # ── right to withdraw ───────────────────────────────────────────────────
    def withdraw(self, contributor: str) -> dict[str, int]:
        """Delete a contributor's pending items and recordings; anonymise their reviewed items.

        Approved/rejected *text* stays (it may already be in the TM / a released dataset)
        but is re-attributed to ``anonymous`` with demographic metadata cleared.
        Voice recordings are deleted outright, since a voice cannot be anonymised.
        Their reviews are kept for vote integrity but the reviewer name is replaced.
        """
        key = contributor_key(contributor)
        if key == ANONYMOUS:
            raise InvalidInputError("a contributor name is required")
        pseudo = anonymous_id(key)
        files: list[str] = []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, status, audio_path FROM contributions WHERE contributor_key=?", (key,)
            ).fetchall()
            pending = [r["id"] for r in rows if r["status"] == "pending"]
            reviewed = [r["id"] for r in rows if r["status"] != "pending"]
            files = [r["audio_path"] for r in rows if r["audio_path"]]
            for cid in pending:
                conn.execute("DELETE FROM contributions WHERE id=?", (cid,))
            for cid in reviewed:
                conn.execute(
                    "UPDATE contributions SET contributor=?, contributor_key=?, accent=NULL, gender=NULL, "
                    "age_range=NULL, region=NULL, audio_path=NULL, audio_duration_s=NULL, updated_at=? WHERE id=?",
                    (ANONYMOUS, pseudo, _now(), cid),
                )
            conn.execute(
                "UPDATE reviews SET reviewer=?, reviewer_key=? WHERE reviewer_key=?", (ANONYMOUS, pseudo, key)
            )
            conn.execute("DELETE FROM contributors WHERE id=?", (key,))
        for rel in files:
            try:
                (self.path / rel).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete withdrawn audio %s", rel)
        return {"deleted": len(pending), "anonymized": len(reviewed), "audio_deleted": len(files)}


def _dataset_card(records: list[dict[str, Any]]) -> str:
    reg = get_registry()
    langs = sorted({r["src_lang"] for r in records} | {r["tgt_lang"] for r in records})
    codes = []
    for name in langs:
        try:
            spec = reg.get(name)
            codes.append(spec.iso639_1 or spec.iso639_3 or name)
        except Exception:
            codes.append(name)
    licenses = sorted({r["license"] for r in records}) or ["cc-by-4.0"]
    hf_license = "cc-by-4.0" if "CC-BY-4.0" in licenses or not records else "cc0-1.0"
    contributors = sorted(
        {r["contributor"] for r in records if r["contributor"] and r["contributor"] != ANONYMOUS}
    )
    n_audio = sum(1 for r in records if r.get("audio"))
    hours = sum(r.get("audio_duration_s") or 0 for r in records) / 3600
    header = "\n".join(
        [
            "---",
            f"license: {hf_license}",
            "language:",
            *[f"- {c}" for c in codes],
            "task_categories:",
            "- translation",
            "- automatic-speech-recognition",
            "pretty_name: AfriLingua community contributions",
            "configs:",
            "- config_name: default",
            "  data_files: data.jsonl",
            "---",
        ]
    )
    credit = ", ".join(contributors) if contributors else "anonymous contributors"
    return f"""{header}

# AfriLingua community contributions

Peer-reviewed translations, corrections, glossary terms and voice recordings
contributed to [AfriLingua](https://github.com/) by native speakers.

* Items: {len(records)} (recordings: {n_audio}, {hours:.2f} h of audio)
* Languages: {", ".join(langs) or "n/a"}
* Licenses: {", ".join(licenses) if records else "CC-BY-4.0"}. Each row carries its own `license`
  field; CC-BY-4.0 rows require attribution to the `contributor` named in the row.
* Only items approved by at least two community reviewers are included.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="data.jsonl", split="train")
```

Audio paths in the `audio` column are relative to this folder (16 kHz mono FLAC).

## Fields

`id, kind, src_lang, tgt_lang, source_text, target_text, original_mt, domain, dialect, region,
accent, gender, age_range, contributor, license, created_at, audio, audio_duration_s`

## Consent & use

Contributors gave explicit consent for their text and recordings to be used to train and
evaluate language technology. Voice recordings must **not** be used for voice cloning or
speaker identification. Withdrawn contributors' recordings are removed from future exports.

## Credits

Thanks to: {credit}.
"""


def open_contribution_store(tm: Any | None = None, path: str | Path | None = None) -> ContributionStore:
    """Open (creating if needed) the contribution store (default ``settings.contributions_dir``)."""
    directory = Path(path) if path is not None else get_settings().contributions_dir
    return ContributionStore(directory, tm=tm)


# Keep dataclass import referenced for type checkers that inspect ReviewPolicy.
_ = asdict
