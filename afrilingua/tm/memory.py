"""SQLite-backed translation memory with a fast vector index.

Storage layout (``<dir>`` defaults to ``settings.data_dir / "tm"``)::

    <dir>/tm.sqlite                      entries table (WAL mode)
    <dir>/index/<src>__<tgt>/            char n-gram index per language pair
    <dir>/semantic/<src>__<tgt>/         optional dense (LaBSE) index per pair

Lookup path for ``search(query, src, tgt, k)``:

1. **Exact**: the normalised key (:func:`~afrilingua.tm.text.tm_key`) is looked up
   through the unique ``(src_lang, tgt_lang, source_norm, target)`` index:
   sub-millisecond, score 1.0, approved entries first. Short-circuits.
2. **Fuzzy**: the char n-gram index proposes ``max(4k, 16)`` candidates (one sparse
   mat-vec), which are reranked with rapidfuzz (``ratio`` / ``token_set_ratio``
   blend). Non-exact matches are capped at 0.99 so only a true normalised match
   can reach 1.0.
3. **Semantic** (optional): when an embedder is configured, its candidates are
   merged and the final score is ``0.75 * fuzzy + 0.25 * cosine``.

Indexes are loaded lazily per pair, reconciled against SQLite (missing ids added,
stale ids removed) and rebuilt from SQLite if the files are corrupt, so SQLite is
always the source of truth. Indexes are persisted on :meth:`TranslationMemory.flush`
/ :meth:`~TranslationMemory.close`, after imports and after a rebuild.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afrilingua.core.errors import InvalidInputError, NotFoundError, UnsupportedLanguageError
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import get_settings
from afrilingua.engine import TMMatch, normalize_text
from afrilingua.tm.index import CharNgramIndex, DenseIndex, IndexCorruptError
from afrilingua.tm.io import iter_pairs
from afrilingua.tm.text import base_form, similarity, tm_key
from afrilingua.tm.vectorizer import DEFAULT_DIM

logger = logging.getLogger(__name__)

__all__ = ["MAX_TEXT_CHARS", "TranslationMemory", "open_translation_memory"]

MAX_TEXT_CHARS = 2000
NON_EXACT_CAP = 0.99
SEMANTIC_WEIGHT = 0.25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          TEXT PRIMARY KEY,
    src_lang    TEXT NOT NULL,
    tgt_lang    TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_norm TEXT NOT NULL,
    target      TEXT NOT NULL,
    approved    INTEGER NOT NULL DEFAULT 0,
    origin      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    uses        INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_entries_dedupe ON entries (src_lang, tgt_lang, source_norm, target);
CREATE INDEX IF NOT EXISTS ix_entries_approved ON entries (approved);
"""

_COLUMNS = (
    "id, src_lang, tgt_lang, source, source_norm, target, approved, origin, created_at, updated_at, uses"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["approved"] = bool(d["approved"])
    return d


@dataclass(frozen=True)
class _Prepared:
    source: str
    target: str
    key: str


class TranslationMemory:
    """Implements :class:`afrilingua.engine.TranslationMemoryProtocol` (see module docs)."""

    def __init__(self, path: Path, *, embedder: Any | None = None, dim: int = DEFAULT_DIM) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "tm.sqlite"
        self.dim = dim
        self.embedder = embedder
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._indexes: dict[tuple[str, str], CharNgramIndex] = {}
        self._semantic: dict[tuple[str, str], DenseIndex] = {}
        self._closed = False
        #: Bumped on every write (add, approve, delete). Caches built on top of the TM
        #: (the engine's translation cache) compare it to know when to drop stale entries.
        self.generation = 0
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── connections ─────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA cache_size=-32000")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA temp_store=MEMORY")
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise RuntimeError("TranslationMemory is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
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

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _langs(src_lang: str, tgt_lang: str) -> tuple[str, str]:
        reg = get_registry()
        return reg.normalize(src_lang), reg.normalize(tgt_lang)

    @staticmethod
    def _prepare(source: str, target: str) -> _Prepared:
        if not isinstance(source, str) or not isinstance(target, str):
            raise InvalidInputError("source and target must be strings")
        src = normalize_text(source)
        tgt = normalize_text(target)
        if not src or not tgt:
            raise InvalidInputError("source and target must be non-empty")
        if len(src) > MAX_TEXT_CHARS or len(tgt) > MAX_TEXT_CHARS:
            raise InvalidInputError(f"source/target must be at most {MAX_TEXT_CHARS} characters")
        key = tm_key(src)
        if not key:
            raise InvalidInputError("source has no letters or digits")
        return _Prepared(src, tgt, key)

    def _index_dir(self, src: str, tgt: str) -> Path:
        return self.path / "index" / f"{src}__{tgt}"

    def _semantic_dir(self, src: str, tgt: str) -> Path:
        return self.path / "semantic" / f"{src}__{tgt}"

    # ── index management ────────────────────────────────────────────────────
    def _pair_index(self, src: str, tgt: str) -> CharNgramIndex:
        pair = (src, tgt)
        index = self._indexes.get(pair)
        if index is not None:
            return index
        with self._index_lock:
            index = self._indexes.get(pair)
            if index is None:
                index = self._load_or_build(src, tgt)
                self._indexes[pair] = index
            return index

    def _pair_rows(self, src: str, tgt: str, ids: Iterable[str] | None = None) -> list[tuple[str, str]]:
        with self._conn() as conn:
            if ids is None:
                cur = conn.execute(
                    "SELECT id, source_norm FROM entries WHERE src_lang=? AND tgt_lang=?", (src, tgt)
                )
                return [(r[0], r[1]) for r in cur]
            out: list[tuple[str, str]] = []
            id_list = list(ids)
            for i in range(0, len(id_list), 500):
                chunk = id_list[i : i + 500]
                q = f"SELECT id, source_norm FROM entries WHERE id IN ({','.join('?' * len(chunk))})"
                out.extend((r[0], r[1]) for r in conn.execute(q, chunk))
            return out

    def _load_or_build(self, src: str, tgt: str) -> CharNgramIndex:
        directory = self._index_dir(src, tgt)
        index: CharNgramIndex | None = None
        if (directory / "meta.json").exists():
            try:
                index = CharNgramIndex.load(directory)
                if index.dim != self.dim:
                    raise IndexCorruptError("dimension changed")
            except IndexCorruptError as exc:
                logger.warning("TM index %s→%s is unusable (%s); rebuilding from SQLite", src, tgt, exc)
                index = None
        if index is None:
            t0 = time.perf_counter()
            rows = self._pair_rows(src, tgt)
            index = CharNgramIndex(self.dim)
            index.add([r[0] for r in rows], [base_form(r[1]) for r in rows])
            if rows:
                logger.info(
                    "Built TM index %s→%s: %d entries in %.0f ms",
                    src,
                    tgt,
                    len(rows),
                    (time.perf_counter() - t0) * 1000,
                )
                self._safe_save(index, directory)
            return index
        self._reconcile(index, src, tgt)
        return index

    def _reconcile(self, index: CharNgramIndex | DenseIndex, src: str, tgt: str) -> None:
        with self._conn() as conn:
            db_ids = {
                r[0]
                for r in conn.execute("SELECT id FROM entries WHERE src_lang=? AND tgt_lang=?", (src, tgt))
            }
        have = index.ids()
        for stale in have - db_ids:
            index.remove(stale)
        missing = db_ids - have
        if missing:
            rows = self._pair_rows(src, tgt, missing)
            index.add(
                [r[0] for r in rows],
                [base_form(r[1]) if isinstance(index, CharNgramIndex) else r[1] for r in rows],
            )
        if missing or (have - db_ids):
            logger.info("Reconciled TM index %s→%s (+%d / -%d)", src, tgt, len(missing), len(have - db_ids))

    def _semantic_index(self, src: str, tgt: str) -> DenseIndex | None:
        if self.embedder is None:
            return None
        pair = (src, tgt)
        idx = self._semantic.get(pair)
        if idx is not None:
            return idx
        with self._index_lock:
            idx = self._semantic.get(pair)
            if idx is None:
                directory = self._semantic_dir(src, tgt)
                try:
                    idx = (
                        DenseIndex.load(directory, self.embedder)
                        if (directory / "meta.json").exists()
                        else None
                    )
                except IndexCorruptError as exc:
                    logger.warning("Semantic TM index unusable (%s); rebuilding", exc)
                    idx = None
                if idx is None:
                    idx = DenseIndex(self.embedder)
                self._reconcile(idx, src, tgt)
                self._semantic[pair] = idx
            return idx

    @staticmethod
    def _safe_save(index: CharNgramIndex | DenseIndex, directory: Path) -> None:
        try:
            index.save(directory)
        except OSError:
            logger.exception("Could not persist TM index to %s", directory)

    def rebuild_index(self, src_lang: str | None = None, tgt_lang: str | None = None) -> int:
        """Drop and rebuild char n-gram indexes (all pairs, or one pair) from SQLite."""
        with self._conn() as conn:
            pairs = [(r[0], r[1]) for r in conn.execute("SELECT DISTINCT src_lang, tgt_lang FROM entries")]
        if src_lang and tgt_lang:
            pairs = [self._langs(src_lang, tgt_lang)]
        with self._index_lock:
            for src, tgt in pairs:
                self._indexes.pop((src, tgt), None)
                shutil.rmtree(self._index_dir(src, tgt), ignore_errors=True)
                self._pair_index(src, tgt)
        return len(pairs)

    # ── protocol: add / lookup ──────────────────────────────────────────────
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
        """Insert a pair; re-adding an existing pair bumps ``uses`` and can upgrade ``approved``."""
        src, tgt = self._langs(src_lang, tgt_lang)
        p = self._prepare(source, target)
        ids = self._upsert(src, tgt, [p], origin=origin, approved=approved)
        return ids[0]

    def add_many(
        self,
        pairs: Iterable[tuple[str, str]],
        src_lang: str,
        tgt_lang: str,
        *,
        origin: str = "import",
        approved: bool = False,
        skip_invalid: bool = True,
    ) -> int:
        """Bulk insert in one transaction; returns the number of pairs stored/updated."""
        src, tgt = self._langs(src_lang, tgt_lang)
        prepared: list[_Prepared] = []
        for source, target in pairs:
            try:
                prepared.append(self._prepare(source, target))
            except InvalidInputError:
                if not skip_invalid:
                    raise
        if not prepared:
            return 0
        return len(self._upsert(src, tgt, prepared, origin=origin, approved=approved))

    def _upsert(
        self, src: str, tgt: str, items: list[_Prepared], *, origin: str, approved: bool
    ) -> list[str]:
        index = self._pair_index(src, tgt)
        sem = self._semantic_index(src, tgt)
        now = _now()
        sql = (
            f"INSERT INTO entries ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,1) "
            "ON CONFLICT(src_lang, tgt_lang, source_norm, target) DO UPDATE SET "
            "uses = uses + 1, updated_at = excluded.updated_at, "
            "origin = CASE WHEN excluded.approved > approved THEN excluded.origin ELSE origin END, "
            "approved = MAX(approved, excluded.approved)"
        )
        gen = [uuid.uuid4().hex for _ in items]
        params = [
            (g, src, tgt, p.source, p.key, p.target, int(approved), origin, now, now)
            for g, p in zip(gen, items, strict=True)
        ]
        with self._tx() as conn:
            conn.executemany(sql, params)
            # Resolve the stored id of every item (new uuid, or the existing row on conflict).
            find = "SELECT id FROM entries WHERE src_lang=? AND tgt_lang=? AND source_norm=? AND target=?"
            ids = [conn.execute(find, (src, tgt, p.key, p.target)).fetchone()[0] for p in items]
        self.generation += 1
        gen_set = set(gen)
        new_pairs = {i: p.key for i, p in zip(ids, items, strict=True) if i in gen_set}
        if new_pairs:
            new_ids = list(new_pairs)
            index.add(new_ids, [base_form(new_pairs[i]) for i in new_ids])
            if sem is not None:
                sem.add(new_ids, [new_pairs[i] for i in new_ids])
        return ids

    def lookup(self, text: str, src_lang: str, tgt_lang: str, *, k: int = 3) -> list[TMMatch]:
        """Protocol method used by the engine: best ``k`` matches, best first."""
        return self.search(text, src_lang, tgt_lang, k=k)[0]

    def search(
        self, query: str, src_lang: str, tgt_lang: str, k: int = 5, *, min_score: float = 0.3
    ) -> tuple[list[TMMatch], float]:
        """Return ``(matches, took_ms)``; see module docstring for the algorithm."""
        t0 = time.perf_counter()
        src, tgt = self._langs(src_lang, tgt_lang)
        k = max(1, min(int(k), 100))
        key = tm_key(query or "")
        if not key:
            return [], (time.perf_counter() - t0) * 1000
        with self._conn() as conn:
            exact = conn.execute(
                f"SELECT {_COLUMNS} FROM entries WHERE src_lang=? AND tgt_lang=? AND source_norm=? "
                "ORDER BY approved DESC, uses DESC, updated_at DESC LIMIT ?",
                (src, tgt, key, k),
            ).fetchall()
        if exact:
            matches = [
                TMMatch(r["id"], r["source"], r["target"], 1.0, bool(r["approved"]), r["origin"])
                for r in exact
            ]
            return matches, (time.perf_counter() - t0) * 1000

        index = self._pair_index(src, tgt)
        n_cand = max(4 * k, 16)
        cands = dict(index.search(base_form(key), n_cand))
        sem = self._semantic_index(src, tgt)
        sem_scores: dict[str, float] = {}
        if sem is not None:
            qv = sem.vector_for_query(key)
            for entry_id, score in sem.search_vector(qv, n_cand):
                cands.setdefault(entry_id, 0.0)
                sem_scores[entry_id] = score
            missing = [i for i in cands if i not in sem_scores]
            sem_scores.update(sem.similarities(qv, missing))
        if not cands:
            return [], (time.perf_counter() - t0) * 1000
        rows = self._fetch(list(cands))
        scored: list[tuple[float, bool, int, sqlite3.Row]] = []
        for r in rows:
            score = min(similarity(key, r["source_norm"]), NON_EXACT_CAP)
            if sem is not None:
                score = (1 - SEMANTIC_WEIGHT) * score + SEMANTIC_WEIGHT * max(
                    0.0, sem_scores.get(r["id"], 0.0)
                )
            if score >= min_score:
                scored.append((score, bool(r["approved"]), int(r["uses"]), r))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        matches = [
            TMMatch(r["id"], r["source"], r["target"], round(float(s), 4), appr, r["origin"])
            for s, appr, _u, r in scored[:k]
        ]
        return matches, (time.perf_counter() - t0) * 1000

    def _fetch(self, ids: list[str]) -> list[sqlite3.Row]:
        if not ids:
            return []
        with self._conn() as conn:
            q = f"SELECT {_COLUMNS} FROM entries WHERE id IN ({','.join('?' * len(ids))})"
            return conn.execute(q, ids).fetchall()

    # ── management ──────────────────────────────────────────────────────────
    def get(self, entry_id: str) -> dict[str, Any] | None:
        rows = self._fetch([entry_id])
        return _row_to_dict(rows[0]) if rows else None

    def approve(self, entry_id: str) -> dict[str, Any]:
        """Mark an entry approved. Raises :class:`NotFoundError` for unknown ids."""
        with self._tx() as conn:
            cur = conn.execute("UPDATE entries SET approved=1, updated_at=? WHERE id=?", (_now(), entry_id))
            if cur.rowcount == 0:
                raise NotFoundError(f"Translation memory entry {entry_id!r} not found")
        self.generation += 1
        entry = self.get(entry_id)
        assert entry is not None
        return entry

    def delete(self, entry_id: str) -> bool:
        """Delete an entry. Raises :class:`NotFoundError` for unknown ids."""
        with self._tx() as conn:
            row = conn.execute("SELECT src_lang, tgt_lang FROM entries WHERE id=?", (entry_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Translation memory entry {entry_id!r} not found")
            conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        self.generation += 1
        pair = (row[0], row[1])
        with self._index_lock:
            if pair in self._indexes:
                self._indexes[pair].remove(entry_id)
            if pair in self._semantic:
                self._semantic[pair].remove(entry_id)
        return True

    def delete_by_origin(self, origin: str) -> int:
        """Delete every entry added under ``origin`` (e.g. ``corpus:<id>``) and refresh indexes."""
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, src_lang, tgt_lang FROM entries WHERE origin=?", (origin,)
            ).fetchall()
            if not rows:
                return 0
            conn.execute("DELETE FROM entries WHERE origin=?", (origin,))
        self.generation += 1
        by_pair: dict[tuple[str, str], list[str]] = {}
        for r in rows:
            by_pair.setdefault((r[1], r[2]), []).append(r[0])
        with self._index_lock:
            for pair, ids in by_pair.items():
                if pair in self._indexes:
                    for eid in ids:
                        self._indexes[pair].remove(eid)
                if pair in self._semantic:
                    for eid in ids:
                        self._semantic[pair].remove(eid)
        self.flush()
        return len(rows)

    def stats(self) -> dict[str, Any]:
        """``{"entries", "approved", "by_pair": {"english→yoruba": n}}`` (contract §4)."""
        with self._conn() as conn:
            total, approved = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(approved), 0) FROM entries"
            ).fetchone()
            by_pair = {
                f"{r[0]}→{r[1]}": int(r[2])
                for r in conn.execute(
                    "SELECT src_lang, tgt_lang, COUNT(*) FROM entries GROUP BY src_lang, tgt_lang ORDER BY 3 DESC"
                )
            }
        return {"entries": int(total), "approved": int(approved), "by_pair": by_pair}

    def iter_entries(
        self, *, approved_only: bool = False, src_lang: str | None = None, tgt_lang: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Iterate entries (used by export and the training data builder)."""
        sql = f"SELECT {_COLUMNS} FROM entries WHERE 1=1"
        args: list[Any] = []
        if approved_only:
            sql += " AND approved=1"
        if src_lang:
            sql += " AND src_lang=?"
            args.append(get_registry().normalize(src_lang))
        if tgt_lang:
            sql += " AND tgt_lang=?"
            args.append(get_registry().normalize(tgt_lang))
        sql += " ORDER BY created_at, id"
        with self._conn() as conn:
            for row in conn.execute(sql, args).fetchall():
                yield _row_to_dict(row)

    # ── import / export ─────────────────────────────────────────────────────
    def import_file(
        self,
        path: str | Path,
        src_lang: str,
        tgt_lang: str,
        fmt: str = "auto",
        *,
        origin: str = "import",
        approved: bool = False,
        batch_size: int = 5000,
    ) -> int:
        """Import TSV/CSV (2 columns), JSONL (``source``/``target``) or TMX. Returns pairs imported."""
        src, tgt = self._langs(src_lang, tgt_lang)
        reg = get_registry()

        def resolve(code: str) -> str | None:
            for candidate in (code, code.replace("-", "_"), code.split("-")[0].split("_")[0]):
                try:
                    return reg.normalize(candidate)
                except UnsupportedLanguageError:
                    continue
            return None

        aliases = {
            name: {a.lower() for a in (*reg.get(name).aliases, reg.get(name).nllb_code)}
            for name in (src, tgt)
        }
        total = 0
        batch: list[tuple[str, str]] = []
        for pair in iter_pairs(Path(path), src, tgt, fmt=fmt, resolve_lang=resolve, lang_aliases=aliases):
            batch.append(pair)
            if len(batch) >= batch_size:
                total += self.add_many(batch, src, tgt, origin=origin, approved=approved)
                batch = []
        if batch:
            total += self.add_many(batch, src, tgt, origin=origin, approved=approved)
        self.flush()
        self.checkpoint()
        return total

    def checkpoint(self) -> None:
        """Fold the WAL back into the main database file (after bulk writes)."""
        with self._write_lock, self._conn() as conn:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
            except sqlite3.Error:  # pragma: no cover - busy readers; harmless
                logger.debug("WAL checkpoint skipped", exc_info=True)

    def export_jsonl(self, path: str | Path, approved_only: bool = True) -> int:
        """Write entries as JSON lines; returns the number written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for entry in self.iter_entries(approved_only=approved_only):
                entry.pop("source_norm", None)
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1
        return n

    # ── lifecycle ───────────────────────────────────────────────────────────
    def flush(self) -> None:
        """Persist dirty indexes to disk."""
        with self._index_lock:
            for (src, tgt), index in list(self._indexes.items()):
                if index.dirty:
                    self._safe_save(index, self._index_dir(src, tgt))
            for (src, tgt), sidx in list(self._semantic.items()):
                if sidx.dirty:
                    self._safe_save(sidx, self._semantic_dir(src, tgt))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            with self._conns_lock:
                for conn in self._conns:
                    with contextlib.suppress(sqlite3.Error):
                        conn.close()
                self._conns.clear()

    def __enter__(self) -> TranslationMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_translation_memory(path: str | Path | None = None) -> TranslationMemory:
    """Open (creating if needed) the translation memory at ``path``.

    Default location: ``settings.data_dir / "tm"``. If ``settings.model_embedder`` is
    set, semantic blending with that model is enabled (loaded lazily).
    """
    settings = get_settings()
    directory = Path(path) if path is not None else settings.data_dir / "tm"
    embedder = None
    if settings.model_embedder.strip():
        from afrilingua.tm.embedder import TransformersEmbedder

        embedder = TransformersEmbedder(settings.model_embedder.strip())
    return TranslationMemory(directory, embedder=embedder)
