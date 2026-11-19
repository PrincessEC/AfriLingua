"""Vector indexes for translation-memory candidate generation.

:class:`VectorIndex` is the small interface the :class:`~afrilingua.tm.memory.TranslationMemory`
talks to, so FAISS / hnswlib / usearch backends can be plugged in later without
touching the store. Two implementations ship:

:class:`CharNgramIndex` (default)
    Hashed char n-gram vectors (:mod:`afrilingua.tm.vectorizer`) in a sparse
    *column-major* (inverted, CSC) matrix plus a small dense "delta" matrix for
    recent additions. A query is a sparse matrix-vector product: the query's
    non-zero buckets are visited rarest-first and their postings are accumulated
    with ``np.bincount`` until a posting budget is reached (very common buckets
    such as ``" th"`` carry little signal and are skipped on large indexes). The
    delta matrix gets one dense mat-vec. Deletions are tombstones; the structure is
    compacted (delta + tombstones folded into the CSC arrays, with an O(nnz) radix
    sort on the uint16 bucket ids) when the delta is full or 25% of rows are dead.

    Why not one dense ``(N, 2048)`` float32 matrix? At 100k entries that is 820 MB
    and ~70 ms per mat-vec on a laptop CPU; the sparse layout is ~80 MB and ~1-2 ms.

:class:`DenseIndex`
    Brute-force cosine search over L2-normalised dense embeddings (optional semantic
    mode, e.g. LaBSE).

Both persist to a directory (``.npy`` arrays + ``ids.json`` + ``meta.json``) with
atomic replace, and raise on any inconsistency when loading so the caller can
rebuild from SQLite.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from afrilingua.tm.vectorizer import DEFAULT_DIM, vectorize_many, vectorize_one

__all__ = ["CharNgramIndex", "DenseIndex", "IndexCorruptError", "VectorIndex"]

_FORMAT_VERSION = 1


class IndexCorruptError(RuntimeError):
    """A persisted index is missing, truncated or inconsistent."""


@runtime_checkable
class VectorIndex(Protocol):
    """Approximate nearest-neighbour index over entry ids."""

    dirty: bool

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None: ...

    def remove(self, entry_id: str) -> bool: ...

    def search(self, text: str, n: int) -> list[tuple[str, float]]: ...

    def ids(self) -> set[str]: ...

    def save(self, directory: Path) -> None: ...

    def __len__(self) -> int: ...


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as fh:
        np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:  # missing, truncated, wrong format
        raise IndexCorruptError(f"cannot read {path.name}: {exc}") from exc


def _load_meta(directory: Path, kind: str) -> tuple[dict, list[str]]:
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise IndexCorruptError(f"cannot read index metadata: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("kind") != kind or meta.get("version") != _FORMAT_VERSION:
        raise IndexCorruptError("index metadata has an unexpected kind/version")
    if not isinstance(ids, list) or len(ids) != meta.get("n") or not all(isinstance(i, str) for i in ids):
        raise IndexCorruptError("index id list does not match metadata")
    return meta, ids


class CharNgramIndex:
    """Sparse hashed char n-gram index (see module docstring)."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        *,
        delta_capacity: int = 1024,
        posting_budget: int = 20_000,
    ) -> None:
        self.dim = dim
        self.delta_capacity = delta_capacity
        self.posting_budget = posting_budget
        self.dirty = False
        self._lock = threading.RLock()
        # main (CSC) part
        self._colptr = np.zeros(dim + 1, dtype=np.int64)
        self._crow = np.zeros(0, dtype=np.int32)
        self._cval = np.zeros(0, dtype=np.float32)
        self._n_main = 0
        # dense delta part (rows n_main .. n_main + n_delta - 1)
        self._delta = np.zeros((0, dim), dtype=np.float32)
        self._n_delta = 0
        # row bookkeeping
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._alive = np.zeros(0, dtype=bool)
        self._dead = 0

    # ── introspection ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._row)

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._row)

    @property
    def nnz(self) -> int:
        return int(self._crow.shape[0])

    # ── mutation ────────────────────────────────────────────────────────────
    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """Add (or replace) entries. ``texts`` are the strings to vectorise."""
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if not ids:
            return
        with self._lock:
            for entry_id in ids:
                if entry_id in self._row:
                    self._kill(entry_id)
            # de-duplicate within the batch (last one wins)
            seen: dict[str, int] = {}
            for i, entry_id in enumerate(ids):
                seen[entry_id] = i
            order = sorted(seen.values())
            ids = [ids[i] for i in order]
            texts = [texts[i] for i in order]
            indptr, dims, vals = vectorize_many(texts, self.dim)
            if self._n_delta + len(ids) <= self.delta_capacity:
                self._append_delta(ids, indptr, dims, vals)
            else:
                self._compact(extra=(list(ids), indptr, dims, vals))
            self.dirty = True

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id not in self._row:
                return False
            self._kill(entry_id)
            self.dirty = True
            if self._dead > 64 and self._dead * 4 > len(self._ids):
                self._compact()
            return True

    def _kill(self, entry_id: str) -> None:
        row = self._row.pop(entry_id)
        self._alive[row] = False
        self._dead += 1
        if row >= self._n_main:
            self._delta[row - self._n_main] = 0.0

    def _append_delta(
        self, ids: Sequence[str], indptr: np.ndarray, dims: np.ndarray, vals: np.ndarray
    ) -> None:
        if self._delta.shape[0] < self.delta_capacity:
            grown = np.zeros((self.delta_capacity, self.dim), dtype=np.float32)
            grown[: self._n_delta] = self._delta[: self._n_delta]
            self._delta = grown
        m = len(ids)
        local = np.repeat(np.arange(m), np.diff(indptr)) + self._n_delta
        self._delta[local, dims.astype(np.int64)] = vals
        base = len(self._ids)
        for i, entry_id in enumerate(ids):
            self._row[entry_id] = base + i
        self._ids.extend(ids)
        self._alive = np.concatenate((self._alive, np.ones(m, dtype=bool)))
        self._n_delta += m

    def _compact(self, extra: tuple[list[str], np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
        """Fold delta rows, drop tombstones and optionally merge a new CSR batch."""
        total_old = len(self._ids)
        parts_rows: list[np.ndarray] = []
        parts_dims: list[np.ndarray] = []
        parts_vals: list[np.ndarray] = []
        if self._crow.shape[0]:
            parts_rows.append(self._crow.astype(np.int64))
            parts_dims.append(np.repeat(np.arange(self.dim, dtype=np.uint16), np.diff(self._colptr)))
            parts_vals.append(np.asarray(self._cval, dtype=np.float32))
        if self._n_delta:
            r, d = np.nonzero(self._delta[: self._n_delta])
            parts_rows.append(r.astype(np.int64) + self._n_main)
            parts_dims.append(d.astype(np.uint16))
            parts_vals.append(self._delta[r, d])
        new_ids = list(self._ids)
        alive_ext = self._alive
        if extra is not None:
            e_ids, e_indptr, e_dims, e_vals = extra
            parts_rows.append(np.repeat(np.arange(len(e_ids), dtype=np.int64), np.diff(e_indptr)) + total_old)
            parts_dims.append(e_dims.astype(np.uint16))
            parts_vals.append(e_vals.astype(np.float32))
            new_ids.extend(e_ids)
            alive_ext = np.concatenate((alive_ext, np.ones(len(e_ids), dtype=bool)))

        if parts_rows:
            rows = np.concatenate(parts_rows)
            dims = np.concatenate(parts_dims)
            vals = np.concatenate(parts_vals)
        else:
            rows = np.zeros(0, np.int64)
            dims = np.zeros(0, np.uint16)
            vals = np.zeros(0, np.float32)
        keep = alive_ext[rows] if rows.size else np.zeros(0, dtype=bool)
        rows, dims, vals = rows[keep], dims[keep], vals[keep]
        remap = np.cumsum(alive_ext, dtype=np.int64) - 1
        rows = remap[rows] if rows.size else rows
        order = np.argsort(dims, kind="stable")  # radix sort for uint16 keys
        dims_sorted = dims[order]
        self._crow = rows[order].astype(np.int32)
        self._cval = vals[order].astype(np.float32)
        counts = (
            np.bincount(dims_sorted, minlength=self.dim) if dims_sorted.size else np.zeros(self.dim, np.int64)
        )
        self._colptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._ids = [i for i, a in zip(new_ids, alive_ext.tolist(), strict=True) if a]
        self._row = {entry_id: r for r, entry_id in enumerate(self._ids)}
        self._n_main = len(self._ids)
        self._alive = np.ones(self._n_main, dtype=bool)
        self._dead = 0
        self._n_delta = 0
        self._delta = np.zeros((0, self.dim), dtype=np.float32)

    # ── search ──────────────────────────────────────────────────────────────
    def search(self, text: str, n: int) -> list[tuple[str, float]]:
        """Return up to ``n`` ``(id, cosine)`` candidates, best first."""
        qd, qv = vectorize_one(text, self.dim)
        with self._lock:
            total = len(self._ids)
            if total == 0 or n <= 0 or qd.size == 0:
                return []
            cand_rows: list[np.ndarray] = []
            cand_scores: list[np.ndarray] = []
            if self._n_main:
                qdi = qd.astype(np.int64)
                starts = self._colptr[qdi]
                lengths = self._colptr[qdi + 1] - starts
                order = np.argsort(lengths, kind="stable")
                cum = np.cumsum(lengths[order])
                # Small/medium indexes: use every bucket (exact cosine). Large: rarest-first budget.
                budget = int(cum[-1]) if cum[-1] <= 6 * self.posting_budget else self.posting_budget
                take = int(np.searchsorted(cum, budget, side="right"))
                take = max(take, min(len(order), 1))
                sel = order[:take]
                sel = sel[lengths[sel] > 0]
                if sel.size:
                    rows = np.concatenate([self._crow[starts[i] : starts[i] + lengths[i]] for i in sel])
                    vals = np.concatenate([self._cval[starts[i] : starts[i] + lengths[i]] for i in sel])
                    vals *= np.repeat(qv[sel], lengths[sel])
                    acc = np.bincount(rows, weights=vals, minlength=self._n_main)
                    # Pick candidates among the touched rows only (avoids an O(N) selection).
                    touched_scores = acc[rows]
                    pool = min(rows.shape[0], 8 * n)
                    if pool < rows.shape[0]:
                        part = np.argpartition(touched_scores, rows.shape[0] - pool)[rows.shape[0] - pool :]
                        touched = np.unique(rows[part])
                    else:
                        touched = np.unique(rows)
                    cand_rows.append(touched.astype(np.int64))
                    cand_scores.append(acc[touched])
            if self._n_delta:
                ds = self._delta[: self._n_delta][:, qd.astype(np.int64)] @ qv
                touched = np.flatnonzero(ds > 0)
                cand_rows.append(touched + self._n_main)
                cand_scores.append(ds[touched].astype(np.float64))
            if not cand_rows:
                return []
            rows_all = np.concatenate(cand_rows)
            scores = np.concatenate(cand_scores)
            if self._dead:
                live = self._alive[rows_all]
                rows_all, scores = rows_all[live], scores[live]
            m = rows_all.shape[0]
            if m == 0:
                return []
            n = min(n, m)
            top = np.argpartition(-scores, n - 1)[:n] if n < m else np.arange(m)
            top = top[np.argsort(-scores[top], kind="stable")]
            return [(self._ids[int(rows_all[i])], float(scores[i])) for i in top.tolist() if scores[i] > 0.0]

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._n_delta or self._dead:
                self._compact()
            _atomic_save_npy(directory / "colptr.npy", self._colptr)
            _atomic_save_npy(directory / "rows.npy", self._crow)
            _atomic_save_npy(directory / "vals.npy", self._cval)
            _atomic_write_text(directory / "ids.json", json.dumps(self._ids))
            meta = {
                "kind": "char-ngram",
                "version": _FORMAT_VERSION,
                "dim": self.dim,
                "n": len(self._ids),
                "nnz": int(self._crow.shape[0]),
            }
            _atomic_write_text(directory / "meta.json", json.dumps(meta))
            self.dirty = False

    @classmethod
    def load(cls, directory: Path, **kwargs: int) -> CharNgramIndex:
        meta, ids = _load_meta(directory, "char-ngram")
        index = cls(int(meta["dim"]), **kwargs)
        colptr = _load_npy(directory / "colptr.npy")
        rows = _load_npy(directory / "rows.npy")
        vals = _load_npy(directory / "vals.npy")
        n = len(ids)
        if (
            colptr.shape != (index.dim + 1,)
            or rows.shape != vals.shape
            or rows.ndim != 1
            or int(colptr[0]) != 0
            or int(colptr[-1]) != rows.shape[0]
            or np.any(np.diff(colptr) < 0)
            or (rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n))
            or not np.all(np.isfinite(vals))
            or len(set(ids)) != n
        ):
            raise IndexCorruptError("index arrays are inconsistent")
        index._colptr = colptr.astype(np.int64)
        index._crow = rows.astype(np.int32)
        index._cval = vals.astype(np.float32)
        index._ids = list(ids)
        index._row = {entry_id: r for r, entry_id in enumerate(ids)}
        index._n_main = n
        index._alive = np.ones(n, dtype=bool)
        return index


class DenseIndex:
    """Brute-force cosine index over normalised dense embeddings (semantic mode)."""

    def __init__(self, embed: Embed, dim: int | None = None) -> None:
        self._embed = embed
        self.dim = dim
        self.dirty = False
        self._lock = threading.RLock()
        self._mat = np.zeros((0, dim or 0), dtype=np.float32)
        self._n = 0
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._alive = np.zeros(0, dtype=bool)

    def __len__(self) -> int:
        return len(self._row)

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._row)

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if not ids:
            return
        vecs = np.asarray(self._embed(list(texts)), dtype=np.float32)
        with self._lock:
            if self.dim is None or self._mat.shape[1] == 0:
                self.dim = int(vecs.shape[1])
                self._mat = np.zeros((0, self.dim), dtype=np.float32)
            for entry_id in ids:
                self.remove(entry_id)
            need = self._n + len(ids)
            if need > self._mat.shape[0]:
                grown = np.zeros((max(need, 2 * self._mat.shape[0], 256), self.dim), dtype=np.float32)
                grown[: self._n] = self._mat[: self._n]
                self._mat = grown
            self._mat[self._n : need] = vecs
            for i, entry_id in enumerate(ids):
                self._row[entry_id] = self._n + i
            self._ids.extend(ids)
            self._alive = np.concatenate((self._alive, np.ones(len(ids), dtype=bool)))
            self._n = need
            self.dirty = True

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            row = self._row.pop(entry_id, None)
            if row is None:
                return False
            self._alive[row] = False
            self._mat[row] = 0.0
            self.dirty = True
            return True

    def vector_for_query(self, text: str) -> np.ndarray:
        return np.asarray(self._embed([text]), dtype=np.float32)[0]

    def similarities(self, query_vec: np.ndarray, ids: Sequence[str]) -> dict[str, float]:
        with self._lock:
            rows = [(i, self._row[i]) for i in ids if i in self._row]
            if not rows:
                return {}
            mat = self._mat[[r for _, r in rows]]
            sims = mat @ query_vec
            return {i: float(s) for (i, _), s in zip(rows, sims, strict=True)}

    def search(self, text: str, n: int) -> list[tuple[str, float]]:
        return self.search_vector(self.vector_for_query(text), n)

    def search_vector(self, q: np.ndarray, n: int) -> list[tuple[str, float]]:
        with self._lock:
            if not self._row or n <= 0:
                return []
            scores = self._mat[: self._n] @ q
            scores[~self._alive] = -2.0
            n = min(n, self._n)
            top = np.argpartition(-scores, n - 1)[:n] if n < self._n else np.arange(self._n)
            top = top[np.argsort(-scores[top], kind="stable")]
            return [(self._ids[r], float(scores[r])) for r in top.tolist() if self._alive[r]]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            keep = np.flatnonzero(self._alive)
            mat = self._mat[keep]
            ids = [self._ids[r] for r in keep.tolist()]
            _atomic_save_npy(directory / "vectors.npy", mat)
            _atomic_write_text(directory / "ids.json", json.dumps(ids))
            meta = {"kind": "dense", "version": _FORMAT_VERSION, "dim": self.dim or 0, "n": len(ids)}
            _atomic_write_text(directory / "meta.json", json.dumps(meta))
            self.dirty = False

    @classmethod
    def load(cls, directory: Path, embed: Embed) -> DenseIndex:
        meta, ids = _load_meta(directory, "dense")
        mat = _load_npy(directory / "vectors.npy")
        if mat.ndim != 2 or mat.shape[0] != len(ids) or (len(ids) and mat.shape[1] != meta.get("dim")):
            raise IndexCorruptError("dense index arrays are inconsistent")
        index = cls(embed, int(meta.get("dim") or 0) or None)
        index._mat = mat.astype(np.float32)
        index._n = len(ids)
        index._ids = list(ids)
        index._row = {entry_id: r for r, entry_id in enumerate(ids)}
        index._alive = np.ones(len(ids), dtype=bool)
        return index


class Embed(Protocol):
    """Callable turning a list of texts into an ``(n, d)`` array of normalised vectors."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...
