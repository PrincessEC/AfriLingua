"""Hashed character n-gram vectors (language-agnostic, no model download).

A text is padded with spaces, split into character 2-, 3- and 4-grams, and each
n-gram is hashed with a *stable* 64-bit mix (independent of ``PYTHONHASHSEED``)
into one of ``dim`` buckets. Counts get sublinear tf weighting ``1 + ln(tf)`` and
each vector is L2-normalised, so a dot product is a cosine similarity.

Everything is vectorised with numpy: a whole batch of texts is encoded into one
UTF-32 code-point array and hashed in a handful of array operations, so building an
index for 100k sentences takes seconds, and a single query takes ~100 µs.
Vectors are returned in CSR form ``(indptr, dims, vals)``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["DEFAULT_DIM", "NGRAM_RANGE", "vectorize_many", "vectorize_one"]

DEFAULT_DIM = 2048
NGRAM_RANGE = (2, 4)

_SEED_INT = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1
_PRIME = np.uint64(0x100000001B3)
_MIX = np.uint64(0xFF51AFD7ED558CCD)
_S33 = np.uint64(33)


def _hash_windows(cp: np.ndarray, n: int) -> np.ndarray:
    """Stable 64-bit hash of every length-``n`` window of ``cp`` (uint64 array)."""
    length = cp.shape[0] - n + 1
    h = np.full(length, np.uint64((_SEED_INT * n) & _MASK64), dtype=np.uint64)
    for j in range(n):
        h ^= cp[j : j + length]
        h *= _PRIME
    h ^= h >> _S33
    h *= _MIX
    h ^= h >> _S33
    return h


def vectorize_many(
    texts: Sequence[str], dim: int = DEFAULT_DIM, ngram_range: tuple[int, int] = NGRAM_RANGE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise ``texts`` into CSR arrays.

    Returns ``(indptr int64[len+1], dims uint16[nnz], vals float32[nnz])``; within a
    row, ``dims`` are sorted ascending and unique.
    """
    if dim > 65_536:
        raise ValueError("dim must fit in uint16")
    n_docs = len(texts)
    if n_docs == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.float32)
    # NUL separates documents; texts never contain it after normalisation.
    joined = "\x00".join(f" {t.replace(chr(0), ' ')} " for t in texts)
    cp = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    sep = cp == 0
    csum = np.concatenate(([0], np.cumsum(sep, dtype=np.int64)))
    doc_of_pos = csum[1:]  # number of separators up to and including position i

    keys: list[np.ndarray] = []
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        length = cp.shape[0] - n + 1
        if length <= 0:
            continue
        valid = (csum[n : n + length] - csum[:length]) == 0
        h = _hash_windows(cp, n)[valid]
        d = (h % np.uint64(dim)).astype(np.int64)
        keys.append(doc_of_pos[:length][valid] * dim + d)
    all_keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    uniq, counts = np.unique(all_keys, return_counts=True)
    doc_ids = uniq // dim
    dims = (uniq % dim).astype(np.uint16)
    vals = (1.0 + np.log(counts)).astype(np.float32)
    norms = np.sqrt(np.bincount(doc_ids, weights=vals.astype(np.float64) ** 2, minlength=n_docs))
    norms[norms == 0] = 1.0
    vals /= norms[doc_ids].astype(np.float32)
    indptr = np.searchsorted(doc_ids, np.arange(n_docs + 1, dtype=np.int64)).astype(np.int64)
    return indptr, dims, vals


def vectorize_one(text: str, dim: int = DEFAULT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Vectorise a single text; returns ``(dims, vals)`` (sparse, sorted by dim)."""
    _, dims, vals = vectorize_many([text], dim)
    return dims, vals
