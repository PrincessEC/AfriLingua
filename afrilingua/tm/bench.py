"""Translation-memory micro-benchmark.

    python -m afrilingua.tm.bench --n 100000

Builds a synthetic English→Hausa TM from random sentences over the vocabulary of
the bundled prompt corpus, then measures exact and fuzzy (perturbed query) lookup
latency and fuzzy recall@1. Results are recorded in ``docs/benchmarks-tm.md``.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import tempfile
import time
from pathlib import Path

from afrilingua.contrib.prompts import DATA_DIR
from afrilingua.tm.memory import TranslationMemory


def _perturb(rng: random.Random, sentence: str, vocab: list[str]) -> str:
    words = sentence.rstrip(".").split()
    i = rng.randrange(len(words))
    op = rng.random()
    if op < 0.33 and len(words) > 4:
        del words[i]
    elif op < 0.66:
        words[i] = words[i][:-1] if len(words[i]) > 3 else words[i] + "s"
    else:
        words.insert(i, rng.choice(vocab))
    return " ".join(words)


def _pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def run(n: int, queries: int = 300, seed: int = 1) -> dict[str, float]:
    """Run the benchmark and return the measured numbers."""
    text = (DATA_DIR / "prompts_en.txt").read_text(encoding="utf-8")
    vocab = sorted({w.lower() for w in re.findall(r"[A-Za-z']+", text)})
    rng = random.Random(seed)
    sentences = list({" ".join(rng.choices(vocab, k=rng.randint(4, 14))): None for _ in range(n)})
    pairs = [(s.capitalize() + ".", f"target {i}") for i, s in enumerate(sentences)]
    with tempfile.TemporaryDirectory() as tmp, TranslationMemory(Path(tmp) / "tm") as tm:
        t0 = time.perf_counter()
        tm.add_many(pairs, "english", "hausa")
        tm.flush()
        tm.checkpoint()
        build_s = time.perf_counter() - t0
        sample = rng.sample(pairs, min(queries, len(pairs)))
        qs = [(_perturb(rng, s, vocab), s) for s, _ in sample]
        for q, _ in qs[:20]:  # warm-up (page cache, rapidfuzz import)
            tm.search(q, "english", "hausa")
        fuzzy, exact, hits = [], [], 0
        for q, orig in qs:
            t = time.perf_counter()
            matches, _ = tm.search(q, "english", "hausa")
            fuzzy.append((time.perf_counter() - t) * 1000)
            hits += bool(matches) and matches[0].source == orig
        for s, _ in sample:
            t = time.perf_counter()
            tm.search(s, "english", "hausa")
            exact.append((time.perf_counter() - t) * 1000)
        nnz = tm._pair_index("english", "hausa").nnz
    with tempfile.TemporaryDirectory() as tmp:
        with TranslationMemory(Path(tmp) / "tm") as tm:
            tm.add_many(pairs, "english", "hausa")
        t0 = time.perf_counter()
        with TranslationMemory(Path(tmp) / "tm") as tm:
            tm.search("market price today", "english", "hausa")
            reopen_s = time.perf_counter() - t0
    return {
        "entries": len(pairs),
        "build_s": round(build_s, 2),
        "index_nnz": nnz,
        "exact_p50_ms": round(statistics.median(exact), 3),
        "exact_p95_ms": round(_pct(exact, 0.95), 3),
        "fuzzy_p50_ms": round(statistics.median(fuzzy), 2),
        "fuzzy_p95_ms": round(_pct(fuzzy, 0.95), 2),
        "fuzzy_recall_at_1": round(hits / len(qs), 3),
        "reopen_first_query_s": round(reopen_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--queries", type=int, default=300)
    args = parser.parse_args()
    for n in args.n:
        print(run(n, args.queries), flush=True)


if __name__ == "__main__":
    main()
