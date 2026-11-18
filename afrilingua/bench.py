"""Benchmarking script."""
from __future__ import annotations

import time


def run_benchmark(engine, langs=None) -> dict:
    t0 = time.perf_counter()
    # Simple mock benchmark
    time.sleep(0.5)
    return {"mt_latency_ms": 100, "asr_rtf": 0.5}
