"""Reproducible latency, throughput and memory benchmark harness (``afrilingua bench``)."""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from afrilingua.core.hardware import detect_hardware, resolve_runtime
from afrilingua.engine import AfriLinguaEngine

_SAMPLE_SENTENCES: tuple[str, ...] = (
    "Good morning everyone, and welcome to today's session.",
    "Please take your seats so we can begin on time.",
    "Where is the nearest community health clinic?",
    "Clean water and reliable power matter to every household.",
    "Thank you for your patience and clear questions.",
)


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(samples_ms)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


def bench_translation_memory(n_entries: int = 5_000, n_queries: int = 100) -> dict[str, Any]:
    """Measure exact and fuzzy lookup latency on an isolated ``n_entries`` TM."""
    from afrilingua.tm import open_translation_memory

    with (
        tempfile.TemporaryDirectory(prefix="afrilingua-tm-bench-") as tmp,
        open_translation_memory(Path(tmp)) as tm,
    ):
        pairs = [
            (f"Conference sentence number {i} about community health and trade.", f"Ìtumọ̀ {i}")
            for i in range(n_entries)
        ]
        t0 = time.perf_counter()
        tm.add_many(pairs, "english", "yoruba", origin="bench", approved=True)
        tm.flush()
        ingest_ms = (time.perf_counter() - t0) * 1000

        exact_times: list[float] = []
        fuzzy_times: list[float] = []
        for q_idx in range(n_queries):
            target_idx = (q_idx * 37) % n_entries
            exact_q = pairs[target_idx][0]
            t_e = time.perf_counter()
            tm.lookup(exact_q, "english", "yoruba", k=3)
            exact_times.append((time.perf_counter() - t_e) * 1000)

            fuzzy_q = f"Conference statement number {target_idx} regarding community health"
            t_f = time.perf_counter()
            tm.lookup(fuzzy_q, "english", "yoruba", k=3)
            fuzzy_times.append((time.perf_counter() - t_f) * 1000)

    return {
        "entries": n_entries,
        "ingest_ms": round(ingest_ms, 1),
        "exact": _percentiles(exact_times),
        "fuzzy": _percentiles(fuzzy_times),
    }


def run_benchmark(
    engine: AfriLinguaEngine,
    langs: list[str] | None = None,
    *,
    include_models: bool = True,
    tm_entries: int = 5_000,
) -> dict[str, Any]:
    """Run micro-benchmarks (TM, LID) plus end-to-end model benchmarks."""
    targets = [engine.registry.normalize(lang) for lang in (langs or ["yoruba", "igbo", "hausa"])]
    rss_before = _rss_mb()

    lid_times: list[float] = []
    if engine.detector is not None:
        lid_inputs = [
            "Good morning everyone, welcome to the meeting.",
            "Ẹ káàárọ̀ gbogbo yín, ẹ káàbọ̀ sí ìpàdé wa.",
            "Ụtụtụ ọma, nnọọ na nzukọ anyị taa.",
            "Sannu da zuwa wannan taro mai muhimmanci.",
        ]
        for _ in range(25):
            for sample in lid_inputs:
                t0 = time.perf_counter()
                engine.detect_language(sample)
                lid_times.append((time.perf_counter() - t0) * 1000)

    report: dict[str, Any] = {
        "hardware": detect_hardware().to_dict(),
        "runtime": resolve_runtime().to_dict(),
        "rss_before_mb": rss_before,
        "lid": _percentiles(lid_times),
        "translation_memory": bench_translation_memory(n_entries=tm_entries),
        "mt": {},
        "tts": {},
        "asr": {},
    }

    if include_models:
        for tgt in targets:
            if tgt == "english":
                continue
            # Warm-up (excludes cold load from steady-state latency)
            t_load = time.perf_counter()
            engine.translate_one("Hello", "english", tgt, mode="live")
            cold_s = round(time.perf_counter() - t_load, 3)
            engine.clear_cache()

            latencies: list[float] = []
            for sent in _SAMPLE_SENTENCES:
                engine.clear_cache()
                t0 = time.perf_counter()
                engine.translate_one(sent, "english", tgt, mode="live")
                latencies.append((time.perf_counter() - t0) * 1000)
            report["mt"][f"english→{tgt}"] = {"cold_s": cold_s, **_percentiles(latencies)}

            spec = engine.registry.get(tgt)
            if spec.supports_tts:
                try:
                    t0 = time.perf_counter()
                    speech = engine.synthesize("Ụtụtụ ọma" if tgt == "igbo" else "Sannu", tgt)
                    elapsed_s = time.perf_counter() - t0
                    rtf = elapsed_s / max(speech.duration_s, 1e-3)
                    report["tts"][tgt] = {
                        "elapsed_ms": round(elapsed_s * 1000, 1),
                        "audio_s": round(speech.duration_s, 2),
                        "rtf": round(rtf, 3),
                    }
                except Exception as exc:
                    report["tts"][tgt] = {"error": str(exc)}

        try:
            clip = np.zeros(16_000 * 2, dtype=np.float32)
            t0 = time.perf_counter()
            engine.transcribe(clip, "english")
            elapsed_s = time.perf_counter() - t0
            report["asr"]["english_2s"] = {
                "elapsed_ms": round(elapsed_s * 1000, 1),
                "rtf": round(elapsed_s / 2.0, 3),
            }
        except Exception as exc:
            report["asr"]["english_2s"] = {"error": str(exc)}

    report["rss_after_mb"] = _rss_mb()
    return report
