"""
Live interpretation session: PCM in → VAD → ASR → MT (per target) → TTS → events.

Implements :class:`afrilingua.pipeline.StreamingSessionProtocol`. Threads:

``feed_*`` (caller thread)
    convert/resample and enqueue only — never touches a model.
VAD worker
    :class:`~afrilingua.pipeline.segmenter.UtteranceSegmenter` over a
    per-session Silero state; schedules finals and (optionally) partials.
ASR worker
    finals first (never dropped); a single "latest partial" slot is dropped
    whenever a final is waiting (backpressure).
MT worker
    one batched ``engine.translate(..., mode="live")`` per target language,
    draining any backlog of finals into the same batch.
TTS worker
    synthesizes translations in order per target; runs after captions are
    emitted, so it never delays them. Stale audio is dropped under backlog.

Events follow docs/api-contract.md §6 (``partial``, ``final``,
``translation``, ``audio``, ``metrics``, ``error``). ``translation.latency_ms``
is end-to-end: from the moment the utterance was cut to translation ready.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from afrilingua.audio.pcm import encode_wav, pcm16_to_float32
from afrilingua.audio.resample import StreamResampler
from afrilingua.core.errors import AfriLinguaError, UnsupportedLanguageError
from afrilingua.pipeline import Event, EventCallback
from afrilingua.pipeline.segmenter import ProbFn, SegmenterConfig, Utterance, UtteranceSegmenter

if TYPE_CHECKING:
    from afrilingua.engine import AfriLinguaEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_CLOSE = object()


class EnergyVAD:
    """Adaptive-noise-floor energy VAD, used only if Silero cannot be loaded."""

    def __init__(self) -> None:
        self.noise = 1e-3

    def __call__(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) if window.size else 0.0
        if rms < self.noise:
            self.noise = 0.9 * self.noise + 0.1 * max(rms, 1e-5)
        else:
            self.noise = min(self.noise * 1.002, 0.05)
        snr_db = 20.0 * np.log10((rms + 1e-9) / (self.noise + 1e-9))
        return float(np.clip((snr_db - 6.0) / 6.0, 0.0, 1.0))


@dataclass
class _Final:
    utterance: Utterance
    cut_at: float  # time.perf_counter() when VAD closed the utterance


@dataclass
class _MTJob:
    seq: int
    text: str
    cut_at: float
    asr_ms: float


@dataclass
class _TTSJob:
    seq: int
    lang: str
    text: str


@dataclass
class _SeqMetrics:
    cut_at: float
    asr_ms: float
    utterance_s: float
    mt_ms: float = 0.0
    tts_ms: float = 0.0
    pending_tts: int = 0
    dropped_tts: int = 0
    done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class StreamingSession:
    """One speaker's live stream fanned out to many target languages."""

    def __init__(
        self,
        engine: AfriLinguaEngine,
        src_lang: str,
        targets: Sequence[str] = (),
        *,
        synthesize_audio: bool = False,
        on_event: EventCallback | None = None,
        vad: ProbFn | None = None,
        partials: bool | None = None,
        partial_interval_s: float = 1.0,
        segmenter_config: SegmenterConfig | None = None,
        max_audio_queue: int = 4000,
        max_tts_backlog: int = 16,
        mt_batch_max: int = 8,
        close_timeout_s: float = 120.0,
    ) -> None:
        self.engine = engine
        self.registry = engine.registry
        self.settings = engine.settings
        self.src_lang = self.registry.normalize(src_lang)
        self.synthesize_audio = synthesize_audio
        self.on_event = on_event
        self.partial_interval_s = partial_interval_s
        self.partials = self._default_partials() if partials is None else partials
        s = self.settings
        self.segmenter_config = segmenter_config or SegmenterConfig(
            sample_rate=SAMPLE_RATE,
            threshold=s.vad_threshold,
            preroll_s=s.vad_preroll_s,
            silence_s=s.vad_silence_s,
            max_utterance_s=s.vad_max_utterance_s,
        )
        self.max_tts_backlog = max(1, max_tts_backlog)
        self.mt_batch_max = max(1, mt_batch_max)
        self.close_timeout_s = close_timeout_s
        self._vad_override = vad

        self._targets_lock = threading.Lock()
        self._targets: list[str] = self._normalize_targets(targets, report=False)
        self._emit_lock = threading.Lock()
        self._resamplers: dict[int, StreamResampler] = {}
        self._feed_lock = threading.Lock()

        self._audio_q: queue.Queue[Any] = queue.Queue(maxsize=max(16, max_audio_queue))
        self._asr_cv = threading.Condition()
        self._finals: deque[_Final] = deque()
        self._partial_audio: np.ndarray | None = None
        self._asr_closing = False
        self._mt_q: queue.Queue[Any] = queue.Queue()
        self._tts_cv = threading.Condition()
        self._tts_jobs: deque[_TTSJob] = deque()
        self._tts_closing = False

        self._metrics: dict[int, _SeqMetrics] = {}
        self._metrics_lock = threading.Lock()
        self._seq = 0
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False
        self._state_lock = threading.Lock()
        self.stats: dict[str, int] = {
            "chunks": 0,
            "utterances": 0,
            "finals": 0,
            "partials": 0,
            "partials_dropped": 0,
            "audio_dropped_chunks": 0,
            "tts_dropped": 0,
            "errors": 0,
        }

    # ── configuration ───────────────────────────────────────────────────────
    def _default_partials(self) -> bool:
        spec = self.registry.get(self.src_lang)
        if spec.whisper_lang:
            return True
        try:
            from afrilingua.core.hardware import resolve_runtime

            return resolve_runtime().device == "cuda"
        except Exception:
            return False

    def _normalize_targets(self, targets: Sequence[str], *, report: bool = True) -> list[str]:
        out: list[str] = []
        for t in targets:
            try:
                name = self.registry.normalize(t)
            except UnsupportedLanguageError as exc:
                if report:
                    self._error(exc, stage="targets")
                    continue
                raise
            if name not in out:
                out.append(name)
        return out

    @property
    def targets(self) -> list[str]:
        with self._targets_lock:
            return list(self._targets)

    def set_targets(self, targets: list[str]) -> None:
        """Replace the target languages (thread-safe; applies to the next MT batch)."""
        normalized = self._normalize_targets(targets)
        with self._targets_lock:
            self._targets = normalized

    # ── events ──────────────────────────────────────────────────────────────
    def _emit(self, event: Event) -> None:
        cb = self.on_event
        if cb is None:
            return
        with self._emit_lock:
            try:
                cb(event)
            except Exception:
                logger.exception("on_event callback raised")

    def _error(
        self, exc: BaseException | str, *, stage: str, seq: int | None = None, code: str | None = None
    ) -> None:
        self.stats["errors"] += 1
        if isinstance(exc, BaseException):
            message = str(exc) or type(exc).__name__
            code = code or (exc.code if isinstance(exc, AfriLinguaError) else "internal_error")
            if not isinstance(exc, AfriLinguaError):
                logger.error("Streaming %s error: %s", stage, message, exc_info=exc)
        else:
            message = exc
            code = code or "internal_error"
        event: Event = {"type": "error", "code": code, "message": message, "stage": stage}
        if seq is not None:
            event["seq"] = seq
        self._emit(event)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the worker threads (idempotent)."""
        with self._state_lock:
            if self._started or self._closed:
                return
            self._started = True
            for name, target in (
                ("vad", self._vad_worker),
                ("asr", self._asr_worker),
                ("mt", self._mt_worker),
                ("tts", self._tts_worker),
            ):
                t = threading.Thread(target=target, name=f"stream-{name}", daemon=True)
                t.start()
                self._threads.append(t)

    def close(self) -> None:
        """Flush the last utterance through every stage, then stop the workers."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if not started:
            return
        with self._feed_lock:
            for rs in self._resamplers.values():
                tail = rs.process(np.zeros(0, dtype=np.float32), last=True)
                if tail.size:
                    self._enqueue_audio(tail)
        self._audio_q.put(_CLOSE)
        deadline = time.monotonic() + self.close_timeout_s
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
            if t.is_alive():
                logger.warning("Streaming worker %s did not stop within the close timeout", t.name)

    @property
    def closed(self) -> bool:
        return self._closed

    # ── input ───────────────────────────────────────────────────────────────
    def feed_pcm16(self, data: bytes, sample_rate: int = 16_000) -> None:
        """Feed little-endian PCM16 mono bytes (thread-safe, never blocks on models)."""
        if self._closed or not data:
            return
        self.feed_float32(pcm16_to_float32(data), sample_rate)

    def feed_float32(self, samples: Any, sample_rate: int = 16_000) -> None:
        """Feed float32 mono samples in [-1, 1] at ``sample_rate``."""
        if self._closed:
            return
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.reshape(-1)
        if x.size == 0:
            return
        sr = int(sample_rate)
        if sr != SAMPLE_RATE:
            with self._feed_lock:
                rs = self._resamplers.get(sr)
                if rs is None:
                    rs = self._resamplers[sr] = StreamResampler(sr, SAMPLE_RATE)
                x = rs.process(x)
            if x.size == 0:
                return
        self._enqueue_audio(x)

    def _enqueue_audio(self, x: np.ndarray) -> None:
        self.stats["chunks"] += 1
        try:
            self._audio_q.put_nowait(x)
        except queue.Full:
            # The VAD worker is far behind (should not happen: VAD is ~1 ms/window).
            self.stats["audio_dropped_chunks"] += 1
            with contextlib.suppress(queue.Empty):
                self._audio_q.get_nowait()
            with contextlib.suppress(queue.Full):
                self._audio_q.put_nowait(x)
            if self.stats["audio_dropped_chunks"] in (1, 100, 1000):
                self._error(
                    "Audio input is arriving faster than it can be processed.", stage="vad", code="overload"
                )

    # ── VAD worker ──────────────────────────────────────────────────────────
    def _make_vad(self) -> ProbFn:
        if self._vad_override is not None:
            return self._vad_override
        try:
            from afrilingua.backends.factory import VAD_KEY

            manager = self.engine.manager
            if manager.has(VAD_KEY):
                backend = manager.get(VAD_KEY)
            else:
                from afrilingua.backends.vad import SileroVADBackend

                backend = SileroVADBackend()
            return backend.new_stream()
        except Exception as exc:
            self._error(
                f"Silero VAD unavailable ({exc}); using energy-based VAD.", stage="vad", code="vad_fallback"
            )
            return EnergyVAD()

    def _vad_worker(self) -> None:
        try:
            seg = UtteranceSegmenter(self._make_vad(), self.segmenter_config)
        except Exception as exc:  # pragma: no cover - defensive
            self._error(exc, stage="vad")
            seg = UtteranceSegmenter(EnergyVAD(), self.segmenter_config)
        last_partial = 0.0
        while True:
            item = self._audio_q.get()
            if item is _CLOSE:
                try:
                    for utt in seg.flush():
                        self._submit_final(utt)
                except Exception as exc:
                    self._error(exc, stage="vad")
                break
            try:
                for utt in seg.push(item):
                    self._submit_final(utt)
                    last_partial = time.monotonic()
                if (
                    self.partials
                    and seg.in_speech
                    and seg.speech_seconds >= 0.6
                    and time.monotonic() - last_partial >= self.partial_interval_s
                ) and self._submit_partial(seg.current_audio()):
                    last_partial = time.monotonic()
            except Exception as exc:
                self._error(exc, stage="vad")
        with self._asr_cv:
            self._asr_closing = True
            self._asr_cv.notify_all()

    def _submit_final(self, utt: Utterance) -> None:
        self.stats["utterances"] += 1
        with self._asr_cv:
            self._finals.append(_Final(utt, time.perf_counter()))
            if self._partial_audio is not None:
                self._partial_audio = None
                self.stats["partials_dropped"] += 1
            self._asr_cv.notify_all()

    def _submit_partial(self, audio: np.ndarray) -> bool:
        with self._asr_cv:
            if self._finals:  # backpressure: finals first, partials are expendable
                self.stats["partials_dropped"] += 1
                return False
            if self._partial_audio is not None:
                self.stats["partials_dropped"] += 1
            self._partial_audio = audio
            self._asr_cv.notify_all()
            return True

    # ── ASR worker ──────────────────────────────────────────────────────────
    def _asr_worker(self) -> None:
        while True:
            with self._asr_cv:
                while not self._finals and self._partial_audio is None and not self._asr_closing:
                    self._asr_cv.wait()
                final = self._finals.popleft() if self._finals else None
                partial = None
                if final is None and self._partial_audio is not None:
                    partial, self._partial_audio = self._partial_audio, None
                if final is None and partial is None and self._asr_closing:
                    break
            if final is not None:
                self._run_final(final)
            elif partial is not None:
                self._run_partial(partial)
        self._mt_q.put(_CLOSE)

    def _run_partial(self, audio: np.ndarray) -> None:
        try:
            text = self.engine.transcribe(audio, self.src_lang).text.strip()
        except Exception as exc:
            self._error(exc, stage="asr_partial")
            return
        if text:
            self.stats["partials"] += 1
            self._emit({"type": "partial", "seq": self._seq, "lang": self.src_lang, "text": text})

    def _run_final(self, final: _Final) -> None:
        utt = final.utterance
        t0 = time.perf_counter()
        try:
            transcript = self.engine.transcribe(utt.audio, self.src_lang)
        except Exception as exc:
            self._error(exc, stage="asr", seq=self._seq)
            return
        asr_ms = (time.perf_counter() - t0) * 1000
        text = transcript.text.strip()
        if not text:
            return
        seq = self._seq
        self._seq += 1
        self.stats["finals"] += 1
        with self._metrics_lock:
            self._metrics[seq] = _SeqMetrics(cut_at=final.cut_at, asr_ms=asr_ms, utterance_s=utt.duration_s)
        event: Event = {
            "type": "final",
            "seq": seq,
            "lang": self.src_lang,
            "text": text,
            "ts": time.time(),
            "duration_s": round(utt.duration_s, 3),
        }
        if transcript.confidence is not None:
            event["confidence"] = round(float(transcript.confidence), 3)
        self._emit(event)
        self._mt_q.put(_MTJob(seq, text, final.cut_at, asr_ms))

    # ── MT worker ───────────────────────────────────────────────────────────
    def _mt_worker(self) -> None:
        stop = False
        while not stop:
            item = self._mt_q.get()
            if item is _CLOSE:
                break
            batch: list[_MTJob] = [item]
            while len(batch) < self.mt_batch_max:
                try:
                    nxt = self._mt_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is _CLOSE:
                    stop = True
                    break
                batch.append(nxt)
            self._translate_batch(batch)
        with self._tts_cv:
            self._tts_closing = True
            self._tts_cv.notify_all()

    def _translate_batch(self, batch: list[_MTJob]) -> None:
        targets = [t for t in self.targets if t != self.src_lang]
        tts_jobs: list[_TTSJob] = []
        for tgt in targets:
            t0 = time.perf_counter()
            try:
                results = self.engine.translate([j.text for j in batch], self.src_lang, tgt, mode="live")
            except Exception as exc:
                for j in batch:
                    self._error(exc, stage="mt", seq=j.seq)
                continue
            mt_ms = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            want_audio = self.synthesize_audio and self._tts_capable(tgt)
            for job, res in zip(batch, results, strict=True):
                with self._metrics_lock:
                    m = self._metrics.get(job.seq)
                    if m is not None:
                        m.mt_ms += mt_ms / len(batch)
                        if want_audio and res.text:
                            m.pending_tts += 1
                self._emit(
                    {
                        "type": "translation",
                        "seq": job.seq,
                        "lang": tgt,
                        "text": res.text,
                        "origin": res.origin,
                        "latency_ms": round((now - job.cut_at) * 1000, 1),
                        "mt_ms": round(mt_ms / len(batch), 1),
                    }
                )
                if want_audio and res.text:
                    tts_jobs.append(_TTSJob(job.seq, tgt, res.text))
        if tts_jobs:
            self._enqueue_tts(tts_jobs)
        for job in batch:
            self._maybe_emit_metrics(job.seq)

    def _tts_capable(self, lang: str) -> bool:
        try:
            return self.registry.get(lang).supports_tts
        except Exception:
            return False

    # ── TTS worker ──────────────────────────────────────────────────────────
    def _enqueue_tts(self, jobs: list[_TTSJob]) -> None:
        dropped: list[_TTSJob] = []
        with self._tts_cv:
            self._tts_jobs.extend(jobs)
            while len(self._tts_jobs) > self.max_tts_backlog:
                dropped.append(self._tts_jobs.popleft())
            self._tts_cv.notify_all()
        for job in dropped:
            self.stats["tts_dropped"] += 1
            logger.info("Dropping stale TTS for seq %s (%s): backlog full", job.seq, job.lang)
            self._finish_tts(job.seq, 0.0, dropped=True)

    def _tts_worker(self) -> None:
        while True:
            with self._tts_cv:
                while not self._tts_jobs and not self._tts_closing:
                    self._tts_cv.wait()
                if not self._tts_jobs and self._tts_closing:
                    break
                job = self._tts_jobs.popleft()
            if job.lang not in self.targets:  # listener group left meanwhile
                self._finish_tts(job.seq, 0.0, dropped=True)
                continue
            t0 = time.perf_counter()
            try:
                speech = self.engine.synthesize(job.text, job.lang)
                tts_ms = (time.perf_counter() - t0) * 1000
                if speech.waveform.size:
                    self._emit(
                        {
                            "type": "audio",
                            "seq": job.seq,
                            "lang": job.lang,
                            "format": "wav",
                            "sample_rate": int(speech.sample_rate),
                            "duration_s": round(speech.duration_s, 3),
                            "data": base64.b64encode(encode_wav(speech.waveform, speech.sample_rate)).decode(
                                "ascii"
                            ),
                        }
                    )
                self._finish_tts(job.seq, tts_ms)
            except Exception as exc:
                self._error(exc, stage="tts", seq=job.seq)
                self._finish_tts(job.seq, (time.perf_counter() - t0) * 1000)

    def _finish_tts(self, seq: int, tts_ms: float, *, dropped: bool = False) -> None:
        with self._metrics_lock:
            m = self._metrics.get(seq)
            if m is None:
                return
            m.tts_ms += tts_ms
            m.pending_tts = max(0, m.pending_tts - 1)
            if dropped:
                m.dropped_tts += 1
        self._maybe_emit_metrics(seq)

    # ── metrics ─────────────────────────────────────────────────────────────
    def _maybe_emit_metrics(self, seq: int) -> None:
        with self._metrics_lock:
            m = self._metrics.get(seq)
            if m is None or m.done or m.pending_tts > 0:
                return
            m.done = True
            del self._metrics[seq]
        event: Event = {
            "type": "metrics",
            "seq": seq,
            "asr_ms": round(m.asr_ms, 1),
            "mt_ms": round(m.mt_ms, 1),
            "tts_ms": round(m.tts_ms, 1),
            "total_ms": round((time.perf_counter() - m.cut_at) * 1000, 1),
            "utterance_s": round(m.utterance_s, 3),
        }
        if m.dropped_tts:
            event["tts_dropped"] = m.dropped_tts
        self._emit(event)

    # ── context manager ─────────────────────────────────────────────────────
    def __enter__(self) -> StreamingSession:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def collect_events() -> tuple[list[Event], Callable[[Event], None]]:
    """Convenience for scripts/tests: returns (events list, callback appending to it)."""
    events: list[Event] = []
    return events, events.append
