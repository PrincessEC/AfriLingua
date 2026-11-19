"""
Voice-activity segmentation of a live 16 kHz stream into utterances.

Pure NumPy + a pluggable speech-probability function (Silero in production, a
fake in tests), so the boundary logic is unit-testable without models.

Rules
-----
* Audio is scored in fixed windows (512 samples = 32 ms for Silero).
* **Pre-roll**: the last ``preroll_s`` of non-speech audio is prepended when
  speech starts, so word onsets are not clipped.
* **Silence end**: an utterance ends after ``silence_s`` of continuous
  non-speech; up to ``trailing_keep_s`` of that silence stays attached, the
  rest becomes the next utterance's pre-roll (no sample is used twice).
* **Max length**: continuous speech is cut at ``max_utterance_s``, at the
  lowest-energy 20 ms frame within the last ``cut_search_s`` rather than
  mid-word; the remainder starts the next utterance, so consecutive cuts tile
  the stream with no lost samples.
* Blips shorter than ``min_speech_s`` of speech are discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ProbFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class SegmenterConfig:
    sample_rate: int = 16_000
    window: int = 512
    threshold: float = 0.5
    neg_threshold: float | None = None  # hysteresis; default threshold - 0.15
    preroll_s: float = 0.3
    silence_s: float = 0.6
    max_utterance_s: float = 8.0
    min_speech_s: float = 0.25
    cut_search_s: float = 1.0
    trailing_keep_s: float = 0.2


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray
    start: int  # absolute sample index of audio[0] in the stream
    end: int  # absolute sample index one past the last sample
    reason: str  # "silence" | "max_length" | "flush"
    speech_s: float
    sample_rate: int = 16_000

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / self.sample_rate


class _GrowBuffer:
    """Append-only float32 buffer with amortised O(1) appends (no Python lists)."""

    def __init__(self, capacity: int) -> None:
        self._data = np.zeros(max(1024, capacity), dtype=np.float32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def append(self, x: np.ndarray) -> None:
        need = self._n + x.size
        if need > self._data.size:
            new = np.zeros(max(need, self._data.size * 2), dtype=np.float32)
            new[: self._n] = self._data[: self._n]
            self._data = new
        self._data[self._n : need] = x
        self._n = need

    def view(self) -> np.ndarray:
        return self._data[: self._n]

    def take(self, n: int) -> np.ndarray:
        """Remove and return the first ``n`` samples."""
        n = min(n, self._n)
        head = self._data[:n].copy()
        rest = self._data[n : self._n].copy()
        self._data[: rest.size] = rest
        self._n = rest.size
        return head

    def clear(self) -> np.ndarray:
        out = self._data[: self._n].copy()
        self._n = 0
        return out


class _Ring:
    """Fixed-size ring holding the most recent ``size`` samples."""

    def __init__(self, size: int) -> None:
        self._buf = np.zeros(max(1, size), dtype=np.float32)
        self._size = size
        self._pos = 0
        self._filled = 0

    def push(self, x: np.ndarray) -> None:
        if self._size <= 0:
            return
        if x.size >= self._size:
            self._buf[:] = x[-self._size :]
            self._pos, self._filled = 0, self._size
            return
        end = self._pos + x.size
        if end <= self._size:
            self._buf[self._pos : end] = x
        else:
            first = self._size - self._pos
            self._buf[self._pos :] = x[:first]
            self._buf[: x.size - first] = x[first:]
        self._pos = end % self._size
        self._filled = min(self._size, self._filled + x.size)

    def get(self) -> np.ndarray:
        if self._filled < self._size:
            return (
                self._buf[self._pos - self._filled : self._pos].copy()
                if self._filled
                else np.zeros(0, np.float32)
            )
        return np.concatenate([self._buf[self._pos :], self._buf[: self._pos]])

    def __len__(self) -> int:
        return self._filled

    def clear(self) -> None:
        self._pos = self._filled = 0


class UtteranceSegmenter:
    """Turn pushed audio chunks into :class:`Utterance` objects."""

    def __init__(self, prob_fn: ProbFn, config: SegmenterConfig | None = None) -> None:
        self.prob_fn = prob_fn
        self.cfg = config or SegmenterConfig()
        c = self.cfg
        sr = c.sample_rate
        self._neg = c.neg_threshold if c.neg_threshold is not None else max(0.0, c.threshold - 0.15)
        self._preroll_n = int(c.preroll_s * sr)
        self._silence_n = int(c.silence_s * sr)
        self._max_n = int(c.max_utterance_s * sr)
        self._min_speech_n = int(c.min_speech_s * sr)
        self._search_n = int(c.cut_search_s * sr)
        self._keep_n = int(c.trailing_keep_s * sr)
        self._pending = _GrowBuffer(c.window * 4)
        self._preroll = _Ring(self._preroll_n)
        self._utt = _GrowBuffer(self._max_n + self._preroll_n + c.window)
        self._abs = 0  # absolute index of the next window's first sample
        self._utt_start = 0
        self._in_speech = False
        self._speech_n = 0
        self._silence_run = 0
        self._continuation = False

    # ── state ───────────────────────────────────────────────────────────────
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def speech_seconds(self) -> float:
        """Seconds of speech in the current (unfinished) utterance."""
        return self._speech_n / self.cfg.sample_rate if self._in_speech else 0.0

    def current_audio(self) -> np.ndarray:
        """Copy of the in-progress utterance (for partial transcripts)."""
        return self._utt.view().copy() if self._in_speech else np.zeros(0, dtype=np.float32)

    # ── processing ──────────────────────────────────────────────────────────
    def push(self, samples: np.ndarray) -> list[Utterance]:
        """Feed audio; returns utterances completed by this chunk."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        out: list[Utterance] = []
        if x.size == 0:
            return out
        self._pending.append(x)
        w = self.cfg.window
        while len(self._pending) >= w:
            window = self._pending.take(w)
            self._process_window(window, out)
        return out

    def flush(self) -> list[Utterance]:
        """End of stream: emit the in-progress utterance (including leftover samples)."""
        out: list[Utterance] = []
        rest = self._pending.clear()
        if self._in_speech:
            if rest.size:
                self._utt.append(rest)
            self._abs += rest.size
            utt = self._emit(len(self._utt), "flush", trim_silence=False)
            if utt is not None:
                out.append(utt)
        else:
            self._abs += rest.size
        self._reset_speech()
        self._preroll.clear()
        return out

    def _process_window(self, window: np.ndarray, out: list[Utterance]) -> None:
        p = float(self.prob_fn(window))
        w = window.size
        if not self._in_speech:
            if p >= self.cfg.threshold:
                pre = self._preroll.get()
                self._preroll.clear()
                self._utt.clear()
                self._utt.append(pre)
                self._utt.append(window)
                self._utt_start = self._abs - pre.size
                self._in_speech = True
                self._speech_n = w
                self._silence_run = 0
                self._continuation = False
            else:
                self._preroll.push(window)
            self._abs += w
            return

        self._utt.append(window)
        self._abs += w
        if p >= self._neg:
            self._silence_run = 0
            self._speech_n += w
        else:
            self._silence_run += w

        if self._silence_run >= self._silence_n:
            utt = self._emit(len(self._utt), "silence", trim_silence=True)
            if utt is not None:
                out.append(utt)
            self._reset_speech()
        elif len(self._utt) >= self._max_n:
            cut = self._lowest_energy_cut()
            utt = self._emit(cut, "max_length", trim_silence=False)
            if utt is not None:
                out.append(utt)
            # remainder stays as the start of the next (continuation) utterance
            self._continuation = True
            self._speech_n = len(self._utt)
            self._silence_run = min(self._silence_run, len(self._utt))

    def _lowest_energy_cut(self) -> int:
        """Index (within the utterance buffer) of the quietest 20 ms frame centre in the last ``cut_search_s``."""
        buf = self._utt.view()
        n = buf.size
        frame = max(1, self.cfg.sample_rate // 50)
        start = max(frame, n - self._search_n)
        region = buf[start:n]
        nframes = region.size // frame
        if nframes < 1:
            return n
        energy = np.sqrt(np.mean(region[: nframes * frame].reshape(nframes, frame) ** 2, axis=1))
        idx = int(np.argmin(energy))
        return start + idx * frame + frame // 2

    def _emit(self, length: int, reason: str, *, trim_silence: bool) -> Utterance | None:
        """Pop the first ``length`` samples of the utterance buffer as an Utterance."""
        speech_n = self._speech_n
        if trim_silence:
            drop = max(0, self._silence_run - self._keep_n)
            length = max(0, length - drop)
        audio = self._utt.take(length)
        start = self._utt_start
        end = start + audio.size
        self._utt_start = end
        if trim_silence:
            # the dropped silence becomes pre-roll for the next utterance
            leftover = self._utt.clear()
            self._preroll.clear()
            self._preroll.push(leftover)
        min_needed = 1 if self._continuation else self._min_speech_n
        if speech_n < min_needed or audio.size == 0:
            return None
        sr = self.cfg.sample_rate
        return Utterance(audio, start, end, reason, round(min(speech_n, audio.size) / sr, 3), sr)

    def _reset_speech(self) -> None:
        self._in_speech = False
        self._speech_n = 0
        self._silence_run = 0
        self._continuation = False
        if len(self._utt):
            leftover = self._utt.clear()
            self._preroll.push(leftover)
