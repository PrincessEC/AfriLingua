"""
Audio playback: one worker thread, a bounded queue, interruptible, restartable.

Audio is resampled to the output device's native rate before playback (the
same WASAPI sample-rate issue that affects capture) and written in short
blocks so :meth:`AudioPlayer.clear` takes effect within ~50 ms.
"""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from afrilingua.audio.resample import resample

logger = logging.getLogger(__name__)

Sink = Callable[[np.ndarray, int], None]
_STOP = object()


class AudioPlayer:
    """Sequential player for ``(waveform, sample_rate)`` items.

    ``sink`` replaces the sound device (used by tests and headless servers); it
    receives each block of float32 audio at the output rate.
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        max_queue: int = 32,
        block_s: float = 0.05,
        sink: Sink | None = None,
        output_sr: int | None = None,
    ) -> None:
        self.device = device
        self.max_queue = max_queue
        self.block_s = block_s
        self._sink = sink
        self._output_sr = output_sr
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._lock = threading.Lock()
        self._stream: Any = None
        self.dropped = 0
        self.playing = threading.Event()

    # ── public API ──────────────────────────────────────────────────────────
    def start(self) -> AudioPlayer:
        """Start (or restart) the worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._queue = queue.Queue(maxsize=self.max_queue)
            self._thread = threading.Thread(target=self._run, name="audio-player", daemon=True)
            self._thread.start()
        return self

    def play(self, audio: Any, sample_rate: int) -> None:
        """Queue audio; when the queue is full the oldest item is dropped."""
        if self._thread is None or not self._thread.is_alive():
            self.start()
        item = (np.asarray(audio, dtype=np.float32).reshape(-1), int(sample_rate), self._generation)
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                self.dropped += 1
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
                    self._queue.task_done()

    def clear(self) -> None:
        """Drop queued audio and interrupt the item currently playing."""
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker (it can be started again later)."""
        thread = self._thread
        if thread is None:
            return
        self.clear()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_STOP)
        thread.join(timeout=timeout)
        self._thread = None
        self._close_stream()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until the queue is empty and nothing is playing."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── worker ──────────────────────────────────────────────────────────────
    def _device_rate(self) -> int:
        if self._output_sr:
            return self._output_sr
        if self._sink is not None:
            return 0  # play at source rate
        import sounddevice as sd

        info = sd.query_devices(self.device, "output")
        self._output_sr = int(round(float(info["default_samplerate"]) or 48_000))
        return self._output_sr

    def _write(self, block: np.ndarray, sr: int) -> None:
        if self._sink is not None:
            self._sink(block, sr)
            return
        import sounddevice as sd

        if self._stream is None or int(self._stream.samplerate) != sr:
            self._close_stream()
            self._stream = sd.OutputStream(device=self.device, samplerate=sr, channels=1, dtype="float32")
            self._stream.start()
        self._stream.write(block.reshape(-1, 1))

    def _close_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            audio, sr, gen = item
            if gen != self._generation or audio.size == 0:
                self._queue.task_done()
                continue
            self.playing.set()
            try:
                out_sr = self._device_rate() or sr
                if out_sr != sr:
                    audio = resample(audio, sr, out_sr)
                step = max(1, int(out_sr * self.block_s))
                for i in range(0, audio.size, step):
                    if gen != self._generation:
                        break
                    self._write(audio[i : i + step], out_sr)
            except Exception:
                logger.exception("Audio playback failed")
                self._close_stream()
            finally:
                self.playing.clear()
                self._queue.task_done()
