"""
src/tts/audio_player.py
Non-blocking audio playback queue.
Audio arrays are queued and played back without blocking the main pipeline.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_SENTINEL = None   # poison pill to stop the worker thread


class AudioPlayer:
    """
    Async audio output queue.
    Call `play(waveform, sample_rate)` from any thread; playback happens
    sequentially in a background thread.

    Usage::
        player = AudioPlayer()
        player.start()
        player.play(waveform_np, 16000)
        player.stop()
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            waveform, sample_rate = item
            try:
                sd.play(waveform, samplerate=sample_rate)
                sd.wait()
            except Exception as exc:
                logger.error("Audio playback error: %s", exc)
            finally:
                self._queue.task_done()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("AudioPlayer started.")

    def play(self, waveform: np.ndarray, sample_rate: int) -> None:
        """Enqueue audio for playback (non-blocking)."""
        if not self._running:
            self.start()
        self._queue.put((waveform, sample_rate))

    def stop(self) -> None:
        """Stop playback after draining the current queue."""
        if not self._running:
            return
        self._queue.put(_SENTINEL)
        if self._thread:
            self._thread.join()
        self._running = False
        logger.info("AudioPlayer stopped.")

    def clear(self) -> None:
        """Discard all pending audio (e.g. when muting)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
