"""
Microphone capture.

The input stream is opened at the device's **native default sample rate**
(fixes v1's ``Invalid sample rate [-9997]`` on Windows WASAPI, which rejects
16 kHz) and resampled to 16 kHz on a consumer thread. The PortAudio callback
only copies frames into a bounded queue — it never runs models or resampling.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from afrilingua.audio.resample import StreamResampler

logger = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], None]


def list_input_devices() -> list[dict[str, Any]]:
    """Input-capable devices: ``[{index, name, hostapi, channels, default_samplerate, default}]``."""
    import sounddevice as sd

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    hostapis = sd.query_hostapis()
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "hostapi": hostapis[dev["hostapi"]]["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "default": idx == default_in,
                }
            )
    return out


class MicrophoneCapture:
    """Capture mono 16 kHz float32 audio and hand it to ``on_audio`` in ~``block_s`` chunks."""

    def __init__(
        self,
        on_audio: AudioCallback,
        *,
        device: int | str | None = None,
        target_sr: int = 16_000,
        block_s: float = 0.03,
        max_queue_s: float = 10.0,
    ) -> None:
        self.on_audio = on_audio
        self.device = device
        self.target_sr = target_sr
        self.block_s = block_s
        self.native_sr: int | None = None
        self.channels = 1
        self.dropped_blocks = 0
        self._max_blocks = max(8, int(max_queue_s / block_s))
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self._max_blocks)
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # ── PortAudio callback: enqueue only ────────────────────────────────────
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.debug("Input stream status: %s", status)
        block = indata.copy()
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(block)

    def _consume(self) -> None:
        assert self.native_sr is not None
        rs = StreamResampler(self.native_sr, self.target_sr)
        while True:
            block = self._queue.get()
            if block is None:
                tail = rs.process(np.zeros(0, dtype=np.float32), last=True)
                if tail.size:
                    self._deliver(tail)
                return
            mono = block.mean(axis=1) if block.ndim > 1 else block.reshape(-1)
            out = rs.process(mono)
            if out.size:
                self._deliver(out)

    def _deliver(self, audio: np.ndarray) -> None:
        try:
            self.on_audio(audio)
        except Exception:
            logger.exception("Microphone consumer callback failed")

    # ── control ─────────────────────────────────────────────────────────────
    def _open_stream(self) -> Any:
        import sounddevice as sd

        info = sd.query_devices(self.device, "input")
        self.native_sr = round(float(info["default_samplerate"]) or 48_000)
        blocksize = max(64, int(self.native_sr * self.block_s))
        extra = None
        try:
            hostapi = sd.query_hostapis(info["hostapi"])["name"]
            if "WASAPI" in hostapi and hasattr(sd, "WasapiSettings"):
                with contextlib.suppress(TypeError):
                    extra = sd.WasapiSettings(auto_convert=True)
        except Exception:
            pass
        attempts = [(1, extra), (int(info.get("max_input_channels", 1) or 1), extra), (1, None)]
        last_exc: Exception | None = None
        for channels, settings in attempts:
            try:
                stream = sd.InputStream(
                    device=self.device,
                    samplerate=self.native_sr,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=self._callback,
                    extra_settings=settings,
                )
                self.channels = channels
                return stream
            except Exception as exc:
                last_exc = exc
                logger.debug("InputStream(channels=%s) failed: %s", channels, exc)
        raise RuntimeError(f"Could not open microphone {self.device!r} at {self.native_sr} Hz: {last_exc}")

    def start(self) -> MicrophoneCapture:
        """Open the device and start capturing (idempotent)."""
        if self._running.is_set():
            return self
        self._queue = queue.Queue(maxsize=self._max_blocks)
        self._stream = self._open_stream()
        self._thread = threading.Thread(target=self._consume, name="mic-consumer", daemon=True)
        self._thread.start()
        self._stream.start()
        self._running.set()
        logger.info(
            "Microphone capture started (%s Hz native → %s Hz, %d ch)",
            self.native_sr,
            self.target_sr,
            self.channels,
        )
        return self

    def stop(self) -> None:
        """Stop the stream and flush remaining audio to ``on_audio``."""
        if not self._running.is_set():
            return
        self._running.clear()
        with contextlib.suppress(Exception):
            self._stream.stop()
            self._stream.close()
        self._stream = None
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def __enter__(self) -> MicrophoneCapture:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
