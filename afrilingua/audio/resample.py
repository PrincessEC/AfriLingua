"""
Sample-rate conversion.

Uses ``soxr`` (high quality, fast) when installed and a NumPy fallback
(windowed-sinc low-pass + linear interpolation) otherwise. The streaming
variant keeps filter state between chunks, so resampling a live stream chunk
by chunk has no clicks at chunk boundaries.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from afrilingua.backends._common import module_available

logger = logging.getLogger(__name__)


def _lowpass_kernel(orig_sr: int, target_sr: int, taps: int = 63) -> np.ndarray | None:
    """Windowed-sinc anti-aliasing kernel for downsampling (None when upsampling)."""
    if target_sr >= orig_sr:
        return None
    cutoff = 0.5 * target_sr / orig_sr * 0.9  # normalized to orig Nyquist=0.5, with margin
    n = np.arange(taps) - (taps - 1) / 2
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
    return (kernel / kernel.sum()).astype(np.float32)


def _as_mono_f32(audio: Any) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.reshape(-1)


def resample(audio: Any, orig_sr: int, target_sr: int = 16_000) -> np.ndarray:
    """Resample a whole mono clip (float32 out)."""
    x = _as_mono_f32(audio)
    if orig_sr == target_sr or x.size == 0:
        return x.copy()
    if module_available("soxr"):
        import soxr

        return np.asarray(soxr.resample(x, orig_sr, target_sr, quality="HQ"), dtype=np.float32)
    kernel = _lowpass_kernel(orig_sr, target_sr)
    if kernel is not None:
        x = np.convolve(x, kernel, mode="same").astype(np.float32)
    n_out = int(round(x.size * target_sr / orig_sr))
    positions = np.arange(n_out, dtype=np.float64) * (orig_sr / target_sr)
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)


class StreamResampler:
    """Chunk-by-chunk resampler that preserves continuity across chunks."""

    def __init__(self, orig_sr: int, target_sr: int = 16_000, *, use_soxr: bool | None = None) -> None:
        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        self.passthrough = self.orig_sr == self.target_sr
        use_soxr = module_available("soxr") if use_soxr is None else use_soxr
        self._soxr: Any = None
        if not self.passthrough and use_soxr:
            import soxr

            self._soxr = soxr.ResampleStream(self.orig_sr, self.target_sr, 1, dtype="float32", quality="HQ")
        self._kernel = _lowpass_kernel(self.orig_sr, self.target_sr)
        self._hist = np.zeros(0 if self._kernel is None else len(self._kernel) - 1, dtype=np.float32)
        self._step = self.orig_sr / self.target_sr
        self._pos = 0.0
        self._prev: float | None = None

    def process(self, chunk: Any, *, last: bool = False) -> np.ndarray:
        """Resample the next chunk; pass ``last=True`` to flush internal buffers."""
        x = _as_mono_f32(chunk)
        if self.passthrough:
            return x.copy()
        if self._soxr is not None:
            return np.asarray(self._soxr.resample_chunk(x, last=last), dtype=np.float32).reshape(-1)
        return self._np_process(x)

    def _np_process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._kernel is not None:
            padded = np.concatenate([self._hist, x])
            y = np.convolve(padded, self._kernel, mode="valid").astype(np.float32)
            self._hist = padded[len(padded) - len(self._hist) :] if len(self._hist) else self._hist
        else:
            y = x
        if self._prev is None:
            ext, offset = y, 0
        else:
            ext, offset = np.concatenate([[self._prev], y]).astype(np.float32), 1
        m = y.size
        if self._pos > m - 1:
            self._pos -= m
            self._prev = float(y[-1])
            return np.zeros(0, dtype=np.float32)
        count = int(np.floor((m - 1 - self._pos) / self._step)) + 1
        positions = self._pos + np.arange(count) * self._step
        out = np.interp(positions + offset, np.arange(ext.size), ext).astype(np.float32)
        self._pos = positions[-1] + self._step - m
        self._prev = float(y[-1])
        return out
