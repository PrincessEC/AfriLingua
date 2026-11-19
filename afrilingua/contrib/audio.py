"""Audio intake for voice contributions.

Uploads (wav / flac / ogg / mp3 / webm …) are decoded, down-mixed to mono,
resampled to 16 kHz and stored as 16-bit FLAC. Decoding order:

1. ``soundfile`` (libsndfile: wav, flac, ogg/vorbis, opus-in-ogg, mp3 with
   libsndfile ≥ 1.1),
2. ``PyAV`` (``av``) if importable: webm/opus, m4a/aac and anything FFmpeg reads,
3. otherwise the original bytes are kept and the item is marked
   ``needs_transcode`` (duration/quality checks are skipped for it).

Simple signal checks reject silence (RMS below −50 dBFS) and heavy clipping
(more than 1% of samples at full scale).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from afrilingua.core.errors import InvalidInputError

logger = logging.getLogger(__name__)

__all__ = ["MAX_DURATION_S", "MIN_DURATION_S", "TARGET_SR", "ProcessedAudio", "decode_audio", "process_audio"]

TARGET_SR = 16_000
MAX_DURATION_S = 30.0
MIN_DURATION_S = 0.3
SILENCE_RMS = 10 ** (-50 / 20)  # −50 dBFS
CLIP_LEVEL = 0.999
MAX_CLIP_FRACTION = 0.01


@dataclass
class ProcessedAudio:
    """Result of :func:`process_audio`."""

    waveform: np.ndarray | None  # float32 mono 16 kHz, None if not decodable
    duration_s: float | None
    needs_transcode: bool
    original: bytes
    original_ext: str


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(x, sr, TARGET_SR, quality="HQ").astype(np.float32)
    except ImportError:  # pragma: no cover - soxr is a core dependency
        n_out = round(len(x) * TARGET_SR / sr)
        t_old = np.arange(len(x)) / sr
        t_new = np.arange(n_out) / TARGET_SR
        return np.interp(t_new, t_old, x).astype(np.float32)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return x.mean(axis=1), int(sr)
    except Exception:
        return None


def _decode_av(data: bytes) -> tuple[np.ndarray, int] | None:
    try:
        import av
    except ImportError:
        return None
    try:
        chunks: list[np.ndarray] = []
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32), TARGET_SR
    except Exception:
        logger.debug("PyAV could not decode upload", exc_info=True)
        return None


def decode_audio(data: bytes) -> np.ndarray | None:
    """Decode arbitrary audio bytes to float32 mono 16 kHz, or ``None`` if impossible."""
    decoded = _decode_soundfile(data) or _decode_av(data)
    if decoded is None:
        return None
    x, sr = decoded
    x = np.nan_to_num(np.asarray(x, dtype=np.float32).reshape(-1))
    return _resample(x, sr)


def _ext_from(filename: str | None, content_type: str | None) -> str:
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext and ext.isalnum() and len(ext) <= 5:
        return ext
    ct = (content_type or "").split(";")[0].strip().lower()
    return {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/flac": "flac",
        "audio/mp4": "m4a",
    }.get(ct, "bin")


def process_audio(
    data: bytes, filename: str | None = None, content_type: str | None = None
) -> ProcessedAudio:
    """Decode + validate an upload. Raises :class:`InvalidInputError` on bad audio."""
    if not data:
        raise InvalidInputError("audio file is empty")
    ext = _ext_from(filename, content_type)
    x = decode_audio(data)
    if x is None:
        logger.warning("Could not decode %s upload; storing original bytes (needs_transcode)", ext)
        return ProcessedAudio(None, None, True, data, ext)
    duration = len(x) / TARGET_SR
    if duration > MAX_DURATION_S + 0.5:
        raise InvalidInputError(f"recording is {duration:.1f}s; the maximum is {MAX_DURATION_S:.0f}s")
    if duration < MIN_DURATION_S:
        raise InvalidInputError("recording is too short")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < SILENCE_RMS:
        raise InvalidInputError("recording appears to be silent; check your microphone")
    clipped = float(np.mean(np.abs(x) >= CLIP_LEVEL))
    if clipped > MAX_CLIP_FRACTION:
        raise InvalidInputError("recording is clipped (too loud); move further from the microphone")
    return ProcessedAudio(np.clip(x, -1.0, 1.0), round(duration, 3), False, data, ext)


def write_flac(path: Path, waveform: np.ndarray) -> None:
    """Write 16 kHz mono 16-bit FLAC."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), waveform, TARGET_SR, format="FLAC", subtype="PCM_16")
    tmp.replace(path)
