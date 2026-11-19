"""PCM16 ↔ float32 conversion and WAV/compressed-audio encode/decode."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from afrilingua.audio.resample import resample
from afrilingua.core.errors import InvalidInputError

logger = logging.getLogger(__name__)


def pcm16_to_float32(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Little-endian signed 16-bit PCM bytes → float32 in [-1, 1].

    A trailing odd byte (a split sample) is ignored.
    """
    buf = bytes(data)
    if len(buf) % 2:
        buf = buf[:-1]
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: Any) -> bytes:
    """float32 samples (clipped to [-1, 1]) → little-endian PCM16 bytes."""
    x = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (x * 32767.0).round().astype("<i2").tobytes()


def encode_wav(audio: Any, sample_rate: int) -> bytes:
    """Encode mono float32 audio as a PCM16 WAV file (bytes)."""
    pcm = float32_to_pcm16(audio)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return bio.getvalue()


def _decode_ffmpeg(data: bytes, target_sr: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise InvalidInputError("Unsupported audio format (install ffmpeg to decode mp3/webm/ogg-opus).")
    proc = subprocess.run(
        [
            exe,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if proc.returncode != 0:
        raise InvalidInputError(f"Could not decode audio: {proc.stderr.decode(errors='replace')[:300]}")
    return pcm16_to_float32(proc.stdout)


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an audio file's bytes → (mono float32, sample_rate).

    Uses soundfile (WAV/FLAC/OGG/MP3 with libsndfile ≥ 1.1), then the stdlib
    ``wave`` module, then ffmpeg (decoded at 16 kHz) as a last resort.
    """
    if not data:
        raise InvalidInputError("Empty audio payload.")
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return audio.mean(axis=1).astype(np.float32), int(sr)
    except Exception:
        logger.debug("soundfile could not decode audio", exc_info=True)
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            if wf.getsampwidth() != 2:
                raise ValueError("only 16-bit WAV supported by the stdlib fallback")
            x = pcm16_to_float32(frames)
            ch = wf.getnchannels()
            if ch > 1:
                x = x[: len(x) - len(x) % ch].reshape(-1, ch).mean(axis=1)
            return x, wf.getframerate()
    except Exception:
        logger.debug("wave could not decode audio", exc_info=True)
    return _decode_ffmpeg(data, 16_000), 16_000


def load_audio(source: bytes | str | Path, target_sr: int = 16_000) -> np.ndarray:
    """Decode bytes or a file path and resample to ``target_sr`` mono float32."""
    data = Path(source).read_bytes() if isinstance(source, str | Path) else source
    audio, sr = decode_audio(data)
    return resample(audio, sr, target_sr)
