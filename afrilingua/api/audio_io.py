"""Audio decode / encode helpers for the HTTP + MCP layers.

* :func:`decode_audio` — any common container (wav/flac/ogg/mp3 via libsndfile,
  webm/opus/m4a via PyAV when installed) → float32 mono at 16 kHz.
* :func:`encode_audio` — float32 waveform → WAV/FLAC/PCM bytes.

Prefers ``afrilingua.audio`` helpers when the engine workstream provides them, but
works standalone with ``soundfile`` + numpy.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from afrilingua.core.errors import CapabilityUnavailableError, InvalidInputError

logger = logging.getLogger(__name__)

TARGET_SR = 16_000

AUDIO_MEDIA_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
}


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or audio.size == 0:
        return audio
    try:
        import soxr

        return np.asarray(soxr.resample(audio, src_sr, dst_sr, quality="HQ"), dtype=np.float32)
    except ImportError:
        pass
    n_out = round(audio.size * dst_sr / src_sr)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.reshape(-1)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _to_mono(samples), int(sr)


def _decode_pyav(data: bytes) -> tuple[np.ndarray, int]:
    import av  # type: ignore[import-not-found]

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise InvalidInputError("The uploaded file contains no audio stream.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    return audio, TARGET_SR


def decode_audio(data: bytes, *, filename: str | None = None) -> np.ndarray:
    """Decode an uploaded audio file to float32 mono 16 kHz in [-1, 1]."""
    if not data:
        raise InvalidInputError("Empty audio upload.")
    errors: list[str] = []
    for decoder in (_decode_soundfile, _decode_pyav):
        try:
            audio, sr = decoder(data)
            return np.clip(resample(audio, sr, TARGET_SR), -1.0, 1.0)
        except ImportError:
            continue
        except InvalidInputError:
            raise
        except Exception as exc:
            errors.append(f"{decoder.__name__}: {exc}")
    hint = ""
    try:
        import av  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        hint = " For webm/opus/m4a uploads install PyAV (`pip install av`)."
    logger.debug("Audio decode failed for %s: %s", filename, errors)
    raise InvalidInputError(
        f"Could not decode audio{f' {filename!r}' if filename else ''}. Supported: wav, flac, ogg, mp3.{hint}"
    )


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_audio(waveform: np.ndarray, sample_rate: int, fmt: str = "wav") -> bytes:
    """Encode a float32 waveform. ``fmt``: wav | flac | pcm (raw s16le) | mp3 | opus (if libsndfile supports)."""
    fmt = (fmt or "wav").lower()
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if fmt == "pcm":
        return float32_to_pcm16(audio)
    import soundfile as sf

    buf = io.BytesIO()
    options: dict[str, str] = {
        "wav": {"format": "WAV", "subtype": "PCM_16"},
        "flac": {"format": "FLAC", "subtype": "PCM_16"},
        "mp3": {"format": "MP3", "subtype": "MPEG_LAYER_III"},
        "opus": {"format": "OGG", "subtype": "OPUS"},
        "ogg": {"format": "OGG", "subtype": "VORBIS"},
    }.get(fmt, {})
    if not options:
        raise InvalidInputError(f"Unsupported audio format {fmt!r}. Use wav, flac or pcm.")
    try:
        sf.write(buf, audio, int(sample_rate), **options)
    except Exception as exc:
        raise CapabilityUnavailableError(
            f"Audio format {fmt!r} is not supported by this build: {exc}"
        ) from exc
    return buf.getvalue()


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    return encode_audio(waveform, sample_rate, "wav")
