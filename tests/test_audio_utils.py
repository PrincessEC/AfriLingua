"""Audio utilities: PCM conversion, WAV round trip, resampling, player, capture plumbing."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from afrilingua.audio.pcm import decode_audio, encode_wav, float32_to_pcm16, load_audio, pcm16_to_float32
from afrilingua.audio.player import AudioPlayer
from afrilingua.audio.resample import StreamResampler, resample
from afrilingua.core.errors import InvalidInputError


def sine(freq: float, seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))])


# ── PCM ─────────────────────────────────────────────────────────────────────


def test_pcm16_round_trip_and_clipping():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = float32_to_pcm16(x)
    assert len(pcm) == 2 * len(x)
    y = pcm16_to_float32(pcm)
    assert np.allclose(y[:5], x[:5], atol=1e-4)
    assert y[5] == pytest.approx(1.0, abs=1e-3) and y[6] == pytest.approx(-1.0, abs=1e-3)


def test_pcm16_odd_length_ignores_trailing_byte():
    assert pcm16_to_float32(b"\x00\x40\x01").shape == (1,)


def test_wav_encode_decode_round_trip():
    x = sine(440, 0.25, 22_050)
    wav = encode_wav(x, 22_050)
    assert wav[:4] == b"RIFF"
    y, sr = decode_audio(wav)
    assert sr == 22_050 and y.shape == x.shape
    assert np.allclose(x, y, atol=1e-3)


def test_load_audio_resamples(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(encode_wav(sine(300, 1.0, 44_100), 44_100))
    y = load_audio(p, 16_000)
    assert abs(len(y) - 16_000) <= 2
    assert abs(dominant_freq(y, 16_000) - 300) < 5


def test_decode_rejects_empty():
    with pytest.raises(InvalidInputError):
        decode_audio(b"")


# ── resampling ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("orig", [8_000, 16_000, 44_100, 48_000])
def test_resample_length_and_pitch(orig):
    x = sine(440, 1.0, orig)
    y = resample(x, orig, 16_000)
    assert abs(len(y) - 16_000) <= 2
    assert abs(dominant_freq(y, 16_000) - 440) < 3


@pytest.mark.parametrize("use_soxr", [True, False])
def test_stream_resampler_matches_length_and_is_continuous(use_soxr):
    if use_soxr:
        pytest.importorskip("soxr")
    sr = 48_000
    x = sine(500, 2.0, sr)
    rs = StreamResampler(sr, 16_000, use_soxr=use_soxr)
    rng = np.random.default_rng(1)
    out, i = [], 0
    while i < len(x):
        n = int(rng.integers(100, 2000))
        out.append(rs.process(x[i : i + n]))
        i += n
    out.append(rs.process(np.zeros(0, dtype=np.float32), last=True))
    y = np.concatenate(out)
    assert abs(len(y) - 32_000) <= 64
    assert abs(dominant_freq(y, 16_000) - 500) < 3
    # continuity: no sample-to-sample jumps larger than a clean 500 Hz sine allows
    core = y[200:-200]
    assert np.max(np.abs(np.diff(core))) < 0.25


def test_numpy_resampler_attenuates_aliases():
    sr = 48_000
    x = sine(12_000, 1.0, sr)  # above the 8 kHz Nyquist of the 16 kHz output
    rs = StreamResampler(sr, 16_000, use_soxr=False)
    y = rs.process(x)
    assert float(np.sqrt(np.mean(y[100:-100] ** 2))) < 0.05


def test_stream_resampler_passthrough():
    rs = StreamResampler(16_000, 16_000)
    x = sine(100, 0.1, 16_000)
    assert np.array_equal(rs.process(x), x)


# ── player ──────────────────────────────────────────────────────────────────


def test_player_plays_sequentially_and_restarts():
    played: list[tuple[int, int]] = []
    player = AudioPlayer(sink=lambda block, sr: played.append((len(block), sr)), block_s=0.05)
    player.play(np.ones(1600, dtype=np.float32), 16_000)
    player.play(np.ones(800, dtype=np.float32), 16_000)
    assert player.wait_idle(5)
    assert sum(n for n, _ in played) == 2400 and {sr for _, sr in played} == {16_000}
    player.stop()
    assert not player.running
    player.play(np.ones(160, dtype=np.float32), 16_000)  # restarts automatically
    assert player.wait_idle(5)
    assert sum(n for n, _ in played) == 2560
    player.stop()


def test_player_clear_interrupts_and_drops_queue():
    started = threading.Event()
    blocks: list[int] = []

    def slow_sink(block, sr):
        started.set()
        blocks.append(len(block))
        time.sleep(0.02)

    player = AudioPlayer(sink=slow_sink, block_s=0.01)
    player.play(np.ones(16_000, dtype=np.float32), 16_000)  # 1 s = 100 blocks
    player.play(np.ones(16_000, dtype=np.float32), 16_000)
    assert started.wait(2)
    player.clear()
    assert player.wait_idle(2)
    assert len(blocks) < 100
    player.stop()


def test_player_bounded_queue_drops_oldest():
    gate = threading.Event()
    player = AudioPlayer(sink=lambda b, sr: gate.wait(2), max_queue=2)
    for _ in range(6):
        player.play(np.ones(10, dtype=np.float32), 16_000)
    assert player.dropped >= 3
    gate.set()
    assert player.wait_idle(5)
    player.stop()


def test_player_resamples_to_output_rate():
    rates: list[int] = []
    player = AudioPlayer(sink=lambda b, sr: rates.append(sr), output_sr=48_000)
    player.play(np.ones(1600, dtype=np.float32), 16_000)
    assert player.wait_idle(5)
    assert set(rates) == {48_000}
    player.stop()


# ── capture plumbing (no sound device) ──────────────────────────────────────


def test_capture_callback_only_enqueues_and_consumer_resamples():
    from afrilingua.audio.capture import MicrophoneCapture

    got: list[np.ndarray] = []
    cap = MicrophoneCapture(got.append, block_s=0.01)
    cap.native_sr = 48_000
    t = threading.Thread(target=cap._consume, daemon=True)
    t.start()
    x = sine(440, 0.5, 48_000).reshape(-1, 1)
    for i in range(0, len(x), 480):
        cap._callback(x[i : i + 480], 480, None, None)
    cap._queue.put(None)
    t.join(5)
    y = np.concatenate(got)
    assert abs(len(y) - 8_000) <= 64
    assert abs(dominant_freq(y, 16_000) - 440) < 5


def test_capture_queue_overflow_drops_oldest():
    from afrilingua.audio.capture import MicrophoneCapture

    cap = MicrophoneCapture(lambda a: None, block_s=0.01, max_queue_s=0.08)
    for _ in range(50):
        cap._callback(np.zeros((160, 1), dtype=np.float32), 160, None, None)
    assert cap.dropped_blocks > 0
    assert cap._queue.qsize() <= cap._max_blocks
