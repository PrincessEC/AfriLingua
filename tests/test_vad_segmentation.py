"""VAD segmentation: pre-roll, silence cut, max-length cut, no lost samples (fake VAD, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from afrilingua.pipeline.segmenter import SegmenterConfig, UtteranceSegmenter

SR = 16_000
W = 512


class ScheduleVAD:
    """Fake VAD: speech probability comes from a per-sample mask, by stream position."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask.astype(np.float32)
        self.pos = 0

    def __call__(self, window: np.ndarray) -> float:
        seg = self.mask[self.pos : self.pos + len(window)]
        self.pos += len(window)
        return float(seg.mean()) if seg.size else 0.0


def make_stream(total_s: float, speech: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    n = int(total_s * SR)
    # identity signal: sample i has a unique value so we can check exact positions
    audio = (np.arange(n, dtype=np.float64) / n).astype(np.float32) * 0.5 + 0.1
    mask = np.zeros(n, dtype=bool)
    for a, b in speech:
        mask[int(a * SR) : int(b * SR)] = True
    return audio, mask


def run(
    audio: np.ndarray, mask: np.ndarray, cfg: SegmenterConfig | None = None, chunk: int = 1600, flush=True
):
    seg = UtteranceSegmenter(ScheduleVAD(mask), cfg or SegmenterConfig())
    out = []
    for i in range(0, len(audio), chunk):
        out.extend(seg.push(audio[i : i + chunk]))
    if flush:
        out.extend(seg.flush())
    return out


def first_speech_window(start_s: float) -> int:
    """Index of the first 512-window whose speech fraction reaches the 0.5 threshold."""
    s = int(start_s * SR)
    k = s // W
    while ((k + 1) * W - max(k * W, s)) / W < 0.5:
        k += 1
    return k * W


def test_preroll_and_silence_cut():
    audio, mask = make_stream(4.0, [(1.0, 2.0)])
    utts = run(audio, mask)
    assert len(utts) == 1
    u = utts[0]
    assert u.reason == "silence"
    expected_start = first_speech_window(1.0) - int(0.3 * SR)
    assert u.start == expected_start  # 300 ms pre-roll
    assert np.array_equal(u.audio, audio[u.start : u.end])  # exact samples, nothing reordered
    # trailing silence kept ≤ trailing_keep_s (+1 window of detection granularity)
    assert 0 <= u.end - 2 * SR <= int(0.2 * SR) + W


def test_utterance_emitted_promptly_after_silence():
    audio, mask = make_stream(3.0, [(0.5, 1.5)])
    seg = UtteranceSegmenter(ScheduleVAD(mask), SegmenterConfig(silence_s=0.6))
    end_of_silence = int((1.5 + 0.6) * SR) + 2 * W
    got = seg.push(audio[:end_of_silence])
    assert len(got) == 1 and got[0].reason == "silence"


def test_short_blip_discarded():
    audio, mask = make_stream(2.0, [(0.5, 0.6)])  # 100 ms < min_speech 250 ms
    assert run(audio, mask) == []


def test_max_length_cut_tiles_without_loss():
    cfg = SegmenterConfig(max_utterance_s=8.0)
    audio, mask = make_stream(22.0, [(1.0, 21.0)])
    utts = run(audio, mask, cfg)
    assert len(utts) >= 3
    assert [u.reason for u in utts[:2]] == ["max_length", "max_length"]
    for a, b in zip(utts, utts[1:], strict=False):
        assert a.end == b.start  # contiguous: no gap, no overlap
    joined = np.concatenate([u.audio for u in utts])
    assert np.array_equal(joined, audio[utts[0].start : utts[-1].end])
    for u in utts[:2]:
        assert len(u.audio) <= int(8.0 * SR)
        assert len(u.audio) >= int(8.0 * SR) - int(1.0 * SR) - W


def test_max_length_cut_prefers_quiet_point():
    cfg = SegmenterConfig(max_utterance_s=8.0, preroll_s=0.0)
    n = int(12 * SR)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(n) * 0.3).astype(np.float32)
    dip = (int(7.5 * SR), int(7.6 * SR))  # quiet stretch inside the last second before 8 s
    audio[dip[0] : dip[1]] *= 0.01
    mask = np.ones(n, dtype=bool)
    utts = run(audio, mask, cfg)
    first = utts[0]
    assert first.reason == "max_length"
    assert dip[0] <= first.end <= dip[1]


def test_flush_emits_ongoing_speech_including_partial_window():
    audio, mask = make_stream(3.0, [(1.0, 3.0)])
    audio = audio[: int(2.5 * SR) + 100]  # not a multiple of the window size
    mask = mask[: audio.size]
    utts = run(audio, mask)
    assert len(utts) == 1 and utts[0].reason == "flush"
    assert utts[0].end == audio.size
    assert np.array_equal(utts[0].audio, audio[utts[0].start :])


def test_two_utterances_do_not_overlap():
    audio, mask = make_stream(5.0, [(0.5, 1.5), (2.4, 3.5)])
    utts = run(audio, mask)
    assert len(utts) == 2
    assert utts[1].start >= utts[0].end
    for u in utts:
        assert np.array_equal(u.audio, audio[u.start : u.end])


@pytest.mark.parametrize("chunk", [160, 512, 777, 4000])
def test_chunk_size_invariance(chunk):
    audio, mask = make_stream(12.0, [(0.5, 1.7), (2.5, 11.5)])
    ref = run(audio, mask, chunk=len(audio))
    got = run(audio, mask, chunk=chunk)
    assert [(u.start, u.end, u.reason) for u in got] == [(u.start, u.end, u.reason) for u in ref]


def test_current_audio_and_speech_seconds():
    audio, mask = make_stream(3.0, [(0.5, 3.0)])
    seg = UtteranceSegmenter(ScheduleVAD(mask))
    seg.push(audio[: int(2.0 * SR)])
    assert seg.in_speech
    assert 1.3 <= seg.speech_seconds <= 1.6
    cur = seg.current_audio()
    end = (int(2.0 * SR) // W) * W  # samples < one window stay pending until the next push
    assert cur.size > 0 and np.array_equal(cur, audio[end - cur.size : end])
