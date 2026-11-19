"""StreamingSession with the fake engine + fake VAD (no models)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from afrilingua.audio.pcm import float32_to_pcm16
from afrilingua.pipeline import StreamingSessionProtocol
from afrilingua.pipeline.segmenter import SegmenterConfig
from afrilingua.pipeline.streaming import EnergyVAD, StreamingSession
from tests.conftest import FakeASR, FakeMT, FakeTTS

SR = 16_000


def amp_vad(window: np.ndarray) -> float:
    """Fake VAD: loud = speech."""
    return 1.0 if float(np.abs(window).max()) > 0.05 else 0.0


def tone(seconds: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def feed(
    session: StreamingSession, audio: np.ndarray, sr: int = SR, chunk_s: float = 0.02, sleep: float = 0.0
) -> None:
    step = int(sr * chunk_s)
    for i in range(0, len(audio), step):
        session.feed_pcm16(float32_to_pcm16(audio[i : i + step]), sample_rate=sr)
        if sleep:
            time.sleep(sleep)


def wait_for(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def make_session(engine, targets, **kw):
    events: list[dict] = []
    lock = threading.Lock()

    def on_event(e):
        with lock:
            events.append(e)

    kw.setdefault("partials", False)
    kw.setdefault("vad", amp_vad)
    kw.setdefault("segmenter_config", SegmenterConfig(silence_s=0.3, max_utterance_s=8.0))
    s = StreamingSession(engine, "english", targets, on_event=on_event, **kw)
    return s, events


def types(events, **match):
    return [e for e in events if all(e.get(k) == v for k, v in match.items())]


def test_implements_protocol(fake_engine):
    s, _ = make_session(fake_engine, ["yoruba"])
    assert isinstance(s, StreamingSessionProtocol)
    assert s.src_lang == "english"


def test_event_order_and_fanout(fake_engine):
    s, events = make_session(fake_engine, ["yoruba", "ig"], synthesize_audio=True)
    s.start()
    feed(s, np.concatenate([silence(0.4), tone(1.0), silence(0.6)]))
    s.close()

    finals = types(events, type="final")
    assert len(finals) == 1 and finals[0]["seq"] == 0 and finals[0]["text"] == "hello world"
    trans = types(events, type="translation")
    assert {t["lang"]: t["text"] for t in trans} == {
        "yoruba": "[yoruba] hello world",
        "igbo": "[igbo] hello world",
    }
    assert all(t["origin"] == "mt" and t["latency_ms"] >= 0 for t in trans)
    audio = types(events, type="audio")
    assert sorted(a["lang"] for a in audio) == ["igbo", "yoruba"]
    assert all(a["format"] == "wav" and a["data"] for a in audio)
    metrics = types(events, type="metrics")
    assert len(metrics) == 1 and {"asr_ms", "mt_ms", "tts_ms", "total_ms"} <= metrics[0].keys()

    idx = {id(e): i for i, e in enumerate(events)}
    i_final = idx[id(finals[0])]
    for lang in ("yoruba", "igbo"):
        i_tr = idx[id(types(events, type="translation", lang=lang)[0])]
        i_au = idx[id(types(events, type="audio", lang=lang)[0])]
        assert i_final < i_tr < i_au
    assert idx[id(metrics[0])] == len(events) - 1


def test_audio_is_decodable_wav(fake_engine):
    import base64

    from afrilingua.audio.pcm import decode_audio

    s, events = make_session(fake_engine, ["hausa"], synthesize_audio=True)
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    a = types(events, type="audio")[0]
    wav, sr = decode_audio(base64.b64decode(a["data"]))
    assert sr == 16000 and wav.size == 1600  # FakeTTS returns 1600 samples


def test_no_audio_events_without_synthesis(fake_engine):
    s, events = make_session(fake_engine, ["yoruba"])
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    assert types(events, type="translation") and not types(events, type="audio")


def test_set_targets_between_utterances(fake_engine):
    s, events = make_session(fake_engine, ["yoruba"])
    s.start()
    feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    assert wait_for(lambda: types(events, type="translation", seq=0))
    s.set_targets(["hausa", "klingon"])  # unknown language → error event, others kept
    assert s.targets == ["hausa"]
    feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    s.close()
    assert [t["lang"] for t in types(events, type="translation", seq=1)] == ["hausa"]
    assert [t["lang"] for t in types(events, type="translation", seq=0)] == ["yoruba"]
    assert types(events, type="error", code="unsupported_language")


def test_source_language_target_is_skipped(fake_engine):
    s, events = make_session(fake_engine, ["english", "yoruba"])
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    assert [t["lang"] for t in types(events, type="translation")] == ["yoruba"]


def test_close_flushes_unfinished_utterance(fake_engine):
    s, events = make_session(fake_engine, ["yoruba"])
    s.start()
    feed(s, np.concatenate([silence(0.3), tone(1.2)]))  # no trailing silence
    s.close()
    assert len(types(events, type="final")) == 1
    assert len(types(events, type="translation")) == 1
    # idempotent close, feeding afterwards is ignored
    s.close()
    s.feed_pcm16(float32_to_pcm16(tone(0.5)))
    assert len(types(events, type="final")) == 1


def test_resamples_non_16k_input(fake_engine):
    s, events = make_session(fake_engine, ["yoruba"])
    with s:
        feed(s, np.concatenate([silence(0.3, 48_000), tone(1.0, 48_000), silence(0.5, 48_000)]), sr=48_000)
    finals = types(events, type="final")
    assert len(finals) == 1
    assert 1.0 <= finals[0]["duration_s"] <= 1.8


def test_mt_errors_become_events_and_session_survives(fake_engine):
    class BrokenMT(FakeMT):
        def translate_batch(self, texts, src_lang, tgt_lang, **kw):
            if tgt_lang == "igbo":
                raise RuntimeError("boom")
            return super().translate_batch(texts, src_lang, tgt_lang, **kw)

    from afrilingua.engine import MT_KEY

    fake_engine.manager.register(MT_KEY, BrokenMT, replace=True)
    s, events = make_session(fake_engine, ["igbo", "yoruba"])
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5), tone(0.8), silence(0.5)]))
    errs = types(events, type="error", stage="mt")
    assert errs and errs[0]["message"] == "boom" and errs[0]["code"] == "internal_error"
    assert len(types(events, type="final")) == 2
    assert [t["lang"] for t in types(events, type="translation")] == ["yoruba", "yoruba"]
    assert len(types(events, type="metrics")) == 2


def test_asr_errors_become_events(fake_engine):
    calls = {"n": 0}

    class FlakyASR(FakeASR):
        def transcribe(self, audio, lang):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("asr down")
            return super().transcribe(audio, lang)

    from afrilingua.engine import ASR_EN_KEY

    fake_engine.manager.register(ASR_EN_KEY, FlakyASR, replace=True)
    s, events = make_session(fake_engine, ["yoruba"])
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5), tone(0.8), silence(0.5)]))
    assert types(events, type="error", stage="asr")
    finals = types(events, type="final")
    assert len(finals) == 1 and finals[0]["seq"] == 0


def test_tts_errors_do_not_block_captions(fake_engine):
    class BrokenTTS(FakeTTS):
        def synthesize(self, text, lang):
            raise RuntimeError("no voice")

    from afrilingua.engine import TTS_KEY

    fake_engine.manager.register(TTS_KEY, BrokenTTS, replace=True)
    s, events = make_session(fake_engine, ["yoruba"], synthesize_audio=True)
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    assert types(events, type="translation")
    assert types(events, type="error", stage="tts")
    assert types(events, type="metrics")


def test_empty_transcripts_produce_no_final(fake_engine):
    from afrilingua.engine import ASR_EN_KEY

    fake_engine.manager.register(ASR_EN_KEY, lambda: FakeASR(text="  "), replace=True)
    s, events = make_session(fake_engine, ["yoruba"])
    with s:
        feed(s, np.concatenate([tone(0.8), silence(0.5)]))
    assert not types(events, type="final") and not types(events, type="translation")


def test_partials_precede_final(fake_engine):
    s, events = make_session(fake_engine, ["yoruba"], partials=True, partial_interval_s=0.05)
    s.start()
    feed(s, tone(2.0), chunk_s=0.1, sleep=0.03)
    feed(s, silence(0.5))
    s.close()
    partials = types(events, type="partial")
    finals = types(events, type="final")
    assert partials and len(finals) == 1
    assert all(p["seq"] == 0 and p["lang"] == "english" for p in partials)
    assert events.index(partials[-1]) < events.index(finals[0])


def test_backpressure_drops_partials_not_finals(fake_engine):
    class SlowASR(FakeASR):
        def transcribe(self, audio, lang):
            time.sleep(0.15)
            return super().transcribe(audio, lang)

    from afrilingua.engine import ASR_EN_KEY

    fake_engine.manager.register(ASR_EN_KEY, SlowASR, replace=True)
    s, events = make_session(fake_engine, [], partials=True, partial_interval_s=0.01)
    s.start()
    for _ in range(4):
        feed(s, np.concatenate([tone(0.7), silence(0.4)]), chunk_s=0.05, sleep=0.005)
    s.close()
    assert len(types(events, type="final")) == 4
    assert s.stats["partials_dropped"] >= 1


def test_feed_does_not_block_on_models(fake_engine):
    class VerySlowASR(FakeASR):
        def transcribe(self, audio, lang):
            time.sleep(0.5)
            return super().transcribe(audio, lang)

    from afrilingua.engine import ASR_EN_KEY

    fake_engine.manager.register(ASR_EN_KEY, VerySlowASR, replace=True)
    s, _ = make_session(fake_engine, ["yoruba"])
    s.start()
    t0 = time.perf_counter()
    feed(s, np.concatenate([tone(0.6), silence(0.4)] * 3))
    assert time.perf_counter() - t0 < 0.4
    s.close()


def test_energy_vad_detects_tone_vs_silence():
    vad = EnergyVAD()
    for _ in range(30):
        vad(np.zeros(512, dtype=np.float32) + 1e-4)
    assert vad(tone(0.032)[:512]) > 0.9
    assert vad(np.zeros(512, dtype=np.float32) + 1e-4) < 0.1


def test_unknown_source_language_rejected(fake_engine):
    from afrilingua.core.errors import UnsupportedLanguageError

    with pytest.raises(UnsupportedLanguageError):
        StreamingSession(fake_engine, "klingon", ["yoruba"], vad=amp_vad)
