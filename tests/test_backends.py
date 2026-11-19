"""Backend wiring and pure helpers (no model downloads)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from afrilingua.backends import _common
from afrilingua.backends.factory import VAD_KEY, FallbackBackend, build_detector, register_default_backends
from afrilingua.core.errors import ModelLoadError
from afrilingua.core.settings import get_settings
from afrilingua.engine import ASR_EN_KEY, ASR_MMS_KEY, MT_KEY, TTS_KEY
from afrilingua.models.manager import ModelManager
from tests.conftest import FakeMT

# ── factory ─────────────────────────────────────────────────────────────────


def test_register_default_backends_is_lazy():
    mgr = ModelManager()
    register_default_backends(mgr)
    assert set(mgr.keys()) == {MT_KEY, ASR_EN_KEY, ASR_MMS_KEY, TTS_KEY, VAD_KEY}
    assert not any(s["loaded"] for s in mgr.status().values())


@pytest.mark.parametrize(
    ("env", "key", "cls_name"),
    [
        ({"MT_BACKEND": "transformers"}, MT_KEY, "HFNLLBBackend"),
        ({"MT_BACKEND": "ctranslate2"}, MT_KEY, "CT2NLLBBackend"),
        ({"ASR_EN_BACKEND": "openai-whisper"}, ASR_EN_KEY, "OpenAIWhisperBackend"),
        ({"ASR_EN_BACKEND": "faster-whisper"}, ASR_EN_KEY, "FasterWhisperBackend"),
    ],
)
def test_explicit_backend_choice(monkeypatch, env, key, cls_name):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    mgr = ModelManager()
    register_default_backends(mgr)
    backend = mgr._slots[key].factory()
    assert type(backend).__name__ == cls_name


def test_auto_falls_back_when_package_missing(monkeypatch):
    monkeypatch.setattr(
        _common, "module_available", lambda name: name not in {"ctranslate2", "faster_whisper"}
    )
    import afrilingua.backends.factory as factory

    monkeypatch.setattr(factory, "module_available", _common.module_available)
    mgr = ModelManager()
    register_default_backends(mgr)
    assert type(mgr._slots[MT_KEY].factory()).__name__ == "HFNLLBBackend"
    assert type(mgr._slots[ASR_EN_KEY].factory()).__name__ == "OpenAIWhisperBackend"


def test_fallback_backend_switches_on_load_failure():
    class Broken(FakeMT):
        name = "broken"

        def load(self):
            raise RuntimeError("no ct2")

    fb = FallbackBackend(Broken, FakeMT, label="mt")
    assert not fb.is_loaded()
    fb.load()
    assert fb.is_loaded() and fb.name == "fake-mt"
    assert "no ct2" in (fb.fallback_reason or "")
    assert fb.translate_batch(["hi"], "english", "yoruba") == ["[yoruba] hi"]
    mgr = ModelManager()
    mgr.register("mt", lambda: FallbackBackend(Broken, FakeMT, label="mt"))
    with mgr.use("mt") as mt:
        assert mt.translate_batch(["a"], "english", "igbo") == ["[igbo] a"]
    assert mgr.status()["mt"]["backend"] == "fake-mt"


def test_fallback_backend_uses_primary_when_ok():
    fb = FallbackBackend(FakeMT, lambda: pytest.fail("fallback used"), label="mt")
    fb.load()
    assert fb.name == "fake-mt" and fb.fallback_reason is None


def test_build_detector():
    assert build_detector().detect("Sannu, yaya kake? Nagode.")[0] == "hausa"


# ── checkpoint helpers ──────────────────────────────────────────────────────


def test_finetuned_dir_requires_config_and_weights(tmp_path, monkeypatch):
    ft = tmp_path / "ft"
    monkeypatch.setenv("FINETUNED_MODEL_DIR", str(ft))
    get_settings.cache_clear()
    s = get_settings()
    assert _common.resolve_mt_source(s, "facebook/nllb") == "facebook/nllb"
    ft.mkdir()
    (ft / "trainer_state.json").write_text("{}")  # v1 loaded this non-empty dir
    assert _common.resolve_mt_source(s, "facebook/nllb") == "facebook/nllb"
    (ft / "config.json").write_text("{}")
    assert _common.resolve_mt_source(s, "facebook/nllb") == "facebook/nllb"
    (ft / "model.safetensors").write_bytes(b"x")
    assert _common.resolve_mt_source(s, "facebook/nllb") == str(ft)


def test_ct2_target_dir_naming(tmp_path, monkeypatch):
    from afrilingua.backends.mt_ct2 import ct2_target_dir, ensure_ct2_model

    monkeypatch.setenv("CT2_CACHE", str(tmp_path / "ct2"))
    get_settings.cache_clear()
    s = get_settings()
    d = ct2_target_dir("facebook/nllb-200-distilled-600M", "int8", s)
    assert d == tmp_path / "ct2" / "facebook--nllb-200-distilled-600M-int8"
    # an existing conversion is reused without importing ctranslate2
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    assert ensure_ct2_model("facebook/nllb-200-distilled-600M", "int8", s) == d


def test_ct2_device_fallback(monkeypatch):
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 0)
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", fake)
    assert _common.ct2_device_for("cuda") == "cpu"
    assert _common.ct2_device_for("cpu") == "cpu"
    assert _common.ct2_compute_type("cpu", "float16") == "int8"
    assert _common.ct2_compute_type("cpu", "float32") == "float32"
    assert _common.ct2_compute_type("cuda", "int8_float16") == "int8_float16"


def test_slugify():
    assert _common.slugify("facebook/nllb-200") == "facebook--nllb-200"
    slug = _common.slugify("C:\\models\\my model/")
    assert slug.endswith("models--my-model") and ":" not in slug and " " not in slug


# ── MMS checkpoint validation (replaces v1 tests/test_mms_asr_config.py) ────


def test_mms_rejects_pretraining_checkpoint():
    from afrilingua.backends.asr_mms import validate_mms_config

    cfg = types.SimpleNamespace(architectures=["Wav2Vec2ForPreTraining"], adapter_attn_dim=None)
    with pytest.raises(ModelLoadError, match="fine-tuned MMS ASR checkpoint"):
        validate_mms_config(cfg, "facebook/mms-300m")


def test_mms_accepts_adapter_ctc_checkpoint():
    from afrilingua.backends.asr_mms import validate_mms_config

    validate_mms_config(
        types.SimpleNamespace(architectures=["Wav2Vec2ForCTC"], adapter_attn_dim=16), "mms-1b-all"
    )


def test_mms_languages_and_quantize_default():
    from afrilingua.backends.asr_mms import MMSASRBackend
    from afrilingua.core.hardware import HardwareInfo, plan_for

    hw = HardwareInfo(8, 16, False, False, "None", 0)
    b = MMSASRBackend(plan=plan_for(get_settings(), hw))
    assert b.languages() == ["igbo", "yoruba", "hausa"]
    assert b.quantize is True and not b.is_loaded()


def test_whisper_languages_and_repo():
    from afrilingua.backends.asr_whisper import FasterWhisperBackend, faster_whisper_repo

    assert faster_whisper_repo("small") == "Systran/faster-whisper-small"
    assert faster_whisper_repo("distil-large-v3") == "Systran/faster-distil-whisper-large-v3"
    from afrilingua.core.hardware import HardwareInfo, plan_for

    b = FasterWhisperBackend(plan=plan_for(get_settings(), HardwareInfo(8, 16, False, False, "None", 0)))
    assert b.languages() == ["english"]


# ── TTS front-end ───────────────────────────────────────────────────────────


def test_number_expansion_en():
    from afrilingua.backends.text_frontend import expand_numbers_en, number_to_words_en

    assert number_to_words_en(0) == "zero"
    assert number_to_words_en(1204) == "one thousand two hundred four"
    assert number_to_words_en(2_000_015) == "two million fifteen"
    assert expand_numbers_en("I have 3 cats and 25% off, 1,500 people, 3.14") == (
        "I have three cats and twenty five percent off, one thousand five hundred people, three point one four"
    )


def test_filter_to_vocab_decomposes_unknown_precomposed():
    from afrilingua.backends.text_frontend import filter_to_vocab

    vocab = set("abdeẹgiklmnoọrsṣtuwy ") | {"\u0300", "\u0301"}
    # "é" is not in the vocab but e + U+0301 are → kept in decomposed form
    assert filter_to_vocab("Ẹ kú ilé!", vocab | {"Ẹ"}) == "Ẹ ku\u0301 ile\u0301"
    assert filter_to_vocab("hello\t\n world 123", set("helowrd ")) == "hello world"


def test_tts_synthesize_with_fake_voice(monkeypatch):
    """MMSTTSBackend front-end + sentence concatenation with a fake VITS voice."""
    import torch

    from afrilingua.backends import tts_mms

    seen: list[str] = []

    class Tok:
        def __call__(self, text, return_tensors="pt"):
            seen.append(text)
            return {"input_ids": torch.ones(1, len(text), dtype=torch.long)}

    class Model:
        def __call__(self, input_ids):
            return types.SimpleNamespace(waveform=torch.ones(1, 100 * input_ids.shape[-1]))

    def fake_load(self, lang, model_id):
        vocab = frozenset("abcdefghijklmnopqrstuvwxyz '")
        return tts_mms._Voice(model_id, Tok(), Model(), vocab, True, 16_000, "cpu")

    monkeypatch.setattr(tts_mms.MMSTTSBackend, "_load_voice", fake_load)
    tts = tts_mms.MMSTTSBackend(max_voices=2)
    audio = tts.synthesize("Hello there. I have 2 dogs!", "english")
    assert seen == ["hello there", "i have two dogs"]  # "." and "!" are not in the vocab
    gap = int(tts_mms._SENTENCE_GAP_S * 16_000)
    assert audio.sample_rate == 16_000
    assert audio.waveform.size == 100 * len("hello there") + gap + 100 * len("i have two dogs")
    assert audio.waveform.dtype == np.float32
    # LRU of voices
    tts.synthesize("Sannu", "hausa")
    tts.synthesize("Bawo", "yoruba")
    assert tts.loaded_voices() == ["hausa", "yoruba"]


def test_tts_english_falls_back_to_os_voice(monkeypatch):
    from afrilingua.backends import tts_mms
    from afrilingua.backends.base import SpeechAudio

    def broken(self, lang, model_id):
        raise OSError("offline")

    monkeypatch.setattr(tts_mms.MMSTTSBackend, "_load_voice", broken)
    monkeypatch.setattr(tts_mms, "module_available", lambda name: True)
    monkeypatch.setattr(
        tts_mms.Pyttsx3Voice, "synthesize", lambda self, text: SpeechAudio(np.ones(10, np.float32), 22_050)
    )
    tts = tts_mms.MMSTTSBackend(english_backend="auto")
    assert tts.synthesize("Hello", "english").sample_rate == 22_050
    with pytest.raises(ModelLoadError):
        tts.synthesize("Sannu", "hausa")
