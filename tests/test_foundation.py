"""Foundation tests: settings, hardware plan, language registry, model manager, engine facade."""
from __future__ import annotations

import threading

import pytest

from afrilingua.core.errors import UnsupportedLanguageError
from afrilingua.core.hardware import HardwareInfo, plan_for
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import Settings
from afrilingua.models.manager import ModelManager
from tests.conftest import FakeMT

# ── hardware ────────────────────────────────────────────────────────────────

CPU_HW = HardwareInfo(cpu_count=8, ram_gb=16, cuda_available=False, cuda_built=False, gpu_name="None", vram_gb=0)
LOW_HW = HardwareInfo(cpu_count=4, ram_gb=4, cuda_available=False, cuda_built=False, gpu_name="None", vram_gb=0)
GPU_HW = HardwareInfo(cpu_count=8, ram_gb=32, cuda_available=True, cuda_built=True, gpu_name="RTX", vram_gb=12)


def test_cpu_plan_uses_int8():
    plan = plan_for(Settings(), CPU_HW)
    assert plan.device == "cpu"
    assert plan.compute_type == "int8"
    assert plan.profile == "cpu-standard"
    assert plan.model_whisper == "small"


def test_low_ram_selects_cpu_lite():
    assert plan_for(Settings(), LOW_HW).profile == "cpu-lite"


def test_gpu_plan_and_cuda_fallback():
    assert plan_for(Settings(), GPU_HW).device == "cuda"
    assert plan_for(Settings(device="cuda"), CPU_HW).device == "cpu"
    assert plan_for(Settings(device="cpu"), GPU_HW).device == "cpu"


def test_whisper_id_normalized():
    assert plan_for(Settings(model_whisper="openai/whisper-medium"), CPU_HW).model_whisper == "medium"


# ── languages ───────────────────────────────────────────────────────────────

def test_registry_aliases():
    reg = get_registry()
    assert reg.normalize("YO") == "yoruba"
    assert reg.normalize("ibo_Latn") == "igbo"
    assert reg.normalize("hau") == "hausa"
    with pytest.raises(UnsupportedLanguageError):
        reg.normalize("klingon")


def test_register_language_persists(tmp_path):
    reg = get_registry()
    spec = reg.register({"name": "Swahili", "nllb_code": "swh_Latn", "aliases": ["sw"], "mms_asr_code": "swh"})
    assert spec.name == "swahili"
    assert reg.normalize("sw") == "swahili"
    assert (reg._user_file).exists()


# ── model manager ───────────────────────────────────────────────────────────

def test_manager_loads_once_under_concurrency():
    created = []

    def factory():
        m = FakeMT()
        created.append(m)
        return m

    mgr = ModelManager()
    mgr.register("mt", factory)
    threads = [threading.Thread(target=lambda: mgr.get("mt")) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
    assert mgr.status()["mt"]["loaded"] is True


# ── engine facade ───────────────────────────────────────────────────────────

def test_translate_passthrough_cache_and_mt(fake_engine):
    r = fake_engine.translate(["Good  morning", "", "Good morning"], "english", "yoruba")
    assert r[0].text == "[yoruba] Good morning" and r[0].origin == "mt"
    assert r[1].origin == "passthrough"
    # identical normalized text in the same batch both go to MT once each batch; second call is cached
    again = fake_engine.translate_one("Good morning", "en", "yo")
    assert again.origin == "cache"
    same = fake_engine.translate_one("Hello", "english", "english")
    assert same.origin == "passthrough" and same.text == "Hello"


def test_translate_uses_approved_tm_hit(fake_engine):
    from afrilingua.engine import TMMatch

    class TM:
        def lookup(self, text, src, tgt, *, k=3):
            return [TMMatch("1", text, "Ẹ káàárọ̀", 1.0, True, "community")]

        def add(self, *a, **kw):
            return "x"

    fake_engine.tm = TM()
    r = fake_engine.translate_one("Good morning", "english", "yoruba")
    assert r.origin == "tm_exact" and r.text == "Ẹ káàárọ̀"


def test_nfc_normalization(fake_engine):
    decomposed = "Ẹ ka\u0300a\u0301ro\u0300"  # combining marks
    r = fake_engine.translate_one(decomposed, "yoruba", "english")
    import unicodedata

    assert r.source_text == unicodedata.normalize("NFC", decomposed)
