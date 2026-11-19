"""Shared pytest fixtures: isolated settings + fake backends (no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from afrilingua.backends.base import SpeechAudio, Transcript


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point data dirs at a temp folder and reset cached singletons."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.setenv("LANGUAGE_REGISTRY_FILE", "")
    monkeypatch.setenv("AFRILINGUA_EXTRA_LANGUAGES", "")
    for key in ("API_HOST", "API_PORT", "AFRILINGUA_SERVE_WEB", "AFRILINGUA_LAN_PORT"):
        monkeypatch.delenv(key, raising=False)
    from afrilingua.core import hardware, languages, settings

    settings.get_settings.cache_clear()
    languages.get_registry.cache_clear()
    hardware.resolve_runtime.cache_clear()
    yield
    settings.get_settings.cache_clear()
    languages.get_registry.cache_clear()
    hardware.resolve_runtime.cache_clear()


class FakeMT:
    name = "fake-mt"

    def __init__(self) -> None:
        self._loaded = False
        self.calls: list[tuple[list[str], str, str]] = []

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def translate_batch(self, texts, src_lang, tgt_lang, *, beam_size=1, max_length=256):
        self.calls.append((list(texts), src_lang, tgt_lang))
        return [f"[{tgt_lang}] {t}" for t in texts]


class FakeASR:
    name = "fake-asr"

    def __init__(self, text: str = "hello world") -> None:
        self._loaded = False
        self.text = text

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def languages(self):
        return ["english", "igbo", "yoruba", "hausa"]

    def transcribe(self, audio, lang):
        return Transcript(text=self.text, language=lang, duration_s=len(audio) / 16000)


class FakeTTS:
    name = "fake-tts"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def languages(self):
        return ["english", "igbo", "yoruba", "hausa"]

    def synthesize(self, text, lang):
        return SpeechAudio(np.zeros(1600, dtype=np.float32), 16000)


@pytest.fixture
def fake_engine():
    """An AfriLinguaEngine wired with fake backends."""
    from afrilingua.engine import ASR_EN_KEY, ASR_MMS_KEY, MT_KEY, TTS_KEY, AfriLinguaEngine
    from afrilingua.models.manager import ModelManager

    manager = ModelManager()
    manager.register(MT_KEY, FakeMT)
    manager.register(ASR_EN_KEY, FakeASR)
    manager.register(ASR_MMS_KEY, FakeASR)
    manager.register(TTS_KEY, FakeTTS)
    return AfriLinguaEngine(manager)
