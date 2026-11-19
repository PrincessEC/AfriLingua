"""Top-level library API (afrilingua.translate / transcribe / speak / languages)."""

from __future__ import annotations

import numpy as np

import afrilingua


def test_top_level_api_uses_shared_engine(fake_engine, monkeypatch):
    monkeypatch.setattr(afrilingua, "_engine", fake_engine)

    assert afrilingua.translate("Good morning", to="yoruba", source="english") == "[yoruba] Good morning"
    assert afrilingua.translate(["Hello", "Thanks"], to="hausa", source="en") == [
        "[hausa] Hello",
        "[hausa] Thanks",
    ]
    assert afrilingua.transcribe(np.zeros(16000, dtype=np.float32), lang="igbo") == "hello world"
    speech = afrilingua.speak("Ndewo", lang="igbo")
    assert speech.sample_rate == 16000


def test_languages_lists_builtins():
    names = {lang["name"] for lang in afrilingua.languages()}
    assert {"english", "igbo", "yoruba", "hausa"} <= names
