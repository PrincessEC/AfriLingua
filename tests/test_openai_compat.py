"""OpenAI-compatible /v1 endpoints, exercised with the official `openai` SDK when installed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from afrilingua.api.app import create_app
from tests._api_fakes import wav_bytes


@pytest.fixture
def client(fake_engine):
    with TestClient(create_app(fake_engine, mount_mcp=False)) as c:
        yield c


@pytest.fixture
def openai_client(client):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url="http://testserver/v1", api_key="local", http_client=client, max_retries=0)


def test_models(client):
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert {m["id"] for m in data["data"]} == {"afrilingua-mt", "afrilingua-asr", "afrilingua-tts"}
    assert client.get("/v1/models/afrilingua-mt").json()["id"] == "afrilingua-mt"
    assert client.get("/v1/models/gpt-4").status_code == 404


def test_transcriptions_formats(client):
    def post(fmt: str):
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
            data={"model": "whisper-1", "language": "yo", "response_format": fmt},
        )

    assert post("json").json() == {"text": "hello world"}
    assert post("text").text == "hello world"
    verbose = post("verbose_json").json()
    assert verbose["task"] == "transcribe" and verbose["language"] == "yoruba"
    assert verbose["duration"] == pytest.approx(1.0, abs=0.01)
    assert verbose["segments"][0]["text"] == "hello world"
    srt = post("srt").text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\nhello world")
    vtt = post("vtt").text
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world")
    bad = post("xml")
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_input"
    assert bad.json()["error"]["type"] == "invalid_request_error"


def test_translations_with_target_language(client):
    r = client.post(
        "/v1/audio/translations",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"language": "english", "target_language": "hausa"},
    )
    assert r.json() == {"text": "[hausa] hello world"}
    verbose = client.post(
        "/v1/audio/translations",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"language": "english", "target_language": "ig", "response_format": "verbose_json"},
    ).json()
    assert verbose["source_text"] == "hello world" and verbose["target_language"] == "igbo"
    assert verbose["segments"][0]["text"] == "[igbo] hello world"


def test_speech(client):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Bawo ni", "voice": "yoruba"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert r.headers["x-voice-language"] == "yoruba"
    # OpenAI voice names fall back to English
    assert (
        client.post("/v1/audio/speech", json={"input": "hi", "voice": "alloy"}).headers["x-voice-language"]
        == "english"
    )
    pcm = client.post("/v1/audio/speech", json={"input": "hi", "voice": "ha", "response_format": "pcm"})
    assert pcm.status_code == 200 and len(pcm.content) == 1600 * 2
    bad = client.post("/v1/audio/speech", json={"input": "hi", "voice": "klingon"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "unsupported_language"


def test_official_openai_sdk(openai_client):
    models = openai_client.models.list()
    assert "afrilingua-asr" in {m.id for m in models.data}

    tr = openai_client.audio.transcriptions.create(
        model="afrilingua", file=("a.wav", wav_bytes(), "audio/wav"), language="ig"
    )
    assert tr.text == "hello world"

    text = openai_client.audio.transcriptions.create(
        model="afrilingua", file=("a.wav", wav_bytes(), "audio/wav"), language="en", response_format="text"
    )
    assert "hello world" in str(text)

    tl = openai_client.audio.translations.create(
        model="afrilingua",
        file=("a.wav", wav_bytes(), "audio/wav"),
        extra_body={"language": "english", "target_language": "yoruba"},
    )
    assert tl.text == "[yoruba] hello world"

    speech = openai_client.audio.speech.create(
        model="afrilingua-tts", input="Sannu", voice="hausa", response_format="wav"
    )
    assert speech.content[:4] == b"RIFF"
