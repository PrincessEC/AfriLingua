"""REST API tests (fake engine, no model downloads)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from afrilingua.api.app import create_app
from afrilingua.core.settings import get_settings
from tests._api_fakes import FakeDetector, wav_bytes


@pytest.fixture
def client(fake_engine):
    fake_engine.detector = FakeDetector()
    app = create_app(fake_engine, mount_mcp=False)
    with TestClient(app) as c:
        yield c


def _keyed_app(monkeypatch, fake_engine, keys: str = "secret-key"):
    monkeypatch.setenv("API_KEYS", keys)
    get_settings.cache_clear()
    return create_app(fake_engine, mount_mcp=False)


# ── system ──────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "2.0.0.dev0"}
    assert r.headers["server-timing"].startswith("app;dur=")


def test_languages(client):
    langs = client.get("/api/v1/languages").json()["languages"]
    by_name = {lang["name"]: lang for lang in langs}
    assert {"english", "igbo", "yoruba", "hausa"} <= set(by_name)
    yo = by_name["yoruba"]
    for key in (
        "nllb_code",
        "display_name",
        "native_name",
        "iso639_1",
        "aliases",
        "supports_asr",
        "supports_tts",
        "builtin",
    ):
        assert key in yo
    assert yo["nllb_code"] == "yor_Latn"


def test_status(client):
    data = client.get("/api/v1/status").json()
    assert {"runtime", "hardware", "models", "languages", "translation_memory"} <= set(data)
    assert data["rooms"] == {"rooms": 0, "live": 0, "listeners": 0}
    assert "warmup" in data


# ── translation ─────────────────────────────────────────────────────────────
def test_translate_single(client):
    r = client.post(
        "/api/v1/translate", json={"text": "Good morning", "src_lang": "english", "tgt_lang": "yo"}
    )
    assert r.status_code == 200
    (res,) = r.json()["results"]
    assert res["text"] == "[yoruba] Good morning"
    assert res["tgt_lang"] == "yoruba" and res["src_lang"] == "english"
    assert res["origin"] == "mt" and res["detected"] is False
    assert set(res) == {
        "text",
        "source_text",
        "src_lang",
        "tgt_lang",
        "origin",
        "latency_ms",
        "detected",
        "matches",
    }


def test_translate_batch_auto_detect_and_cache(client):
    body = {"texts": ["Hello", "Thank you"], "src_lang": "auto", "tgt_lang": "igbo", "mode": "document"}
    first = client.post("/api/v1/translate", json=body).json()["results"]
    assert [r["text"] for r in first] == ["[igbo] Hello", "[igbo] Thank you"]
    assert all(r["detected"] for r in first)
    second = client.post("/api/v1/translate", json=body).json()["results"]
    assert {r["origin"] for r in second} == {"cache"}


def test_translate_errors(client):
    r = client.post("/api/v1/translate", json={"text": "hi", "tgt_lang": "klingon"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_language"
    assert "klingon" in r.json()["error"]["message"]

    r = client.post("/api/v1/translate", json={"tgt_lang": "yoruba"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_input"

    r = client.post("/api/v1/translate", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input"


def test_detect(client):
    assert client.post("/api/v1/detect", json={"text": "Ẹ káàárọ̀"}).json() == {
        "language": "yoruba",
        "confidence": 0.91,
    }
    assert client.post("/api/v1/detect", json={"text": ""}).status_code == 400


# ── speech ──────────────────────────────────────────────────────────────────
def test_transcribe_wav(client):
    r = client.post(
        "/api/v1/transcribe", files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")}, data={"lang": "yoruba"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world" and body["language"] == "yoruba"
    assert body["duration_s"] == pytest.approx(1.0, abs=0.01)
    assert set(body) == {"text", "language", "segments", "confidence", "duration_s"}


def test_transcribe_bad_audio(client):
    r = client.post(
        "/api/v1/transcribe", files={"file": ("a.wav", b"garbage-bytes", "audio/wav")}, data={"lang": "en"}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input"


def test_speech_returns_wav(client):
    r = client.post("/api/v1/speech", json={"text": "Ẹ káàbọ̀", "lang": "yoruba"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.headers["x-sample-rate"] == "16000"


# ── errors / middleware ─────────────────────────────────────────────────────
def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_upload_limit(monkeypatch, fake_engine):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    with TestClient(create_app(fake_engine, mount_mcp=False)) as c:
        big = b"\x00" * (2 * 1024 * 1024)
        r = c.post("/api/v1/transcribe", files={"file": ("a.wav", big, "audio/wav")}, data={"lang": "en"})
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"
        # chunked body without Content-Length
        r = c.post(
            "/api/v1/translate",
            content=iter([b"{" + b" " * (700 * 1024), b" " * (700 * 1024) + b"}"]),
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413


def test_cors_preflight(client):
    r = client.options(
        "/api/v1/translate",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_auth_off_by_default(client):
    assert client.get("/api/v1/languages").status_code == 200


def test_auth_on(monkeypatch, fake_engine):
    with TestClient(_keyed_app(monkeypatch, fake_engine)) as c:
        assert c.get("/api/v1/health").status_code == 200  # exempt
        assert c.get("/").status_code == 200  # SPA / static exempt
        r = c.get("/api/v1/languages")
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"
        assert r.headers["www-authenticate"] == "Bearer"
        assert c.get("/api/v1/languages", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/v1/languages", headers={"Authorization": "Bearer secret-key"}).status_code == 200
        assert c.get("/api/v1/languages?key=secret-key").status_code == 200
        r = c.get("/v1/models")
        assert r.status_code == 401 and r.json()["error"]["type"] == "invalid_request_error"
        # CORS preflight still works without a key
        pre = c.options(
            "/api/v1/languages",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
        assert pre.status_code == 200


# ── documents ───────────────────────────────────────────────────────────────
def test_document_job_lifecycle(client):
    def fake_translate(engine, src_path, out_dir, tgt_lang, src_lang=None, on_progress=None):
        for i in range(1, 5):
            on_progress(i, 4)
        out = Path(out_dir) / f"{Path(src_path).stem}.{tgt_lang}{Path(src_path).suffix}"
        out.write_text(f"[{tgt_lang}] " + Path(src_path).read_text(encoding="utf-8"), encoding="utf-8")
        return out

    client.app.state.jobs._translator = fake_translate
    r = client.post(
        "/api/v1/documents",
        files={"file": ("note.txt", b"Hello there", "text/plain")},
        data={"tgt_lang": "ha"},
    )
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued" and job["id"]
    for _ in range(100):
        status = client.get(f"/api/v1/documents/{job['id']}").json()
        if status["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert status["status"] == "done", status
    assert status["progress"] == 1.0 and status["filename"] == "note.txt"
    dl = client.get(f"/api/v1/documents/{job['id']}/download")
    assert dl.status_code == 200 and dl.text == "[hausa] Hello there"
    assert "attachment" in dl.headers["content-disposition"]


def test_document_errors(client):
    r = client.post(
        "/api/v1/documents",
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
        data={"tgt_lang": "ha"},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input"
    assert client.get("/api/v1/documents/nope").status_code == 404


def test_document_job_failure_is_reported(client):
    def boom(*_a, **_k):
        raise RuntimeError("converter crashed")

    client.app.state.jobs._translator = boom
    job = client.post(
        "/api/v1/documents", files={"file": ("a.txt", b"x", "text/plain")}, data={"tgt_lang": "igbo"}
    ).json()
    for _ in range(100):
        status = client.get(f"/api/v1/documents/{job['id']}").json()
        if status["status"] == "error":
            break
        time.sleep(0.02)
    assert status["status"] == "error" and "converter crashed" in status["error"]
    assert client.get(f"/api/v1/documents/{job['id']}/download").status_code == 409


# ── SPA hosting ─────────────────────────────────────────────────────────────
def test_spa_fallback_page_when_not_built(fake_engine, tmp_path):
    with TestClient(create_app(fake_engine, mount_mcp=False, web_dist=tmp_path / "missing")) as c:
        r = c.get("/")
        assert r.status_code == 200 and "pnpm --dir web build" in r.text


def _write_site(dist):
    """A miniature `vike build` output: dist/client with prerendered pages."""
    client = dist / "client"
    pages = {
        "index.html": "<html>HOME</html>",
        "translate/index.html": "<html>TRANSLATE</html>",
        "room/_/index.html": "<html>ROOM-TEMPLATE</html>",
        "404.html": "<html>NOT-FOUND</html>",
        "assets/app.js": "console.log(1)",
        "robots.txt": "User-agent: *\nAllow: /\n",
        "translate/index.pageContext.json": "{}",
    }
    for rel, body in pages.items():
        f = client / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return client


def test_spa_fallback_page_when_client_dir_missing(fake_engine, tmp_path):
    # dist/ exists but has no client/ (e.g. a stale pre-Vike build): still the build hint.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>OLD</html>", encoding="utf-8")
    with TestClient(create_app(fake_engine, mount_mcp=False, web_dist=dist)) as c:
        r = c.get("/translate")
        assert r.status_code == 200 and "pnpm --dir web build" in r.text


def test_spa_serves_index_and_assets(fake_engine, tmp_path):
    dist = tmp_path / "dist"
    _write_site(dist)
    with TestClient(create_app(fake_engine, mount_mcp=False, web_dist=dist)) as c:
        assert c.get("/").text == "<html>HOME</html>"
        assert c.get("/translate").text == "<html>TRANSLATE</html>"  # <route>/index.html
        assert c.get("/translate/").text == "<html>TRANSLATE</html>"
        assert c.get("/translate/index.html").text == "<html>TRANSLATE</html>"
        # Dynamic client-only route: every /room/<id> gets the prerendered template.
        assert c.get("/room/abc-123").text == "<html>ROOM-TEMPLATE</html>"
        assert c.get("/room/ABC123/?pin=1").text == "<html>ROOM-TEMPLATE</html>"
        assert "no-cache" in c.get("/room/abc-123").headers["cache-control"]
        # Unknown routes get Vike's prerendered 404.html with a real 404 status.
        for bad in ("/room", "/room/a/b", "/nope", "/translate/deeper"):
            r = c.get(bad)
            assert r.status_code == 404 and r.text == "<html>NOT-FOUND</html>", bad
        asset = c.get("/assets/app.js")
        assert asset.text == "console.log(1)" and "immutable" in asset.headers["cache-control"]
        assert c.get("/assets/missing.js").status_code == 404
        assert c.get("/robots.txt").text.startswith("User-agent")
        ctx = c.get("/translate/index.pageContext.json")
        assert ctx.status_code == 200 and ctx.headers["cache-control"] == "no-cache"
        assert c.get("/api/v1/nope").json()["error"]["code"] == "not_found"
        assert c.get("/v1/nope").status_code == 404
        assert c.get("/../../etc/passwd").status_code in {200, 404}  # never escapes dist/client
        assert c.get("/docs").status_code == 200
        assert c.head("/translate").status_code == 200


def test_spa_resolver_blocks_traversal(tmp_path):
    from afrilingua.api.app import _resolve_spa_path

    dist = tmp_path / "dist"
    _write_site(dist)
    (dist / "secret.txt").write_text("nope", encoding="utf-8")  # sibling of client/, outside the root
    assert _resolve_spa_path(dist, "../secret.txt") == ("missing", None)
    assert _resolve_spa_path(dist, "secret.txt")[0] == "missing"
    assert _resolve_spa_path(dist, "translate/../../secret.txt") == ("missing", None)


def test_no_web(fake_engine):
    with TestClient(create_app(fake_engine, serve_web=False, mount_mcp=False)) as c:
        assert c.get("/").status_code == 404


def test_openapi_has_tags(client):
    spec = client.get("/openapi.json").json()
    tags = {t["name"] for t in spec["tags"]}
    assert {"system", "translation", "speech", "documents", "rooms", "openai-compatible"} <= tags
    assert "/api/v1/translate" in spec["paths"]
