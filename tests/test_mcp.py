"""MCP server tests: tools/resources/prompts in-process via the official SDK client, plus the /mcp HTTP mount."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.client.client import Client

from afrilingua.api.app import create_app
from afrilingua.api.rooms import RoomManager
from afrilingua.mcp.server import build_mcp
from tests._api_fakes import FakeContrib, FakeDetector, FakeStreamingSession, FakeTM, wav_b64

EXPECTED_TOOLS = {
    "translate_text",
    "translate_batch",
    "detect_language",
    "transcribe_audio",
    "synthesize_speech",
    "translate_speech",
    "search_translation_memory",
    "list_languages",
    "submit_translation",
    "submit_correction",
    "create_live_room",
}


@pytest.fixture
def tm():
    t = FakeTM()
    t.add("Good morning", "Ẹ káàárọ̀", "english", "yoruba", approved=True)
    return t


@pytest.fixture
def contrib():
    return FakeContrib()


@pytest.fixture
def server(fake_engine, tm, contrib):
    fake_engine.detector = FakeDetector()
    fake_engine.tm = tm
    return build_mcp(lambda: fake_engine, contrib_provider=lambda: contrib)


def payload(result) -> Any:
    assert not result.is_error, result.content
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_lists_all_tools_with_descriptions(server):
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 40, tool.name
    translate = next(t for t in tools if t.name == "translate_text")
    assert set(translate.input_schema["required"]) == {"text", "target_language"}
    room = next(t for t in tools if t.name == "create_live_room")
    assert "ctx" not in room.input_schema.get("properties", {})


async def test_translate_and_detect(server):
    async with Client(server) as client:
        res = payload(
            await client.call_tool(
                "translate_text", {"text": "Thank you", "target_language": "ig", "source_language": "en"}
            )
        )
        assert res["text"] == "[igbo] Thank you" and res["tgt_lang"] == "igbo"
        tm_hit = payload(
            await client.call_tool(
                "translate_text",
                {"text": "Good morning", "target_language": "yoruba", "source_language": "english"},
            )
        )
        assert tm_hit["text"] == "Ẹ káàárọ̀" and tm_hit["origin"] == "tm_exact"
        batch = payload(
            await client.call_tool("translate_batch", {"texts": ["a", "b"], "target_language": "hausa"})
        )
        assert [r["text"] for r in batch["results"]] == ["[hausa] a", "[hausa] b"]
        det = payload(await client.call_tool("detect_language", {"text": "Ẹ ṣé"}))
        assert det == {"language": "yoruba", "confidence": 0.91}
        langs = payload(await client.call_tool("list_languages", {}))
        assert {"english", "igbo", "yoruba", "hausa"} <= {lang["name"] for lang in langs["languages"]}


async def test_errors_are_readable(server):
    async with Client(server) as client:
        bad = await client.call_tool("translate_text", {"text": "hi", "target_language": "klingon"})
        assert bad.is_error and "unsupported_language" in bad.content[0].text


async def test_speech_tools(server):
    async with Client(server) as client:
        tr = payload(
            await client.call_tool("transcribe_audio", {"language": "hausa", "audio_base64": wav_b64()})
        )
        assert tr["text"] == "hello world" and tr["language"] == "hausa"

        no_audio = await client.call_tool("transcribe_audio", {"language": "hausa"})
        assert no_audio.is_error and "audio_base64" in no_audio.content[0].text

        path_denied = await client.call_tool("transcribe_audio", {"language": "en", "file_path": "C:/x.wav"})
        assert path_denied.is_error and "stdio" in path_denied.content[0].text

        speech = await client.call_tool("synthesize_speech", {"text": "Ndewo", "language": "igbo"})
        assert not speech.is_error
        kinds = [c.type for c in speech.content]
        assert kinds == ["text", "audio"]
        audio = speech.content[1]
        assert audio.mime_type == "audio/wav" and base64.b64decode(audio.data)[:4] == b"RIFF"

        both = await client.call_tool(
            "translate_speech",
            {
                "target_language": "yoruba",
                "source_language": "english",
                "audio_base64": wav_b64(),
                "include_audio": True,
            },
        )
        assert [c.type for c in both.content] == ["text", "audio"]
        info = json.loads(both.content[0].text)
        assert info["transcript"] == "hello world" and info["translation"] == "[yoruba] hello world"


async def test_file_path_allowed_for_stdio(fake_engine, tmp_path):
    from tests._api_fakes import wav_bytes

    f = tmp_path / "clip.wav"
    f.write_bytes(wav_bytes())
    server = build_mcp(lambda: fake_engine, allow_file_paths=True)
    async with Client(server) as client:
        res = payload(
            await client.call_tool("transcribe_audio", {"language": "english", "file_path": str(f)})
        )
        assert res["text"] == "hello world"


async def test_translation_memory_search(server):
    async with Client(server) as client:
        res = payload(
            await client.call_tool(
                "search_translation_memory",
                {"query": "Good morning", "source_language": "en", "target_language": "yo"},
            )
        )
    assert res["matches"][0]["target"] == "Ẹ káàárọ̀" and res["matches"][0]["score"] == 1.0
    assert "took_ms" in res


async def test_contributions_require_consent(server, contrib):
    base = {
        "source_text": "Welcome",
        "target_text": "Ẹ káàbọ̀",
        "source_language": "english",
        "target_language": "yoruba",
        "license": "CC-BY-4.0",
    }
    async with Client(server) as client:
        denied = await client.call_tool("submit_translation", {**base, "consent": False})
        assert denied.is_error and "consent" in denied.content[0].text
        assert contrib.submitted == []

        ok = payload(
            await client.call_tool("submit_translation", {**base, "consent": True, "dialect": "Ọ̀yọ́"})
        )
        assert ok["status"] == "pending" and ok["id"] == "c1"
        sub = contrib.submitted[0]
        assert sub["kind"] == "translation" and sub["consent"] is True and sub["license"] == "CC-BY-4.0"
        assert sub["src_lang"] == "english" and sub["dialect"] == "Ọ̀yọ́"

        bad_license = await client.call_tool(
            "submit_translation", {**base, "consent": True, "license": "MIT"}
        )
        assert bad_license.is_error

        corr = payload(
            await client.call_tool(
                "submit_correction",
                {
                    "source_text": "Good night",
                    "machine_translation": "O dabọ",
                    "corrected_translation": "O dàárọ̀",
                    "source_language": "en",
                    "target_language": "yo",
                    "consent": True,
                    "license": "CC0-1.0",
                },
            )
        )
        assert corr["kind"] == "correction" and contrib.submitted[1]["original_mt"] == "O dabọ"


async def test_create_live_room_without_http_server(server):
    async with Client(server) as client:
        res = payload(await client.call_tool("create_live_room", {"source_language": "english"}))
    assert res["error"] == "http_server_required" and "afrilingua serve" in res["message"]


async def test_create_live_room_with_rooms(fake_engine):
    rooms = RoomManager(fake_engine, session_factory=FakeStreamingSession)
    server = build_mcp(lambda: fake_engine, rooms_provider=lambda: rooms)
    async with Client(server) as client:
        res = payload(
            await client.call_tool(
                "create_live_room",
                {"source_language": "en", "target_languages": ["yo", "ha"], "title": "Keynote"},
            )
        )
    room = rooms.get(res["id"])
    assert room is not None and room.targets == ["yoruba", "hausa"]
    assert res["host_token"] == room.host_token
    assert res["host_ws_url"].startswith("ws://") and res["host_ws_url"].endswith(room.host_token)
    assert res["join_url"].endswith(f"/room/{room.id}")


async def test_resources_and_prompts(server):
    async with Client(server) as client:
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert {"afrilingua://languages", "afrilingua://status", "afrilingua://tm/stats"} <= uris
        langs = json.loads((await client.read_resource("afrilingua://languages")).contents[0].text)
        assert any(lang["name"] == "yoruba" for lang in langs["languages"])
        stats = json.loads((await client.read_resource("afrilingua://tm/stats")).contents[0].text)
        assert stats == {"available": True, "entries": 1, "approved": 1, "by_pair": {}}
        status = json.loads((await client.read_resource("afrilingua://status")).contents[0].text)
        assert "models" in status
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert prompts == {"interpret_for_audience", "localize_document"}
        prompt = await client.get_prompt(
            "interpret_for_audience", {"target_languages": "yoruba, igbo", "topic": "health"}
        )
        assert "create_live_room" in prompt.messages[0].content.text


# ── Streamable HTTP mount ───────────────────────────────────────────────────
MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _rpc(method: str, params: dict[str, Any] | None = None, rid: int = 1) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


def _sse_json(text: str) -> dict[str, Any]:
    if text.lstrip().startswith("{"):
        return json.loads(text)
    data = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
    return json.loads(data[-1])


def test_http_mount_lists_tools_and_creates_rooms(fake_engine):
    app = create_app(fake_engine, session_factory=FakeStreamingSession)
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        r = c.post("/mcp", json=_rpc("tools/list"), headers=MCP_HEADERS)
        assert r.status_code == 200, r.text
        names = {t["name"] for t in _sse_json(r.text)["result"]["tools"]}
        assert names == EXPECTED_TOOLS

        r = c.post(
            "/mcp",
            json=_rpc(
                "tools/call", {"name": "create_live_room", "arguments": {"source_language": "english"}}, 2
            ),
            headers=MCP_HEADERS,
        )
        result = _sse_json(r.text)["result"]
        room = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        assert room["join_url"].startswith("http://127.0.0.1:8000/room/")
        assert c.get(f"/api/v1/rooms/{room['id']}").status_code == 200

        # DNS-rebinding protection on loopback
        evil = c.post("/mcp", json=_rpc("tools/list"), headers={**MCP_HEADERS, "host": "evil.example"})
        assert evil.status_code in {400, 403, 421}


def test_http_mount_respects_api_keys(monkeypatch, fake_engine):
    from afrilingua.core.settings import get_settings

    monkeypatch.setenv("API_KEYS", "mcp-key")
    get_settings.cache_clear()
    with TestClient(create_app(fake_engine), base_url="http://127.0.0.1:8000") as c:
        assert c.post("/mcp", json=_rpc("tools/list"), headers=MCP_HEADERS).status_code == 401
        ok = c.post(
            "/mcp", json=_rpc("tools/list"), headers={**MCP_HEADERS, "authorization": "Bearer mcp-key"}
        )
        assert ok.status_code == 200
