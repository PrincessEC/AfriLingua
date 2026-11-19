"""Live room tests: host + listeners over TestClient WebSockets with a fake StreamingSession."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from afrilingua.api.app import create_app
from afrilingua.api.rooms import RoomManager, Subscriber
from afrilingua.core.settings import get_settings
from tests._api_fakes import FakeStreamingSession

START = json.dumps({"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le"})
FRAME = b"\x01\x00" * 320  # 20 ms of PCM16 @ 16 kHz


@pytest.fixture(autouse=True)
def _reset_sessions():
    FakeStreamingSession.instances.clear()
    yield
    FakeStreamingSession.instances.clear()


@pytest.fixture
def client(fake_engine):
    app = create_app(fake_engine, mount_mcp=False, session_factory=FakeStreamingSession)
    with TestClient(app) as c:
        yield c


def recv_until(ws, pred, limit: int = 60) -> list[dict[str, Any]]:
    """Receive events until one matches ``pred``; returns every event received (inclusive)."""
    seen = []
    for _ in range(limit):
        event = ws.receive_json()
        seen.append(event)
        if pred(event):
            return seen
    raise AssertionError(f"expected event never arrived; got {seen}")


def is_type(t: str, seq: int | None = None, lang: str | None = None):
    def pred(e: dict[str, Any]) -> bool:
        return (
            e.get("type") == t
            and (seq is None or e.get("seq") == seq)
            and (lang is None or e.get("lang") == lang)
            and not e.get("replay")
        )

    return pred


def create_room(client, **kw) -> dict[str, Any]:
    body = {"title": "Town hall", "src_lang": "english", "targets": ["yoruba"], **kw}
    r = client.post("/api/v1/rooms", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_create_and_get_room(client):
    room = create_room(client, pin="4821")
    assert len(room["id"]) == 7 and room["id"][3] == "-"
    assert room["host_token"] and room["join_url"].endswith(f"/room/{room['id']}")
    assert room["host_ws"] == f"/api/v1/rooms/{room['id']}/host"
    info = client.get(f"/api/v1/rooms/{room['id'].upper()}").json()  # case-insensitive
    assert info["id"] == room["id"] and info["title"] == "Town hall"
    assert info["src_lang"] == "english" and info["targets"] == ["yoruba"]
    assert info["listeners"] == 0 and info["live"] is False and info["pin_required"] is True
    assert "host_token" not in info and "pin" not in info
    assert client.get("/api/v1/rooms/zzz-zzz").status_code == 404


def test_join_url_uses_browser_origin(client):
    r = client.post(
        "/api/v1/rooms",
        json={"src_lang": "english", "targets": ["igbo"]},
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.json()["join_url"].startswith("http://localhost:5173/room/")


def test_create_room_validates_languages(client):
    r = client.post("/api/v1/rooms", json={"src_lang": "english", "targets": ["klingon"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_language"


def test_fanout_per_language_audio_opt_in_and_late_join(client):
    room = create_room(client)
    rid, token = room["id"], room["host_token"]
    with (
        client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yo&audio=1") as yo,
        client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=igbo") as ig,
    ):
        recv_until(yo, lambda e: e["type"] == "status" and e["listeners"] == 2)
        recv_until(ig, lambda e: e["type"] == "status" and e["listeners"] == 2)
        assert client.get(f"/api/v1/rooms/{rid}").json()["languages"] == {"yoruba": 1, "igbo": 1}

        with client.websocket_connect(f"/api/v1/rooms/{rid}/host?token={token}") as host:
            host.send_text(START)
            live = recv_until(host, lambda e: e["type"] == "status" and e["live"])[-1]
            assert live["listeners"] == 2
            recv_until(yo, lambda e: e["type"] == "status" and e["live"])
            recv_until(ig, lambda e: e["type"] == "status" and e["live"])

            session = FakeStreamingSession.instances[0]
            assert session.started and session.src_lang == "english"
            assert sorted(session.targets) == ["igbo", "yoruba"]  # room targets ∪ listener languages
            assert session.synthesize_audio is True

            host.send_bytes(FRAME)
            yo_events = recv_until(yo, is_type("audio", 1))
            ig_events = recv_until(ig, is_type("translation", 1))
            host_events = recv_until(host, is_type("metrics", 1))

            assert [e["type"] for e in yo_events] == ["partial", "final", "translation", "audio"]
            assert all(e["lang"] in {"english", "yoruba"} for e in yo_events)
            assert yo_events[3]["lang"] == "yoruba" and yo_events[3]["format"] == "wav"
            assert [e["type"] for e in ig_events] == ["partial", "final", "translation"]
            assert ig_events[-1]["lang"] == "igbo" and ig_events[-1]["text"] == "[igbo] 1"
            # host sees captions + every translation + metrics, never audio
            assert {e["type"] for e in host_events} == {"partial", "final", "translation", "metrics"}
            assert {e["lang"] for e in host_events if e["type"] == "translation"} == {"igbo", "yoruba"}

            # igbo listener did not opt into audio: nothing between translation#1 and partial#2
            host.send_bytes(FRAME)
            between = recv_until(ig, is_type("partial", 2))
            assert [e["type"] for e in between] == ["partial"]
            recv_until(yo, is_type("audio", 2))
            assert session.fed == 2 * len(FRAME)

            # late joiner gets recent captions replayed
            with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yoruba") as late:
                replay = recv_until(late, lambda e: e["type"] == "status")
                replayed = [e for e in replay if e.get("replay")]
                assert [(e["type"], e["seq"]) for e in replayed] == [
                    ("final", 1),
                    ("translation", 1),
                    ("final", 2),
                    ("translation", 2),
                ]
                assert all(e["lang"] in {"english", "yoruba"} for e in replayed)
                assert replay[-1]["listeners"] == 3
            recv_until(host, lambda e: e["type"] == "status" and e["listeners"] == 2)

        # host left: live → false, session closed
        status = recv_until(yo, lambda e: e["type"] == "status" and not e["live"])[-1]
        assert status["host"] is False
        assert session.closed


def test_targets_follow_listeners(client):
    room = create_room(client, targets=[])
    rid = room["id"]
    with client.websocket_connect(f"/api/v1/rooms/{rid}/host?token={room['host_token']}") as host:
        host.send_text(START)
        recv_until(host, lambda e: e["type"] == "status" and e["live"])
        session = FakeStreamingSession.instances[0]
        assert session.targets == []
        with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=hausa&audio=1") as ha:
            recv_until(host, lambda e: e["type"] == "status" and e["listeners"] == 1)
            assert session.targets == ["hausa"]
            assert session.audio_history[-1] == ["hausa"]
            ha.send_text(json.dumps({"type": "subscribe", "lang": "igbo", "audio": False}))
            recv_until(host, lambda e: e["type"] == "status" and e["languages"] == {"igbo": 1})
            assert session.targets == ["igbo"]
            assert session.audio_history[-1] == []
        recv_until(host, lambda e: e["type"] == "status" and e["listeners"] == 0)
        assert session.targets == []
        host.send_text(json.dumps({"type": "stop"}))
        recv_until(host, lambda e: e["type"] == "status" and not e["live"])
        assert session.closed


def test_audio_before_start_and_bad_commands(client):
    room = create_room(client)
    with client.websocket_connect(f"/api/v1/rooms/{room['id']}/host?token={room['host_token']}") as host:
        host.send_bytes(FRAME)
        assert recv_until(host, lambda e: e["type"] == "error")[-1]["code"] == "not_started"
        host.send_text("not json")
        assert recv_until(host, lambda e: e["type"] == "error")[-1]["code"] == "invalid_input"
        host.send_text(json.dumps({"type": "start", "encoding": "opus"}))
        assert recv_until(host, lambda e: e["type"] == "error")[-1]["code"] == "invalid_input"
        host.send_text(json.dumps({"type": "ping"}))
        recv_until(host, lambda e: e["type"] == "pong")


def _expect_rejected(ws, code: str, close_code: int) -> None:
    err = ws.receive_json()
    assert err["type"] == "error" and err["code"] == code
    with pytest.raises(WebSocketDisconnect) as exc:
        ws.receive_json()
    assert exc.value.code == close_code


def test_host_auth_and_single_host(client):
    room = create_room(client)
    rid = room["id"]
    with client.websocket_connect(f"/api/v1/rooms/{rid}/host?token=wrong") as ws:
        _expect_rejected(ws, "unauthorized", 4401)
    with client.websocket_connect("/api/v1/rooms/nop-nop/host?token=x") as ws:
        _expect_rejected(ws, "not_found", 4404)
    with client.websocket_connect(f"/api/v1/rooms/{rid}/host?token={room['host_token']}") as host:
        recv_until(host, lambda e: e["type"] == "status")
        with client.websocket_connect(f"/api/v1/rooms/{rid}/host?token={room['host_token']}") as second:
            _expect_rejected(second, "host_already_connected", 4409)


def test_listener_pin_and_language(client):
    room = create_room(client, pin="1234")
    rid = room["id"]
    with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yoruba") as ws:
        _expect_rejected(ws, "unauthorized", 4403)
    with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yoruba&pin=0000") as ws:
        _expect_rejected(ws, "unauthorized", 4403)
    with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=klingon&pin=1234") as ws:
        _expect_rejected(ws, "unsupported_language", 4400)
    with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yoruba&pin=1234") as ws:
        assert recv_until(ws, lambda e: e["type"] == "status")[-1]["listeners"] == 1


def test_rooms_with_api_keys(monkeypatch, fake_engine):
    monkeypatch.setenv("API_KEYS", "k1")
    get_settings.cache_clear()
    app = create_app(fake_engine, mount_mcp=False, session_factory=FakeStreamingSession)
    with TestClient(app) as c:
        assert c.post("/api/v1/rooms", json={"src_lang": "english"}).status_code == 401
        room = c.post(
            "/api/v1/rooms", json={"src_lang": "english"}, headers={"Authorization": "Bearer k1"}
        ).json()
        rid = room["id"]
        assert c.get(f"/api/v1/rooms/{rid}").status_code == 200  # public info for the audience
        with c.websocket_connect(f"/api/v1/rooms/{rid}/host?token={room['host_token']}") as ws:
            _expect_rejected(ws, "unauthorized", 4401)  # API key required for hosts
        with c.websocket_connect(f"/api/v1/rooms/{rid}/host?token={room['host_token']}&key=k1") as host:
            recv_until(host, lambda e: e["type"] == "status")
        with c.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=igbo") as listener:  # audience: no key
            recv_until(listener, lambda e: e["type"] == "status")


def test_delete_room_requires_token(client):
    room = create_room(client)
    rid = room["id"]
    assert client.delete(f"/api/v1/rooms/{rid}?token=nope").status_code == 401
    with client.websocket_connect(f"/api/v1/rooms/{rid}/listen?lang=yoruba") as ws:
        recv_until(ws, lambda e: e["type"] == "status")
        assert client.delete(f"/api/v1/rooms/{rid}?token={room['host_token']}").status_code == 204
        closed = recv_until(ws, lambda e: e.get("closed"))[-1]
        assert closed["reason"] == "closed_by_host"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert client.get(f"/api/v1/rooms/{rid}").status_code == 404


# ── unit tests ──────────────────────────────────────────────────────────────
class _NullWS:
    async def send_text(self, data: str) -> None:  # pragma: no cover - not used
        pass

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        pass


async def test_subscriber_queue_drops_oldest_audio_and_coalesces_partials():
    sub = Subscriber(_NullWS(), role="listener", lang="yoruba", audio=True, max_queue=10, max_audio=2)
    for i in range(5):
        sub.push({"type": "audio", "seq": i})
    assert [e["seq"] for e in sub._q] == [3, 4] and sub.dropped == 3
    sub.push({"type": "partial", "seq": 1, "text": "a"})
    sub.push({"type": "partial", "seq": 1, "text": "ab"})
    assert [e.get("text") for e in sub._q if e["type"] == "partial"] == ["ab"]
    for i in range(20):
        sub.push({"type": "final", "seq": i})
    assert len(sub._q) == 10
    assert all(e["type"] == "final" for e in sub._q)  # audio/partials were sacrificed first


async def test_room_reaper(fake_engine):
    manager = RoomManager(fake_engine, session_factory=FakeStreamingSession, idle_ttl_s=600)
    room = manager.create(title="t", src_lang="en", targets=["yo"])
    assert await manager.reap(now=room.idle_since + 599) == []
    assert await manager.reap(now=room.idle_since + 601) == [room.id]
    assert manager.get(room.id) is None


async def test_dispatch_from_worker_thread(fake_engine):
    manager = RoomManager(fake_engine, session_factory=FakeStreamingSession)
    await manager.start()
    try:
        room = manager.create(title="t", src_lang="english", targets=["igbo"])

        class Recorder(_NullWS):
            def __init__(self) -> None:
                self.sent: list[dict[str, Any]] = []

            async def send_text(self, data: str) -> None:
                self.sent.append(json.loads(data))

        ws = Recorder()
        sub = await manager.add_listener(room, ws, lang="igbo", audio=False)
        task = asyncio.create_task(sub.run_sender())
        await manager.start_session(room)
        manager.feed(room, FRAME)
        for _ in range(200):
            if any(e["type"] == "translation" for e in ws.sent):
                break
            await asyncio.sleep(0.01)
        assert [e["type"] for e in ws.sent if e["type"] != "status"] == ["partial", "final", "translation"]
        task.cancel()
    finally:
        await manager.shutdown()
