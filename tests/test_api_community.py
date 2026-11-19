"""Integration tests for ``/api/v1/tm``, ``/api/v1/contributions`` and ``/api/v1/datasets``."""

from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from afrilingua.api.app import create_app


def _sine_wav_bytes(freq: float = 220.0, duration_s: float = 0.8, sr: int = 16_000) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    pcm = (0.35 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_tm_router_crud_search_and_import(fake_engine):
    app = create_app(fake_engine, mount_mcp=False, serve_web=False)
    with TestClient(app) as c:
        r = c.post(
            "/api/v1/tm/entries",
            json={
                "source": "Good morning everyone",
                "target": "Ẹ káàárọ̀ gbogbo yín",
                "src_lang": "english",
                "tgt_lang": "yoruba",
                "approved": False,
            },
        )
        assert r.status_code == 200
        entry_id = r.json()["id"]

        appr = c.post(f"/api/v1/tm/entries/{entry_id}/approve")
        assert appr.status_code == 200 and appr.json()["approved"] is True

        srch = c.get(
            "/api/v1/tm/search",
            params={"q": "Good morning everyone", "src_lang": "en", "tgt_lang": "yo", "k": 3},
        )
        assert srch.status_code == 200
        body = srch.json()
        assert body["matches"][0]["target"] == "Ẹ káàárọ̀ gbogbo yín"
        assert body["matches"][0]["score"] == 1.0
        assert "took_ms" in body

        tsv = "Thank you\tDaalụ\nWelcome\tNnọọ\n".encode()
        imp = c.post(
            "/api/v1/tm/import",
            data={"src_lang": "english", "tgt_lang": "igbo", "approved": "true"},
            files={"file": ("pairs.tsv", tsv, "text/tab-separated-values")},
        )
        assert imp.status_code == 200 and imp.json()["imported"] == 2

        stats = c.get("/api/v1/tm/stats").json()
        assert stats["entries"] == 3 and stats["approved"] == 3

        assert c.delete(f"/api/v1/tm/entries/{entry_id}").status_code == 204
        assert c.delete(f"/api/v1/tm/entries/{entry_id}").status_code == 404


def test_contributions_router_full_workflow(fake_engine):
    app = create_app(fake_engine, mount_mcp=False, serve_web=False)
    with TestClient(app) as c:
        prompts = c.get("/api/v1/contributions/prompts", params={"lang": "yoruba", "kind": "translate", "n": 5})
        assert prompts.status_code == 200 and len(prompts.json()["prompts"]) == 5

        sub = c.post(
            "/api/v1/contributions",
            json={
                "kind": "translation",
                "src_lang": "english",
                "tgt_lang": "yoruba",
                "source_text": "Where is the clinic?",
                "target_text": "ibo ni ile-iwosan wa?",
                "domain": "medical",
                "dialect": "Ọ̀yọ́",
                "contributor": "Ada",
                "license": "CC-BY-4.0",
                "consent": True,
            },
        )
        assert sub.status_code == 200
        cid = sub.json()["id"]

        # Self-review is blocked; two independent reviewers approve and promote to TM
        assert (
            c.post(f"/api/v1/contributions/{cid}/review", json={"vote": "approve", "reviewer": "Ada"}).status_code
            == 400
        )
        q = c.get("/api/v1/contributions/review-queue", params={"reviewer": "Tunde"}).json()["items"]
        assert any(item["id"] == cid for item in q)

        r1 = c.post(f"/api/v1/contributions/{cid}/review", json={"vote": "approve", "reviewer": "Tunde"})
        assert r1.status_code == 200 and r1.json()["status"] == "pending"
        r2 = c.post(f"/api/v1/contributions/{cid}/review", json={"vote": "approve", "reviewer": "Bola"})
        assert r2.status_code == 200 and r2.json()["status"] == "approved"

        # Approved contribution is now immediately queryable in the TM!
        tm_hit = c.get(
            "/api/v1/tm/search",
            params={"q": "Where is the clinic?", "src_lang": "en", "tgt_lang": "yo"},
        ).json()
        assert tm_hit["matches"][0]["origin"] == "community"

        # Audio recording upload + streaming download
        wav = _sine_wav_bytes()
        aud = c.post(
            "/api/v1/contributions/audio",
            data={
                "text": "Ẹ káàárọ̀",
                "lang": "yoruba",
                "dialect": "Ọ̀yọ́",
                "consent": "true",
                "license": "CC-BY-4.0",
                "contributor": "Tunde",
            },
            files={"audio": ("greeting.wav", wav, "audio/wav")},
        )
        assert aud.status_code == 200 and aud.json()["has_audio"] is True
        aid = aud.json()["id"]
        assert aud.json()["text"] == "Ẹ káàárọ̀"

        stream = c.get(f"/api/v1/contributions/{aid}/audio")
        assert stream.status_code == 200 and len(stream.content) > 100

        stats = c.get("/api/v1/contributions/stats").json()
        assert stats["total"] == 2 and stats["approved"] == 1
        assert stats["leaderboard"][0]["count"] >= 1

        exp = c.get("/api/v1/contributions/export", params={"format": "jsonl"})
        assert exp.status_code == 200 and "Where is the clinic?" in exp.text


def test_datasets_router_upload_and_delete_removes_tm_entries(fake_engine):
    app = create_app(fake_engine, mount_mcp=False, serve_web=False)
    with TestClient(app) as c:
        srt_en = (
            "1\n00:00:01,000 --> 00:00:03,500\nWelcome back to the village.\n\n"
            "2\n00:00:04,000 --> 00:00:06,500\nPlease sit down with us.\n"
        ).encode()
        srt_yo = (
            "1\n00:00:01,100 --> 00:00:03,400\nẸ káàbọ̀ padà sí abúlé wa.\n\n"
            "2\n00:00:04,050 --> 00:00:06,450\nẸ jọ̀wọ́ ẹ jókòó pẹ̀lú wa.\n"
        ).encode()

        up = c.post(
            "/api/v1/datasets",
            data={
                "format": "srt",
                "src_lang": "english",
                "tgt_lang": "yoruba",
                "source_name": "Open Short Film Subtitles",
                "license": "CC-BY-4.0",
                "domain": "media",
                "add_to_tm": "true",
            },
            files=[
                ("files", ("en.srt", srt_en, "text/plain")),
                ("files", ("yo.srt", srt_yo, "text/plain")),
            ],
        )
        assert up.status_code == 200, up.text
        corpus = up.json()
        assert corpus["accepted_pairs"] == 2 and corpus["tm_imported"] == 2

        hit = c.get(
            "/api/v1/tm/search",
            params={"q": "Welcome back to the village.", "src_lang": "en", "tgt_lang": "yo"},
        ).json()
        assert hit["matches"][0]["target"] == "Ẹ káàbọ̀ padà sí abúlé wa."

        listed = c.get("/api/v1/datasets").json()["items"]
        assert len(listed) == 1 and listed[0]["id"] == corpus["id"]

        deleted = c.delete(f"/api/v1/datasets/{corpus['id']}").json()
        assert deleted["deleted"] is True and deleted["tm_entries_removed"] == 2

        after = c.get(
            "/api/v1/tm/search",
            params={"q": "Welcome back to the village.", "src_lang": "en", "tgt_lang": "yo"},
        ).json()
        assert after["matches"] == []
