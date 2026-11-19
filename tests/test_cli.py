"""CLI argument parsing and light command behaviour (no servers or models are started)."""

from __future__ import annotations

import json

import pytest

from afrilingua import cli


def parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_serve_args():
    args = parse("serve", "--port", "9000", "--lan", "--reload", "--no-web")
    assert args.func is cli.cmd_serve
    assert args.port == 9000 and args.lan and args.reload and args.no_web
    args = parse("serve")
    assert args.host is None and args.port is None and not args.lan


def test_subcommands_parse():
    assert parse("mcp").func is cli.cmd_mcp
    assert parse("desktop", "--port", "8123").port == 8123
    pull = parse("models", "pull", "--profile", "cpu-lite", "--langs", "yoruba,igbo")
    assert pull.func is cli.cmd_models and pull.models_cmd == "pull" and pull.profile == "cpu-lite"
    assert cli._split_langs(pull.langs) == ["yoruba", "igbo"]
    assert parse("models", "list").models_cmd == "list"
    assert parse("status").func is cli.cmd_status
    tr = parse("translate", "Good morning", "--to", "yoruba", "--from", "english")
    assert (tr.text, tr.to, tr.src, tr.mode) == ("Good morning", "yoruba", "english", "document")
    assert parse("bench", "--langs", "hausa").langs == "hausa"
    cimp = parse(
        "corpora",
        "import",
        "en.srt",
        "yo.srt",
        "--format",
        "srt",
        "--from",
        "en",
        "--to",
        "yo",
        "--source-name",
        "Open Film",
        "--license",
        "CC-BY-4.0",
        "--domain",
        "media",
        "--add-to-tm",
    )
    assert cimp.func is cli.cmd_corpora and cimp.files == ["en.srt", "yo.srt"] and cimp.add_to_tm
    assert parse("corpora", "list").corpora_cmd == "list"
    assert parse("corpora", "delete", "abc123").id == "abc123"
    sp = parse("train", "splits", "--commercial-safe", "--max-domain-share", "0.35")
    assert sp.func is cli.cmd_train and sp.commercial_safe and sp.max_domain_share == 0.35
    mt = parse("train", "mt", "--dry-run", "--lora-r", "16")
    assert mt.dry_run and mt.lora_r == 16
    asr = parse("train", "asr", "--lang", "yoruba", "--dry-run")
    assert asr.lang == "yoruba" and asr.dry_run


def test_invalid_args_exit():
    with pytest.raises(SystemExit):
        parse()  # command required
    with pytest.raises(SystemExit):
        parse("translate", "hi")  # --to required
    with pytest.raises(SystemExit):
        parse("models")  # pull|list required
    with pytest.raises(SystemExit):
        parse("models", "pull", "--profile", "huge")


def test_serve_lan_warns_and_binds_all_interfaces(monkeypatch, capsys):
    for key in ("API_HOST", "API_PORT", cli.SERVE_WEB_ENV, cli.LAN_PORT_ENV):
        monkeypatch.delenv(key, raising=False)
    calls = {}
    import os
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(args=a, kwargs=kw))
    assert cli.main(["serve", "--lan", "--port", "8765", "--no-web"]) == 0
    assert calls["kwargs"]["host"] == "0.0.0.0" and calls["kwargs"]["port"] == 8765
    assert calls["kwargs"]["factory"] is True and calls["args"] == ("afrilingua.cli:app_factory",)
    err = capsys.readouterr().err
    assert "WARNING" in err and "0.0.0.0" in err
    assert os.environ[cli.SERVE_WEB_ENV] == "0" and os.environ[cli.LAN_PORT_ENV] == "8765"


def test_app_factory_honours_env(monkeypatch):
    monkeypatch.setenv(cli.SERVE_WEB_ENV, "0")
    monkeypatch.setenv(cli.LAN_PORT_ENV, "8001")
    app = cli.app_factory()
    assert app.state.lan_port == 8001
    assert not any(getattr(r, "path", "") == "/{full_path:path}" for r in app.routes)


def test_translate_command(monkeypatch, capsys, fake_engine):
    import afrilingua.engine as engine_mod

    monkeypatch.setattr(engine_mod, "build_engine", lambda **_kw: fake_engine)
    assert cli.main(["translate", "Good morning", "--to", "yo", "--from", "en"]) == 0
    assert capsys.readouterr().out.strip() == "[yoruba] Good morning"
    assert cli.main(["translate", "Hi", "--to", "ig", "--from", "en", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tgt_lang"] == "igbo"
    assert cli.main(["translate", "Hi", "--to", "klingon"]) == 2
    assert "unsupported_language" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit):
        parse("--version")
    assert "2.0.0.dev0" in capsys.readouterr().out


def test_corpora_and_train_commands_end_to_end(tmp_path, monkeypatch, capsys):
    from afrilingua.core.settings import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text(
        "Good morning everyone\tẸ káàárọ̀ gbogbo yín\n"
        "Thank you very much\tẸ ṣé púpọ̀ gan-an\n"
        "Welcome to the meeting\tẸ káàbọ̀ sí ìpàdé náà\n",
        encoding="utf-8",
    )
    rc = cli.main(
        [
            "corpora",
            "import",
            str(tsv),
            "--format",
            "tsv",
            "--from",
            "en",
            "--to",
            "yo",
            "--source-name",
            "Greeting Phrases",
            "--license",
            "CC-BY-4.0",
            "--add-to-tm",
        ]
    )
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["accepted_pairs"] == 3 and rec["added_to_tm"] is True

    assert cli.main(["corpora", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed["items"]) == 1

    out_dir = tmp_path / "splits"
    assert cli.main(["train", "splits", "--out-dir", str(out_dir), "--commercial-safe"]) == 0
    split_res = json.loads(capsys.readouterr().out)
    assert split_res["train"] >= 2

    assert cli.main(["train", "mt", "--data-dir", str(out_dir), "--dry-run"]) == 0
    mt_res = json.loads(capsys.readouterr().out)
    assert mt_res["dry_run"] is True and mt_res["train_rows"] == split_res["train"]

    assert cli.main(["corpora", "delete", rec["id"]]) == 0
    del_res = json.loads(capsys.readouterr().out)
    assert del_res["deleted"] is True
    get_settings.cache_clear()
