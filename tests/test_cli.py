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
    calls = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(args=a, kwargs=kw))
    assert cli.main(["serve", "--lan", "--port", "8765", "--no-web"]) == 0
    assert calls["kwargs"]["host"] == "0.0.0.0" and calls["kwargs"]["port"] == 8765
    assert calls["kwargs"]["factory"] is True and calls["args"] == ("afrilingua.cli:app_factory",)
    err = capsys.readouterr().err
    assert "WARNING" in err and "0.0.0.0" in err
    import os

    assert os.environ[cli.SERVE_WEB_ENV] == "0" and os.environ[cli.LAN_PORT_ENV] == "8765"
    for key in ("API_HOST", "API_PORT", cli.SERVE_WEB_ENV, cli.LAN_PORT_ENV):
        monkeypatch.delenv(key, raising=False)


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
