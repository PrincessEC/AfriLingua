"""
``afrilingua`` command-line interface (stdlib argparse, no extra dependencies).

    afrilingua serve [--host H] [--port P] [--lan] [--reload] [--no-web]
    afrilingua mcp                      # MCP over stdio (Claude Desktop, Cursor, Gemini CLI …)
    afrilingua desktop                  # local server + native window (pywebview) or browser
    afrilingua models pull [--profile cpu-lite|cpu-standard|gpu] [--langs yoruba,igbo]
    afrilingua models list
    afrilingua status
    afrilingua translate "Good morning" --to yoruba [--from english]
    afrilingua corpora import <files...> --format srt --from en --to yo --source-name ... --license CC-BY-4.0
    afrilingua corpora list | delete <id>
    afrilingua train splits | mt | asr
    afrilingua bench [--langs yoruba,igbo]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from afrilingua import __version__

logger = logging.getLogger("afrilingua.cli")

SERVE_WEB_ENV = "AFRILINGUA_SERVE_WEB"
LAN_PORT_ENV = "AFRILINGUA_LAN_PORT"


# ── helpers ────────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    from afrilingua.core.logging import configure_logging

    configure_logging(level, json_output=os.environ.get("AFRILINGUA_LOG_JSON", "") == "1")


def _split_langs(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _configure_server_env(host: str, port: int, *, serve_web: bool, lan: bool) -> None:
    """Export server settings so the app factory (also in --reload workers) sees them."""
    from afrilingua.core.settings import get_settings

    os.environ["API_HOST"] = host
    os.environ["API_PORT"] = str(port)
    os.environ[SERVE_WEB_ENV] = "1" if serve_web else "0"
    if lan:
        os.environ[LAN_PORT_ENV] = str(port)
    else:
        os.environ.pop(LAN_PORT_ENV, None)
    get_settings.cache_clear()


def app_factory() -> Any:
    """uvicorn ``--factory`` entry point (used by ``serve`` and ``--reload``)."""
    from afrilingua.api.app import create_app

    app = create_app(serve_web=os.environ.get(SERVE_WEB_ENV, "1") != "0")
    lan_port = os.environ.get(LAN_PORT_ENV)
    if lan_port:
        app.state.lan_port = int(lan_port)
    return app


# ── commands ───────────────────────────────────────────────────────────────
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from afrilingua.core.settings import get_settings

    settings = get_settings()
    host = "0.0.0.0" if args.lan else (args.host or settings.api_host)
    port = args.port or settings.api_port
    if args.lan:
        print(
            "WARNING: --lan binds 0.0.0.0, exposing AfriLingua to every device on your network. "
            "Set API_KEYS=... to require an API key (room listeners stay PIN-protected).",
            file=sys.stderr,
        )
        if not settings.api_key_set:
            print("WARNING: API_KEYS is empty — the API is unauthenticated.", file=sys.stderr)
    _configure_server_env(host, port, serve_web=not args.no_web, lan=args.lan)
    shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(
        f"AfriLingua {__version__} → http://{shown}:{port}  (docs: /docs · MCP: /mcp · OpenAI: /v1)",
        file=sys.stderr,
    )
    uvicorn.run(
        "afrilingua.cli:app_factory",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        ws_max_size=16 * 1024 * 1024,
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    # stdout carries the MCP protocol: logs go to stderr only.
    _setup_logging(args.log_level)
    from afrilingua.mcp.server import run_stdio

    run_stdio()
    return 0


def cmd_desktop(args: argparse.Namespace) -> int:
    import uvicorn

    port = args.port or _free_port()
    _configure_server_env("127.0.0.1", port, serve_web=True, lan=False)
    config = uvicorn.Config(
        app_factory, factory=True, host="127.0.0.1", port=port, log_level=args.log_level.lower()
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="afrilingua-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not server.started:
        print("AfriLingua server failed to start.", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/"
    try:
        import webview  # type: ignore[import-not-found]

        webview.create_window("AfriLingua", url, width=1280, height=840, min_size=(900, 600))
        webview.start()
    except ImportError:
        import webbrowser

        print(f"pywebview not installed; opening {url} in your browser (Ctrl+C to quit).", file=sys.stderr)
        webbrowser.open(url)
        try:
            while thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    server.should_exit = True
    thread.join(timeout=10)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    if args.models_cmd == "pull":
        try:
            from afrilingua.models.pull import pull_models
        except ImportError as exc:
            print(f"Model pulling is not available in this build: {exc}", file=sys.stderr)
            return 1

        def on_progress(*parts: Any, **_kw: Any) -> None:
            print(" ".join(str(p) for p in parts), file=sys.stderr, flush=True)

        result = pull_models(profile=args.profile, langs=_split_langs(args.langs), on_progress=on_progress)
        if result is not None:
            _print_json(result)
        return 0
    from afrilingua.engine import build_engine

    engine = build_engine(with_tm=False, with_detector=False)
    _print_json(engine.manager.status())
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from afrilingua.engine import build_engine

    engine = build_engine(with_tm=True, with_detector=False)
    status = engine.status()
    status["version"] = __version__
    _print_json(status)
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    from afrilingua.core.errors import AfriLinguaError
    from afrilingua.engine import build_engine

    text = args.text if args.text != "-" else sys.stdin.read()
    engine = build_engine()
    try:
        result = engine.translate_one(text, args.src, args.to, mode=args.mode)
    except AfriLinguaError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.to_dict())
    else:
        print(result.text)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        from afrilingua.bench import run_benchmark
    except ImportError as exc:
        print(f"Benchmarking is not available in this build: {exc}", file=sys.stderr)
        return 1
    from afrilingua.engine import build_engine

    _print_json(run_benchmark(build_engine(), langs=_split_langs(args.langs)))
    return 0


def cmd_corpora(args: argparse.Namespace) -> int:
    from afrilingua.core.errors import AfriLinguaError
    from afrilingua.tm import open_translation_memory
    from afrilingua.training.corpora import open_corpus_store

    tm = open_translation_memory()
    store = open_corpus_store(tm=tm)
    try:
        if args.corpora_cmd == "list":
            _print_json({"items": [r.to_dict() for r in store.list()]})
            return 0
        if args.corpora_cmd == "delete":
            _print_json(store.delete(args.id))
            return 0
        rec = store.import_corpus(
            fmt=args.format,
            files=[Path(p) for p in args.files],
            src_lang=args.src,
            tgt_lang=args.to,
            source_name=args.source_name,
            license=args.license,
            domain=args.domain,
            add_to_tm=args.add_to_tm,
        )
        _print_json(rec.to_dict())
        return 0
    except AfriLinguaError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return 2


def cmd_train(args: argparse.Namespace) -> int:
    from afrilingua.core.errors import AfriLinguaError

    try:
        if args.train_cmd == "splits":
            from afrilingua.contrib import open_contribution_store
            from afrilingua.tm import open_translation_memory
            from afrilingua.training.corpora import open_corpus_store
            from afrilingua.training.datasets import build_training_splits

            tm = open_translation_memory()
            summary = build_training_splits(
                out_dir=args.out_dir,
                contrib=open_contribution_store(tm=tm),
                corpora=open_corpus_store(tm=tm),
                tm=tm,
                lang=args.lang,
                commercial_safe=args.commercial_safe,
                include_tm=args.include_tm,
                max_domain_share=args.max_domain_share,
            )
            _print_json(summary.to_dict())
            return 0
        if args.train_cmd == "mt":
            from afrilingua.training.mt_lora import MTLoRAConfig, run_mt_lora

            res = run_mt_lora(
                MTLoRAConfig(
                    base_model=args.base_model,
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    lora_r=args.lora_r,
                    export_ct2=not args.no_ct2,
                    dry_run=args.dry_run,
                )
            )
            _print_json(res)
            return 0
        if args.train_cmd == "asr":
            from afrilingua.training.asr_adapter import ASRAdapterConfig, run_asr_adapter_training

            res = run_asr_adapter_training(
                ASRAdapterConfig(
                    lang=args.lang,
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    dry_run=args.dry_run,
                )
            )
            _print_json(res)
            return 0
    except (AfriLinguaError, ValueError, RuntimeError) as exc:
        code = getattr(exc, "code", "train_error")
        print(f"error [{code}]: {exc}", file=sys.stderr)
        return 2
    return 0


# ── parser ─────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afrilingua",
        description="Offline-first speech & translation engine for African languages.",
    )
    parser.add_argument("--version", action="version", version=f"afrilingua {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p = sub.add_parser("serve", help="Run the HTTP server (REST, WebSockets, /v1, /mcp, web app).")
    p.add_argument("--host", default=None, help="Bind address (default: API_HOST or 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Port (default: API_PORT or 8000).")
    p.add_argument(
        "--lan", action="store_true", help="Bind 0.0.0.0 so phones/laptops on your network can join."
    )
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development).")
    p.add_argument("--no-web", action="store_true", help="Do not serve the web app (API only).")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="Run the MCP server over stdio (for AI assistants / IDEs).")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("desktop", help="Open AfriLingua in a desktop window (local server).")
    p.add_argument("--port", type=int, default=None, help="Port (default: a free one).")
    p.set_defaults(func=cmd_desktop)

    p = sub.add_parser("models", help="Download or list models.")
    msub = p.add_subparsers(dest="models_cmd", metavar="<pull|list>")
    msub.required = True
    pull = msub.add_parser("pull", help="Download (and convert) models for offline use.")
    pull.add_argument("--profile", default=None, choices=["auto", "cpu-lite", "cpu-standard", "gpu"])
    pull.add_argument("--langs", default=None, help="Comma-separated languages (default: all built-in).")
    msub.add_parser("list", help="Show registered models and their load state.")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("status", help="Print runtime, hardware and model status as JSON.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("translate", help="Translate text from the command line ('-' reads stdin).")
    p.add_argument("text")
    p.add_argument("--to", required=True, help="Target language (e.g. yoruba, ig, hau).")
    p.add_argument("--from", dest="src", default="auto", help="Source language (default: auto-detect).")
    p.add_argument("--mode", default="document", choices=["live", "document"])
    p.add_argument("--json", action="store_true", help="Print the full TranslationResult as JSON.")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("bench", help="Benchmark translation/speech latency on this machine.")
    p.add_argument("--langs", default=None, help="Comma-separated languages.")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "corpora", help="Import, list or delete bulk parallel corpora (Bibles, subtitles, docs)."
    )
    csub = p.add_subparsers(dest="corpora_cmd", metavar="<import|list|delete>")
    csub.required = True
    cimp = csub.add_parser("import", help="Import parallel files into the corpus store (and optionally TM).")
    cimp.add_argument(
        "files",
        nargs="+",
        help="Input file(s): 1 for tsv/csv/jsonl/tmx, 2+ for ebible/usfm/srt/vtt/lines/unaligned.",
    )
    cimp.add_argument(
        "--format", required=True, help="ebible, usfm, srt, vtt, lines, tsv, csv, jsonl, tmx, unaligned."
    )
    cimp.add_argument("--from", dest="src", required=True, help="Source language.")
    cimp.add_argument("--to", required=True, help="Target language.")
    cimp.add_argument(
        "--source-name", required=True, help="Dataset provenance name (e.g. 'Yoruba Open Bible 1900')."
    )
    cimp.add_argument(
        "--license",
        required=True,
        help="public-domain, CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0, research-only, proprietary-permission.",
    )
    cimp.add_argument(
        "--domain",
        default="general",
        help="general, religious, media, medical, legal, education, government.",
    )
    cimp.add_argument(
        "--add-to-tm", action="store_true", help="Also ingest accepted pairs into Translation Memory."
    )
    csub.add_parser("list", help="List imported parallel corpora.")
    cdel = csub.add_parser("delete", help="Delete an imported corpus (and its TM entries if ingested).")
    cdel.add_argument("id", help="Corpus ID.")
    p.set_defaults(func=cmd_corpora)

    p = sub.add_parser("train", help="Build training splits and fine-tune LoRA / MMS adapters.")
    tsub = p.add_subparsers(dest="train_cmd", metavar="<splits|mt|asr>")
    tsub.required = True
    tsplits = tsub.add_parser("splits", help="Build deduplicated, domain-capped train/val/test JSONL splits.")
    tsplits.add_argument("--out-dir", default=None, help="Output directory (default: data/processed).")
    tsplits.add_argument("--lang", default=None, help="Restrict to pairs involving this language.")
    tsplits.add_argument(
        "--commercial-safe", action="store_true", help="Exclude copyleft and research-only corpora."
    )
    tsplits.add_argument("--include-tm", action="store_true", help="Include approved ad-hoc TM entries.")
    tsplits.add_argument(
        "--max-domain-share", type=float, default=0.40, help="Cap per-domain share (default: 0.40)."
    )
    tmt = tsub.add_parser(
        "mt", help="LoRA fine-tune NLLB-200 on parallel splits (+ optional CTranslate2 int8 export)."
    )
    tmt.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    tmt.add_argument("--data-dir", default="data/processed")
    tmt.add_argument("--output-dir", default="data/finetuned/nllb-afrilingua")
    tmt.add_argument("--epochs", type=int, default=3)
    tmt.add_argument("--max-steps", type=int, default=-1)
    tmt.add_argument("--lora-r", type=int, default=8)
    tmt.add_argument("--no-ct2", action="store_true", help="Skip CTranslate2 quantized export.")
    tmt.add_argument(
        "--dry-run", action="store_true", help="Validate splits and runtime config without loading weights."
    )
    tasr = tsub.add_parser("asr", help="Fine-tune an MMS CTC language adapter on approved community audio.")
    tasr.add_argument("--lang", required=True, help="Target language (e.g. yoruba, igbo, hausa).")
    tasr.add_argument("--epochs", type=int, default=5)
    tasr.add_argument("--max-steps", type=int, default=-1)
    tasr.add_argument(
        "--dry-run", action="store_true", help="Count approved recordings without loading weights."
    )
    p.set_defaults(func=cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "mcp":
        _setup_logging(args.log_level)
    from afrilingua.core.settings import apply_runtime_environment

    apply_runtime_environment()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
