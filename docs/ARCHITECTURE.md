# AfriLingua v2 — Architecture & Ownership Map

```
afrilingua/
├── core/            settings · hardware (CPU-first plans) · languages · errors · logging      [foundation]
├── backends/        base.py Protocols · concrete MT/ASR/TTS/VAD/LID engines · factory.py      [engine]
├── models/          manager.py (shared, locked, lazy) · pull.py (prefetch/convert)            [foundation/engine]
├── engine.py        AfriLinguaEngine facade: normalize → cache → TM → MT; ASR; TTS          [foundation]
├── audio/           capture (native rate + soxr resample) · vad (Silero ONNX) · player        [engine]
├── pipeline/        StreamingSession: VAD → ASR → MT(per target) → TTS, events               [engine]
├── documents/       format-preserving DOCX / PDF→DOCX / TXT translation jobs                  [engine]
├── tm/              translation memory: SQLite + vector index (char n-gram + optional LaBSE)  [community]
├── contrib/         community contributions: store, prompts, review/votes, QE, export         [community]
├── training/        LoRA MT fine-tune, ASR adapter training, CT2 export, eval gate           [community]
├── api/             FastAPI app, /api/v1 routers, rooms WS, OpenAI-compat /v1, SPA hosting    [api]
├── mcp/             MCP server (stdio + Streamable HTTP at /mcp)                              [api]
├── cli.py           `afrilingua serve | mcp | desktop | models pull | bench | translate`      [api]
web/                 React + TS + Vite SPA: landing, translate, live host, listener, contribute [web]
tests/               fast tests with fake backends (no downloads); `-m models` for real ones
docs/                api-contract.md (source of truth for REST/WS/MCP) · ARCHITECTURE.md
```

## Cross-workstream contracts

| Contract | Defined in | Implemented by | Consumed by |
|---|---|---|---|
| `MTBackend` / `ASRBackend` / `TTSBackend` / `LanguageIdentifier` | `backends/base.py` | engine | engine facade |
| `register_default_backends(manager)`, `build_detector()` | `backends/factory.py` | engine | `engine.build_engine()` |
| `AfriLinguaEngine` | `engine.py` | foundation | api, mcp, pipeline, cli |
| `TranslationMemoryProtocol`, `TMMatch` | `engine.py` | community (`tm.open_translation_memory()`) | engine, api, mcp |
| `StreamingSessionProtocol` + event dicts | `pipeline/__init__.py` | engine (`pipeline.streaming.StreamingSession`) | api rooms, mcp |
| `translate_document(engine, src_path, out_dir, tgt_lang, src_lang=None, on_progress=None) -> Path` | `documents/__init__.py` | engine | api documents router |
| `pull_models(profile=None, langs=None, on_progress=None)` | `models/pull.py` | engine | cli |
| `run_benchmark(engine, langs=None) -> dict` | `bench.py` | engine | cli |
| `ContributionStore` (`contrib.open_contribution_store(tm)`) | `contrib/` | community | api |
| Routers `tm_router`, `contributions_router` | `api/routers/tm.py`, `api/routers/contributions.py` | community | api `create_app` (via `request.app.state.tm` / `.contrib`) |
| REST / WS / MCP surface | `docs/api-contract.md` | api | web, external clients |

## Runtime principles
1. **CPU-first**: int8 CTranslate2 / faster-whisper / ONNX by default; GPU accelerates when present.
2. **One model owner**: `ModelManager` — every request, room and MCP call shares the same instances.
3. **Never block audio**: capture threads only enqueue; models run on pipeline workers.
4. **Offline by default after first pull**: `AFRILINGUA_OFFLINE=1` forbids network calls.
5. **Community loop**: approved contributions → translation memory immediately → training data for the next model pack.
