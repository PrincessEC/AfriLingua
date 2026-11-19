# AfriLingua v2 — API Contract

Single FastAPI server (`afrilingua serve`, default `http://127.0.0.1:8000`).
The React SPA is served from `/` (built files in `web/dist`). In dev the SPA runs
on Vite `:5173` and proxies `/api`, `/v1`, `/mcp` to `:8000`.

Auth: if `API_KEYS` is set, send `Authorization: Bearer <key>` (HTTP) or
`?key=<key>` (WebSocket). Empty `API_KEYS` = local mode, no auth.

Errors (all JSON endpoints):
```json
{ "error": { "code": "unsupported_language", "message": "…" } }
```
Codes: `unsupported_language` (400), `invalid_input` (400), `capability_unavailable` (422),
`model_load_failed` / `model_not_downloaded` (503), `not_found` (404), `unauthorized` (401), `payload_too_large` (413).

Languages are canonical names (`english`, `igbo`, `yoruba`, `hausa`, …); aliases
(`yo`, `yor`, `yor_Latn`) are accepted on input.

---

## 1. System

| Method | Path | Response |
|---|---|---|
| GET | `/api/v1/health` | `{ "status": "ok", "version": "2.0.0.dev0" }` |
| GET | `/api/v1/status` | `{ runtime, hardware, models, languages, translation_memory }` (see `AfriLinguaEngine.status`) |
| GET | `/api/v1/languages` | `{ "languages": LanguageSpec[] }` |

`LanguageSpec`: `{ name, nllb_code, display_name, native_name, iso639_1, iso639_3, mms_asr_code, mms_tts_model, whisper_lang, aliases[], supports_asr, supports_tts, builtin }`

## 2. Translation & speech

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/v1/translate` | `{ "texts": string[] \| "text": string, "src_lang"?: string\|"auto", "tgt_lang": string, "mode"?: "live"\|"document", "use_tm"?: bool }` | `{ "results": TranslationResult[] }` |
| POST | `/api/v1/detect` | `{ "text": string }` | `{ "language": string, "confidence": number }` |
| POST | `/api/v1/transcribe` | multipart: `file` (wav/mp3/ogg/webm/flac), `lang` | `Transcript` |
| POST | `/api/v1/speech` | `{ "text": string, "lang": string, "format"?: "wav" }` | `audio/wav` bytes |

`TranslationResult`: `{ text, source_text, src_lang, tgt_lang, origin: "passthrough"|"cache"|"tm_exact"|"mt"|"mt_tm_assisted", latency_ms, detected, matches: TMMatch[] }`
`Transcript`: `{ text, language, segments: [{start, end, text}], confidence, duration_s }`

## 3. Documents (background jobs)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/documents` | multipart `file` (.txt/.docx/.pdf), `tgt_lang`, `src_lang?` → `{ "id", "status": "queued" }` |
| GET | `/api/v1/documents/{id}` | `{ id, status: queued\|running\|done\|error, progress: 0..1, filename, error? }` |
| GET | `/api/v1/documents/{id}/download` | translated file (DOCX stays DOCX; PDF → DOCX; TXT → TXT) |

## 4. Translation memory (fast search)

| Method | Path | Body / Query | Response |
|---|---|---|---|
| GET | `/api/v1/tm/search` | `q, src_lang, tgt_lang, k=5` | `{ "matches": TMMatch[], "took_ms": number }` |
| POST | `/api/v1/tm/entries` | `{ source, target, src_lang, tgt_lang, approved?, origin? }` | `{ "id" }` |
| POST | `/api/v1/tm/entries/{id}/approve` | – | `{ "id", "approved": true }` |
| DELETE | `/api/v1/tm/entries/{id}` | – | `204` |
| POST | `/api/v1/tm/import` | multipart `file` (TSV/CSV/JSONL/TMX), `src_lang`, `tgt_lang` | `{ "imported": n }` |
| GET | `/api/v1/tm/stats` | – | `{ "entries": n, "approved": n, "by_pair": {"english→yoruba": n} }` |

`TMMatch`: `{ id, source, target, score, approved, origin }`

## 5. Community contributions

| Method | Path | Body / Query | Response |
|---|---|---|---|
| GET | `/api/v1/contributions/prompts` | `lang, kind=translate\|record, n=10` | `{ "prompts": [{ id, text, lang, domain }] }` |
| POST | `/api/v1/contributions` | `ContributionIn` (JSON) | `Contribution` |
| POST | `/api/v1/contributions/audio` | multipart: `audio`, `text`, `lang`, `dialect?`, `accent?`, `gender?`, `age_range?`, `consent=true`, `license`, `contributor?` | `Contribution` |
| GET | `/api/v1/contributions` | `status?, lang?, kind?, limit=50, offset=0` | `{ items: Contribution[], total }` |
| GET | `/api/v1/contributions/review-queue` | `lang?, reviewer` | `{ items: Contribution[] }` (excludes reviewer's own) |
| POST | `/api/v1/contributions/{id}/review` | `{ "vote": "approve"\|"reject", "reviewer": string, "comment"? }` | `Contribution` |
| GET | `/api/v1/contributions/{id}/audio` | – | audio bytes |
| GET | `/api/v1/contributions/stats` | – | `{ total, approved, pending, by_lang: {lang: {text, audio, approved}}, contributors, hours_audio }` |
| GET | `/api/v1/contributions/export` | `lang?, kind?, format=jsonl\|hf` | file download (approved only, with licenses) |

`ContributionIn`:
```json
{ "kind": "translation" | "correction" | "glossary" | "transcription",
  "src_lang": "english", "tgt_lang": "yoruba",
  "source_text": "Good morning", "target_text": "Ẹ káàárọ̀",
  "original_mt"?: "…",            // for corrections: what the machine said
  "domain"?: "general|medical|legal|religious|education|government|media",
  "dialect"?: "Ọ̀yọ́", "region"?: "Lagos",
  "contributor"?: "display name or anonymous id",
  "license": "CC-BY-4.0" | "CC0-1.0",
  "consent": true }
```
`Contribution` = `ContributionIn` + `{ id, status: "pending"|"approved"|"rejected", votes_up, votes_down, quality_score, created_at, has_audio }`.
Approval rule: ≥ 2 approvals and approvals > rejections → `approved`, which also writes the pair into the translation memory as `approved=true, origin="community"`.

## 6. Live rooms (conference interpretation)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/rooms` | `{ "title", "src_lang", "targets": string[], "pin"? }` → `{ id, host_token, join_url }` |
| GET | `/api/v1/rooms/{id}` | `{ id, title, src_lang, targets, listeners, live }` |
| WS | `/api/v1/rooms/{id}/host?token=…` | host streams mic audio |
| WS | `/api/v1/rooms/{id}/listen?lang=yoruba&audio=1&pin=…` | listener receives events |

Host → server: first text frame `{"type":"start","sample_rate":16000,"encoding":"pcm_s16le"}`, then **binary** PCM16 LE mono frames (any size, ~20–100 ms recommended), `{"type":"stop"}` to end.

Server → clients (JSON text frames):
```json
{"type":"status","live":true,"listeners":12}
{"type":"partial","seq":7,"lang":"english","text":"Good morning every"}
{"type":"final","seq":7,"lang":"english","text":"Good morning everyone.","ts":1731950000.1}
{"type":"translation","seq":7,"lang":"yoruba","text":"Ẹ káàárọ̀ gbogbo yín.","origin":"mt","latency_ms":412}
{"type":"audio","seq":7,"lang":"yoruba","format":"wav","data":"<base64>"}
{"type":"error","code":"…","message":"…"}
```

## 7. OpenAI-compatible endpoints

Drop-in for apps written against OpenAI/Whisper audio APIs (change `base_url` to `http://127.0.0.1:8000/v1`).

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/audio/transcriptions` | multipart `file`, `model` (ignored/“afrilingua”), `language` (ISO or name), `response_format=json\|text\|verbose_json` |
| POST | `/v1/audio/translations` | multipart `file`, `language` (source), extra `target_language` (default `english`) |
| POST | `/v1/audio/speech` | `{ "model", "input", "voice": "<lang>", "response_format": "wav" }` |
| GET | `/v1/models` | lists `afrilingua-mt`, `afrilingua-asr`, `afrilingua-tts` |

## 8. MCP

* `afrilingua mcp` → stdio transport (for Claude Desktop, Cursor, Gemini CLI, IDEs).
* Streamable HTTP mounted at `/mcp` on the main server.

Tools: `translate_text`, `translate_batch`, `detect_language`, `transcribe_audio`, `synthesize_speech`,
`translate_speech`, `search_translation_memory`, `list_languages`, `submit_translation`, `submit_correction`,
`create_live_room`. Resources: `afrilingua://languages`, `afrilingua://status`, `afrilingua://tm/stats`.
