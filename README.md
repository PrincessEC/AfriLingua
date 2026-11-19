# AfriLingua

**Fast, accurate AI translation for African languages. Lightweight enough for any laptop.**

AfriLingua is an open-source translation library for **Yoruba, Igbo, Hausa and English**. It handles text, speech and documents, and more languages can be added through configuration. It runs fully offline on an ordinary CPU, with no GPU and no API bill. Your data never leaves the machine.

[![CI](https://github.com/fegig/AfriLingua/actions/workflows/ci.yml/badge.svg)](https://github.com/fegig/AfriLingua/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
![License](https://img.shields.io/badge/code-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

```python
import afrilingua

afrilingua.translate("Where is the nearest clinic?", to="hausa")
# 'A ina ne asibitin da ya fi kusa?'

afrilingua.translate("Where is the nearest clinic?", to="yoruba")
# 'Ibo ni ilé ìwòsàn tó sún mọ́ ẹ jù lọ wà?'
```

These are real outputs from the default model (NLLB-200 600M, int8) on a laptop CPU.

- **Fast.** Approved phrases return in about **0.1 ms**. New sentences take about a second on a laptop CPU.
- **Accurate, and measurably improving.** Corrections take effect on the next request, and a new model ships only if it scores better.
- **Lightweight.** About 2 GB of RAM with three languages loaded. No GPU needed.
- **Plugs in anywhere.** Use it as a Python library, a local REST API, an OpenAI-compatible audio API, or an MCP server for AI agents.
- **Trainable on your data.** Upload any translated text you have, and fine-tune on it.

> [!NOTE]
> AfriLingua is in **alpha**. The library API and HTTP contracts are stable enough to build on. Read the [security model](#security-model) before you expose the server on a network.

---

## Install

AfriLingua is not on PyPI yet. Install it from GitHub:

```bash
pip install "afrilingua[all] @ git+https://github.com/fegig/AfriLingua.git@dev"
afrilingua models pull --langs yoruba,igbo,hausa     # one-time download, then fully offline
```

You need Python 3.11–3.13, about 3 GB of free RAM and 2–3 GB of disk for models. A GPU is used automatically when present, but is never required.

The first translation into a language converts the model to int8 once, which takes about 2–3 minutes. Every run after that loads in under a second.

| Extra | Adds |
|---|---|
| `fast` | int8 CPU backends: CTranslate2, faster-whisper, ONNX |
| `server` | REST, WebSocket and MCP server |
| `audio` | Microphone capture |
| `documents` | DOCX and PDF translation |
| `train` | Fine-tuning |
| `all` | Everything except `train` |

## Use it

### Python

```python
import afrilingua

afrilingua.translate("Thank you", to="igbo")                                 # source is detected automatically
afrilingua.translate(["Thank you", "Welcome"], to="igbo", source="english")  # batches
afrilingua.translate(report_text, to="yoruba", mode="document")             # beam search for documents

text  = afrilingua.transcribe(audio, lang="yoruba")    # speech to text (16 kHz mono float32)
voice = afrilingua.speak("Ndewo", lang="igbo")         # text to speech -> waveform + sample rate
afrilingua.languages()                                 # what is installed and what each language supports
```

For lower-level control, use the engine directly. Every result tells you where it came from and how long it took:

```python
from afrilingua.engine import build_engine

engine = build_engine()
r = engine.translate_one("Good morning", "english", "yoruba")
r.text, r.origin, r.latency_ms    # origin: "tm_exact" | "cache" | "mt" | "mt_tm_assisted"
```

### REST API

```bash
afrilingua serve                  # http://127.0.0.1:8000, interactive docs at /docs

curl http://127.0.0.1:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning", "tgt_lang": "igbo"}'
```

Full contract: [docs/api-contract.md](docs/api-contract.md).

### OpenAI-compatible audio

Apps already written for the OpenAI or Whisper audio APIs work by changing the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
client.audio.transcriptions.create(model="afrilingua", file=open("sermon.wav", "rb"), language="yo")
```

### MCP, for AI agents

```json
{ "mcpServers": { "afrilingua": { "command": "afrilingua", "args": ["mcp"] } } }
```

Agents get tools such as `translate_text`, `translate_batch`, `detect_language`, `transcribe_audio`, `synthesize_speech`, `translate_speech` and `search_translation_memory`. All of them run locally.

### Live interpretation app

```bash
afrilingua serve --lan            # set API_KEYS first, see "Security model"
```

Open the web app and create a room, then share the QR code. Listeners follow on their phones with captions and an optional voice in their language. Projector mode shows large captions in two languages.

## Speed

Speed is the first thing we optimise. **Every number below was measured, and you can reproduce it** with `afrilingua bench`.

Measured 19 Nov 2026 on a Ryzen 9 5900HS laptop (8 cores), CPU only, int8, 8 threads. The laptop was **on battery and running other work**, so expect faster results when plugged in.

| Operation | Median | p95 |
|---|---|---|
| Approved phrase from translation memory (exact, 5,000 entries) | **0.12 ms** | 0.19 ms |
| Similar-sentence lookup in translation memory (5,000 entries) | 1.9 ms | 2.8 ms |
| Sentence already translated this session | 0.05 ms | |
| Language detection | 0.10 ms | 0.18 ms |
| New sentence, English → Igbo | 730 ms | 1.19 s |
| New sentence, English → Hausa | 773 ms | 1.15 s |
| New sentence, English → Yoruba | 1.10 s | 1.63 s |
| Five-sentence paragraph (Yoruba / Igbo / Hausa) | 1.3 / 1.1 / 1.6 s | |
| Memory with translation loaded for three languages | 2.0 GB | |

**How it stays fast**

- **Most sentences never reach the model.** Conferences, clinics and churches repeat themselves. Text is split into sentences; approved translations of exact sentences come from translation memory in a fraction of a millisecond, and so do repeats within a session. Similar-but-different sentences are never reused verbatim.
- **New sentences are decoded together.** A paragraph's new sentences go to the model as one batch: five sentences took 1.1–1.6 s instead of 2.6–7.0 s as one block.
- **Only the scripts you use.** Decoding is restricted to the alphabets of your configured languages, skipping most of the model's 256k-token vocabulary with identical output (6–20% faster per sentence).
- **int8 everywhere it matters.** Translation runs on CTranslate2, English speech on faster-whisper, and voice on ONNX Runtime.
- **One warm model manager.** Every request, room and agent shares the same loaded models. Languages load lazily and idle models are released.
- **Greedy decoding for live speech, beam search only for documents.**

**What comes next for speed:** captions that stream word by word as they are decoded, a smaller distilled student model per language pair, and published numbers for plugged-in, idle and low-end (4-core, 8 GB) machines. See [docs/benchmarks.md](docs/benchmarks.md).

## Accuracy

- **Strong open baselines.** Meta NLLB-200 for translation, MMS for African-language speech, Whisper for English.
- **Corrections apply on the next request.** An approved correction becomes a translation-memory entry and is served exactly as written.
- **A new model ships only if it wins.** Fine-tuned models must beat the current one on held-out data: chrF++ on FLORES-200 for text, WER on FLEURS for speech. FLORES is never used for training.
- **Tone marks survive.** Yoruba tone marks and Igbo dot-below letters are normalised before translation and speech, so *ọkọ*, *ọkọ̀* and *ọ̀kọ̀* stay three different words.

Published scores per language pair will accompany each model pack release.

## Train it on your data

Already have translated text? Upload any parallel text you have. AfriLingua aligns, cleans and balances it, then fine-tunes on it.

| Formats | How it is aligned |
|---|---|
| TMX, CSV, TSV, JSONL, line-aligned text | Row by row, matching language codes |
| SRT, WebVTT | By timestamp overlap, including 1:2 and 2:1 cue merges |
| USFM, eBible verse files | By chapter and verse reference |
| Plain text, not pre-aligned | Sentence-split and length-aligned automatically, keeping only confident pairs |

```bash
# Import a tab-separated file of English / Yoruba pairs
afrilingua corpora import phrases.tsv --format tsv --from english --to yoruba \
  --source-name "Clinic phrasebook" --license CC-BY-4.0 --domain medical

# Build deduplicated, domain-balanced splits (no domain over 40%), then fine-tune with LoRA
afrilingua train splits --lang yoruba --max-domain-share 0.4
afrilingua train mt --data-dir data/processed
```

Every corpus records its source, licence and domain. Cleaning drops duplicates, URLs, emails, wrong-language lines and bad length ratios. Domain capping stops any single kind of text from dominating the model.

**Only import text you have the right to use.** `--commercial-safe` builds leave out research-only and share-alike data. Uploaded corpora will require a second person's approval before they reach translation memory or training (in progress; see [Security model](#security-model)).

Corpora can also be uploaded over the API (`POST /api/v1/datasets`) or from the web app.

## Languages

| Language | Translate | Speech → text | Text → speech |
|---|---|---|---|
| English | ✓ | ✓ | ✓ |
| Yoruba | ✓ | ✓ | ✓ |
| Igbo | ✓ | ✓ | ✓ |
| Hausa | ✓ | ✓ | ✓ |

Any of the 200 languages NLLB-200 covers can be added without code changes. Speech support follows once there are recordings:

```bash
AFRILINGUA_EXTRA_LANGUAGES='[{"name":"swahili","nllb_code":"swh_Latn","mms_asr_code":"swh","mms_tts_model":"facebook/mms-tts-swh","aliases":["sw"]}]'
```

## Community

Speakers can translate prompt sentences, correct machine output, record their voice tagged with dialect and region (for example Ọ̀yọ́, Ìjẹ̀bú, Owerri, Onitsha, Kano, Sokoto) and review other people's work.

Contributions are licensed **CC-BY-4.0** by default, or CC0 if you prefer, and are credited. They stay on your device until you choose to share them. Voice recordings are never used for voice cloning without a separate opt-in.

## Security model

AfriLingua is built to run on your own machine, so the defaults assume a single trusted user:

- **The server binds to `127.0.0.1` and has no authentication by default.** Anything that can reach the port can translate, approve and delete translation-memory entries, import data and read contributions.
- **Set `API_KEYS` before you use `--lan` or expose the server anywhere.** Treat every key as an administrator credential.
- **A web page in your browser can reach `127.0.0.1`.** Until Origin and Host checks land, close untrusted tabs or set `API_KEYS` while the server runs.

A source review in November 2026 found issues in this area that are being fixed. They include no Origin/Host validation on the REST API, unauthenticated approval of translation-memory entries, no attempt limit on room PINs, no decoded-size limits on audio and DOCX uploads, and fuzzy translation-memory matches being served as exact. Please report security issues privately to the maintainers rather than in public issues.

## Configuration

Everything is optional and set through environment variables or `.env`. See [.env.example](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `AFRILINGUA_PROFILE` | `auto` | `cpu-lite` (8 GB RAM or less), `cpu-standard` or `gpu` |
| `DEVICE` | `auto` | `cpu` or `cuda` |
| `AFRILINGUA_OFFLINE` | `0` | `1` = never touch the network |
| `API_KEYS` | empty | Comma-separated keys. Required before exposing the server |
| `MODEL_NLLB` | from profile | Swap the translation model |

## Architecture

```
afrilingua/
  core/        settings, hardware profiles, language registry
  backends/    CTranslate2 / Transformers MT, faster-whisper & MMS ASR, MMS TTS, VAD, language ID
  models/      shared model manager (lazy loading, per-model locks), model downloads
  engine.py    normalize → cache → translation memory → MT; ASR; TTS
  tm/          translation memory and vector index
  training/    corpus import and alignment, dataset splits, LoRA / adapter training, evaluation
  contrib/     community contributions, review, export
  pipeline/    streaming interpretation sessions
  api/  mcp/   REST, WebSocket rooms, OpenAI-compatible routes, MCP server
web/           web app (Vike + React): landing, translate, documents, live rooms, community
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Licensing

- **Code:** MIT.
- **Models:** AfriLingua downloads third-party models under their own licences. **NLLB-200 and MMS are CC-BY-NC 4.0 (non-commercial).** Whisper is MIT. Review these before commercial use. A commercial-friendly model pack built on Apache-2.0 models is on the roadmap.
- **Community data:** CC-BY-4.0 by default.

## Contributing

Bug reports, language data, benchmark results from your hardware and code are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/fegig/AfriLingua.git && cd AfriLingua
pip install -e ".[dev]"
pytest -q                      # fast tests, no model downloads
pytest -m models               # real-model tests (downloads models)
pnpm --dir web install && pnpm --dir web dev
```

## Roadmap

- **Speed:** vocabulary-restricted decoding, a distilled student model, and published numbers for 4-core / 8 GB machines
- **Accuracy:** published chrF++ and WER per language pair with every model pack
- **Security:** Origin/Host validation, reviewer-only approval, PIN rate limits, upload decode limits
- Signed, versioned model packs (`afrilingua pull yoruba`), including domain packs (medical, legal, faith)
- A commercial-friendly model pack (Omnilingual ASR, MADLAD-400)
- Native desktop app with installers for Windows and macOS (today: `afrilingua desktop`)
- Live captions inside meetings: Google Meet, Zoom and Microsoft Teams integrations
- More languages: Swahili, Nigerian Pidgin, Twi, Amharic, Zulu, Fula
- Code-switching: English mixed with Yoruba, Igbo or Hausa in one sentence

## Acknowledgements

Meta AI (NLLB-200, MMS), OpenAI (Whisper), the CTranslate2, faster-whisper and Silero VAD projects, the Masakhane, NaijaVoices and Common Voice communities, and the Igbo, Yoruba and Hausa speakers this project exists to serve.
