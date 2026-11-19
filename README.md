# AfriLingua

**Fast, lightweight AI translation and speech for African languages.**
Igbo, Yoruba, Hausa and English today, with more added by configuration and community data. It runs offline on an ordinary laptop CPU.

Use it as a **Python library**, a **local REST API** (OpenAI-compatible), or an **MCP server** that AI agents can call. Or run the built-in **live interpretation app** for conferences, churches, clinics and classrooms.

[![CI](https://github.com/fegig/AfriLingua/actions/workflows/ci.yml/badge.svg)](https://github.com/fegig/AfriLingua/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
![License](https://img.shields.io/badge/code-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

```python
import afrilingua

afrilingua.translate("Good morning, everyone.", to="yoruba")
# e.g. 'Ẹ káàárọ̀, gbogbo yín.'  (illustrative; actual output depends on the model)
```

> [!NOTE]
> AfriLingua v2 is in **alpha**. The engine, library API and contracts are stable enough to build on; the interpretation app and community hub are under active development. Benchmarks are published as they are measured. See [Speed and accuracy](#speed-and-accuracy).

---

## Why AfriLingua

| | Typical cloud speech/translation API | AfriLingua |
|---|---|---|
| Runs without internet | No | **Yes**, after a one-time model download |
| Runs on a CPU-only laptop | n/a (server-side) | **Yes**: int8 models, about 3 GB RAM target |
| Your audio leaves the device | Yes | **No**. Data stays local unless you choose to share it |
| Igbo / Yoruba / Hausa speech **and** text | Varies | **Yes**: speech-to-text, translation, text-to-speech |
| Plugs into AI agents | Rarely | **Yes**: MCP server and OpenAI-compatible endpoints |
| Improves from its users | No | **Yes**: corrections and recordings feed the next model |
| Cost per minute | Metered | **Free** to run on your own hardware |

## Install

> Not on PyPI yet. Install from GitHub:

```bash
pip install "afrilingua[all] @ git+https://github.com/fegig/AfriLingua.git@dev"
```

Extras: `fast` (int8 CPU backends: CTranslate2, faster-whisper, ONNX), `server` (REST and MCP), `audio` (microphone), `documents` (DOCX/PDF), `all` (everything above), `train` (fine-tuning).

Requirements: Python 3.11–3.13, about 3 GB free RAM, 2–3 GB disk for models. A GPU is optional; it is used automatically when present.

Download models once (then work fully offline):

```bash
afrilingua models pull --langs igbo,yoruba,hausa
```

## Use it

### 1. As a Python library

```python
import afrilingua

afrilingua.translate("Where is the clinic?", to="hausa")                 # auto-detects the source
afrilingua.translate(["Thank you", "Welcome"], to="igbo", source="english")
afrilingua.translate(long_text, to="yoruba", mode="document")            # beam search for higher quality

text  = afrilingua.transcribe(audio, lang="yoruba")   # mono float32, 16 kHz
voice = afrilingua.speak("Ndewo", lang="igbo")        # SpeechAudio(waveform, sample_rate)

afrilingua.languages()                                # what is installed and what each language supports
```

Need control over backends, the translation memory or live streaming? Use the engine directly:

```python
from afrilingua.engine import build_engine

engine = build_engine()
result = engine.translate_one("Good morning", "english", "yoruba")
result.text, result.origin, result.latency_ms   # origin: "tm_exact" | "cache" | "mt" | "mt_tm_assisted"
```

### 2. As a local API (OpenAI-compatible)

```bash
afrilingua serve            # http://127.0.0.1:8000  (docs at /docs)
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning", "tgt_lang": "igbo"}'
```

Apps already written for OpenAI or Whisper audio work by changing the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
client.audio.transcriptions.create(model="afrilingua", file=open("sermon.wav", "rb"), language="yo")
```

Full contract: [docs/api-contract.md](docs/api-contract.md).

### 3. As an MCP server for AI agents

```json
{ "mcpServers": { "afrilingua": { "command": "afrilingua", "args": ["mcp"] } } }
```

Agents get tools such as `translate_text`, `transcribe_audio`, `synthesize_speech`, `translate_speech`, `search_translation_memory` and `create_live_room`. All of them run locally. See [docs/mcp.md](docs/mcp.md).

### 4. Live interpretation app

```bash
afrilingua serve --lan      # set API_KEYS first when exposing on a network
```

Open the web app, create a room and share the QR code. Listeners pick their language on their phones and get live captions and audio. Projector mode shows large two-language captions.

## Speed and accuracy

These are the two things we optimise for and measure on every release. **We do not publish a number we cannot reproduce.** Run `afrilingua bench` on your own machine.

**How it stays fast on a CPU**

- **int8 inference everywhere it matters.** Translation runs on CTranslate2, English speech recognition on faster-whisper, and voice/TTS on ONNX Runtime.
- **Translation memory with vector search.** Previously approved translations come back in about a millisecond, before any model runs. Repeated phrases (agendas, liturgy, speaker names, medical terms) are instant and consistent.
- **One shared model manager.** Every request, room and agent call reuses the same loaded models. Models load lazily and idle ones are evicted.
- **Streaming pipeline.** Voice activity detection cuts speech at natural pauses, and captions are never blocked by speech synthesis.

**Targets for a modern 8-core laptop CPU** (measured results will replace this table in [docs/benchmarks.md](docs/benchmarks.md)):

| Stage | Target |
|---|---|
| Translation, one sentence (int8, greedy) | 150–400 ms |
| Translation memory hit | under 5 ms |
| End of speech → caption on screen | about 1–1.5 s |
| End of speech → translated audio | about 1.5–2.5 s |
| RAM with 4 languages loaded | about 2.5–3.5 GB |

**How accuracy improves**

- Strong open baselines: Meta NLLB-200 for translation, MMS for African-language speech, Whisper for English.
- **Community corrections become approved translation-memory entries immediately**, and become training data for the next model pack.
- Fine-tuning (LoRA for translation, adapters for speech) runs on community data, with an **evaluation gate**: a new model ships only if it beats the current one on held-out data.
- Published per release: chrF++ / BLEU on FLORES-200 devtest and WER on FLEURS, per language pair. FLORES is never used for training.

## Languages

| Language | Translate | Speech → text | Text → speech |
|---|---|---|---|
| English | ✓ | ✓ | ✓ |
| Igbo | ✓ | ✓ | ✓ |
| Yoruba | ✓ | ✓ | ✓ |
| Hausa | ✓ | ✓ | ✓ |

Add a language without code changes. Any language covered by NLLB-200 and MMS (hundreds of them) can be registered:

```bash
AFRILINGUA_EXTRA_LANGUAGES='[{"name":"swahili","nllb_code":"swh_Latn","mms_asr_code":"swh","mms_tts_model":"facebook/mms-tts-swh","aliases":["sw"]}]'
```

## Help train it

Pre-trained models only go so far for African languages, especially with accents, dialects and tone. The community hub in the app lets speakers:

- translate prompt sentences and correct machine output,
- record their voice, tagged with dialect and region (for example Ọ̀yọ́, Ìjẹ̀bú, Owerri, Onitsha, Kano, Sokoto),
- review other people's contributions (two approvals publish an entry).

Contributions are licensed **CC-BY-4.0** by default (CC0 optional), are credited, and stay on your device until you choose to share them. Voice recordings are never used for voice cloning without a separate opt-in. See [docs/community.md](docs/community.md).

## Configuration

Everything is optional and set through environment variables or `.env`; see [.env.example](.env.example). Common ones:

| Variable | Default | Purpose |
|---|---|---|
| `AFRILINGUA_PROFILE` | `auto` | `cpu-lite` (≤8 GB RAM), `cpu-standard` or `gpu` |
| `DEVICE` | `auto` | `cpu` or `cuda` |
| `AFRILINGUA_OFFLINE` | `0` | `1` = never touch the network |
| `API_KEYS` | empty | Required before exposing the server on a network |
| `MODEL_NLLB` | from profile | Swap the translation model |

## Architecture

```
afrilingua/
  core/        settings, hardware profiles, language registry
  backends/    CTranslate2 / Transformers MT, faster-whisper & MMS ASR, MMS TTS, VAD, language ID
  models/      shared model manager (lazy loading, per-model locks)
  engine.py    facade: normalize → cache → translation memory → MT; ASR; TTS
  pipeline/    streaming interpretation sessions
  tm/          translation memory and vector index
  contrib/     community contributions, review, export
  api/  mcp/   REST, WebSocket rooms, OpenAI-compatible routes, MCP server
web/           React app: interpretation, translation, community hub
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Licensing

- **Code**: MIT.
- **Models**: AfriLingua downloads third-party models with their own licenses. **NLLB-200 and MMS are CC-BY-NC 4.0 (non-commercial).** Whisper is MIT. If you plan commercial use, review these licenses. A commercial-friendly model pack built on Apache-2.0 models is on the roadmap.
- **Community data**: CC-BY-4.0 by default.

## Contributing

Bug reports, language data, benchmarks on your hardware and code are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Quick dev setup:

```bash
git clone https://github.com/fegig/AfriLingua.git && cd AfriLingua
pip install -e ".[dev]"
pytest -q                      # fast tests, no model downloads
pytest -m models               # real-model tests (downloads models)
pnpm --dir web install && pnpm --dir web dev
```

## Roadmap

- Measured benchmarks per release, for CPU-lite and GPU profiles
- Signed, versioned model packs (`afrilingua pull yoruba`), including domain packs (medical, legal, faith)
- Commercial-friendly model pack (Omnilingual ASR, MADLAD-400)
- Desktop installer for Windows and macOS
- More languages: Swahili, Pidgin, Twi, Amharic, Zulu, Fula
- Code-switching (English mixed with Yoruba, Igbo or Hausa in one sentence)

## Acknowledgements

Meta AI (NLLB-200, MMS), OpenAI (Whisper), the CTranslate2, faster-whisper and Silero VAD projects, the Masakhane, NaijaVoices and Common Voice communities, and the Igbo, Yoruba and Hausa speakers this project exists to serve.
