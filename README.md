# AfriLingua

**Real-time AI translation for African indigenous languages.**
English ↔ Igbo · Yoruba · Hausa — live speech, documents, and text.

AfriLingua is an open-source project that makes language barriers disappear for speakers of African indigenous languages. Whether you are attending a conference, watching a presentation, or participating in a meeting conducted in English (or in Igbo, Yoruba, or Hausa), AfriLingua provides live transcribed translation on screen and through audio — in real time.

> **Use case:** An Igbo speaker at an English-language conference hears the translated speech in Igbo through their earphones while the original English transcript appears side by side on their screen. An English speaker in the same room gets the reverse — automatic translation of everything said in Igbo, Yoruba, or Hausa.

---

## Features

| Feature | Description |
|---|---|
| **Live conversation** | Speak into a microphone; hear and read the translation in real time |
| **Auto language detection** | Automatically identifies whether speech or text is English, Igbo, Yoruba, or Hausa |
| **Document translation** | Upload `.txt`, `.pdf`, or `.docx` files and download the translated version |
| **Text translation** | Paste or type any text and translate instantly, with optional audio playback |
| **Bidirectional** | Translates both ways: English → native language and native → English |
| **Offline-capable** | All models run locally on CPU — no cloud API required after setup |
| **Fine-tunable** | Built-in data pipeline to collect open corpora and fine-tune the translation model on domain-specific data |
| **Extensible** | Designed so any new language can be added by the community |

---

## Supported Languages

| Language | Code | ASR | TTS | Translation |
|---|---|---|---|---|
| English | `eng_Latn` | Whisper-small | pyttsx3 | NLLB-200 |
| Igbo | `ibo_Latn` | MMS-1B-All | MMS-TTS | NLLB-200 |
| Yoruba | `yor_Latn` | MMS-1B-All | MMS-TTS | NLLB-200 |
| Hausa | `hau_Latn` | MMS-1B-All | MMS-TTS | NLLB-200 |

Want to add another language? See [Adding a New Language](#adding-a-new-language).

---

## Model Stack

AfriLingua supports multiple model variants per role. The defaults are chosen for CPU compatibility. Swap any model by editing `.env` — no code changes required.

### Translation — NLLB-200

| Model ID | Size | Peak RAM | Speed (CPU) | Recommended For |
|---|---|---|---|---|
| `facebook/nllb-200-distilled-600M` ⭐ | 2.5 GB | ~4 GB | ~2 s/sentence | CPU default |
| `facebook/nllb-200-distilled-1.3B` | 5.1 GB | ~8 GB | ~5 s/sentence | GPU / high-RAM CPU |
| `facebook/nllb-200-3.3B` | 13 GB | ~16 GB | GPU only | Maximum accuracy |

```env
# .env
MODEL_NLLB=facebook/nllb-200-distilled-600M
```

### ASR — English (Whisper)

| Model ID | Size | Peak RAM | WER | Recommended For |
|---|---|---|---|---|
| `openai/whisper-tiny` | 75 MB | ~1 GB | Highest | Very low-spec CPU |
| `openai/whisper-base` | 145 MB | ~1 GB | High | Low-spec CPU |
| `openai/whisper-small` ⭐ | 460 MB | ~2 GB | Good | CPU default |
| `openai/whisper-medium` | 1.5 GB | ~5 GB | Better | GPU recommended |
| `openai/whisper-large-v3` | 3.1 GB | ~10 GB | Best | GPU |

```env
# .env
MODEL_WHISPER=openai/whisper-small
```

### ASR — African Languages (Meta MMS)

| Model ID | Size | Peak RAM | Recommended For |
|---|---|---|---|
| `facebook/mms-1b-all` ⭐ | ~4.0 GB | ~8 GB | Default ASR checkpoint; best coverage/accuracy |
| `facebook/mms-1b-l1107` | ~4.0 GB | ~8 GB | ASR checkpoint for 1,100+ languages |
| `facebook/mms-1b-fl102` | ~4.0 GB | ~8 GB | ASR checkpoint for a smaller language set |

```env
# .env
MODEL_MMS_ASR=facebook/mms-1b-all
```

`facebook/mms-300m` is a pretrained MMS checkpoint, not a fine-tuned ASR
checkpoint. It is useful for fine-tuning experiments, but it does not include
the ASR adapters required by runtime speech recognition.

### TTS — African Languages (Meta MMS-VITS)

Each language has its own dedicated model. These are small and fast on both CPU and GPU.

| Language | Model ID | Size |
|---|---|---|
| Igbo | `rnjema-unima/mms-tts-ibo-baseline` | ~50 MB |
| Yoruba | `facebook/mms-tts-yor` | ~50 MB |
| Hausa | `facebook/mms-tts-hau` | ~50 MB |

Meta does not currently publish an official `facebook/mms-tts-ibo` checkpoint,
so AfriLingua defaults to a public community Igbo VITS checkpoint. You can
replace it with another compatible model by changing `MODEL_MMS_TTS_IGBO`.

### TTS — English

| Engine | Notes |
|---|---|
| `pyttsx3` ⭐ | Fully offline, uses OS-level speech synthesis — no download required |
| `openai/whisper` (via MMS) | Can be swapped for a neural English TTS if preferred |

### Voice Activity Detection

| Model | Size | Notes |
|---|---|---|
| `snakers4/silero-vad` ⭐ | ~5 MB | Runs on CPU in real time; detects speech boundaries |

### Language Detection

| Library | Notes |
|---|---|
| `lingua-language-detector` ⭐ | Offline; English + Yoruba via ML model; Igbo + Hausa via word-list heuristic |

---

## Hardware & Device Configuration

Set `DEVICE` in `.env` to control where models run:

```env
# Options: auto | cpu | cuda
DEVICE=auto

# Options: auto | float32 | float16
TORCH_DTYPE=auto
```

| Value | Behaviour |
|---|---|
| `auto` | Detects CUDA at startup. Uses GPU if available, falls back to CPU. |
| `cpu` | Forces CPU regardless of hardware. Safe on any machine. |
| `cuda` | Forces GPU. Warns and falls back to CPU if CUDA is unavailable. |

| `TORCH_DTYPE` | Behaviour |
|---|---|
| `auto` | `float16` on GPU (2× faster, half the VRAM); `float32` on CPU |
| `float32` | Full precision — always safe, higher memory usage |
| `float16` | Half precision — GPU only; do not use on CPU |

### Choosing a hardware profile

| Profile | Recommended settings | Expected RAM | Translation speed |
|---|---|---|---|
| Low-spec laptop (no GPU) | `DEVICE=cpu`, `MODEL_NLLB=…-600M`, `MODEL_WHISPER=…-base` | ~6 GB | ~3–5 s/sentence |
| Standard laptop / desktop | `DEVICE=auto`, `MODEL_NLLB=…-600M`, `MODEL_WHISPER=…-small` | ~6–8 GB | ~1–3 s/sentence |
| GPU workstation (8 GB VRAM) | `DEVICE=cuda`, `MODEL_NLLB=…-1.3B`, `MODEL_WHISPER=…-medium` | ~8 GB VRAM | <1 s/sentence |
| GPU workstation (24 GB VRAM) | `DEVICE=cuda`, `MODEL_NLLB=…-3.3B`, `MODEL_WHISPER=…-large-v3` | ~20 GB VRAM | <0.5 s/sentence |

---

## Deployment Modes

Set `DEPLOYMENT_MODE` in `.env`:

```env
# Options: web | desktop
DEPLOYMENT_MODE=web
```

### Web Application (default)

The default mode. A Gradio browser UI is served alongside a FastAPI backend. Accessible from any browser on the local network — ideal for conference rooms, shared workstations, or cloud hosting.

```bash
# Start the web UI
python ui/app.py
# → http://localhost:7860

# Or start only the API backend
uvicorn api.main:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000/docs
```

**When to use:** shared environments, multi-user setups, browser-based access, server deployment, VS Code / browser plugin integration.

**To expose on your local network:**
```env
# .env
API_HOST=0.0.0.0
```

### Desktop Standalone Application

A self-contained native desktop application built with PyQt6. No browser required. Suitable for offline use on a single machine. **Minimises to the system tray** and runs as a background process — right-click the tray icon to start/stop interpretation or restore the window.

**Additional dependencies:**
```bash
pip install PyQt6 pyinstaller
```

**Run directly:**
```bash
python ui/desktop_app.py
```

The desktop app provides the same four surfaces as the web UI:

| Tab | Description |
|---|---|
| 🎙️ Live Interpretation | Simultaneous interpretation — mic → translated audio + live scrolling transcript |
| ✍️ Text Translation | Side-by-side input/output with optional TTS playback |
| 📄 Document Translation | Browse for `.txt` / `.pdf` / `.docx`, progress bar, save translated file |
| ⚙️ Settings | Device/GPU info, VAD chunk interval slider, model info |

**System tray behaviour:**
- Clicking **X** hides the window to the tray — the process keeps running
- Right-click the tray icon → **Start Interpretation**, **Stop**, **Show Window**, or **Quit**
- Left-click the tray icon → restore window

**Package as a standalone executable:**
```bash
# Windows (.exe — no Python required on target machine)
pyinstaller --onefile --windowed --name AfriLingua ui/desktop_app.py

# macOS (.app bundle)
pyinstaller --onefile --windowed --name AfriLingua ui/desktop_app.py

# Linux (ELF binary)
pyinstaller --onefile --name AfriLingua ui/desktop_app.py
```

The packaged executable is output to `dist/AfriLingua`. Models are downloaded on first launch and cached in `models/cache/`.

**When to use:** conference interpreters, field workers, fully offline environments, or distributing AfriLingua as a standalone product without requiring a browser.

### Browser Extension / Plugin

The FastAPI WebSocket endpoint (`ws://127.0.0.1:8000/translate/stream`) can power a browser extension that overlays translated subtitles on top of any web meeting (Google Meet, Zoom web, Teams web). The extension streams mic audio to the local API and renders text on screen.

Extension development is on the roadmap. The API contract is stable and documented at `http://127.0.0.1:8000/docs`.

---

## Project Structure

```
afrilingua/
├── src/
│   ├── config.py              # Centralised config (reads .env)
│   ├── translation/
│   │   ├── engine.py          # NLLB TranslationEngine
│   │   └── finetune.py        # Fine-tuning script (Seq2SeqTrainer)
│   ├── asr/
│   │   ├── whisper_asr.py     # Whisper ASR (English)
│   │   ├── mms_asr.py         # Meta MMS ASR (African languages)
│   │   ├── vad.py             # Silero voice activity detection
│   │   └── audio_capture.py   # Microphone capture
│   ├── tts/
│   │   ├── mms_tts.py         # Meta MMS TTS (African languages)
│   │   ├── english_tts.py     # pyttsx3 TTS (English)
│   │   └── audio_player.py    # Async audio playback queue
│   ├── detection/
│   │   └── lang_detector.py   # Lingua language detector
│   ├── pipeline/
│   │   └── realtime_pipeline.py  # End-to-end orchestration
│   └── document/
│       ├── reader.py          # TXT / PDF / DOCX reader
│       └── translator.py      # Document translation
├── data/
│   ├── collect_data.py        # Download open corpora
│   ├── preprocess.py          # Clean, filter, split datasets
│   ├── raw/                   # Downloaded corpus files
│   ├── processed/             # Cleaned train/val/test JSONL
│   └── finetuned/             # Fine-tuned model checkpoints
├── api/
│   └── main.py                # FastAPI REST + WebSocket backend
├── ui/
│   └── app.py                 # Gradio web UI (3 tabs)
├── tests/
│   └── test_translation.py    # Round-trip translation tests
├── models/
│   └── cache/                 # HuggingFace model cache
├── .env                       # Environment configuration
├── pyproject.toml
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10 or higher.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/afrilingua.git
cd afrilingua

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

Models are downloaded automatically from HuggingFace on first use and cached to `models/cache/`. Total download is approximately 7–8 GB with the default ASR checkpoint.

---

## Quick Start

### Web UI (recommended)

```bash
python ui/app.py
```

Open `http://localhost:7860` in your browser. Three tabs are available:

- **Live Conversation** — select your language, click the microphone, speak. Translation appears as text and plays as audio.
- **Document Translation** — upload a `.txt`, `.pdf`, or `.docx` file and download the translated version.
- **Text Translation** — type or paste any text, choose source and target languages, translate.

### API Server

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API documentation is available at `http://127.0.0.1:8000/docs`.

**Translate text (HTTP):**
```bash
curl -X POST http://127.0.0.1:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning everyone", "tgt_lang": "igbo"}'
```

**Translate a document:**
```bash
curl -X POST http://127.0.0.1:8000/translate/document \
  -F "file=@speech.txt" \
  -F "tgt_lang=yoruba"
```

**Real-time streaming (WebSocket):**
Connect to `ws://127.0.0.1:8000/translate/stream`, send a JSON config first:
```json
{"src_lang": "english", "tgt_lang": "hausa", "audio_output": true}
```
Then stream raw float32 PCM audio frames (16 kHz, mono). The server responds with:
```json
{"original": "...", "translated": "...", "audio_b64": "..."}
```

---

## Fine-Tuning on Domain Data

The base NLLB-200 model already supports all three languages. Fine-tuning on open corpora (FLORES-200, OPUS-100, Tatoeba) can further improve quality for specific domains like conference speech.

```bash
# Step 1: Download open parallel corpora (~30 min, requires internet)
python data/collect_data.py

# Step 2: Clean, deduplicate and split into train/val/test
python data/preprocess.py

# Step 3a: Quick smoke test (500 samples, ~10 min on CPU)
python src/translation/finetune.py --max_train_samples 500

# Step 3b: Full fine-tuning (all collected data, several hours on CPU)
python src/translation/finetune.py
```

The fine-tuned checkpoint is saved to `data/finetuned/nllb-afrilingua/`. The `TranslationEngine` automatically loads it on next startup if it exists.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Round-trip translation for all three language pairs
- Batch translation
- Same-language passthrough
- Empty string handling
- Unsupported language error

---

## Configuration

All settings live in `.env` at the project root. Copy `.env.example` to `.env` and edit as needed. No source code changes are required.

```env
# ── Device ──────────────────────────────────────────────────────────────────
DEVICE=auto          # auto | cpu | cuda
TORCH_DTYPE=auto     # auto | float32 | float16

# ── Deployment ──────────────────────────────────────────────────────────────
DEPLOYMENT_MODE=web  # web | desktop

# ── Models (swap to upgrade quality or reduce memory) ───────────────────────
MODEL_NLLB=facebook/nllb-200-distilled-600M
MODEL_WHISPER=openai/whisper-small
MODEL_MMS_ASR=facebook/mms-1b-all
MODEL_MMS_TTS_IGBO=rnjema-unima/mms-tts-ibo-baseline
MODEL_MMS_TTS_YORUBA=facebook/mms-tts-yor
MODEL_MMS_TTS_HAUSA=facebook/mms-tts-hau

# ── Paths ────────────────────────────────────────────────────────────────────
HF_CACHE=./models/cache
FINETUNED_MODEL_DIR=./data/finetuned/nllb-afrilingua

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE=16000
VAD_THRESHOLD=0.5

# ── API / UI ─────────────────────────────────────────────────────────────────
API_HOST=127.0.0.1
API_PORT=8000
GRADIO_PORT=7860
```

See [Hardware & Device Configuration](#hardware--device-configuration) and [Deployment Modes](#deployment-modes) for full details on each option.

---

## Adding a New Language

AfriLingua is designed to be extended. To add a new language (e.g. Swahili):

**1. Check NLLB and MMS support**

Verify the language has an NLLB code at [NLLB supported languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) and an MMS ASR/TTS model at [MMS model list](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html).

**2. Add the language code**

In `src/translation/engine.py`, add to `LANG_CODES`:
```python
LANG_CODES = {
    ...
    "swahili": "swh_Latn",   # add this line
}
```

**3. Add MMS ASR support**

In `src/asr/mms_asr.py`, add to `MMS_LANG_MAP`:
```python
MMS_LANG_MAP = {
    ...
    "swahili": "swh",   # MMS language tag
}
```

**4. Add MMS TTS support**

In `.env` and `src/config.py`, add the TTS model ID:
```env
MODEL_MMS_TTS_SWAHILI=facebook/mms-tts-swh
```

Then in `src/tts/mms_tts.py`, add to `_LANG_TO_MODEL_ID`:
```python
_LANG_TO_MODEL_ID = {
    ...
    "swahili": MODEL_MMS_TTS_SWAHILI,
}
```

**5. Add to lingua detection**

In `src/detection/lang_detector.py`, add `Language.SWAHILI` to `_SUPPORTED` and `_LINGUA_TO_NAME`.

**6. Contribute training data**

Add a new corpus collector function in `data/collect_data.py` following the existing pattern, then open a pull request.

---

## Contributing

Contributions are very welcome. AfriLingua exists to serve communities that have historically been underrepresented in AI and language technology. Every improvement — whether a bug fix, a new language, better training data, or UI polish — makes a real difference.

### How to contribute

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature/add-swahili`
3. Make your changes and add tests where applicable
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a pull request with a clear description of what you changed and why

### What we especially need

- **Training data** — Parallel sentence pairs for any African language (see `data/collect_data.py` for the pattern to follow)
- **New languages** — Any language supported by NLLB-200 and Meta MMS can be added (see [Adding a New Language](#adding-a-new-language))
- **Domain-specific corpora** — Medical, legal, news, and conference speech data in African languages
- **ASR improvements** — Better speech recognition for low-resource languages
- **TTS quality** — Naturalness improvements for synthesised African language speech
- **UI/UX** — Accessibility, mobile layout, additional display modes
- **Documentation** — Translations of this README into Igbo, Yoruba, and Hausa

### Code style

- Python 3.10+, type hints preferred
- Follow the existing module structure — one concern per file
- No external API calls from core modules (keep it offline-capable)
- Add a test for any new translation language pair

### Reporting issues

Open a GitHub issue and include:
- Your OS and Python version
- The exact command you ran
- The full error message or unexpected behaviour

---

## Roadmap

- [ ] React/TypeScript production frontend with conference-room layout
- [ ] ONNX export + INT8 quantization for low-power devices
- [x] Desktop app — `ui/desktop_app.py` with PyQt6 + system tray + PyInstaller packaging (Windows / macOS / Linux)
- [ ] Browser extension — subtitle overlay for web meetings (Google Meet, Zoom, Teams)
- [ ] Mobile app (React Native / Flutter)
- [ ] Speaker diarization — label who is speaking in multi-speaker sessions
- [ ] Yoruba tonal marking improvement
- [ ] Igbo dialect handling (Standard Igbo vs. regional variants)
- [ ] More languages: Swahili, Amharic, Twi, Zulu, Wolof, Fula, and more

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute it. See `LICENSE` for full terms.

The underlying AI models each have their own licenses:
- Meta NLLB-200: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- Meta MMS: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- OpenAI Whisper: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)

Please review the model licenses before using this project in a commercial product.

---

## Acknowledgements

- [Meta AI — NLLB-200](https://ai.meta.com/research/no-language-left-behind/) for multilingual translation
- [Meta AI — MMS](https://ai.meta.com/blog/multilingual-model-speech-recognition/) for speech recognition and synthesis across 1,100+ languages
- [OpenAI — Whisper](https://github.com/openai/whisper) for robust English speech recognition
- [Silero VAD](https://github.com/snakers4/silero-vad) for lightweight voice activity detection
- [OPUS corpora](https://opus.nlpl.eu/) and [FLORES-200](https://github.com/facebookresearch/flores) for open parallel data
- The Igbo, Yoruba, and Hausa communities whose languages this project is built to serve
