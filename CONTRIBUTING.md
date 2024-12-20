# Contributing to AfriLingua

Thank you for your interest in contributing. AfriLingua exists to serve African language communities that are underrepresented in AI and language technology. Your contribution — whether code, data, documentation, or a bug report — makes a real difference.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Adding a New Language](#adding-a-new-language)
- [Adding Training Data](#adding-training-data)
- [Pull Request Process](#pull-request-process)
- [Commit Style](#commit-style)
- [Project Layout](#project-layout)

---

## Code of Conduct

Be respectful of all contributors and communities. Discussions should stay focused on technical merit. Discrimination of any kind will not be tolerated.

---

## Ways to Contribute

### High priority

- **New languages** — Add any language supported by NLLB-200 and Meta MMS (Swahili, Amharic, Twi, Zulu, Wolof, Fula, etc.)
- **Training data** — Contribute parallel sentence pairs for existing or new languages
- **Domain corpora** — Medical, legal, news, and conference speech data in African languages
- **Bug reports** — Open a GitHub issue with reproduction steps

### General

- Improve ASR accuracy for low-resource languages
- Improve TTS naturalness
- UI/UX improvements and accessibility
- Translate this documentation into Igbo, Yoruba, or Hausa
- Write more tests

---

## Development Setup

```bash
# Fork and clone
git clone https://github.com/<your-username>/afrilingua.git
cd afrilingua

# Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests to confirm everything works
pytest tests/ -v
```

---

## Adding a New Language

Follow these steps to add full support for a new language. Use the existing Igbo/Yoruba/Hausa implementation as a reference throughout.

### 1. Verify model support

Before starting, confirm that:
- The language has an NLLB-200 code — check the [FLORES-200 language list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- Meta MMS has an ASR model for the language — check the [MMS language coverage page](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)
- Meta MMS has a TTS model for the language — same page, TTS column

### 2. Translation support

In `src/translation/engine.py`, add the NLLB language code to `LANG_CODES`:

```python
LANG_CODES: dict[str, str] = {
    "english": "eng_Latn",
    "igbo":    "ibo_Latn",
    "yoruba":  "yor_Latn",
    "hausa":   "hau_Latn",
    "swahili": "swh_Latn",   # <-- add your language here
}
```

### 3. ASR support

In `src/asr/mms_asr.py`, add the MMS ASR language tag to `MMS_LANG_MAP`:

```python
MMS_LANG_MAP: dict[str, str] = {
    "igbo":    "ibo",
    "yoruba":  "yor",
    "hausa":   "hau",
    "swahili": "swh",   # <-- add your language here
}
```

### 4. TTS support

Add the model ID to `.env`:
```env
MODEL_MMS_TTS_SWAHILI=facebook/mms-tts-swh
```

Add the config variable in `src/config.py`:
```python
MODEL_MMS_TTS_SWAHILI: str = _get("MODEL_MMS_TTS_SWAHILI", "facebook/mms-tts-swh")
```

Register it in `src/tts/mms_tts.py`:
```python
from src.config import ..., MODEL_MMS_TTS_SWAHILI

_LANG_TO_MODEL_ID: dict[str, str] = {
    "igbo":    MODEL_MMS_TTS_IGBO,
    "yoruba":  MODEL_MMS_TTS_YORUBA,
    "hausa":   MODEL_MMS_TTS_HAUSA,
    "swahili": MODEL_MMS_TTS_SWAHILI,   # <-- add here
}
```

### 5. Language detection

In `src/detection/lang_detector.py`, add the Lingua language to `_SUPPORTED` and `_LINGUA_TO_NAME`:

```python
from lingua import Language, LanguageDetectorBuilder

_SUPPORTED = [
    Language.ENGLISH,
    Language.IGBO,
    Language.YORUBA,
    Language.HAUSA,
    Language.SWAHILI,   # <-- add here
]

_LINGUA_TO_NAME: dict[Language, str] = {
    Language.ENGLISH:  "english",
    Language.IGBO:     "igbo",
    Language.YORUBA:   "yoruba",
    Language.HAUSA:    "hausa",
    Language.SWAHILI:  "swahili",   # <-- add here
}
```

> Check [lingua-language-detector supported languages](https://github.com/pemistahl/lingua-rs#which-languages-are-supported) to confirm Lingua supports your language.

### 6. Add tests

In `tests/test_translation.py`, add a round-trip test:

```python
def test_english_to_swahili(engine):
    result = engine.translate("Good morning", "english", "swahili")
    assert isinstance(result, str) and len(result) > 0
```

### 7. Add training data

See [Adding Training Data](#adding-training-data) below.

### 8. Update the README

Add your language to the **Supported Languages** table in `README.md` and remove it from the **Roadmap** if it was listed there.

---

## Adding Training Data

Training data improves translation quality for specific domains. All data collection is done in `data/collect_data.py`.

### Using existing open sources

The file already collects from FLORES-200, OPUS-100, and Tatoeba. If your language is available on those platforms, simply add its code to the relevant maps at the top of `collect_data.py`:

```python
_FLORES_CODES = {
    "igbo":    "ibo_Latn",
    "yoruba":  "yor_Latn",
    "hausa":   "hau_Latn",
    "swahili": "swh_Latn",   # <-- add here
}

_OPUS_CODES = {
    "igbo":    "ig",
    "yoruba":  "yo",
    "hausa":   "ha",
    "swahili": "sw",   # <-- ISO 639-1 code
}
```

### Adding a custom corpus

If you have a custom parallel dataset (e.g. scraped news articles, community translations, recorded speech), add a collector function following this pattern:

```python
def collect_my_corpus(lang_name: str) -> list[dict]:
    """
    Return a list of {"english": "...", "translation": "...", "lang": lang_name}
    records from your custom source.
    """
    records = []
    # load your data source here
    for english_sentence, native_sentence in my_data_source:
        if english_sentence.strip() and native_sentence.strip():
            records.append({
                "english":     english_sentence.strip(),
                "translation": native_sentence.strip(),
                "lang":        lang_name,
            })
    return records
```

Then call it inside `collect_all()`:

```python
def collect_all() -> None:
    for lang_name, flores_code in _FLORES_CODES.items():
        all_records = []
        all_records.extend(collect_flores(lang_name, flores_code))
        all_records.extend(collect_opus100(lang_name, _OPUS_CODES[lang_name]))
        all_records.extend(collect_tatoeba(lang_name, _OPUS_CODES[lang_name]))

        if lang_name == "swahili":
            all_records.extend(collect_my_corpus("swahili"))   # <-- add here
        ...
```

### Data quality guidelines

- Each record must have both an English sentence and a native language sentence
- Minimum 3 words per sentence; maximum ~200 tokens
- Avoid machine-translated data as training input — it creates a feedback loop
- Include domain diversity: news, conversation, formal speech, technical text
- Do not include personally identifiable information
- Provide a source URL or citation in your pull request description

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add or update tests if you changed any logic
4. Run the full test suite: `pytest tests/ -v`
5. Push to your fork and open a pull request against the `main` branch
6. Fill in the pull request template — describe what you changed, why, and how to test it
7. A maintainer will review within a few days

### Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] No hard-coded file paths or API keys
- [ ] New languages documented in `README.md`
- [ ] New dependencies added to `requirements.txt`
- [ ] Docstrings updated if you changed a public function signature

---

## Commit Style

Use short, imperative commit messages:

```
Add Swahili translation support
Fix VAD silence threshold for Hausa speech
Improve MMS TTS audio queue flushing
Add Tatoeba corpus collector for Yoruba
```

For larger changes, add a body paragraph after a blank line explaining the motivation.

---

## Project Layout (quick reference)

```
src/translation/    — NLLB engine and fine-tuning script
src/asr/            — Whisper, MMS ASR, VAD, mic capture
src/tts/            — MMS TTS, English TTS, audio player
src/detection/      — Language detection (lingua)
src/pipeline/       — Real-time orchestration
src/document/       — Document reading and translation
data/               — Corpus collection and preprocessing scripts
api/                — FastAPI backend
ui/                 — Gradio web interface
tests/              — Pytest test suite
```

---

Questions? Open a GitHub issue or start a discussion. We are happy to help you get your contribution merged.
