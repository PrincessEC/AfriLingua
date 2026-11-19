"""
Central language registry for AfriLingua.

The built-in languages are English, Igbo, Yoruba, and Hausa. Extra languages can
be added without code changes by setting AFRILINGUA_EXTRA_LANGUAGES to JSON, or
by pointing LANGUAGE_REGISTRY_FILE at a JSON file.

Example:
[
  {
    "name": "swahili",
    "nllb_code": "swh_Latn",
    "mms_asr_code": "swh",
    "mms_tts_model": "facebook/mms-tts-swh",
    "aliases": ["kiswahili"],
    "detection_words": ["habari", "asante"]
  }
]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
_USER_LANGUAGE_REGISTRY_FILE = _ROOT / "data" / "languages.json"


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    nllb_code: str
    mms_asr_code: str | None = None
    mms_tts_model: str | None = None
    whisper_lang: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    detection_words: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_asr(self) -> bool:
        return bool(self.whisper_lang or self.mms_asr_code)

    @property
    def supports_tts(self) -> bool:
        return self.name == "english" or bool(self.mms_tts_model)


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _env_model(name: str, default: str | None) -> str | None:
    key = f"MODEL_MMS_TTS_{name.upper().replace('-', '_')}"
    return os.environ.get(key, default)


_DEFAULT_SPECS: list[LanguageSpec] = [
    LanguageSpec(
        name="english",
        nllb_code="eng_Latn",
        whisper_lang="en",
        aliases=("en",),
    ),
    LanguageSpec(
        name="igbo",
        nllb_code="ibo_Latn",
        mms_asr_code="ibo",
        mms_tts_model=_env_model("igbo", "rnjema-unima/mms-tts-ibo-baseline"),
        aliases=("ibo", "ig"),
        detection_words=("kedu", "nne", "nna", "anyi", "daalu", "nnoo", "biko", "ulo", "aha"),
    ),
    LanguageSpec(
        name="yoruba",
        nllb_code="yor_Latn",
        mms_asr_code="yor",
        mms_tts_model=_env_model("yoruba", "facebook/mms-tts-yor"),
        aliases=("yor", "yo"),
    ),
    LanguageSpec(
        name="hausa",
        nllb_code="hau_Latn",
        mms_asr_code="hau",
        mms_tts_model=_env_model("hausa", "facebook/mms-tts-hau"),
        aliases=("hau", "ha"),
        detection_words=("yaya", "kai", "ina", "gida", "kana", "tafi", "yaro", "wannan", "nagode", "sannu"),
    ),
]


def _coerce_spec(raw: dict[str, Any]) -> LanguageSpec:
    name = _norm(str(raw.get("name", "")))
    if not name:
        raise ValueError("language entry is missing 'name'")

    nllb_code = raw.get("nllb_code") or raw.get("nllb") or raw.get("code")
    if not nllb_code:
        raise ValueError(f"language {name!r} is missing 'nllb_code'")

    aliases = tuple(_norm(str(item)) for item in raw.get("aliases", []) if str(item).strip())
    words = tuple(str(item).strip().lower() for item in raw.get("detection_words", []) if str(item).strip())
    tts_model = raw.get("mms_tts_model") or raw.get("tts_model")

    return LanguageSpec(
        name=name,
        nllb_code=str(nllb_code).strip(),
        mms_asr_code=(str(raw["mms_asr_code"]).strip() if raw.get("mms_asr_code") else None),
        mms_tts_model=_env_model(name, str(tts_model).strip() if tts_model else None),
        whisper_lang=(str(raw["whisper_lang"]).strip() if raw.get("whisper_lang") else None),
        aliases=aliases,
        detection_words=words,
    )


def _iter_extra_specs() -> Iterable[LanguageSpec]:
    payloads: list[Any] = []
    if _USER_LANGUAGE_REGISTRY_FILE.exists():
        try:
            payloads.append(json.loads(_USER_LANGUAGE_REGISTRY_FILE.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Could not read user language registry %s: %s", _USER_LANGUAGE_REGISTRY_FILE, exc)

    registry_file = os.environ.get("LANGUAGE_REGISTRY_FILE", "").strip()
    if registry_file:
        path = Path(registry_file)
        if not path.is_absolute():
            path = _ROOT / path
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Could not read LANGUAGE_REGISTRY_FILE=%s: %s", path, exc)

    inline_json = os.environ.get("AFRILINGUA_EXTRA_LANGUAGES", "").strip()
    if inline_json:
        try:
            payloads.append(json.loads(inline_json))
        except json.JSONDecodeError as exc:
            logger.warning("AFRILINGUA_EXTRA_LANGUAGES is not valid JSON: %s", exc)

    for payload in payloads:
        entries = payload.values() if isinstance(payload, dict) else payload
        if not isinstance(entries, list) and not isinstance(entries, tuple):
            logger.warning("Language registry payload must be a list or object.")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("Skipping non-object language entry: %r", entry)
                continue
            try:
                yield _coerce_spec(entry)
            except ValueError as exc:
                logger.warning("Skipping language entry: %s", exc)


def _build_registry() -> dict[str, LanguageSpec]:
    registry = {spec.name: spec for spec in _DEFAULT_SPECS}
    for spec in _iter_extra_specs():
        registry[spec.name] = spec
    return registry


LANGUAGE_REGISTRY: dict[str, LanguageSpec] = _build_registry()
LANGUAGE_ALIASES: dict[str, str] = {}

for _name, _spec in LANGUAGE_REGISTRY.items():
    LANGUAGE_ALIASES[_norm(_name)] = _name
    LANGUAGE_ALIASES[_spec.nllb_code.lower()] = _name
    for _alias in _spec.aliases:
        LANGUAGE_ALIASES[_norm(_alias)] = _name


def normalize_language(lang: str) -> str:
    key = _norm(lang)
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    raise ValueError(
        f"Unsupported language: {lang!r}. Choose from: {sorted(LANGUAGE_REGISTRY)}"
    )


def get_language_spec(lang: str) -> LanguageSpec:
    return LANGUAGE_REGISTRY[normalize_language(lang)]


LANG_CODES: dict[str, str] = {
    name: spec.nllb_code for name, spec in LANGUAGE_REGISTRY.items()
}
CODE_TO_LANG: dict[str, str] = {
    spec.nllb_code.lower(): name for name, spec in LANGUAGE_REGISTRY.items()
}
SUPPORTED_LANGUAGES: list[str] = list(LANGUAGE_REGISTRY.keys())
ASR_LANG_CODES: dict[str, str] = {
    name: spec.mms_asr_code
    for name, spec in LANGUAGE_REGISTRY.items()
    if spec.mms_asr_code
}
TTS_MODEL_IDS: dict[str, str] = {
    name: spec.mms_tts_model
    for name, spec in LANGUAGE_REGISTRY.items()
    if spec.mms_tts_model
}
TEXT_LANGUAGE_CHOICES: list[str] = SUPPORTED_LANGUAGES
ASR_LANGUAGE_CHOICES: list[str] = [
    name for name, spec in LANGUAGE_REGISTRY.items() if spec.supports_asr
]


def _refresh_language_views() -> None:
    LANGUAGE_ALIASES.clear()
    for name, spec in LANGUAGE_REGISTRY.items():
        LANGUAGE_ALIASES[_norm(name)] = name
        LANGUAGE_ALIASES[spec.nllb_code.lower()] = name
        for alias in spec.aliases:
            LANGUAGE_ALIASES[_norm(alias)] = name

    LANG_CODES.clear()
    LANG_CODES.update({name: spec.nllb_code for name, spec in LANGUAGE_REGISTRY.items()})
    CODE_TO_LANG.clear()
    CODE_TO_LANG.update({spec.nllb_code.lower(): name for name, spec in LANGUAGE_REGISTRY.items()})
    SUPPORTED_LANGUAGES.clear()
    SUPPORTED_LANGUAGES.extend(LANGUAGE_REGISTRY.keys())
    ASR_LANG_CODES.clear()
    ASR_LANG_CODES.update(
        {
            name: spec.mms_asr_code
            for name, spec in LANGUAGE_REGISTRY.items()
            if spec.mms_asr_code
        }
    )
    TTS_MODEL_IDS.clear()
    TTS_MODEL_IDS.update(
        {
            name: spec.mms_tts_model
            for name, spec in LANGUAGE_REGISTRY.items()
            if spec.mms_tts_model
        }
    )
    ASR_LANGUAGE_CHOICES.clear()
    ASR_LANGUAGE_CHOICES.extend(
        name for name, spec in LANGUAGE_REGISTRY.items() if spec.supports_asr
    )


def _persist_user_language(spec: LanguageSpec) -> None:
    _USER_LANGUAGE_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    if _USER_LANGUAGE_REGISTRY_FILE.exists():
        try:
            raw = json.loads(_USER_LANGUAGE_REGISTRY_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                payload = [item for item in raw if isinstance(item, dict)]
        except Exception:
            payload = []

    spec_dict = asdict(spec)
    spec_dict["aliases"] = list(spec.aliases)
    spec_dict["detection_words"] = list(spec.detection_words)
    payload = [item for item in payload if _norm(str(item.get("name", ""))) != spec.name]
    payload.append(spec_dict)
    _USER_LANGUAGE_REGISTRY_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def register_language(spec: LanguageSpec, *, persist: bool = True) -> LanguageSpec:
    """Register a language at runtime and optionally persist it for future launches."""
    normalized = LanguageSpec(
        name=_norm(spec.name),
        nllb_code=spec.nllb_code.strip(),
        mms_asr_code=spec.mms_asr_code.strip() if spec.mms_asr_code else None,
        mms_tts_model=spec.mms_tts_model.strip() if spec.mms_tts_model else None,
        whisper_lang=spec.whisper_lang.strip() if spec.whisper_lang else None,
        aliases=tuple(_norm(alias) for alias in spec.aliases if alias.strip()),
        detection_words=tuple(word.strip().lower() for word in spec.detection_words if word.strip()),
    )
    if not normalized.name:
        raise ValueError("Language name is required.")
    if not normalized.nllb_code:
        raise ValueError("NLLB language code is required.")

    LANGUAGE_REGISTRY[normalized.name] = normalized
    _refresh_language_views()
    if persist:
        _persist_user_language(normalized)
    return normalized


def register_language_from_values(
    *,
    name: str,
    nllb_code: str,
    mms_asr_code: str = "",
    mms_tts_model: str = "",
    whisper_lang: str = "",
    aliases: str = "",
    detection_words: str = "",
) -> LanguageSpec:
    split_aliases = tuple(item.strip() for item in aliases.split(",") if item.strip())
    split_words = tuple(item.strip() for item in detection_words.split(",") if item.strip())
    return register_language(
        LanguageSpec(
            name=name,
            nllb_code=nllb_code,
            mms_asr_code=mms_asr_code or None,
            mms_tts_model=mms_tts_model or None,
            whisper_lang=whisper_lang or None,
            aliases=split_aliases,
            detection_words=split_words,
        )
    )
