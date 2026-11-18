"""
Language registry.

Built-in: English, Igbo, Yoruba, Hausa. Extra languages are added without code
changes via ``LANGUAGE_REGISTRY_FILE`` (JSON file), ``AFRILINGUA_EXTRA_LANGUAGES``
(inline JSON) or at runtime through :meth:`LanguageRegistry.register`
(persisted to ``data/languages.json``).

Entry format::

    {"name": "swahili", "nllb_code": "swh_Latn", "mms_asr_code": "swh",
     "mms_tts_model": "facebook/mms-tts-swh", "aliases": ["kiswahili", "sw"],
     "display_name": "Swahili", "native_name": "Kiswahili"}
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from afrilingua.core.errors import UnsupportedLanguageError
from afrilingua.core.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class LanguageSpec:
    name: str                                   # canonical key, e.g. "yoruba"
    nllb_code: str                              # e.g. "yor_Latn"
    display_name: str = ""
    native_name: str = ""
    iso639_1: str | None = None                 # "yo"
    iso639_3: str | None = None                 # "yor"
    mms_asr_code: str | None = None
    mms_tts_model: str | None = None
    whisper_lang: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    detection_words: tuple[str, ...] = field(default_factory=tuple)
    builtin: bool = False

    @property
    def supports_asr(self) -> bool:
        return bool(self.whisper_lang or self.mms_asr_code)

    @property
    def supports_tts(self) -> bool:
        return bool(self.mms_tts_model) or self.name == "english"

    def to_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        d["detection_words"] = list(self.detection_words)
        d["supports_asr"] = self.supports_asr
        d["supports_tts"] = self.supports_tts
        d["display_name"] = self.display_name or self.name.title()
        return d


_BUILTINS: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        name="english", nllb_code="eng_Latn", display_name="English", native_name="English",
        iso639_1="en", iso639_3="eng", whisper_lang="en",
        mms_tts_model="facebook/mms-tts-eng", aliases=("en", "eng"), builtin=True,
    ),
    LanguageSpec(
        name="igbo", nllb_code="ibo_Latn", display_name="Igbo", native_name="Asụsụ Igbo",
        iso639_1="ig", iso639_3="ibo", mms_asr_code="ibo",
        mms_tts_model="rnjema-unima/mms-tts-ibo-baseline", aliases=("ig", "ibo"),
        detection_words=("kedu", "nne", "nna", "anyị", "anyi", "daalụ", "daalu", "biko", "ụlọ", "nnọọ", "ndewo", "gị", "ọ", "dị"),
        builtin=True,
    ),
    LanguageSpec(
        name="yoruba", nllb_code="yor_Latn", display_name="Yoruba", native_name="Èdè Yorùbá",
        iso639_1="yo", iso639_3="yor", mms_asr_code="yor",
        mms_tts_model="facebook/mms-tts-yor", aliases=("yo", "yor"),
        detection_words=("ẹ", "ṣé", "kú", "àárọ̀", "ọ̀rẹ́", "mo", "ni", "àti", "jọ̀wọ́", "ẹ̀kọ́"),
        builtin=True,
    ),
    LanguageSpec(
        name="hausa", nllb_code="hau_Latn", display_name="Hausa", native_name="Harshen Hausa",
        iso639_1="ha", iso639_3="hau", mms_asr_code="hau",
        mms_tts_model="facebook/mms-tts-hau", aliases=("ha", "hau"),
        detection_words=("sannu", "nagode", "yaya", "gida", "wannan", "kuma", "ina", "kai", "ne", "ce", "da", "zuwa"),
        builtin=True,
    ),
)


def _coerce(raw: dict[str, Any]) -> LanguageSpec:
    name = _norm(str(raw.get("name", "")))
    if not name:
        raise ValueError("language entry is missing 'name'")
    nllb = raw.get("nllb_code") or raw.get("nllb") or raw.get("code")
    if not nllb:
        raise ValueError(f"language {name!r} is missing 'nllb_code'")

    def _opt(key: str) -> str | None:
        v = raw.get(key)
        return str(v).strip() if v else None

    return LanguageSpec(
        name=name,
        nllb_code=str(nllb).strip(),
        display_name=_opt("display_name") or name.title(),
        native_name=_opt("native_name") or "",
        iso639_1=_opt("iso639_1"),
        iso639_3=_opt("iso639_3"),
        mms_asr_code=_opt("mms_asr_code"),
        mms_tts_model=_opt("mms_tts_model") or _opt("tts_model"),
        whisper_lang=_opt("whisper_lang"),
        aliases=tuple(_norm(str(a)) for a in raw.get("aliases", []) if str(a).strip()),
        detection_words=tuple(str(w).strip().lower() for w in raw.get("detection_words", []) if str(w).strip()),
    )


class LanguageRegistry:
    """Thread-safe registry of supported languages."""

    def __init__(self, specs: Iterable[LanguageSpec], user_file: Path) -> None:
        self._lock = threading.RLock()
        self._specs: dict[str, LanguageSpec] = {}
        self._aliases: dict[str, str] = {}
        self._user_file = user_file
        for spec in specs:
            self._put(spec)

    # ── internal ────────────────────────────────────────────────────────────
    def _put(self, spec: LanguageSpec) -> None:
        self._specs[spec.name] = spec
        self._aliases[_norm(spec.name)] = spec.name
        self._aliases[_norm(spec.nllb_code)] = spec.name
        for alias in spec.aliases:
            self._aliases[_norm(alias)] = spec.name
        for code in (spec.iso639_1, spec.iso639_3):
            if code:
                self._aliases[_norm(code)] = spec.name

    # ── queries ─────────────────────────────────────────────────────────────
    def __iter__(self) -> Iterator[LanguageSpec]:
        with self._lock:
            return iter(list(self._specs.values()))

    def __contains__(self, lang: object) -> bool:
        return isinstance(lang, str) and _norm(lang) in self._aliases

    def names(self) -> list[str]:
        with self._lock:
            return list(self._specs)

    def normalize(self, lang: str) -> str:
        key = _norm(lang or "")
        with self._lock:
            if key in self._aliases:
                return self._aliases[key]
        raise UnsupportedLanguageError(lang, sorted(self._specs))

    def get(self, lang: str) -> LanguageSpec:
        return self._specs[self.normalize(lang)]

    def nllb_code(self, lang: str) -> str:
        return self.get(lang).nllb_code

    def asr_languages(self) -> list[str]:
        return [s.name for s in self if s.supports_asr]

    def tts_languages(self) -> list[str]:
        return [s.name for s in self if s.supports_tts]

    # ── mutation ────────────────────────────────────────────────────────────
    def register(self, spec: LanguageSpec | dict[str, Any], *, persist: bool = True) -> LanguageSpec:
        spec = _coerce(spec) if isinstance(spec, dict) else replace(spec, name=_norm(spec.name))
        with self._lock:
            self._put(spec)
            if persist:
                self._persist(spec)
        return spec

    def _persist(self, spec: LanguageSpec) -> None:
        self._user_file.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        if self._user_file.exists():
            try:
                raw = json.loads(self._user_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    payload = [i for i in raw if isinstance(i, dict)]
            except Exception:
                logger.warning("Could not parse %s; rewriting it.", self._user_file)
        entry = spec.to_public_dict()
        for k in ("supports_asr", "supports_tts", "builtin"):
            entry.pop(k, None)
        payload = [i for i in payload if _norm(str(i.get("name", ""))) != spec.name] + [entry]
        self._user_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_payloads() -> Iterator[dict[str, Any]]:
    s = get_settings()
    sources: list[tuple[str, str]] = []
    user_file = s.data_dir / "languages.json"
    if user_file.exists():
        sources.append((str(user_file), user_file.read_text(encoding="utf-8")))
    if s.language_registry_file.strip():
        p = Path(s.language_registry_file.strip())
        p = p if p.is_absolute() else PROJECT_ROOT / p
        try:
            sources.append((str(p), p.read_text(encoding="utf-8")))
        except OSError as exc:
            logger.warning("Could not read LANGUAGE_REGISTRY_FILE=%s: %s", p, exc)
    if s.afrilingua_extra_languages.strip():
        sources.append(("AFRILINGUA_EXTRA_LANGUAGES", s.afrilingua_extra_languages))

    for origin, text in sources:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid language JSON in %s: %s", origin, exc)
            continue
        entries = list(payload.values()) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            logger.warning("Language registry payload in %s must be a list or object.", origin)
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry


@lru_cache(maxsize=1)
def get_registry() -> LanguageRegistry:
    s = get_settings()
    reg = LanguageRegistry(_BUILTINS, user_file=s.data_dir / "languages.json")
    for raw in _load_payloads():
        try:
            reg.register(raw, persist=False)
        except ValueError as exc:
            logger.warning("Skipping language entry: %s", exc)
    return reg
