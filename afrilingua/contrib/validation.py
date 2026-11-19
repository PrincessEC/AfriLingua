"""Validation and PII scrubbing for community contributions (contract §5)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from afrilingua.core.errors import InvalidInputError
from afrilingua.core.languages import get_registry
from afrilingua.engine import normalize_text

__all__ = [
    "ALLOWED_LICENSES",
    "DOMAINS",
    "KINDS",
    "TM_KINDS",
    "ValidatedContribution",
    "anonymous_id",
    "contributor_key",
    "find_pii",
    "scrub_pii",
    "validate_contribution",
]

KINDS = ("translation", "correction", "glossary", "transcription")
#: Kinds whose approved pairs are written into the translation memory.
TM_KINDS = frozenset({"translation", "correction", "glossary"})
ALLOWED_LICENSES = ("CC-BY-4.0", "CC0-1.0")
DOMAINS = ("general", "medical", "legal", "religious", "education", "government", "media")

MAX_TEXT = {"translation": 1000, "correction": 1000, "glossary": 200, "transcription": 1000}
MAX_ORIGINAL_MT = 2000
MAX_META = 100
ANONYMOUS = "anonymous"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|org|net|ng|io|gov|edu)(?:/\S*)?\b", re.IGNORECASE)
# Phone numbers: 7+ digits, optionally with +country code, spaces, dashes, dots or parentheses.
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_SLUG = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def find_pii(text: str) -> list[str]:
    """Return the kinds of PII found in ``text`` (``email``, ``url``, ``phone``)."""
    found = []
    if _EMAIL.search(text):
        found.append("email")
    if _URL.search(_EMAIL.sub(" ", text)):
        found.append("url")
    if any(sum(ch.isdigit() for ch in m.group()) >= 7 for m in _PHONE.finditer(text)):
        found.append("phone")
    return found


def scrub_pii(text: str) -> str:
    """Replace e-mail addresses, URLs and phone numbers with placeholders."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)

    def _phone(m: re.Match[str]) -> str:
        return "[phone]" if sum(ch.isdigit() for ch in m.group()) >= 7 else m.group()

    return _PHONE.sub(_phone, text)


def anonymous_id(value: str) -> str:
    """Stable pseudonymous id (``anon-xxxxxxxx``) for a contributor identifier."""
    return "anon-" + hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:10]


def contributor_key(name: str | None) -> str:
    """Canonical contributor id used for self-review checks and the leaderboard."""
    name = normalize_text(name or "")
    return name.casefold() if name else ANONYMOUS


def _clean_meta(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInputError(f"{field_name} must be a string")
    value = normalize_text(value)
    if not value:
        return None
    if len(value) > MAX_META:
        raise InvalidInputError(f"{field_name} must be at most {MAX_META} characters")
    return value


def _clean_contributor(value: Any) -> str:
    name = _clean_meta(value, "contributor")
    if not name:
        return ANONYMOUS
    # Never store an e-mail address or phone number as a public display name.
    if find_pii(name):
        return anonymous_id(name)
    return name


def _norm_lang(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{field_name} is required")
    return get_registry().normalize(value)  # raises UnsupportedLanguageError (400)


def _clean_text(value: Any, field_name: str, limit: int, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise InvalidInputError(f"{field_name} must be a string")
    text = unicodedata.normalize("NFC", normalize_text(value))
    if required and not text:
        raise InvalidInputError(f"{field_name} must not be empty")
    if len(text) > limit:
        raise InvalidInputError(f"{field_name} must be at most {limit} characters")
    return text


@dataclass
class ValidatedContribution:
    """A contribution that passed validation (texts NFC-normalised, PII scrubbed)."""

    kind: str
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    license: str
    consent: bool
    contributor: str
    original_mt: str | None = None
    domain: str = "general"
    dialect: str | None = None
    region: str | None = None
    accent: str | None = None
    gender: str | None = None
    age_range: str | None = None
    pii_found: list[str] = field(default_factory=list)


def validate_contribution(payload: dict[str, Any]) -> ValidatedContribution:
    """Validate a ``ContributionIn`` payload; raises :class:`InvalidInputError`."""
    if not isinstance(payload, dict):
        raise InvalidInputError("contribution payload must be an object")
    if payload.get("consent") is not True:
        raise InvalidInputError("consent must be true: contributions are only accepted with explicit consent")
    license_ = payload.get("license") or "CC-BY-4.0"
    if license_ not in ALLOWED_LICENSES:
        raise InvalidInputError(f"license must be one of {', '.join(ALLOWED_LICENSES)}")
    kind = payload.get("kind") or "translation"
    if kind not in KINDS:
        raise InvalidInputError(f"kind must be one of {', '.join(KINDS)}")
    src = _norm_lang(payload.get("src_lang"), "src_lang")
    tgt = _norm_lang(payload.get("tgt_lang") or (src if kind == "transcription" else None), "tgt_lang")
    if src == tgt and kind != "transcription":
        raise InvalidInputError("src_lang and tgt_lang must differ")
    limit = MAX_TEXT[kind]
    source = _clean_text(payload.get("source_text"), "source_text", limit)
    target = _clean_text(payload.get("target_text"), "target_text", limit, required=kind != "transcription")
    if kind == "transcription" and not target:
        target = source
    original_mt = (
        _clean_text(payload.get("original_mt"), "original_mt", MAX_ORIGINAL_MT, required=False) or None
    )
    domain = (_clean_meta(payload.get("domain"), "domain") or "general").lower()
    if not _SLUG.match(domain):
        raise InvalidInputError(f"domain must be a short slug such as {', '.join(DOMAINS)}")

    pii = sorted(set(find_pii(source) + find_pii(target)))
    return ValidatedContribution(
        kind=kind,
        src_lang=src,
        tgt_lang=tgt,
        source_text=scrub_pii(source),
        target_text=scrub_pii(target),
        license=license_,
        consent=True,
        contributor=_clean_contributor(payload.get("contributor")),
        original_mt=scrub_pii(original_mt) if original_mt else None,
        domain=domain,
        dialect=_clean_meta(payload.get("dialect"), "dialect"),
        region=_clean_meta(payload.get("region"), "region"),
        accent=_clean_meta(payload.get("accent"), "accent"),
        gender=_clean_meta(payload.get("gender"), "gender"),
        age_range=_clean_meta(payload.get("age_range"), "age_range"),
        pii_found=pii,
    )
