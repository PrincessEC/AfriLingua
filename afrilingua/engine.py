"""
AfriLinguaEngine — the single facade used by the API, MCP server, pipeline and
desktop shell. It hides backend choice, model loading, caching and the
translation-memory fast path.

Translation flow (per sentence)::

    normalize → exact LRU cache → TM exact/approved hit → MT (batched misses)
                                   ↘ fuzzy matches are returned as hints
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from afrilingua.backends.base import LanguageIdentifier, SpeechAudio, Transcript
from afrilingua.backends.text_frontend import split_sentences
from afrilingua.core.errors import CapabilityUnavailableError, InvalidInputError
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings
from afrilingua.models.manager import ModelManager

logger = logging.getLogger(__name__)

TranslationMode = Literal["live", "document"]
ResultSource = Literal["passthrough", "cache", "tm_exact", "mt", "mt_tm_assisted"]

# Model keys registered in the ModelManager (see backends.factory).
MT_KEY = "mt"
ASR_EN_KEY = "asr.en"
ASR_MMS_KEY = "asr.mms"
TTS_KEY = "tts"


# ── Translation memory contract (implemented in afrilingua.tm) ─────────────


@dataclass(frozen=True)
class TMMatch:
    id: str
    source: str
    target: str
    score: float  # 0..1 similarity of `source` to the query
    approved: bool
    origin: str = ""  # "community" | "session" | "import" | "mt"
    kind: str = ""  # "exact" | "wording" (similar text) | "meaning" (similar meaning)


@runtime_checkable
class TranslationMemoryProtocol(Protocol):
    def lookup(self, text: str, src_lang: str, tgt_lang: str, *, k: int = 3) -> list[TMMatch]: ...

    def add(
        self,
        source: str,
        target: str,
        src_lang: str,
        tgt_lang: str,
        *,
        origin: str = "mt",
        approved: bool = False,
    ) -> str: ...


@dataclass
class TranslationResult:
    text: str
    source_text: str
    src_lang: str
    tgt_lang: str
    origin: ResultSource
    latency_ms: float = 0.0
    detected: bool = False
    matches: list[TMMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_text": self.source_text,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "origin": self.origin,
            "latency_ms": round(self.latency_ms, 2),
            "detected": self.detected,
            "matches": [m.__dict__ for m in self.matches],
        }


def normalize_text(text: str) -> str:
    """Unicode NFC (critical for Yoruba/Igbo diacritics) + whitespace cleanup."""
    return " ".join(unicodedata.normalize("NFC", text).split())


#: Sentence boundary followed by more text: worth splitting for batching and memory hits.
_SENTENCE_HINT = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"'\u201d\u2019)\]]*\s+\S")
#: Segments longer than this are not cached (bounds cache memory per entry).
_MAX_CACHED_CHARS = 2000


class _LRU:
    """Recent-results cache. Values are ``(text, kind)`` where kind is ``"tm"`` or ``"mt"``."""

    def __init__(self, capacity: int = 4096) -> None:
        self._data: OrderedDict[tuple[str, str, str], tuple[str, str]] = OrderedDict()
        self._cap = capacity
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, str]) -> tuple[str, str] | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: tuple[str, str, str], value: str, kind: str = "mt") -> None:
        if len(key[2]) > _MAX_CACHED_CHARS:
            return
        with self._lock:
            self._data[key] = (value, kind)
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _combine_origins(origins: set[str]) -> str:
    """Origin of a multi-sentence result from the origins of its sentences."""
    if len(origins) == 1:
        return next(iter(origins))
    if origins <= {"tm_exact", "cache"}:
        return "tm_exact"
    if "tm_exact" in origins or "mt_tm_assisted" in origins:
        return "mt_tm_assisted"
    return "mt"


class AfriLinguaEngine:
    def __init__(
        self,
        manager: ModelManager | None = None,
        *,
        settings: Settings | None = None,
        registry: LanguageRegistry | None = None,
        tm: TranslationMemoryProtocol | None = None,
        detector: LanguageIdentifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.manager = manager or ModelManager()
        self.tm = tm
        self.detector = detector
        self._cache = _LRU()
        self._tm_generation: int | None = None

    # ── language helpers ────────────────────────────────────────────────────
    def detect_language(self, text: str) -> tuple[str, float]:
        if self.detector is None:
            return "unknown", 0.0
        return self.detector.detect(text)

    def _resolve_src(self, texts: Sequence[str], src_lang: str | None) -> tuple[str, bool]:
        if src_lang and src_lang.lower() not in {"auto", "auto-detect"}:
            return self.registry.normalize(src_lang), False
        sample = " ".join(texts)[:1000]
        lang, _conf = self.detect_language(sample)
        return (lang if lang != "unknown" else "english"), True

    # ── translation ─────────────────────────────────────────────────────────
    def _segments(self, text: str) -> list[str]:
        """Split ``text`` into sentences so each can hit memory and the rest decode as one batch."""
        if not self.settings.mt_segment_sentences or not _SENTENCE_HINT.search(text):
            return [text]
        segs = [seg for seg in split_sentences(text, max_chars=400) if seg.strip()]
        return segs or [text]

    def _sync_cache_with_tm(self) -> None:
        """Drop cached results whenever the translation memory has changed."""
        generation = getattr(self.tm, "generation", None)
        if generation != self._tm_generation:
            self._cache.clear()
            self._tm_generation = generation

    def translate(
        self,
        texts: Sequence[str],
        src_lang: str | None,
        tgt_lang: str,
        *,
        mode: TranslationMode = "live",
        use_tm: bool = True,
    ) -> list[TranslationResult]:
        if isinstance(texts, str):
            raise InvalidInputError(
                "translate() expects a sequence of strings; use translate_one() for a single string"
            )
        limit = self.settings.max_text_chars
        for t in texts:
            if limit and len(t) > limit:
                raise InvalidInputError(
                    f"Text is {len(t)} characters; the limit is {limit}. "
                    "Split it, or use document translation for long files."
                )
        t0 = time.perf_counter()
        tgt = self.registry.normalize(tgt_lang)
        src, detected = self._resolve_src(texts, src_lang)
        norm = [normalize_text(t) for t in texts]
        self._sync_cache_with_tm()

        results: list[TranslationResult | None] = [None] * len(norm)
        plan: list[tuple[int, list[str]]] = []
        done: dict[str, tuple[str, str]] = {}  # sentence -> (translation, origin)
        pending: list[str] = []
        fuzzy_hint: dict[str, bool] = {}
        text_matches: dict[int, list[TMMatch]] = {}

        for i, text in enumerate(norm):
            if not text or src == tgt:
                results[i] = TranslationResult(text, text, src, tgt, "passthrough", detected=detected)
                continue
            segments = self._segments(text)
            plan.append((i, segments))
            for seg in segments:
                if seg in done or seg in fuzzy_hint:
                    continue
                cached = self._cache.get((src, tgt, seg))
                if cached is not None and (use_tm or cached[1] != "tm"):
                    done[seg] = (cached[0], "cache")
                    continue
                matches: list[TMMatch] = []
                if use_tm and self.tm is not None:
                    try:
                        matches = self.tm.lookup(seg, src, tgt, k=3)
                    except Exception:
                        logger.exception("Translation memory lookup failed")
                if len(segments) == 1:
                    text_matches[i] = matches
                best = matches[0] if matches else None
                # Only an exact (normalised) match is served verbatim. A *similar* sentence's
                # translation is not a translation of this one ("two" vs "ten" tablets).
                if best and best.approved and best.score >= self.settings.tm_exact_threshold:
                    done[seg] = (best.target, "tm_exact")
                    self._cache.put((src, tgt, seg), best.target, "tm")
                    continue
                fuzzy_hint[seg] = bool(best and best.score >= self.settings.tm_fuzzy_threshold)
                pending.append(seg)

        if pending:
            beam = self.settings.live_beam_size if mode == "live" else self.settings.document_beam_size
            with self.manager.use(MT_KEY) as mt:
                outputs = mt.translate_batch(
                    pending, src, tgt, beam_size=beam, max_length=self.settings.max_translation_tokens
                )
            for seg, out in zip(pending, outputs, strict=True):
                done[seg] = (out, "mt_tm_assisted" if fuzzy_hint.get(seg) else "mt")
                self._cache.put((src, tgt, seg), out, "mt")

        for i, segments in plan:
            parts = [done[seg] for seg in segments]
            translated = " ".join(p[0] for p in parts if p[0])
            origin = _combine_origins({p[1] for p in parts})
            results[i] = TranslationResult(
                translated, norm[i], src, tgt, origin, detected=detected, matches=text_matches.get(i, [])
            )

        elapsed = (time.perf_counter() - t0) * 1000
        final = [r for r in results if r is not None]
        for r in final:
            r.latency_ms = elapsed / max(1, len(final))
        return final

    def translate_one(
        self, text: str, src_lang: str | None, tgt_lang: str, *, mode: TranslationMode = "live"
    ) -> TranslationResult:
        return self.translate([text], src_lang, tgt_lang, mode=mode)[0]

    # ── speech ──────────────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        spec = self.registry.get(lang)
        if spec.whisper_lang:
            key = ASR_EN_KEY
        elif spec.mms_asr_code:
            key = ASR_MMS_KEY
        else:
            raise CapabilityUnavailableError(f"No speech recognition configured for {spec.name!r}.")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return Transcript(text="", language=spec.name)
        with self.manager.use(key) as asr:
            return asr.transcribe(audio, spec.name)

    def synthesize(self, text: str, lang: str) -> SpeechAudio:
        spec = self.registry.get(lang)
        if not spec.supports_tts:
            raise CapabilityUnavailableError(f"No speech synthesis configured for {spec.name!r}.")
        text = normalize_text(text)
        if not text:
            return SpeechAudio(np.zeros(0, dtype=np.float32), 16_000)
        with self.manager.use(TTS_KEY) as tts:
            return tts.synthesize(text, spec.name)

    # ── ops ─────────────────────────────────────────────────────────────────
    def clear_cache(self) -> None:
        self._cache.clear()

    def status(self) -> dict:
        from afrilingua.core.hardware import detect_hardware, resolve_runtime

        return {
            "runtime": resolve_runtime().to_dict(),
            "hardware": detect_hardware().to_dict(),
            "models": self.manager.status(),
            "languages": [s.to_public_dict() for s in self.registry],
            "translation_memory": self.tm is not None,
        }


def build_engine(*, with_tm: bool = True, with_detector: bool = True) -> AfriLinguaEngine:
    """Construct the default, fully wired engine (backends registered lazily)."""
    from afrilingua.backends.factory import build_detector, register_default_backends

    manager = ModelManager()
    register_default_backends(manager)
    tm = None
    if with_tm:
        try:
            from afrilingua.tm import open_translation_memory

            tm = open_translation_memory()
        except Exception:
            logger.exception("Translation memory unavailable; continuing without it")
    detector = build_detector() if with_detector else None
    return AfriLinguaEngine(manager, tm=tm, detector=detector)
