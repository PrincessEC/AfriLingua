"""
AfriLingua MCP server (official ``mcp`` SDK 2.x, ``MCPServer``).

Lets any MCP-capable model or agent (Claude Desktop, Cursor, VS Code, Gemini CLI …)
translate text, transcribe and synthesize speech, search the translation memory,
contribute translations and open live interpretation rooms.

Transports
----------
* stdio — ``afrilingua mcp`` (engine built lazily on first tool call).
* Streamable HTTP — mounted at ``/mcp`` on the main FastAPI server
  (see :func:`afrilingua.api.app.create_app`), sharing its engine and rooms.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import dataclasses
import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from mcp.server.mcpserver import Audio, Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import TextContent, ToolAnnotations

from afrilingua import __version__
from afrilingua.core.errors import AfriLinguaError, CapabilityUnavailableError, InvalidInputError
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EngineProvider = Callable[[], Any]
Provider = Callable[[], Any]
F = TypeVar("F", bound=Callable[..., Any])

License = Literal["CC-BY-4.0", "CC0-1.0"]
Domain = Literal["general", "medical", "legal", "religious", "education", "government", "media"]

INSTRUCTIONS = """\
AfriLingua is an offline-first speech and translation engine for African languages
(English ↔ Igbo, Yoruba, Hausa and any languages added to its registry).

Use it to:
- translate text (`translate_text`, `translate_batch`) — languages are names such as
  "english", "yoruba", "igbo", "hausa"; ISO codes (yo, ig, ha, yor) are accepted too;
- detect a text's language (`detect_language`);
- transcribe speech (`transcribe_audio`), speak text (`synthesize_speech`) or do both with
  translation in one step (`translate_speech`);
- look up vetted human translations (`search_translation_memory`) before translating
  domain-specific terms;
- give back to the community: when a user provides or approves a better translation, call
  `submit_translation` / `submit_correction` — ONLY with the user's explicit consent and a license;
- open a live interpretation room for a conference (`create_live_room`).
Call `list_languages` first if unsure which languages are available.
"""

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
COMPUTE = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


# ── helpers ────────────────────────────────────────────────────────────────
def _tool_errors(fn: F) -> F:
    """Turn anticipated AfriLingua errors into ToolErrors the model can read."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except AfriLinguaError as exc:
            raise ToolError(f"[{exc.code}] {exc}") from exc
        except (ValueError, KeyError, FileNotFoundError, PermissionError) as exc:
            raise ToolError(f"[invalid_input] {exc}") from exc

    return wrapper  # type: ignore[return-value]


def to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return to_plain(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_plain(dataclasses.asdict(obj))
    return str(obj)


def _transcript_dict(t: Any) -> dict[str, Any]:
    return {
        "text": t.text,
        "language": t.language,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in t.segments],
        "confidence": t.confidence,
        "duration_s": round(float(t.duration_s), 3),
    }


class _Runtime:
    """Resolves the engine / TM / contribution store / rooms for tool calls."""

    def __init__(
        self,
        engine_provider: EngineProvider,
        *,
        tm_provider: Provider | None,
        contrib_provider: Provider | None,
        rooms_provider: Provider | None,
        settings: Settings,
        allow_file_paths: bool,
    ) -> None:
        self._engine_provider = engine_provider
        self._tm_provider = tm_provider
        self._contrib_provider = contrib_provider
        self._rooms_provider = rooms_provider
        self.settings = settings
        self.allow_file_paths = allow_file_paths
        self._lock = threading.Lock()
        self._contrib: Any = None

    @property
    def engine(self) -> Any:
        engine = self._engine_provider()
        if engine is None:
            raise AfriLinguaError("The AfriLingua engine is still starting; retry shortly.")
        return engine

    @property
    def tm(self) -> Any:
        if self._tm_provider is not None:
            tm = self._tm_provider()
            if tm is not None:
                return tm
        return getattr(self.engine, "tm", None)

    @property
    def contrib(self) -> Any:
        if self._contrib_provider is not None:
            store = self._contrib_provider()
            if store is not None:
                return store
        with self._lock:
            if self._contrib is None:
                try:
                    from afrilingua.contrib import open_contribution_store
                except ImportError as exc:
                    raise CapabilityUnavailableError(
                        f"Community contributions are not available: {exc}"
                    ) from exc
                self._contrib = open_contribution_store(self.tm)
            return self._contrib

    @property
    def rooms(self) -> Any:
        return self._rooms_provider() if self._rooms_provider is not None else None

    def load_audio(self, audio_base64: str | None, file_path: str | None) -> Any:
        from afrilingua.api.audio_io import decode_audio

        if bool(audio_base64) == bool(file_path):
            raise InvalidInputError("Provide exactly one of `audio_base64` or `file_path`.")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if file_path:
            if not self.allow_file_paths:
                raise InvalidInputError(
                    "`file_path` is only allowed over the local stdio transport; send `audio_base64`."
                )
            path = Path(file_path).expanduser()
            if not path.is_file():
                raise InvalidInputError(f"Audio file not found: {file_path}")
            if path.stat().st_size > max_bytes:
                raise InvalidInputError(f"Audio file exceeds {self.settings.max_upload_mb} MB.")
            return decode_audio(path.read_bytes(), filename=path.name)
        payload = str(audio_base64)
        if payload.startswith("data:") and "," in payload:  # data URL
            payload = payload.split(",", 1)[1]
        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise InvalidInputError(f"`audio_base64` is not valid base64: {exc}") from exc
        if len(raw) > max_bytes:
            raise InvalidInputError(f"Audio exceeds {self.settings.max_upload_mb} MB.")
        return decode_audio(raw)


def _submit(store: Any, payload: dict[str, Any]) -> dict[str, Any]:
    for name in ("submit", "create", "add"):
        fn = getattr(store, name, None)
        if callable(fn):
            return to_plain(fn(payload))
    raise CapabilityUnavailableError("The contribution store does not support submissions.")


def _require_consent(consent: bool, license: str) -> None:
    if consent is not True:
        raise InvalidInputError(
            "Contributions require explicit consent: ask the user whether they agree to share this "
            "translation publicly under the chosen license, then call again with consent=true."
        )
    if license not in {"CC-BY-4.0", "CC0-1.0"}:
        raise InvalidInputError("license must be 'CC-BY-4.0' or 'CC0-1.0'.")


def _request_base_url(ctx: Context | None, settings: Settings) -> str:
    request = None
    with contextlib.suppress(Exception):
        request = ctx.request_context.request if ctx is not None else None
    if request is not None and hasattr(request, "base_url"):
        from afrilingua.api.deps import public_base_url

        with contextlib.suppress(Exception):
            return public_base_url(request)
    host = settings.api_host if settings.api_host not in {"0.0.0.0", "::"} else "127.0.0.1"
    return f"http://{host}:{settings.api_port}"


# ── server ─────────────────────────────────────────────────────────────────
def build_mcp(
    engine_provider: EngineProvider,
    *,
    tm_provider: Provider | None = None,
    contrib_provider: Provider | None = None,
    rooms_provider: Provider | None = None,
    settings: Settings | None = None,
    allow_file_paths: bool = False,
) -> MCPServer:
    """Build the AfriLingua MCP server.

    ``engine_provider`` returns the shared :class:`~afrilingua.engine.AfriLinguaEngine`
    (called per tool invocation, so it can be lazy). ``rooms_provider`` returns the
    HTTP server's RoomManager; without it ``create_live_room`` explains how to start one.
    ``allow_file_paths`` lets audio tools read local files (enable for stdio only).
    """
    rt = _Runtime(
        engine_provider,
        tm_provider=tm_provider,
        contrib_provider=contrib_provider,
        rooms_provider=rooms_provider,
        settings=settings or get_settings(),
        allow_file_paths=allow_file_paths,
    )
    # MCPServer() calls logging.basicConfig(); keep the host application's logging untouched
    # (and stdout clean for stdio) by making basicConfig a no-op during construction.
    root = logging.getLogger()
    guard = logging.NullHandler() if not root.handlers else None
    if guard is not None:
        root.addHandler(guard)
    try:
        server: MCPServer = MCPServer(
            "afrilingua",
            title="AfriLingua",
            description="Offline-first speech & translation for African languages (Igbo, Yoruba, Hausa, …).",
            instructions=INSTRUCTIONS,
            version=__version__,
            website_url="https://github.com/afrilingua",
        )
    finally:
        if guard is not None:
            root.removeHandler(guard)

    # ── translation ────────────────────────────────────────────────────────
    @server.tool(annotations=COMPUTE)
    @_tool_errors
    def translate_text(
        text: str,
        target_language: str,
        source_language: str = "auto",
        mode: Literal["live", "document"] = "live",
        use_translation_memory: bool = True,
    ) -> dict[str, Any]:
        """Translate a piece of text between English and African languages.

        Args:
            text: The text to translate (sentence or paragraph).
            target_language: Language to translate into, e.g. "yoruba", "igbo", "hausa", "english"
                (ISO codes like "yo", "ig", "ha" also work).
            source_language: Language of `text`, or "auto" to detect it.
            mode: "live" for fastest greedy decoding, "document" for higher-quality beam search.
            use_translation_memory: Prefer vetted community translations when an exact match exists.

        Returns:
            {text, source_text, src_lang, tgt_lang, origin, latency_ms, detected, matches}; `origin`
            tells whether the result came from the translation memory ("tm_exact"), the cache or the
            MT model ("mt" / "mt_tm_assisted").
        """
        engine = rt.engine
        return engine.translate(
            [text], source_language, target_language, mode=mode, use_tm=use_translation_memory
        )[0].to_dict()

    @server.tool(annotations=COMPUTE)
    @_tool_errors
    def translate_batch(
        texts: list[str],
        target_language: str,
        source_language: str = "auto",
        mode: Literal["live", "document"] = "document",
    ) -> dict[str, Any]:
        """Translate many texts in one efficient batched call (keeps order).

        Args:
            texts: Up to 512 strings (sentences, subtitles, UI strings …).
            target_language: Target language name or ISO code.
            source_language: Source language or "auto" (detected once for the whole batch).
            mode: "document" (beam search, default) or "live" (fastest).

        Returns:
            {"results": [TranslationResult, …]} in the same order as `texts`.
        """
        if len(texts) > 512:
            raise InvalidInputError("translate_batch accepts at most 512 texts per call.")
        results = rt.engine.translate(list(texts), source_language, target_language, mode=mode)
        return {"results": [r.to_dict() for r in results]}

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def detect_language(text: str) -> dict[str, Any]:
        """Detect which supported language a text is written in.

        Args:
            text: Any text (a sentence or more gives the best accuracy).

        Returns:
            {"language": "<canonical name or 'unknown'>", "confidence": 0..1}
        """
        language, confidence = rt.engine.detect_language(text)
        return {"language": language, "confidence": round(float(confidence), 4)}

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def list_languages() -> dict[str, Any]:
        """List supported languages and their capabilities (translation, speech recognition, speech synthesis).

        Returns:
            {"languages": [{name, display_name, native_name, iso639_1, iso639_3, aliases, supports_asr,
            supports_tts}, …]}
        """
        keys = (
            "name",
            "display_name",
            "native_name",
            "iso639_1",
            "iso639_3",
            "aliases",
            "supports_asr",
            "supports_tts",
        )
        langs = [spec.to_public_dict() for spec in rt.engine.registry]
        return {"languages": [{k: d.get(k) for k in keys} for d in langs]}

    # ── speech ─────────────────────────────────────────────────────────────
    @server.tool(annotations=COMPUTE)
    @_tool_errors
    def transcribe_audio(
        language: str = "english",
        audio_base64: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe speech to text (speech recognition).

        Provide the audio as `audio_base64` (base64 of a wav/mp3/ogg/flac/webm file, data: URLs are
        fine) or, over the local stdio transport only, a `file_path`.

        Args:
            language: Spoken language ("english", "yoruba", "igbo", "hausa" …).
            audio_base64: Base64-encoded audio file.
            file_path: Absolute path of a local audio file (stdio transport only).

        Returns:
            {text, language, segments: [{start, end, text}], confidence, duration_s}
        """
        engine = rt.engine
        lang = engine.registry.normalize(language)
        audio = rt.load_audio(audio_base64, file_path)
        return _transcript_dict(engine.transcribe(audio, lang))

    @server.tool(annotations=COMPUTE)
    @_tool_errors
    def synthesize_speech(text: str, language: str) -> list[TextContent | Audio]:
        """Speak text aloud: returns a WAV audio clip (MCP audio content, base64 WAV).

        Args:
            text: Text to speak (max ~5000 characters).
            language: Language of the text ("yoruba", "igbo", "hausa", "english" …).
        """
        from afrilingua.api.audio_io import wav_bytes

        engine = rt.engine
        lang = engine.registry.normalize(language)
        if len(text) > 5000:
            raise InvalidInputError("Text is too long for speech synthesis (max 5000 characters).")
        speech = engine.synthesize(text, lang)
        wav = wav_bytes(speech.waveform, speech.sample_rate)
        meta = {
            "language": lang,
            "sample_rate": speech.sample_rate,
            "duration_s": round(speech.duration_s, 3),
        }
        return [TextContent(type="text", text=json.dumps(meta)), Audio(data=wav, format="wav")]

    @server.tool(annotations=COMPUTE)
    @_tool_errors
    def translate_speech(
        target_language: str,
        source_language: str = "english",
        audio_base64: str | None = None,
        file_path: str | None = None,
        include_audio: bool = False,
    ) -> list[TextContent | Audio]:
        """Speech-to-speech interpretation in one call: transcribe → translate → (optionally) speak.

        Args:
            target_language: Language to translate into.
            source_language: Spoken language of the input audio.
            audio_base64: Base64-encoded audio file (wav/mp3/ogg/flac/webm).
            file_path: Local audio file path (stdio transport only).
            include_audio: Also return the translation as spoken WAV audio.

        Returns:
            A JSON text block {transcript, translation, src_lang, tgt_lang, origin, timings_ms} and, when
            `include_audio` is true, an audio block with the spoken translation.
        """
        from afrilingua.api.audio_io import wav_bytes

        engine = rt.engine
        src = engine.registry.normalize(source_language)
        tgt = engine.registry.normalize(target_language)
        audio = rt.load_audio(audio_base64, file_path)
        t0 = time.perf_counter()
        transcript = engine.transcribe(audio, src)
        t1 = time.perf_counter()
        result = engine.translate([transcript.text], src, tgt, mode="live")[0]
        t2 = time.perf_counter()
        payload: dict[str, Any] = {
            "transcript": transcript.text,
            "translation": result.text,
            "src_lang": src,
            "tgt_lang": tgt,
            "origin": result.origin,
            "duration_s": round(float(transcript.duration_s), 3),
            "timings_ms": {"asr": round((t1 - t0) * 1000), "mt": round((t2 - t1) * 1000)},
        }
        content: list[TextContent | Audio] = []
        audio_block: Audio | None = None
        if include_audio and result.text:
            speech = engine.synthesize(result.text, tgt)
            payload["timings_ms"]["tts"] = round((time.perf_counter() - t2) * 1000)
            audio_block = Audio(data=wav_bytes(speech.waveform, speech.sample_rate), format="wav")
        content.append(TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)))
        if audio_block is not None:
            content.append(audio_block)
        return content

    # ── translation memory & community ─────────────────────────────────────
    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def search_translation_memory(
        query: str, source_language: str, target_language: str, limit: int = 5
    ) -> dict[str, Any]:
        """Search the translation memory for vetted / previous human translations similar to `query`.

        Use before translating terminology (medical, legal, religious …) to stay consistent with
        community-approved wording.

        Args:
            query: Text to look up.
            source_language: Language of `query`.
            target_language: Language of the translations wanted.
            limit: Maximum matches (1–50).

        Returns:
            {"matches": [{id, source, target, score (0..1), approved, origin}], "took_ms": n}
        """
        engine = rt.engine
        src = engine.registry.normalize(source_language)
        tgt = engine.registry.normalize(target_language)
        tm = rt.tm
        if tm is None:
            raise CapabilityUnavailableError("The translation memory is not available on this server.")
        t0 = time.perf_counter()
        matches = tm.lookup(query, src, tgt, k=max(1, min(int(limit), 50)))
        return {"matches": to_plain(list(matches)), "took_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @server.tool(annotations=WRITE)
    @_tool_errors
    def submit_translation(
        source_text: str,
        target_text: str,
        source_language: str,
        target_language: str,
        consent: bool,
        license: License,
        domain: Domain = "general",
        dialect: str | None = None,
        contributor: str | None = None,
    ) -> dict[str, Any]:
        """Contribute a human translation to the AfriLingua community dataset (goes to peer review).

        IMPORTANT: only call this when the user explicitly agreed to share the text publicly. Set
        `consent=true` only after asking, and pass the license they chose.

        Args:
            source_text: Original text.
            target_text: The human translation.
            source_language: Language of `source_text`.
            target_language: Language of `target_text`.
            consent: Must be true — the user consents to publishing under `license`.
            license: "CC-BY-4.0" (attribution) or "CC0-1.0" (public domain).
            domain: Subject area, used to route reviews.
            dialect: Optional dialect/variety (e.g. "Ọ̀yọ́", "Kano").
            contributor: Optional display name or anonymous id.

        Returns:
            The stored contribution {id, status: "pending", …}.
        """
        _require_consent(consent, license)
        engine = rt.engine
        payload = {
            "kind": "translation",
            "src_lang": engine.registry.normalize(source_language),
            "tgt_lang": engine.registry.normalize(target_language),
            "source_text": source_text,
            "target_text": target_text,
            "domain": domain,
            "dialect": dialect,
            "contributor": contributor or "mcp-agent",
            "license": license,
            "consent": True,
        }
        return _submit(rt.contrib, {k: v for k, v in payload.items() if v is not None})

    @server.tool(annotations=WRITE)
    @_tool_errors
    def submit_correction(
        source_text: str,
        machine_translation: str,
        corrected_translation: str,
        source_language: str,
        target_language: str,
        consent: bool,
        license: License,
        domain: Domain = "general",
        contributor: str | None = None,
    ) -> dict[str, Any]:
        """Report a better translation for something AfriLingua got wrong (goes to peer review).

        Corrections are the most valuable training signal. Only call with the user's explicit consent.

        Args:
            source_text: The original text that was translated.
            machine_translation: What AfriLingua produced.
            corrected_translation: The correct human translation.
            source_language: Language of `source_text`.
            target_language: Language of the translations.
            consent: Must be true — the user consents to publishing under `license`.
            license: "CC-BY-4.0" or "CC0-1.0".
            domain: Subject area.
            contributor: Optional display name or anonymous id.

        Returns:
            The stored contribution {id, status: "pending", …}.
        """
        _require_consent(consent, license)
        engine = rt.engine
        payload = {
            "kind": "correction",
            "src_lang": engine.registry.normalize(source_language),
            "tgt_lang": engine.registry.normalize(target_language),
            "source_text": source_text,
            "target_text": corrected_translation,
            "original_mt": machine_translation,
            "domain": domain,
            "contributor": contributor or "mcp-agent",
            "license": license,
            "consent": True,
        }
        return _submit(rt.contrib, payload)

    # ── live rooms ─────────────────────────────────────────────────────────
    @server.tool(annotations=WRITE)
    @_tool_errors
    def create_live_room(
        ctx: Context,
        source_language: str = "english",
        target_languages: list[str] | None = None,
        title: str = "Live session",
        pin: str | None = None,
    ) -> dict[str, Any]:
        """Open a live interpretation room (conference mode): one speaker, many listeners, each hearing
        captions / speech in their own language in real time.

        Requires the AfriLingua HTTP server (`afrilingua serve`); over stdio this returns instructions.

        Args:
            source_language: Language the speaker will talk in.
            target_languages: Languages to interpret into up front (listeners can add more by joining).
            title: Room title shown to listeners.
            pin: Optional PIN listeners must enter.

        Returns:
            {id, join_url (share with the audience), host_ws_url, host_token (secret), listen_ws_url,
            how_to_stream}.
        """
        rooms = rt.rooms
        if rooms is None:
            s = rt.settings
            return {
                "error": "http_server_required",
                "message": (
                    "Live rooms need the AfriLingua HTTP server. Start it with `afrilingua serve` "
                    f"(default http://{s.api_host}:{s.api_port}) and connect this MCP client to "
                    f"http://{s.api_host}:{s.api_port}/mcp, or create a room with POST /api/v1/rooms."
                ),
            }
        room = rooms.create(
            title=title, src_lang=source_language, targets=list(target_languages or []), pin=pin
        )
        base = _request_base_url(ctx, rt.settings)
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        return {
            "id": room.id,
            "title": room.title,
            "src_lang": room.src_lang,
            "targets": room.targets,
            "join_url": f"{base}/room/{room.id}",
            "host_token": room.host_token,
            "host_ws_url": f"{ws_base}/api/v1/rooms/{room.id}/host?token={room.host_token}",
            "listen_ws_url": f"{ws_base}/api/v1/rooms/{room.id}/listen?lang=<language>&audio=1",
            "pin_required": bool(room.pin),
            "how_to_stream": (
                'Connect to host_ws_url, send {"type":"start","sample_rate":16000,"encoding":"pcm_s16le"}, '
                'then binary PCM16 LE mono frames (20–100 ms each), and {"type":"stop"} to end.'
            ),
        }

    # ── resources ──────────────────────────────────────────────────────────
    @server.resource(
        "afrilingua://languages",
        name="languages",
        mime_type="application/json",
        description="Supported languages with capabilities.",
    )
    def languages_resource() -> str:
        return json.dumps({"languages": [s.to_public_dict() for s in rt.engine.registry]}, ensure_ascii=False)

    @server.resource(
        "afrilingua://status",
        name="status",
        mime_type="application/json",
        description="Runtime, hardware and model load status.",
    )
    def status_resource() -> str:
        data = rt.engine.status()
        rooms = rt.rooms
        if rooms is not None:
            data["rooms"] = rooms.summary()
        return json.dumps(to_plain(data), ensure_ascii=False)

    @server.resource(
        "afrilingua://tm/stats",
        name="tm-stats",
        mime_type="application/json",
        description="Translation-memory statistics (entries, approved, per language pair).",
    )
    def tm_stats_resource() -> str:
        tm = rt.tm
        if tm is None:
            return json.dumps({"available": False})
        stats = tm.stats() if callable(getattr(tm, "stats", None)) else {}
        return json.dumps({"available": True, **to_plain(stats)}, ensure_ascii=False)

    # ── prompts ────────────────────────────────────────────────────────────
    @server.prompt(title="Interpret for an audience")
    def interpret_for_audience(
        target_languages: str, source_language: str = "english", topic: str = ""
    ) -> str:
        """Set up live interpretation of a talk for a multilingual audience."""
        topic_line = f" The talk is about: {topic}." if topic else ""
        return (
            f"You are coordinating live interpretation from {source_language} into {target_languages}.{topic_line}\n"
            "1. Call `list_languages` and confirm every requested language is supported.\n"
            "2. If the topic has specialised terms, call `search_translation_memory` for each key term and "
            "keep approved wording consistent.\n"
            f"3. Call `create_live_room` with source_language='{source_language}' and the target languages, "
            "then give the speaker the host instructions and share `join_url` with the audience.\n"
            "4. For short pre-written remarks, use `translate_batch` (mode='document') and "
            "`synthesize_speech` to prepare audio in each language."
        )

    @server.prompt(title="Localize a document")
    def localize_document(
        target_language: str, text: str, domain: str = "general", source_language: str = "auto"
    ) -> str:
        """Translate and culturally localize a document's text, preserving structure."""
        return (
            f"Localize the following {domain} document from {source_language} into {target_language}.\n"
            "- Split it into paragraphs and call `translate_batch` with mode='document'.\n"
            "- Check domain terms with `search_translation_memory` and prefer approved translations.\n"
            "- Keep headings, lists, numbers, names and formatting intact; flag sentences you are unsure about.\n"
            "- If the user corrects any sentence, offer to share it with `submit_correction` (ask for consent "
            "and a license first).\n\n"
            f"Document:\n{text}"
        )

    return server


# ── stdio entry point ──────────────────────────────────────────────────────
class _LazyEngine:
    def __init__(self) -> None:
        self._engine: Any = None
        self._lock = threading.Lock()

    def __call__(self) -> Any:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    from afrilingua.engine import build_engine

                    logger.info("Building AfriLingua engine for MCP (stdio)…")
                    self._engine = build_engine()
        return self._engine


def run_stdio() -> None:
    """Run the MCP server over stdio. Nothing but protocol traffic is written to stdout."""
    from afrilingua.core.settings import apply_runtime_environment

    apply_runtime_environment()
    server = build_mcp(_LazyEngine(), allow_file_paths=True)
    server.run("stdio")
