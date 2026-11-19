"""OpenAI-compatible audio endpoints (contract §7).

Point any OpenAI SDK at ``http://127.0.0.1:8000/v1``::

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
    client.audio.transcriptions.create(model="afrilingua", file=f, language="yo")
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from afrilingua.api.audio_io import AUDIO_MEDIA_TYPES, encode_audio
from afrilingua.api.deps import EngineDep
from afrilingua.api.errors import not_found
from afrilingua.api.routers.speech import transcribe_upload
from afrilingua.backends.base import Segment, Transcript
from afrilingua.core.errors import InvalidInputError, UnsupportedLanguageError

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

ResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]

MODELS = {
    "afrilingua-mt": "Text translation (NLLB family, CTranslate2 int8 on CPU)",
    "afrilingua-asr": "Speech recognition (Whisper for English, MMS for African languages)",
    "afrilingua-tts": "Speech synthesis (MMS-TTS / VITS)",
}
_CREATED = 1_731_888_000

# OpenAI's built-in voice names → fall back to English so drop-in clients keep working.
OPENAI_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


# ── formatting helpers ──────────────────────────────────────────────────────
def _segments(t: Transcript) -> list[Segment]:
    if t.segments:
        return list(t.segments)
    if not t.text:
        return []
    return [Segment(0.0, float(t.duration_s or 0.0), t.text)]


def _ts(seconds: float, sep: str) -> str:
    ms = round(max(seconds, 0.0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{_ts(seg.start, ',')} --> {_ts(seg.end, ',')}\n{seg.text.strip()}\n"
        for i, seg in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    blocks = [f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}\n{seg.text.strip()}\n" for seg in segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def format_transcript(
    text: str,
    segments: list[Segment],
    *,
    language: str,
    duration: float,
    fmt: str,
    task: str,
    extra: dict[str, Any] | None = None,
) -> Response:
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(to_srt(segments), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(to_vtt(segments), media_type="text/vtt")
    if fmt == "verbose_json":
        body: dict[str, Any] = {
            "task": task,
            "language": language,
            "duration": round(duration, 3),
            "text": text,
            "segments": [
                {
                    "id": i,
                    "seek": 0,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                }
                for i, seg in enumerate(segments)
            ],
        }
        body.update(extra or {})
        return JSONResponse(body)
    if fmt == "json":
        return JSONResponse({"text": text})
    raise InvalidInputError(f"Unsupported response_format {fmt!r}. Use json, text, verbose_json, srt or vtt.")


# ── endpoints ───────────────────────────────────────────────────────────────
@router.get("/models", summary="List models (OpenAI format)")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": _CREATED, "owned_by": "afrilingua", "description": desc}
            for mid, desc in MODELS.items()
        ],
    }


@router.get("/models/{model_id}", summary="Retrieve a model (OpenAI format)")
async def get_model(model_id: str) -> dict[str, Any]:
    if model_id not in MODELS:
        raise not_found(f"Model {model_id!r}")
    return {
        "id": model_id,
        "object": "model",
        "created": _CREATED,
        "owned_by": "afrilingua",
        "description": MODELS[model_id],
    }


@router.post("/audio/transcriptions", summary="Transcribe audio (OpenAI Whisper-compatible)")
async def transcriptions(
    file: Annotated[UploadFile, File(description="Audio file.")],
    engine: EngineDep,
    model: Annotated[str, Form(description="Ignored (any value accepted).")] = "afrilingua",
    language: Annotated[
        str | None, Form(description="Spoken language, ISO code or name. Default: english.")
    ] = None,
    prompt: Annotated[str | None, Form(description="Ignored.")] = None,
    response_format: Annotated[str, Form(description="json | text | verbose_json | srt | vtt")] = "json",
    temperature: Annotated[float | None, Form(description="Ignored.")] = None,
) -> Response:
    lang = engine.registry.normalize(language or "english")
    t = await transcribe_upload(engine, file, lang)
    return format_transcript(
        t.text,
        _segments(t),
        language=t.language,
        duration=t.duration_s,
        fmt=response_format,
        task="transcribe",
    )


@router.post(
    "/audio/translations",
    summary="Speech → translated text",
    description="Unlike OpenAI (always → English), AfriLingua accepts an extra `target_language` field.",
)
async def translations(
    file: Annotated[UploadFile, File(description="Audio file.")],
    engine: EngineDep,
    model: Annotated[str, Form()] = "afrilingua",
    language: Annotated[str | None, Form(description="Source (spoken) language. Default: english.")] = None,
    target_language: Annotated[str, Form(description="Target language (AfriLingua extension).")] = "english",
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float | None, Form()] = None,
) -> Response:
    src = engine.registry.normalize(language or "english")
    tgt = engine.registry.normalize(target_language or "english")
    t = await transcribe_upload(engine, file, src)
    segments = _segments(t)
    texts = [t.text] + [s.text for s in segments]
    results = await run_in_threadpool(engine.translate, texts, src, tgt, mode="document")
    translated = results[0].text
    out_segments = [Segment(s.start, s.end, r.text) for s, r in zip(segments, results[1:], strict=True)]
    return format_transcript(
        translated,
        out_segments,
        language=tgt,
        duration=t.duration_s,
        fmt=response_format,
        task="translate",
        extra={"source_language": src, "source_text": t.text, "target_language": tgt},
    )


class SpeechBody(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"examples": [{"model": "afrilingua-tts", "input": "Ẹ káàbọ̀", "voice": "yoruba"}]},
    )
    model: str = "afrilingua-tts"
    input: str = Field(min_length=1, max_length=5000)
    voice: str = Field("english", description="Language name/alias (e.g. `yoruba`, `ig`, `hau`).")
    response_format: str = Field(
        "wav", description="wav | flac | pcm (mp3/opus when libsndfile supports them)"
    )
    speed: float | None = None
    instructions: str | None = None


@router.post(
    "/audio/speech",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
    summary="Text → speech (OpenAI-compatible; voice = language)",
)
async def speech(body: SpeechBody, engine: EngineDep) -> Response:
    voice = body.voice.strip().lower()
    try:
        lang = engine.registry.normalize(voice)
    except UnsupportedLanguageError:
        if voice in OPENAI_VOICES:
            lang = "english"
        else:
            raise
    fmt = (body.response_format or "wav").lower()
    t0 = time.perf_counter()
    audio = await run_in_threadpool(engine.synthesize, body.input, lang)
    data = await run_in_threadpool(encode_audio, audio.waveform, audio.sample_rate, fmt)
    return Response(
        content=data,
        media_type=AUDIO_MEDIA_TYPES.get(fmt, "application/octet-stream"),
        headers={
            "X-Sample-Rate": str(audio.sample_rate),
            "X-Voice-Language": lang,
            "X-Synthesis-Ms": f"{(time.perf_counter() - t0) * 1000:.0f}",
        },
    )
