"""Speech: transcription (any common audio format) and synthesis."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from afrilingua.api.audio_io import AUDIO_MEDIA_TYPES, decode_audio, encode_audio
from afrilingua.api.deps import get_engine
from afrilingua.api.schemas import ERROR_RESPONSES, SpeechRequest, TranscriptModel
from afrilingua.backends.base import Transcript
from afrilingua.engine import AfriLinguaEngine

router = APIRouter(prefix="/api/v1", tags=["speech"])


def transcript_to_dict(t: Transcript) -> dict[str, Any]:
    return {
        "text": t.text,
        "language": t.language,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in t.segments],
        "confidence": t.confidence,
        "duration_s": round(float(t.duration_s), 3),
    }


async def transcribe_upload(engine: AfriLinguaEngine, file: UploadFile, lang: str) -> Transcript:
    lang = engine.registry.normalize(lang)
    data = await file.read()
    audio = await run_in_threadpool(decode_audio, data, filename=file.filename)
    transcript: Transcript = await run_in_threadpool(engine.transcribe, audio, lang)
    if not transcript.duration_s and audio.size:
        transcript = Transcript(
            text=transcript.text,
            language=transcript.language,
            segments=transcript.segments,
            confidence=transcript.confidence,
            duration_s=audio.size / 16_000,
        )
    return transcript


@router.post(
    "/transcribe",
    response_model=TranscriptModel,
    responses=ERROR_RESPONSES,
    summary="Transcribe an audio file (wav/mp3/ogg/webm/flac)",
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav, mp3, ogg, flac; webm/opus/m4a need PyAV)."),
    lang: str = Form("english", description="Spoken language (name or alias)."),
    engine: AfriLinguaEngine = Depends(get_engine),
) -> dict[str, Any]:
    return transcript_to_dict(await transcribe_upload(engine, file, lang))


@router.post(
    "/speech",
    responses={**ERROR_RESPONSES, 200: {"content": {"audio/wav": {}}, "description": "Synthesized audio"}},
    response_class=Response,
    summary="Synthesize speech (returns audio/wav bytes)",
)
async def speech(body: SpeechRequest, engine: AfriLinguaEngine = Depends(get_engine)) -> Response:
    lang = engine.registry.normalize(body.lang)
    audio = await run_in_threadpool(engine.synthesize, body.text, lang)
    data = await run_in_threadpool(encode_audio, audio.waveform, audio.sample_rate, body.format)
    return Response(
        content=data,
        media_type=AUDIO_MEDIA_TYPES[body.format],
        headers={
            "X-Sample-Rate": str(audio.sample_rate),
            "X-Audio-Duration": f"{audio.duration_s:.3f}",
            "Content-Disposition": f'inline; filename="speech-{lang}.{body.format}"',
        },
    )
