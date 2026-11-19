"""Text translation + language detection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from afrilingua.api.deps import EngineDep
from afrilingua.api.schemas import (
    ERROR_RESPONSES,
    DetectRequest,
    DetectResponse,
    TranslateRequest,
    TranslateResponse,
)

router = APIRouter(prefix="/api/v1", tags=["translation"])


@router.post(
    "/translate",
    response_model=TranslateResponse,
    responses=ERROR_RESPONSES,
    summary="Translate one text or a batch",
    description="Normalize → cache → translation memory → machine translation. "
    "`src_lang` may be `auto`; results report `detected: true` when the source was detected.",
)
async def translate(body: TranslateRequest, engine: EngineDep) -> dict[str, Any]:
    texts = body.all_texts()
    src = body.src_lang or "auto"
    engine.registry.normalize(body.tgt_lang)  # fail fast (400) before touching models
    if src.lower() not in {"auto", "auto-detect"}:
        engine.registry.normalize(src)
    results = await run_in_threadpool(
        engine.translate, texts, src, body.tgt_lang, mode=body.mode, use_tm=body.use_tm
    )
    return {"results": [r.to_dict() for r in results]}


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses=ERROR_RESPONSES,
    summary="Detect the language of a text",
)
async def detect(body: DetectRequest, engine: EngineDep) -> dict[str, Any]:
    language, confidence = await run_in_threadpool(engine.detect_language, body.text)
    return {"language": language, "confidence": round(float(confidence), 4)}
