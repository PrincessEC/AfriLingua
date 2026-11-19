"""Community contributions REST router (``/api/v1/contributions``; contract §5)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from afrilingua.core.errors import CapabilityUnavailableError, PayloadTooLargeError
from afrilingua.core.settings import Settings, get_settings

router = APIRouter(prefix="/api/v1/contributions", tags=["contributions"])


class ContributionInBody(BaseModel):
    kind: Literal["translation", "correction", "glossary", "transcription"]
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    original_mt: str | None = None
    domain: str | None = None
    dialect: str | None = None
    region: str | None = None
    accent: str | None = None
    gender: str | None = None
    age_range: str | None = None
    contributor: str | None = None
    license: str = "CC-BY-4.0"
    consent: bool = False


class ReviewInBody(BaseModel):
    vote: Literal["approve", "reject"]
    reviewer: str = Field(..., min_length=1)
    comment: str | None = None


class WithdrawInBody(BaseModel):
    contributor: str = Field(..., min_length=1)


def _require_contrib(request: Request) -> Any:
    contrib = getattr(request.app.state, "contrib", None)
    if contrib is None:
        raise CapabilityUnavailableError("Community contribution store is disabled on this server.")
    return contrib


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.get("/prompts")
async def get_prompts(
    request: Request,
    lang: str = Query(...),
    kind: Literal["translate", "record"] = Query("translate"),
    n: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    contrib = _require_contrib(request)
    prompts = await run_in_threadpool(lambda: contrib.get_prompts(lang, kind=kind, n=n))
    return {"prompts": prompts}


@router.post("")
async def submit_contribution(body: ContributionInBody, request: Request) -> dict[str, Any]:
    contrib = _require_contrib(request)
    payload = body.model_dump(exclude_none=True)
    return await run_in_threadpool(contrib.submit, payload)


@router.post("/audio")
async def submit_audio_contribution(
    request: Request,
    audio: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form(...),
    dialect: str | None = Form(None),
    accent: str | None = Form(None),
    region: str | None = Form(None),
    gender: str | None = Form(None),
    age_range: str | None = Form(None),
    domain: str | None = Form(None),
    consent: bool = Form(False),
    license: str = Form("CC-BY-4.0"),
    contributor: str | None = Form(None),
    voice_cloning_opt_in: bool = Form(False),
) -> dict[str, Any]:
    _ = voice_cloning_opt_in  # never used for voice cloning unless separately tracked in a future voice pipeline
    contrib = _require_contrib(request)
    max_bytes = _settings(request).max_upload_mb * 1024 * 1024
    raw = await audio.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"Audio upload exceeds {_settings(request).max_upload_mb} MB.")
    return await run_in_threadpool(
        lambda: contrib.add_audio(
            raw,
            text=text,
            lang=lang,
            filename=audio.filename,
            content_type=audio.content_type,
            dialect=dialect,
            accent=accent,
            gender=gender,
            age_range=age_range,
            region=region,
            consent=consent,
            license=license,
            contributor=contributor,
            domain=domain,
        )
    )


@router.get("")
async def list_contributions(
    request: Request,
    status: str | None = Query(None),
    lang: str | None = Query(None),
    kind: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    contrib = _require_contrib(request)
    items, total = await run_in_threadpool(
        lambda: contrib.list(status=status, lang=lang, kind=kind, limit=limit, offset=offset)
    )
    return {"items": items, "total": total}


@router.get("/review-queue")
async def review_queue(
    request: Request,
    reviewer: str = Query(..., min_length=1),
    lang: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    contrib = _require_contrib(request)
    items = await run_in_threadpool(lambda: contrib.review_queue(reviewer, lang=lang, limit=limit))
    return {"items": items}


@router.post("/{contribution_id}/review")
async def review_contribution(
    contribution_id: str, body: ReviewInBody, request: Request
) -> dict[str, Any]:
    contrib = _require_contrib(request)
    return await run_in_threadpool(
        contrib.review, contribution_id, body.reviewer, body.vote, body.comment
    )


@router.get("/{contribution_id}/audio")
async def get_contribution_audio(contribution_id: str, request: Request) -> FileResponse:
    contrib = _require_contrib(request)
    path, media_type = await run_in_threadpool(contrib.audio_file, contribution_id)
    return FileResponse(path, media_type=media_type, filename=Path(path).name)


@router.get("/stats")
async def contribution_stats(request: Request) -> dict[str, Any]:
    contrib = _require_contrib(request)
    return await run_in_threadpool(contrib.stats)


@router.get("/export")
async def export_contributions(
    request: Request,
    lang: str | None = Query(None),
    kind: str | None = Query(None),
    format: Literal["jsonl", "hf"] = Query("jsonl"),
) -> FileResponse:
    contrib = _require_contrib(request)

    def _build_export() -> tuple[Path, str]:
        out = contrib.export(lang=lang, kind=kind, fmt=format)
        if format == "jsonl":
            return out, "application/x-ndjson"
        archive = Path(shutil.make_archive(str(out), "zip", root_dir=out))
        return archive, "application/zip"

    path, media_type = await run_in_threadpool(_build_export)
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/withdraw")
async def withdraw_contributor(body: WithdrawInBody, request: Request) -> dict[str, int]:
    contrib = _require_contrib(request)
    return await run_in_threadpool(contrib.withdraw, body.contributor)
