"""Translation memory REST router (``/api/v1/tm``; contract §4)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from afrilingua.core.errors import CapabilityUnavailableError, PayloadTooLargeError
from afrilingua.core.settings import Settings, get_settings

router = APIRouter(prefix="/api/v1/tm", tags=["translation-memory"])


class TMEntryIn(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    src_lang: str
    tgt_lang: str
    approved: bool = False
    origin: str = "manual"


def _require_tm(request: Request) -> Any:
    tm = getattr(request.app.state, "tm", None)
    if tm is None:
        raise CapabilityUnavailableError("Translation memory is disabled on this server.")
    return tm


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.get("/search")
async def search_tm(
    request: Request,
    q: str = Query(..., description="Source phrase to look up."),
    src_lang: str = Query(...),
    tgt_lang: str = Query(...),
    k: int = Query(5, ge=1, le=100),
) -> dict[str, Any]:
    tm = _require_tm(request)
    matches, took_ms = await run_in_threadpool(tm.search, q, src_lang, tgt_lang, k)
    return {
        "matches": [
            {
                "id": m.id,
                "source": m.source,
                "target": m.target,
                "score": m.score,
                "approved": m.approved,
                "origin": m.origin,
            }
            for m in matches
        ],
        "took_ms": round(float(took_ms), 2),
    }


@router.post("/entries")
async def add_entry(body: TMEntryIn, request: Request) -> dict[str, str]:
    tm = _require_tm(request)
    entry_id = await run_in_threadpool(
        lambda: tm.add(
            body.source,
            body.target,
            body.src_lang,
            body.tgt_lang,
            origin=body.origin,
            approved=body.approved,
        )
    )
    return {"id": entry_id}


@router.post("/entries/{entry_id}/approve")
async def approve_entry(entry_id: str, request: Request) -> dict[str, Any]:
    tm = _require_tm(request)
    await run_in_threadpool(tm.approve, entry_id)
    return {"id": entry_id, "approved": True}


@router.delete("/entries/{entry_id}", status_code=204)
async def delete_entry(entry_id: str, request: Request) -> Response:
    tm = _require_tm(request)
    await run_in_threadpool(tm.delete, entry_id)
    return Response(status_code=204)


@router.post("/import")
async def import_entries(
    request: Request,
    file: UploadFile = File(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    approved: bool = Form(False),
) -> dict[str, int]:
    tm = _require_tm(request)
    max_bytes = _settings(request).max_upload_mb * 1024 * 1024
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"Upload exceeds {_settings(request).max_upload_mb} MB.")

    suffix = Path(file.filename or "pairs.tsv").suffix or ".tsv"

    def _run() -> int:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            return int(tm.import_file(tmp_path, src_lang, tgt_lang, approved=approved))
        finally:
            tmp_path.unlink(missing_ok=True)

    imported = await run_in_threadpool(_run)
    return {"imported": imported}


@router.get("/stats")
async def tm_stats(request: Request) -> dict[str, Any]:
    tm = _require_tm(request)
    return await run_in_threadpool(tm.stats)
