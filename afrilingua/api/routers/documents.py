"""Document translation (format-preserving) as background jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from afrilingua.api.deps import get_engine
from afrilingua.api.errors import APIError, not_found
from afrilingua.api.jobs import JobManager
from afrilingua.api.schemas import ERROR_RESPONSES, JobCreated, JobStatus
from afrilingua.core.errors import InvalidInputError
from afrilingua.engine import AfriLinguaEngine

router = APIRouter(prefix="/api/v1", tags=["documents"])

ALLOWED_SUFFIXES = {".txt", ".docx", ".pdf"}
MEDIA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain; charset=utf-8",
    ".pdf": "application/pdf",
}


def get_jobs(request: Request) -> JobManager:
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise APIError(503, "unavailable", "Document jobs are not available yet.")
    return jobs


@router.post(
    "/documents",
    response_model=JobCreated,
    status_code=202,
    responses=ERROR_RESPONSES,
    summary="Upload a .txt/.docx/.pdf for background translation",
)
async def create_document_job(
    file: UploadFile = File(..., description=".txt, .docx or .pdf"),
    tgt_lang: str = Form(..., description="Target language"),
    src_lang: str | None = Form(None, description="Source language (omit or `auto` to detect)"),
    engine: AfriLinguaEngine = Depends(get_engine),
    jobs: JobManager = Depends(get_jobs),
) -> dict[str, Any]:
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InvalidInputError(f"Unsupported document type {suffix or '(none)'!r}. Use .txt, .docx or .pdf.")
    tgt = engine.registry.normalize(tgt_lang)
    src = None if not src_lang or src_lang.lower() in {"auto", "auto-detect"} else engine.registry.normalize(src_lang)
    data = await file.read()
    if not data:
        raise InvalidInputError("The uploaded document is empty.")
    job = await run_in_threadpool(jobs.submit, name, data, tgt, src)
    return {"id": job.id, "status": job.status}


@router.get("/documents/{job_id}", response_model=JobStatus, responses=ERROR_RESPONSES, summary="Job status")
async def document_status(job_id: str, jobs: JobManager = Depends(get_jobs)) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise not_found("Document job")
    return job.public()


@router.get(
    "/documents/{job_id}/download",
    responses={**ERROR_RESPONSES, 200: {"content": {MEDIA_TYPES[".docx"]: {}, "text/plain": {}}}},
    summary="Download the translated file (DOCX→DOCX, PDF→DOCX, TXT→TXT)",
)
async def download_document(job_id: str, jobs: JobManager = Depends(get_jobs)) -> FileResponse:
    job = jobs.get(job_id)
    if job is None:
        raise not_found("Document job")
    if job.status != "done" or not job.result_path:
        raise APIError(409, "not_ready", f"Job is {job.status}; download is available when status is 'done'.")
    path = Path(job.result_path)
    if not path.exists():
        raise not_found("Translated file")
    return FileResponse(path, filename=path.name, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))
