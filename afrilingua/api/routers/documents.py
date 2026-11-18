"""Documents translation router."""
from fastapi import APIRouter, Request, UploadFile, File, BackgroundTasks
from typing import Optional
from afrilingua.core.settings import get_settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

@router.post("")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tgt_lang: str = "english",
    src_lang: Optional[str] = None
):
    # Simplified placeholder for background job submission
    return {"id": "job_123", "status": "queued"}

@router.get("/{job_id}")
async def get_job(job_id: str):
    return {"id": job_id, "status": "done"}
