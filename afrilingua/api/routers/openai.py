"""OpenAI-compatible router."""
from fastapi import APIRouter, Request, UploadFile, File, Form
from typing import Optional

router = APIRouter(prefix="/v1", tags=["openai"])

@router.post("/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: str = "afrilingua", language: Optional[str] = None):
    # Simplified mock for OpenAI compatibility
    return {"text": "mock transcription"}
