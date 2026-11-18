"""Translate and ASR routers."""
from fastapi import APIRouter, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/v1", tags=["translate"])

class TranslateRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    src_lang: Optional[str] = "auto"
    tgt_lang: str
    mode: str = "live"
    use_tm: bool = True

@router.post("/translate")
async def translate(req: TranslateRequest, request: Request):
    texts = req.texts or ([req.text] if req.text else [])
    results = request.app.state.engine.translate(
        texts, req.src_lang, req.tgt_lang, mode=req.mode, use_tm=req.use_tm
    )
    return {"results": [r.to_dict() for r in results]}

@router.post("/detect")
async def detect(req: dict, request: Request):
    lang, conf = request.app.state.engine.detect_language(req["text"])
    return {"language": lang, "confidence": conf}
