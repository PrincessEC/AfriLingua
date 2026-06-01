"""
api/main.py
FastAPI backend for AfriLingua.

Endpoints:
    POST /translate/text         — text in, text out
    POST /translate/document     — file upload, translated file returned
    WS   /translate/stream       — WebSocket: raw audio bytes in,
                                   JSON {original, translated, audio_b64} out

Run with:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import API_HOST, API_PORT
from src.detection.lang_detector import LanguageDetector
from src.document.translator import DocumentTranslator
from src.pipeline.realtime_pipeline import RealtimePipeline, TranslationResult
from src.translation.engine import TranslationEngine, SUPPORTED_LANGUAGES
from src.languages import ASR_LANGUAGE_CHOICES, TTS_MODEL_IDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AfriLingua API",
    description="Real-time English ↔ Igbo / Yoruba / Hausa translation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared singletons ─────────────────────────────────────────────────────────
_engine:   TranslationEngine = TranslationEngine()
_detector: LanguageDetector  = LanguageDetector()
_doc_translator: DocumentTranslator = DocumentTranslator(engine=_engine, detector=_detector)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TextTranslateRequest(BaseModel):
    text: str
    src_lang: Optional[str] = None   # auto-detected if omitted
    tgt_lang: str


class TextTranslateResponse(BaseModel):
    original:   str
    translated: str
    src_lang:   str
    tgt_lang:   str


# ── Text translation endpoint ─────────────────────────────────────────────────

@app.post("/translate/text", response_model=TextTranslateResponse)
async def translate_text(req: TextTranslateRequest) -> TextTranslateResponse:
    src = req.src_lang
    if not src:
        src = _detector.detect_or_default(req.text, default="english")

    try:
        translated = _engine.translate(req.text, src, req.tgt_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return TextTranslateResponse(
        original=req.text,
        translated=translated,
        src_lang=src,
        tgt_lang=req.tgt_lang,
    )


# ── Document translation endpoint ─────────────────────────────────────────────

@app.post("/translate/document")
async def translate_document(
    file: UploadFile = File(...),
    tgt_lang: str    = Form(...),
    src_lang: Optional[str] = Form(None),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Supported formats: .txt .pdf .docx")

    # Write upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        out_path = _doc_translator.translate_file(tmp_path, tgt_lang, src_lang)
        translated_text = out_path.read_text(encoding="utf-8")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "filename": f"translated_{file.filename}",
        "content":  translated_text,
        "tgt_lang": tgt_lang,
    }


# ── WebSocket streaming endpoint ──────────────────────────────────────────────

@app.websocket("/translate/stream")
async def stream_translate(websocket: WebSocket) -> None:
    """
    WebSocket protocol:
        Client sends JSON config first:
            {"src_lang": "english", "tgt_lang": "yoruba", "audio_output": true}
        Then sends binary audio frames (float32 PCM, 16 kHz mono, little-endian).
        Server responds with JSON:
            {"original": "...", "translated": "...", "audio_b64": "..."|null}
    """
    await websocket.accept()
    logger.info("WebSocket connected.")

    pipeline: Optional[RealtimePipeline] = None

    try:
        # First message must be JSON config
        config = await websocket.receive_json()
        src_lang     = config.get("src_lang", "english")
        tgt_lang     = config.get("tgt_lang", "igbo")
        audio_output = bool(config.get("audio_output", True))

        results_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_result(result: TranslationResult) -> None:
            asyncio.run_coroutine_threadsafe(
                results_queue.put(result), loop
            )

        pipeline = RealtimePipeline(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            on_result=on_result,
            audio_output=audio_output,
            local_playback=False,
        )

        async def sender() -> None:
            while True:
                result: TranslationResult = await results_queue.get()
                await websocket.send_json({
                    "original":   result.original_text,
                    "translated": result.translated_text,
                    "src_lang":   result.src_lang,
                    "tgt_lang":   result.tgt_lang,
                    "audio_b64":  result.audio_b64,
                })

        sender_task = asyncio.ensure_future(sender())

        # Receive audio frames
        while True:
            raw = await websocket.receive_bytes()
            # Decode float32 PCM bytes
            n_samples = len(raw) // 4
            audio = np.array(struct.unpack(f"<{n_samples}f", raw), dtype=np.float32)

            # Run the pipeline in a thread pool to avoid blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pipeline.process_audio_chunk(audio, wait_for_audio=audio_output)
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
    finally:
        if pipeline:
            pipeline.stop_microphone()


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status":              "ok",
        "supported_languages": SUPPORTED_LANGUAGES,
        "asr_languages":       ASR_LANGUAGE_CHOICES,
        "tts_languages":       ["english", *TTS_MODEL_IDS.keys()],
    }


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
