"""Pydantic request/response models (drive the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorDetail(BaseModel):
    code: str = Field(examples=["unsupported_language"])
    message: str = Field(examples=["Unsupported language: 'klingon'."])


class ErrorResponse(BaseModel):
    error: ErrorDetail


ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "Invalid input / unsupported language"},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    413: {"model": ErrorResponse, "description": "Payload too large"},
    422: {"model": ErrorResponse, "description": "Capability unavailable for this language"},
    503: {"model": ErrorResponse, "description": "Model failed to load / not downloaded"},
}


# ── system ────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = Field(examples=["2.0.0.dev0"])


# ── translation ───────────────────────────────────────────────────────────
class TMMatchModel(BaseModel):
    id: str
    source: str
    target: str
    score: float
    approved: bool
    origin: str = ""


class TranslationResultModel(BaseModel):
    text: str
    source_text: str
    src_lang: str
    tgt_lang: str
    origin: Literal["passthrough", "cache", "tm_exact", "mt", "mt_tm_assisted"]
    latency_ms: float
    detected: bool
    matches: list[TMMatchModel] = []


class TranslateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "Good morning, how are you?", "src_lang": "english", "tgt_lang": "yoruba"},
                {"texts": ["Thank you", "Welcome"], "src_lang": "auto", "tgt_lang": "igbo", "mode": "document"},
            ]
        }
    )

    texts: list[str] | None = Field(None, max_length=512, description="Batch of sentences/paragraphs.")
    text: str | None = Field(None, description="Single text (alternative to `texts`).")
    src_lang: str | None = Field("auto", description="Source language name/alias, or `auto` to detect.")
    tgt_lang: str = Field(description="Target language name or alias (e.g. `yoruba`, `yo`, `yor_Latn`).")
    mode: Literal["live", "document"] = Field("live", description="`live` = greedy/fast, `document` = beam search.")
    use_tm: bool = Field(True, description="Consult the translation memory first.")

    @model_validator(mode="after")
    def _one_of(self) -> TranslateRequest:
        if self.texts is None and self.text is None:
            raise ValueError("Provide `text` or `texts`.")
        return self

    def all_texts(self) -> list[str]:
        return list(self.texts) if self.texts is not None else [self.text or ""]


class TranslateResponse(BaseModel):
    results: list[TranslationResultModel]


class DetectRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "Ẹ káàárọ̀, ṣé dáadáa ni?"}]})
    text: str = Field(min_length=1)


class DetectResponse(BaseModel):
    language: str = Field(examples=["yoruba"])
    confidence: float = Field(examples=[0.93])


# ── speech ────────────────────────────────────────────────────────────────
class SegmentModel(BaseModel):
    start: float
    end: float
    text: str


class TranscriptModel(BaseModel):
    text: str
    language: str
    segments: list[SegmentModel] = []
    confidence: float | None = None
    duration_s: float = 0.0


class SpeechRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "Ẹ káàbọ̀", "lang": "yoruba", "format": "wav"}]})
    text: str = Field(min_length=1, max_length=5000)
    lang: str
    format: Literal["wav", "flac", "pcm"] = "wav"


# ── documents ─────────────────────────────────────────────────────────────
class JobCreated(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"] = "queued"


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float = Field(ge=0, le=1)
    filename: str
    error: str | None = None
    tgt_lang: str | None = None
    src_lang: str | None = None
    created_at: float | None = None
    download_url: str | None = None


# ── rooms ─────────────────────────────────────────────────────────────────
class RoomCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"title": "Town hall", "src_lang": "english", "targets": ["yoruba", "igbo", "hausa"], "pin": "4821"}]
        }
    )
    title: str = Field("Live session", max_length=200)
    src_lang: str = "english"
    targets: list[str] = Field(default_factory=list, max_length=32)
    pin: str | None = Field(None, max_length=32, description="Optional PIN listeners must provide.")


class RoomCreated(BaseModel):
    id: str = Field(examples=["kq7-m2x"])
    host_token: str
    join_url: str = Field(examples=["http://127.0.0.1:8000/room/kq7-m2x"])
    host_ws: str = Field(description="Relative host WebSocket path (add `?token=`).")
    listen_ws: str = Field(description="Relative listener WebSocket path (add `?lang=&audio=&pin=`).")
    lan_join_url: str | None = Field(None, description="Join URL using this machine's LAN IP (when serving on the LAN).")


class RoomInfo(BaseModel):
    id: str
    title: str
    src_lang: str
    targets: list[str]
    listeners: int
    live: bool
    pin_required: bool = False
    host_connected: bool = False
    languages: dict[str, int] = Field(default_factory=dict, description="Listener count per language.")
    created_at: float | None = None
