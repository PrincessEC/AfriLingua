"""
Typed application settings.

All configuration is read from environment variables and the project ``.env``
file. Variable names are backwards compatible with AfriLingua v1 (``MODEL_NLLB``,
``DEVICE``, ``HF_CACHE`` …), so existing ``.env`` files keep working.

Nothing here has import-time side effects: call :func:`get_settings` to obtain
the cached instance and :func:`apply_runtime_environment` once at process start
to export Hugging Face cache variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DeviceChoice = Literal["auto", "cpu", "cuda"]
DtypeChoice = Literal["auto", "float32", "float16", "int8"]
ProfileChoice = Literal["auto", "cpu-lite", "cpu-standard", "gpu"]
PreloadChoice = Literal["adaptive", "all", "minimal"]
MTBackendChoice = Literal["auto", "ctranslate2", "transformers"]
ASRBackendChoice = Literal["auto", "faster-whisper", "openai-whisper"]
TTSEnglishChoice = Literal["auto", "mms", "pyttsx3"]


class Settings(BaseSettings):
    """Runtime configuration. Field names map 1:1 to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Hardware ────────────────────────────────────────────────────────────
    device: DeviceChoice = "auto"
    torch_dtype: DtypeChoice = "auto"
    profile: ProfileChoice = Field(default="auto", validation_alias="AFRILINGUA_PROFILE")
    preload_mode: PreloadChoice = "adaptive"
    cpu_threads: int = Field(default=0, description="0 = let backends decide")

    # ── Backends ────────────────────────────────────────────────────────────
    mt_backend: MTBackendChoice = "auto"
    asr_en_backend: ASRBackendChoice = "auto"
    tts_en_backend: TTSEnglishChoice = "auto"

    # ── Model IDs (empty string = choose from hardware profile) ────────────
    model_nllb: str = ""
    model_whisper: str = ""
    model_mms_asr: str = "facebook/mms-1b-all"
    model_mms_tts_english: str = "facebook/mms-tts-eng"
    model_embedder: str = Field(
        default="",
        description="Optional semantic embedder for the translation memory (e.g. sentence-transformers/LaBSE). "
        "Empty = fast built-in char n-gram vectors.",
    )

    # Live vs. document decoding
    live_beam_size: int = 1
    document_beam_size: int = 4
    max_translation_tokens: int = 256

    # ── Paths ───────────────────────────────────────────────────────────────
    hf_cache: Path = Path("models/cache")
    ct2_cache: Path = Path("models/ct2")
    data_dir: Path = Path("data")
    finetuned_model_dir: Path = Path("data/finetuned/nllb-afrilingua")
    language_registry_file: str = ""
    afrilingua_extra_languages: str = ""

    # ── Audio ───────────────────────────────────────────────────────────────
    sample_rate: int = 16_000
    vad_threshold: float = 0.5
    chunk_duration_s: float = 0.1
    vad_silence_s: float = 0.6
    vad_max_utterance_s: float = 8.0
    vad_preroll_s: float = 0.3

    # ── Translation memory ─────────────────────────────────────────────────
    #: 1.0 = serve memory verbatim only on an exact (normalised) match. Fuzzy scores are
    #: capped below 1.0, so lowering this serves *similar* sentences' translations: unsafe.
    tm_exact_threshold: float = 1.0
    tm_fuzzy_threshold: float = 0.80

    # ── Latency ─────────────────────────────────────────────────────────────
    #: Split multi-sentence text so known sentences hit memory and the rest decode in one batch.
    mt_segment_sentences: bool = True
    #: Restrict NLLB decoding to the scripts of registered languages (CTranslate2 vocab map).
    mt_vocab_restrict: bool = True
    #: Longest text accepted by translate(); longer input should go through document jobs.
    max_text_chars: int = 20_000

    # ── Server ──────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_keys: str = Field(default="", description="Comma-separated API keys. Empty = no auth (local use).")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 25
    offline: bool = Field(default=False, validation_alias="AFRILINGUA_OFFLINE")

    @field_validator("hf_cache", "ct2_cache", "data_dir", "finetuned_model_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ── Derived helpers ─────────────────────────────────────────────────────
    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "afrilingua.db"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def contributions_dir(self) -> Path:
        return self.data_dir / "contributions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def apply_runtime_environment(settings: Settings | None = None) -> None:
    """Export cache locations / offline flags for Hugging Face libraries.

    Call once at process start-up (CLI entry point, server lifespan, tests).
    """
    s = settings or get_settings()
    s.hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(s.hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(s.hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(s.hf_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if s.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
