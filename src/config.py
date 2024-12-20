"""
src/config.py
Centralised config loaded from .env — import this everywhere instead of
reading os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from src/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE: str = _get("DEVICE", "cpu")
TORCH_DTYPE: str = _get("TORCH_DTYPE", "float32")

# ── Model IDs ─────────────────────────────────────────────────────────────────
MODEL_NLLB: str = _get("MODEL_NLLB", "facebook/nllb-200-distilled-600M")
MODEL_WHISPER: str = _get("MODEL_WHISPER", "openai/whisper-small")
MODEL_MMS_ASR: str = _get("MODEL_MMS_ASR", "facebook/mms-300m")
MODEL_MMS_TTS_IGBO: str = _get("MODEL_MMS_TTS_IGBO", "facebook/mms-tts-ibo")
MODEL_MMS_TTS_YORUBA: str = _get("MODEL_MMS_TTS_YORUBA", "facebook/mms-tts-yor")
MODEL_MMS_TTS_HAUSA: str = _get("MODEL_MMS_TTS_HAUSA", "facebook/mms-tts-hau")

# ── Paths ─────────────────────────────────────────────────────────────────────
HF_CACHE: Path = _ROOT / _get("HF_CACHE", "models/cache")
FINETUNED_MODEL_DIR: Path = _ROOT / _get("FINETUNED_MODEL_DIR", "data/finetuned/nllb-afrilingua")
DATA_RAW_DIR: Path = _ROOT / _get("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR: Path = _ROOT / _get("DATA_PROCESSED_DIR", "data/processed")

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = int(_get("SAMPLE_RATE", "16000"))
VAD_THRESHOLD: float = float(_get("VAD_THRESHOLD", "0.5"))
CHUNK_DURATION_S: float = float(_get("CHUNK_DURATION_S", "0.5"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = _get("API_HOST", "127.0.0.1")
API_PORT: int = int(_get("API_PORT", "8000"))

# ── UI ────────────────────────────────────────────────────────────────────────
GRADIO_PORT: int = int(_get("GRADIO_PORT", "7860"))

# ensure cache dir exists at import time
HF_CACHE.mkdir(parents=True, exist_ok=True)

# tell HuggingFace libraries where to cache
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
