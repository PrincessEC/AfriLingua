"""
src/config.py
Centralised config loaded from .env — import this everywhere instead of
reading os.environ directly.
"""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from src/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── Hardware / GPU detection ─────────────────────────────────────────────────

def _detect_gpu_info() -> dict:
    """
    Returns a dict with GPU information:
        available   — True if a CUDA-capable GPU is present
        name        — GPU model name (e.g. "NVIDIA GeForce RTX 3070")
        vram_gb     — total VRAM in gigabytes
        compute     — compute capability string (e.g. "8.6")
    """
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return {"available": False, "name": "None", "vram_gb": 0.0, "compute": ""}
        p = _t.cuda.get_device_properties(0)
        return {
            "available":  True,
            "name":       p.name,
            "vram_gb":    round(p.total_memory / 1_073_741_824, 1),  # bytes → GiB
            "compute":    f"{p.major}.{p.minor}",
        }
    except Exception:
        return {"available": False, "name": "None", "vram_gb": 0.0, "compute": ""}


GPU_INFO: dict = _detect_gpu_info()

# ── Device resolution ─────────────────────────────────────────────────────────
# DEVICE env var accepts:  cpu | cuda | auto
#   auto  → use CUDA if available, otherwise CPU
#   cuda  → force GPU (warns + falls back if CUDA unavailable)
#   cpu   → always CPU
_DEVICE_PREF: str = _get("DEVICE", "auto").lower()

if _DEVICE_PREF == "auto":
    try:
        import torch as _torch
        DEVICE: str = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
elif _DEVICE_PREF == "cuda":
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            warnings.warn(
                "DEVICE=cuda requested but CUDA is not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    DEVICE = "cpu"

# TORCH_DTYPE env var accepts:  float32 | float16 | auto
#   auto  → float16 on GPU (faster, uses less VRAM), float32 on CPU
_DTYPE_PREF: str = _get("TORCH_DTYPE", "auto").lower()
if _DTYPE_PREF == "auto":
    TORCH_DTYPE: str = "float16" if DEVICE == "cuda" else "float32"
else:
    TORCH_DTYPE = _DTYPE_PREF

# ── Hardware profile ──────────────────────────────────────────────────────────
# Automatically selected based on detected VRAM.  Drives default model sizes.
# .env overrides still take full priority.
#
#   high  — CUDA GPU with ≥ 8 GB VRAM  → best quality models
#   mid   — CUDA GPU with 4–7 GB VRAM  → balanced models
#   low   — CPU only or < 4 GB VRAM    → smallest/fastest models

_vram_gb: float = GPU_INFO["vram_gb"] if GPU_INFO["available"] and DEVICE == "cuda" else 0.0

if _vram_gb >= 8:
    HARDWARE_PROFILE: str = "high"
elif _vram_gb >= 4:
    HARDWARE_PROFILE = "mid"
else:
    HARDWARE_PROFILE = "low"

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "high": {
        # Large distilled models — best translation quality
        "MODEL_NLLB":    "facebook/nllb-200-distilled-1.3B",
        "MODEL_WHISPER": "openai/whisper-medium",
        "MODEL_MMS_ASR": "facebook/mms-300m",
    },
    "mid": {
        # Standard distilled models — good balance of speed and quality
        "MODEL_NLLB":    "facebook/nllb-200-distilled-600M",
        "MODEL_WHISPER": "openai/whisper-small",
        "MODEL_MMS_ASR": "facebook/mms-300m",
    },
    "low": {
        # Smallest footprint — runs well on CPU or integrated GPU
        "MODEL_NLLB":    "facebook/nllb-200-distilled-600M",
        "MODEL_WHISPER": "openai/whisper-base",
        "MODEL_MMS_ASR": "facebook/mms-300m",
    },
}
_profile_defaults = _PROFILE_DEFAULTS[HARDWARE_PROFILE]

# ── Deployment mode ───────────────────────────────────────────────────────────
# Options: web | desktop
DEPLOYMENT_MODE: str = _get("DEPLOYMENT_MODE", "web").lower()

# ── Model IDs ─────────────────────────────────────────────────────────────────
# Default values are chosen automatically based on HARDWARE_PROFILE.
# Override any of these in .env to pin a specific model regardless of hardware.
MODEL_NLLB:        str = _get("MODEL_NLLB",         _profile_defaults["MODEL_NLLB"])
MODEL_WHISPER:     str = _get("MODEL_WHISPER",      _profile_defaults["MODEL_WHISPER"])
MODEL_MMS_ASR:     str = _get("MODEL_MMS_ASR",      _profile_defaults["MODEL_MMS_ASR"])
MODEL_MMS_TTS_IGBO:   str = _get("MODEL_MMS_TTS_IGBO",   "facebook/mms-tts-ibo")
MODEL_MMS_TTS_YORUBA: str = _get("MODEL_MMS_TTS_YORUBA", "facebook/mms-tts-yor")
MODEL_MMS_TTS_HAUSA:  str = _get("MODEL_MMS_TTS_HAUSA",  "facebook/mms-tts-hau")

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
