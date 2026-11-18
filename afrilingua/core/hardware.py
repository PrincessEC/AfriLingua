"""
Hardware detection and profile selection.

CPU is the primary target; a CUDA GPU is treated as an accelerator. Detection
is lazy and cached — importing this module never touches torch.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from functools import lru_cache

from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareInfo:
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    cuda_built: bool
    gpu_name: str
    vram_gb: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved device / precision / model choices for this machine."""

    device: str            # "cpu" | "cuda"
    compute_type: str      # "int8" | "float16" | "float32" | "int8_float16"
    torch_dtype: str       # "float32" | "float16"  (for transformers fallbacks)
    profile: str           # "cpu-lite" | "cpu-standard" | "gpu"
    threads: int
    model_nllb: str
    model_whisper: str
    model_mms_asr: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1_073_741_824, 1)
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareInfo:
    cpu_count = os.cpu_count() or 1
    ram = _ram_gb()
    try:
        import torch

        built = bool(torch.backends.cuda.is_built())
        if not torch.cuda.is_available():
            reason = (
                "CUDA PyTorch is installed but no GPU is visible to this process."
                if built
                else "CPU-only PyTorch build."
            )
            return HardwareInfo(cpu_count, ram, False, built, "None", 0.0, reason)
        props = torch.cuda.get_device_properties(0)
        return HardwareInfo(
            cpu_count,
            ram,
            True,
            True,
            props.name,
            round(props.total_memory / 1_073_741_824, 1),
        )
    except ImportError:
        return HardwareInfo(cpu_count, ram, False, False, "None", 0.0, "PyTorch not installed.")
    except Exception as exc:  # pragma: no cover - driver specific
        return HardwareInfo(cpu_count, ram, False, False, "None", 0.0, str(exc))


_PROFILE_MODELS: dict[str, dict[str, str]] = {
    "cpu-lite": {
        "model_nllb": "facebook/nllb-200-distilled-600M",
        "model_whisper": "base",
    },
    "cpu-standard": {
        "model_nllb": "facebook/nllb-200-distilled-600M",
        "model_whisper": "small",
    },
    "gpu": {
        "model_nllb": "facebook/nllb-200-distilled-1.3B",
        "model_whisper": "small",
    },
}


def _normalize_whisper(name: str) -> str:
    """Accept both ``openai/whisper-small`` and ``small`` style identifiers."""
    name = name.strip()
    if name.startswith("openai/whisper-"):
        return name.removeprefix("openai/whisper-")
    return name


@lru_cache(maxsize=1)
def resolve_runtime() -> RuntimePlan:
    """Resolve the plan from :func:`get_settings` (cached; call ``cache_clear`` in tests)."""
    return plan_for(get_settings(), detect_hardware())


def plan_for(s: Settings, hw: HardwareInfo) -> RuntimePlan:
    """Pure function: settings + hardware → runtime plan (unit-testable)."""

    if s.device == "cuda" and not hw.cuda_available:
        logger.warning("DEVICE=cuda requested but CUDA is unavailable (%s). Using CPU.", hw.reason)
    use_cuda = hw.cuda_available and s.device in {"auto", "cuda"}
    device = "cuda" if use_cuda else "cpu"

    if s.profile != "auto":
        profile = s.profile
    elif use_cuda and hw.vram_gb >= 6:
        profile = "gpu"
    elif hw.ram_gb and hw.ram_gb < 8:
        profile = "cpu-lite"
    else:
        profile = "cpu-standard"

    if s.torch_dtype in {"float16", "float32"}:
        torch_dtype = s.torch_dtype if device == "cuda" else "float32"
    else:
        torch_dtype = "float16" if device == "cuda" else "float32"

    if s.torch_dtype == "float32":
        compute_type = "float32"
    elif device == "cuda":
        compute_type = "int8_float16" if s.torch_dtype == "int8" else "float16"
    else:
        compute_type = "int8"

    threads = s.cpu_threads or max(1, min(8, (hw.cpu_count or 2) - 1))
    defaults = _PROFILE_MODELS[profile]
    return RuntimePlan(
        device=device,
        compute_type=compute_type,
        torch_dtype=torch_dtype,
        profile=profile,
        threads=threads,
        model_nllb=s.model_nllb or defaults["model_nllb"],
        model_whisper=_normalize_whisper(s.model_whisper or defaults["model_whisper"]),
        model_mms_asr=s.model_mms_asr,
    )
