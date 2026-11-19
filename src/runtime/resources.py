"""Runtime resource helpers shared by the web and desktop interfaces."""
from __future__ import annotations

import gc
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.config import (
    DEVICE,
    GPU_INFO,
    HARDWARE_PROFILE,
    MODEL_MMS_ASR,
    MODEL_NLLB,
    MODEL_WHISPER,
    PRELOAD_MODE,
    TORCH_DTYPE,
)

logger = logging.getLogger(__name__)

try:  # Optional at runtime; the app still works without it.
    import psutil
except Exception:  # pragma: no cover - depends on deployment image
    psutil = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - no audio backend on some hosts
    sd = None

try:
    import torch
except Exception:  # pragma: no cover - torch is normally required by the app
    torch = None


def _gb(value: int | float | None) -> float:
    if not value:
        return 0.0
    return float(value) / 1_073_741_824


@dataclass(frozen=True)
class RuntimeSnapshot:
    device: str
    profile: str
    dtype: str
    preload_mode: str
    gpu_available: bool
    gpu_name: str
    gpu_total_gb: float
    gpu_free_gb: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_allocated_pct: float
    process_ram_gb: float
    system_ram_pct: float


def get_runtime_snapshot() -> RuntimeSnapshot:
    """Return a best-effort snapshot of process memory and CUDA usage."""
    process_ram_gb = 0.0
    system_ram_pct = 0.0
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            process_ram_gb = _gb(process.memory_info().rss)
            system_ram_pct = float(psutil.virtual_memory().percent)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("Could not read process memory: %s", exc)

    gpu_available = bool(
        torch is not None
        and getattr(torch, "cuda", None) is not None
        and torch.cuda.is_available()
    )
    gpu_name = str(GPU_INFO.get("name") or "None")
    gpu_total_gb = float(GPU_INFO.get("vram_gb") or 0.0)
    gpu_free_gb = 0.0
    gpu_allocated_gb = 0.0
    gpu_reserved_gb = 0.0
    gpu_allocated_pct = 0.0

    if gpu_available:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_gb = _gb(total_bytes)
            gpu_total_gb = total_gb or gpu_total_gb
            gpu_free_gb = _gb(free_bytes)
            gpu_allocated_gb = _gb(torch.cuda.memory_allocated(0))
            gpu_reserved_gb = _gb(torch.cuda.memory_reserved(0))
            if gpu_total_gb:
                gpu_allocated_pct = min(100.0, gpu_allocated_gb / gpu_total_gb * 100)
        except Exception as exc:  # pragma: no cover - CUDA driver dependent
            logger.debug("Could not read CUDA memory: %s", exc)

    return RuntimeSnapshot(
        device=DEVICE,
        profile=HARDWARE_PROFILE,
        dtype=TORCH_DTYPE,
        preload_mode=PRELOAD_MODE,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_total_gb=gpu_total_gb,
        gpu_free_gb=gpu_free_gb,
        gpu_allocated_gb=gpu_allocated_gb,
        gpu_reserved_gb=gpu_reserved_gb,
        gpu_allocated_pct=gpu_allocated_pct,
        process_ram_gb=process_ram_gb,
        system_ram_pct=system_ram_pct,
    )


def format_resource_strip(snapshot: Optional[RuntimeSnapshot] = None) -> str:
    """Small main-workspace status strip without exposing model internals."""
    snap = snapshot or get_runtime_snapshot()
    if snap.gpu_available:
        gpu_text = (
            f"GPU {snap.gpu_allocated_gb:.2f} GB active"
            f" / {snap.gpu_total_gb:.1f} GB"
        )
        bar_width = max(2.0, min(100.0, snap.gpu_allocated_pct))
    else:
        gpu_text = "CPU mode"
        bar_width = 0.0

    return (
        '<div class="al-resource-strip">'
        '<div class="al-resource-copy">'
        '<span class="al-resource-kicker">Workspace runtime</span>'
        f'<strong>{gpu_text}</strong>'
        f'<span>App RAM {snap.process_ram_gb:.2f} GB</span>'
        '</div>'
        '<div class="al-resource-meter" aria-label="GPU memory allocated">'
        f'<span style="width:{bar_width:.1f}%"></span>'
        '</div>'
        '</div>'
    )


def format_settings_report(loaded_models: Optional[dict[str, object]] = None) -> str:
    """Detailed operational state intended for Settings, not the workspace."""
    snap = get_runtime_snapshot()
    lines = [
        "Runtime",
        f"Device: {snap.device.upper()}",
        f"Profile: {snap.profile}",
        f"Torch dtype: {snap.dtype}",
        f"Preload mode: {snap.preload_mode}",
        "",
        "Memory",
    ]
    if snap.gpu_available:
        lines.extend(
            [
                f"GPU: {snap.gpu_name}",
                f"VRAM total: {snap.gpu_total_gb:.2f} GB",
                f"VRAM free: {snap.gpu_free_gb:.2f} GB",
                f"App allocated: {snap.gpu_allocated_gb:.2f} GB",
                f"App reserved: {snap.gpu_reserved_gb:.2f} GB",
            ]
        )
    else:
        reason = GPU_INFO.get("reason") or "CUDA is not available to this process."
        lines.extend(["GPU: Not active", f"Reason: {reason}"])
    lines.extend(
        [
            f"App RAM: {snap.process_ram_gb:.2f} GB",
            f"System RAM used: {snap.system_ram_pct:.0f}%",
            "",
            "Configured models",
            f"Translation: {MODEL_NLLB}",
            f"English ASR: {MODEL_WHISPER}",
            f"African ASR: {MODEL_MMS_ASR}",
        ]
    )
    if loaded_models:
        lines.append("")
        lines.append("Loaded now")
        for name, value in loaded_models.items():
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


def clear_accelerator_cache() -> str:
    """Release unoccupied CUDA cache blocks and run Python garbage collection."""
    gc.collect()
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        snap = get_runtime_snapshot()
        return (
            "GPU cache cleared. "
            f"Allocated {snap.gpu_allocated_gb:.2f} GB, "
            f"reserved {snap.gpu_reserved_gb:.2f} GB."
        )
    return "CPU mode is active. Python memory cleanup completed."


_WINDOWS_DEVICE_NOISE = (
    "microsoft sound mapper",
    "primary sound",
    "wdm-ks",
    "mapper",
    "output (",
    "input ()",
    "midi",
    "system virtual",
    "line out",
)
_PREFERRED_HOST_APIS = ("Windows WASAPI",)
_FALLBACK_HOST_APIS = ("Windows DirectSound", "MME")


def _clean_device_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace("Microphone Array (Realtek(R) Audio)", "Microphone Array (Realtek Audio)")
    name = name.replace("Speakers (Realtek(R) Audio)", "Speakers (Realtek Audio)")
    return name


def _device_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _is_noise_device(name: str, hostapi: str) -> bool:
    value = f"{name} {hostapi}".lower()
    return any(token in value for token in _WINDOWS_DEVICE_NOISE)


def _audio_device_choices(kind: str) -> list[str]:
    """Return curated user-facing audio choices for server-side capture/playback."""
    is_input = kind == "input"
    default_label = "Default microphone" if is_input else "Default speakers"
    choices = [default_label]
    if sd is None:
        return choices
    try:
        hostapis = sd.query_hostapis()
        devices = list(sd.query_devices())
        seen: set[str] = set()

        def _append(index: int, device: dict, preferred: bool = False) -> None:
            hostapi = hostapis[int(device.get("hostapi", 0))]["name"]
            name = _clean_device_name(str(device.get("name") or f"Device {index}"))
            if _is_noise_device(name, hostapi):
                return
            key = _device_key(name)
            if not key or key in seen:
                return
            seen.add(key)
            suffix = "default" if preferred else hostapi.replace("Windows ", "")
            choices.append(f"{name} ({suffix}) [{index}]")

        def _append_for_hostapis(names: tuple[str, ...]) -> None:
            for index, device in enumerate(devices):
                channels = int(device.get("max_input_channels" if is_input else "max_output_channels", 0))
                if channels <= 0:
                    continue
                hostapi = hostapis[int(device.get("hostapi", 0))]["name"]
                if hostapi not in names:
                    continue
                _append(index, device)

        _append_for_hostapis(_PREFERRED_HOST_APIS)
        if len(choices) == 1:
            _append_for_hostapis(_FALLBACK_HOST_APIS)
    except Exception as exc:
        logger.warning("Could not enumerate %s devices: %s", kind, exc)
    return choices


def input_device_choices() -> list[str]:
    return _audio_device_choices("input")


def output_device_choices() -> list[str]:
    return _audio_device_choices("output")


def parse_audio_device(choice: str | None) -> int | None:
    """Parse a dropdown label from input/output device choices."""
    if not choice or choice.startswith("Default "):
        return None
    bracket_match = re.search(r"\[(\d+)\]\s*$", choice)
    if bracket_match:
        return int(bracket_match.group(1))
    try:
        return int(choice.split(":", 1)[0].strip())
    except (TypeError, ValueError):
        return None


def parse_input_device(choice: str | None) -> int | None:
    return parse_audio_device(choice)


def parse_output_device(choice: str | None) -> int | None:
    return parse_audio_device(choice)
