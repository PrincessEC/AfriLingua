"""
Shared helpers for concrete backends.

Everything here is import-light: heavy libraries (torch, transformers,
huggingface_hub) are imported inside functions only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from afrilingua.core.errors import ModelLoadError, OfflineModelMissingError
from afrilingua.core.settings import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def module_available(name: str) -> bool:
    """Return True if ``name`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def transformers_major() -> int:
    """Major version of the installed ``transformers`` package (4 if unknown)."""
    try:
        import transformers

        return int(transformers.__version__.split(".", 1)[0])
    except Exception:
        return 4


def dtype_kwargs(dtype_name: str) -> dict[str, Any]:
    """``from_pretrained`` dtype kwarg for the installed transformers version.

    Transformers >= 5 uses ``dtype=``; 4.x uses ``torch_dtype=``.
    """
    import torch

    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    key = "dtype" if transformers_major() >= 5 else "torch_dtype"
    return {key: dtype}


def hf_kwargs(settings: Settings) -> dict[str, Any]:
    """Common ``from_pretrained`` kwargs: cache location + offline mode."""
    return {"cache_dir": str(settings.hf_cache), "local_files_only": bool(settings.offline)}


def configure_torch_threads(threads: int) -> None:
    """Set intra-op threads for torch on CPU (best effort, once per process)."""
    if threads <= 0:
        return
    try:
        import torch

        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not set torch threads", exc_info=True)


def is_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors."""
    msg = str(exc).lower()
    return "out of memory" in msg or type(exc).__name__ == "OutOfMemoryError"


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def slugify(value: str) -> str:
    """Filesystem-safe slug for model ids (``facebook/nllb`` → ``facebook--nllb``)."""
    value = value.strip().replace("\\", "/").rstrip("/")
    value = value.replace("/", "--")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def snapshot_path(
    model_id: str,
    settings: Settings,
    *,
    allow_patterns: list[str] | None = None,
    required_any: tuple[str, ...] = (),
) -> Path:
    """Return a local directory for ``model_id`` (download if allowed).

    Local directories are returned unchanged. For hub ids the HF cache is
    tried first (no network); a download is attempted only when not offline.
    If ``required_any`` is given, a snapshot only counts as present when at
    least one of those files exists in it.
    """
    local = Path(model_id)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    def _ok(p: Path) -> bool:
        return not required_any or any((p / f).exists() for f in required_any)

    try:
        path = Path(
            snapshot_download(
                model_id, cache_dir=str(settings.hf_cache), local_files_only=True, allow_patterns=allow_patterns
            )
        )
        if _ok(path):
            return path
    except Exception:
        pass
    if settings.offline:
        raise OfflineModelMissingError(model_id)
    logger.info("Downloading %s into %s ...", model_id, settings.hf_cache)
    try:
        path = Path(snapshot_download(model_id, cache_dir=str(settings.hf_cache), allow_patterns=allow_patterns))
    except Exception as exc:
        raise ModelLoadError(f"Could not download {model_id!r}: {exc}") from exc
    if not _ok(path):
        raise ModelLoadError(f"{model_id!r} has none of the expected files {list(required_any)}")
    return path


# ── NLLB checkpoint resolution ──────────────────────────────────────────────

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def is_valid_hf_checkpoint(path: Path) -> bool:
    """A usable Transformers checkpoint has ``config.json`` and weights.

    (v1 loaded *any* non-empty directory, e.g. one holding only training logs.)
    """
    return path.is_dir() and (path / "config.json").is_file() and any((path / f).is_file() for f in _WEIGHT_FILES)


def is_ct2_model_dir(path: Path) -> bool:
    """A CTranslate2 export contains ``model.bin`` and ``config.json``."""
    return path.is_dir() and (path / "model.bin").is_file() and (path / "config.json").is_file()


def resolve_mt_source(settings: Settings, default_model_id: str) -> str:
    """Fine-tuned checkpoint dir if it is a valid checkpoint, else the hub id."""
    ft = settings.finetuned_model_dir
    if is_valid_hf_checkpoint(ft):
        logger.info("Using fine-tuned NLLB checkpoint at %s", ft)
        return str(ft)
    if ft.is_dir() and any(ft.iterdir()):
        logger.warning(
            "FINETUNED_MODEL_DIR=%s exists but is not a complete checkpoint (needs config.json + weights); "
            "using %s instead.",
            ft,
            default_model_id,
        )
    return default_model_id


def checkpoint_fingerprint(path: Path) -> str:
    """Short hash of a local checkpoint's weight files (size + mtime)."""
    h = hashlib.sha1()
    for name in sorted(_WEIGHT_FILES + ("config.json",)):
        f = path / name
        if f.is_file():
            st = f.stat()
            h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:10]


def ct2_device_for(plan_device: str) -> str:
    """CTranslate2/faster-whisper device: fall back to CPU when CT2 sees no GPU."""
    if plan_device != "cuda":
        return "cpu"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    logger.warning("Runtime plan requests CUDA but CTranslate2 sees no CUDA device; using CPU int8 for CT2.")
    return "cpu"


def ct2_compute_type(device: str, plan_compute_type: str) -> str:
    """Compute type valid for ``device`` (int8 on CPU unless float32 was forced)."""
    if device == "cpu":
        return "float32" if plan_compute_type == "float32" else "int8"
    return plan_compute_type if plan_compute_type in {"float16", "int8_float16", "float32", "int8"} else "float16"
