"""
Pre-fetch every model a deployment needs, so the machine can then run offline.

``pull_models()`` downloads (idempotently) into ``HF_CACHE``:

* NLLB for the profile + its CTranslate2 conversion (if ctranslate2 is installed)
* faster-whisper (or openai-whisper) for Whisper languages
* MMS ASR base weights + only the adapters for the selected languages
* MMS TTS voices for the selected languages
* and verifies the Silero VAD model bundled with the ``silero-vad`` package.

With ``AFRILINGUA_OFFLINE=1`` nothing is downloaded; the summary reports
which models are present and which are missing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from afrilingua.backends._common import ct2_compute_type, ct2_device_for, module_available, snapshot_path
from afrilingua.core.hardware import RuntimePlan, detect_hardware, plan_for
from afrilingua.core.languages import get_registry
from afrilingua.core.settings import Settings, apply_runtime_environment, get_settings

logger = logging.getLogger(__name__)

PullProgress = Callable[[int, int, str], None]


@dataclass
class PullItem:
    kind: str
    model: str
    status: str = "pending"  # ok | missing | error | skipped
    path: str = ""
    seconds: float = 0.0
    error: str = ""


def _plan(settings: Settings, profile: str | None) -> RuntimePlan:
    if profile and profile != "auto":
        settings = settings.model_copy(update={"profile": profile})
    return plan_for(settings, detect_hardware())


def _steps(settings: Settings, plan: RuntimePlan, langs: Sequence[str]) -> list[tuple[PullItem, Callable[[], str]]]:
    from afrilingua.backends.asr_whisper import faster_whisper_repo
    from afrilingua.backends.mt_ct2 import ensure_ct2_model, local_nllb_checkpoint

    reg = get_registry()
    specs = [reg.get(lang) for lang in langs]
    steps: list[tuple[PullItem, Callable[[], str]]] = []

    steps.append((PullItem("mt", plan.model_nllb), lambda: str(local_nllb_checkpoint(plan.model_nllb, settings))))
    if module_available("ctranslate2") and settings.mt_backend != "transformers":
        quant = ct2_compute_type(ct2_device_for(plan.device), plan.compute_type)
        steps.append(
            (
                PullItem("mt-ct2", f"{plan.model_nllb} ({quant})"),
                lambda: str(ensure_ct2_model(plan.model_nllb, quant, settings)),
            )
        )

    if any(s.whisper_lang for s in specs):
        if module_available("faster_whisper") and settings.asr_en_backend != "openai-whisper":
            repo = faster_whisper_repo(plan.model_whisper)
            steps.append((PullItem("asr-whisper", repo), lambda: str(snapshot_path(repo, settings))))
        else:
            steps.append(
                (PullItem("asr-whisper", f"openai-whisper/{plan.model_whisper}"), lambda: _openai_whisper(settings, plan))
            )

    codes = [s.mms_asr_code for s in specs if s.mms_asr_code]
    if codes:
        patterns = ["*.json", "model.safetensors", *(f"adapter.{c}.safetensors" for c in codes)]
        steps.append(
            (
                PullItem("asr-mms", f"{plan.model_mms_asr} [{', '.join(codes)}]"),
                lambda: str(
                    snapshot_path(
                        plan.model_mms_asr,
                        settings,
                        allow_patterns=patterns,
                        required_any=tuple(f"adapter.{c}.safetensors" for c in codes),
                    )
                ),
            )
        )
        # an existing snapshot may lack newly requested adapters → fetch each individually
        for code in codes:
            steps.append(
                (PullItem("asr-mms-adapter", f"{plan.model_mms_asr}:{code}"), _adapter_step(settings, plan, code))
            )

    for spec in specs:
        model_id = (
            settings.model_mms_tts_english or spec.mms_tts_model if spec.name == "english" else spec.mms_tts_model
        )
        if model_id:
            steps.append(
                (
                    PullItem("tts", f"{model_id} ({spec.name})"),
                    lambda m=model_id: str(snapshot_path(m, settings, required_any=("config.json",))),
                )
            )

    steps.append((PullItem("vad", "silero-vad (bundled)"), _silero))
    return steps


def _adapter_step(settings: Settings, plan: RuntimePlan, code: str) -> Callable[[], str]:
    def run() -> str:
        from huggingface_hub import hf_hub_download

        fname = f"adapter.{code}.safetensors"
        local = Path(plan.model_mms_asr) / fname
        if local.is_file():
            return str(local)
        try:
            return hf_hub_download(plan.model_mms_asr, fname, cache_dir=str(settings.hf_cache), local_files_only=True)
        except Exception:
            if settings.offline:
                raise
        return hf_hub_download(plan.model_mms_asr, fname, cache_dir=str(settings.hf_cache))

    return run


def _openai_whisper(settings: Settings, plan: RuntimePlan) -> str:
    root = settings.hf_cache / "whisper"
    target = root / f"{plan.model_whisper}.pt"
    if target.is_file():
        return str(target)
    if settings.offline:
        raise FileNotFoundError(str(target))
    import whisper

    url = whisper._MODELS[plan.model_whisper]
    return str(whisper._download(url, str(root), in_memory=False))


def _silero() -> str:
    from afrilingua.backends.vad import SileroVADBackend

    backend = SileroVADBackend()
    backend.new_stream()
    return "onnx" if backend.onnx else "torchscript"


def pull_models(
    profile: str | None = None,
    langs: Sequence[str] | None = None,
    on_progress: PullProgress | None = None,
) -> dict[str, Any]:
    """Download / convert / verify the models for ``profile`` and ``langs``.

    Idempotent: cached models are detected without network access. Returns a
    summary ``{"profile", "device", "languages", "offline", "ok", "items": [...]}``
    where each item has ``kind, model, status (ok|missing|error), path, seconds, error``.
    ``on_progress(done, total, message)`` is called before each step and at the end.
    """
    settings = get_settings()
    apply_runtime_environment(settings)
    plan = _plan(settings, profile)
    reg = get_registry()
    selected = [reg.normalize(lang) for lang in langs] if langs else reg.names()
    steps = _steps(settings, plan, selected)
    total = len(steps)
    items: list[PullItem] = []
    for i, (item, run) in enumerate(steps):
        if on_progress:
            on_progress(i, total, f"{item.kind}: {item.model}")
        t0 = time.perf_counter()
        try:
            item.path = run()
            item.status = "ok"
        except Exception as exc:
            offline_missing = settings.offline or type(exc).__name__ in {
                "OfflineModelMissingError",
                "LocalEntryNotFoundError",
            }
            item.status = "missing" if offline_missing else "error"
            item.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Model pull %s %s failed: %s", item.kind, item.model, exc)
        item.seconds = round(time.perf_counter() - t0, 2)
        items.append(item)
    if on_progress:
        on_progress(total, total, "Done")
    return {
        "profile": plan.profile,
        "device": plan.device,
        "languages": selected,
        "offline": settings.offline,
        "hf_cache": str(settings.hf_cache),
        "ct2_cache": str(settings.ct2_cache),
        "ok": all(i.status == "ok" for i in items),
        "items": [asdict(i) for i in items],
    }
