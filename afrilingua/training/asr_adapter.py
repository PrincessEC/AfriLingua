"""Fine-tune an MMS CTC language adapter on approved community voice recordings.

Follows the standard Meta MMS adapter recipe: freeze the 1B Wav2Vec2 backbone
and optimize only the ~2.5M adapter + LM-head parameters for the target language,
saving ``adapter.<mms_code>.safetensors``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from afrilingua.core.languages import get_registry
from afrilingua.core.settings import apply_runtime_environment, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ASRAdapterConfig:
    lang: str
    base_model: str = "facebook/mms-1b-all"
    output_dir: Path = Path("data/finetuned/mms-adapters")
    epochs: int = 5
    max_steps: int = -1
    learning_rate: float = 1e-3
    batch_size: int = 2
    dry_run: bool = False


def collect_approved_audio_records(contrib: Any, lang: str) -> list[dict[str, Any]]:
    code = get_registry().normalize(lang)
    records: list[dict[str, Any]] = []
    for rec in contrib.iter_approved(lang=code, kind="transcription"):
        rel = rec.get("audio_path")
        if not rel:
            continue
        full = Path(contrib.path) / rel
        if full.is_file() and full.suffix.lower() in {".flac", ".wav"}:
            records.append({"audio_path": str(full), "text": rec["target_text"], "lang": code})
    return records


def run_asr_adapter_training(cfg: ASRAdapterConfig, *, contrib: Any | None = None) -> dict[str, Any]:
    settings = get_settings()
    apply_runtime_environment(settings)
    spec = get_registry().get(cfg.lang)
    if not spec.mms_asr_code:
        raise ValueError(f"Language {spec.name!r} has no mms_asr_code configured.")

    if contrib is None:
        from afrilingua.contrib import open_contribution_store

        contrib = open_contribution_store()

    records = collect_approved_audio_records(contrib, spec.name)
    if cfg.dry_run:
        return {
            "dry_run": True,
            "lang": spec.name,
            "mms_code": spec.mms_asr_code,
            "approved_recordings": len(records),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        }

    if not records:
        raise ValueError(f"No approved FLAC/WAV recordings available for {spec.name!r}.")

    import soundfile as sf
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(
        cfg.base_model, target_lang=spec.mms_asr_code, cache_dir=str(settings.hf_cache)
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        cfg.base_model,
        target_lang=spec.mms_asr_code,
        ignore_mismatched_sizes=True,
        cache_dir=str(settings.hf_cache),
    )
    model.init_adapter_layers()
    model.freeze_base_model()
    for name, param in model.named_parameters():
        if "adapter" in name or "lm_head" in name:
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    model.train()

    steps = 0
    last_loss = 0.0
    for _epoch in range(cfg.epochs):
        for rec in records:
            audio, sr = sf.read(rec["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16_000:
                from afrilingua.audio.resample import resample_audio

                audio = resample_audio(audio, sr, 16_000)
            inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
            labels = processor(text=rec["text"], return_tensors="pt").input_ids
            out = model(input_values=inputs.input_values, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            last_loss = float(loss.detach().cpu().item())
            steps += 1
            if 0 < cfg.max_steps <= steps:
                break
        if 0 < cfg.max_steps <= steps:
            break

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = out_dir / f"adapter.{spec.mms_asr_code}.safetensors"
    adapter_weights = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "adapter" in k or "lm_head" in k
    }
    save_file(adapter_weights, str(adapter_file))
    return {
        "adapter_file": str(adapter_file),
        "steps": steps,
        "final_loss": round(last_loss, 4),
        "recordings": len(records),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune an MMS ASR language adapter on community audio.")
    p.add_argument("--lang", required=True)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    ns = p.parse_args(argv)
    res = run_asr_adapter_training(
        ASRAdapterConfig(lang=ns.lang, max_steps=ns.max_steps, epochs=ns.epochs, dry_run=ns.dry_run)
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
