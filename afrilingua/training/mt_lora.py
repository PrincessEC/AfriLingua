"""LoRA fine-tuning of NLLB-200 on AfriLingua parallel splits + CTranslate2 int8 export.

Fixes every v1 fine-tuning bug:

* sets **both** ``tokenizer.src_lang`` and ``tokenizer.tgt_lang`` per language pair
  so target BOS/EOS language tokens are encoded properly;
* passes ``processing_class=tokenizer`` on Transformers ≥ 5 (falls back to ``tokenizer=``
  on Transformers 4);
* works on both CPU ("overnight" small-rank LoRA defaults) and CUDA GPUs;
* optionally merges the LoRA adapter and exports an int8 CTranslate2 model straight
  into ``settings.ct2_cache``.

CLI::

    python -m afrilingua.training.mt_lora --data-dir data/processed --max-steps 200
    python -m afrilingua.training.mt_lora --dry-run
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from afrilingua.core.hardware import resolve_runtime
from afrilingua.core.settings import apply_runtime_environment, get_settings

logger = logging.getLogger(__name__)


@dataclass
class MTLoRAConfig:
    base_model: str = "facebook/nllb-200-distilled-600M"
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/finetuned/nllb-afrilingua")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    epochs: int = 3
    max_steps: int = -1
    max_length: int = 160
    export_ct2: bool = True
    dry_run: bool = False


def _load_split_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_parallel_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int = 160
) -> list[dict[str, Any]]:
    """Tokenize rows grouped by ``(src_code, tgt_code)`` so both language tags are exact."""
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        src_code = r.get("src_code") or r["src_lang"]
        tgt_code = r.get("tgt_code") or r["tgt_lang"]
        by_pair.setdefault((src_code, tgt_code), []).append(r)

    encoded: list[dict[str, Any]] = []
    for (src_code, tgt_code), group in by_pair.items():
        tokenizer.src_lang = src_code
        tokenizer.tgt_lang = tgt_code
        batch = tokenizer(
            [g["source"] for g in group],
            text_target=[g["target"] for g in group],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        for i in range(len(group)):
            encoded.append(
                {
                    "input_ids": batch["input_ids"][i],
                    "attention_mask": batch["attention_mask"][i],
                    "labels": batch["labels"][i],
                }
            )
    return encoded


def run_mt_lora(cfg: MTLoRAConfig) -> dict[str, Any]:
    """Run (or dry-run validate) NLLB LoRA fine-tuning."""
    settings = get_settings()
    apply_runtime_environment(settings)
    plan = resolve_runtime()

    train_rows = _load_split_rows(Path(cfg.data_dir) / "train.jsonl")
    val_rows = _load_split_rows(Path(cfg.data_dir) / "val.jsonl")

    if cfg.dry_run:
        return {
            "dry_run": True,
            "device": plan.device,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        }

    if not train_rows:
        raise ValueError(f"No training rows found in {Path(cfg.data_dir) / 'train.jsonl'}.")

    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Training dependencies missing. Install with: pip install -e '.[train]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, cache_dir=str(settings.hf_cache))
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, cache_dir=str(settings.hf_cache))

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(model, lora_cfg)

    train_ds = Dataset.from_list(tokenize_parallel_rows(train_rows, tokenizer, max_length=cfg.max_length))
    val_ds = (
        Dataset.from_list(tokenize_parallel_rows(val_rows, tokenizer, max_length=cfg.max_length))
        if val_rows
        else None
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = plan.device == "cuda"

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        fp16=use_cuda,
        use_cpu=not use_cuda,
        eval_strategy="epoch" if val_ds is not None and cfg.max_steps < 0 else "no",
        save_strategy="no",
        dataloader_num_workers=0,
        logging_steps=25,
        report_to="none",
    )

    trainer_kw: dict[str, Any] = {
        "model": peft_model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True),
    }
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kw)
    train_res = trainer.train()

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    ct2_out: str | None = None
    if cfg.export_ct2:
        try:
            import ctranslate2

            quant = "int8" if plan.device == "cpu" else plan.compute_type
            ct2_dir = settings.ct2_cache / f"nllb-afrilingua-{quant}"
            converter = ctranslate2.converters.TransformersConverter(str(out_dir))
            converter.convert(str(ct2_dir), quantization=quant, force=True)
            ct2_out = str(ct2_dir)
        except Exception:
            logger.exception("CTranslate2 export skipped")

    return {
        "output_dir": str(out_dir),
        "ct2_dir": ct2_out,
        "train_loss": getattr(train_res, "training_loss", None),
        "train_rows": len(train_rows),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune NLLB with LoRA on AfriLingua parallel data.")
    p.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("data/finetuned/nllb-afrilingua"))
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--no-ct2", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    ns = p.parse_args(argv)
    res = run_mt_lora(
        MTLoRAConfig(
            base_model=ns.base_model,
            data_dir=ns.data_dir,
            output_dir=ns.output_dir,
            max_steps=ns.max_steps,
            epochs=ns.epochs,
            lora_r=ns.lora_r,
            export_ct2=not ns.no_ct2,
            dry_run=ns.dry_run,
        )
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
