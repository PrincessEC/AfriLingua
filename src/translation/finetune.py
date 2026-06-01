"""
src/translation/finetune.py
Fine-tunes facebook/nllb-200-distilled-600M on the processed parallel corpus
for English ↔ Igbo / Yoruba / Hausa.

Output checkpoint: data/finetuned/nllb-afrilingua/

Usage (from project root):
    python src/translation/finetune.py

Training is CPU-only, fp32. This will take several hours on large corpora;
use --max_train_samples to run a quick smoke-test.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

from src.config import (
    DATA_PROCESSED_DIR,
    FINETUNED_MODEL_DIR,
    HF_CACHE,
    MODEL_NLLB,
)
from src.hf_compat import fp32_dtype_kwargs

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE        = 4
NUM_EPOCHS        = 3
LEARNING_RATE     = 5e-5


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_splits(max_train_samples: int | None = None):
    train_records = _load_jsonl(DATA_PROCESSED_DIR / "train.jsonl")
    val_records   = _load_jsonl(DATA_PROCESSED_DIR / "val.jsonl")

    if max_train_samples:
        train_records = train_records[:max_train_samples]

    train_ds = Dataset.from_list(train_records)
    val_ds   = Dataset.from_list(val_records)
    return train_ds, val_ds


# ── Tokenisation ──────────────────────────────────────────────────────────────

def make_preprocess_fn(tokenizer: AutoTokenizer):
    def preprocess(examples):
        # Set tokenizer source language per example
        # (HuggingFace handles batches with a single src_lang — we preprocess
        # each language pair separately using a map call per unique src_lang.)
        inputs = tokenizer(
            examples["english"],
            text_target=examples["translation"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        return inputs
    return preprocess


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, src_lang: str) -> Dataset:
    tokenizer.src_lang = src_lang
    ds = ds.filter(lambda x: x["src_lang"] == src_lang)
    return ds.map(
        make_preprocess_fn(tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )


# ── BLEU metric ───────────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer: AutoTokenizer):
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": round(result["score"], 2)}

    return compute_metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train(max_train_samples: int | None = None) -> None:
    logger.info("Loading base model: %s", MODEL_NLLB)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NLLB, cache_dir=str(HF_CACHE))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NLLB,
        cache_dir=str(HF_CACHE),
        **fp32_dtype_kwargs(),
    )

    train_ds, val_ds = load_splits(max_train_samples)

    if len(train_ds) == 0:
        logger.error("Training set is empty. Run data/collect_data.py and data/preprocess.py first.")
        return

    # Tokenise for each source language separately and concatenate
    from datasets import concatenate_datasets
    tokenized_train_parts = []
    tokenized_val_parts   = []

    for src_lang in train_ds.unique("src_lang"):
        tokenized_train_parts.append(tokenize_dataset(train_ds, tokenizer, src_lang))
    for src_lang in val_ds.unique("src_lang"):
        tokenized_val_parts.append(tokenize_dataset(val_ds, tokenizer, src_lang))

    tokenized_train = concatenate_datasets(tokenized_train_parts)
    tokenized_val   = concatenate_datasets(tokenized_val_parts)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(FINETUNED_MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=200,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=False,                 # CPU: no fp16
        dataloader_num_workers=0,   # safer on Windows
        logging_steps=50,
        report_to="none",           # no wandb/tensorboard required
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning …")
    trainer.train()

    logger.info("Saving fine-tuned model to %s", FINETUNED_MODEL_DIR)
    trainer.save_model(str(FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))
    logger.info("Fine-tuning complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB for AfriLingua")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training set size (useful for smoke-testing, e.g. --max_train_samples 500)",
    )
    args = parser.parse_args()
    train(max_train_samples=args.max_train_samples)
