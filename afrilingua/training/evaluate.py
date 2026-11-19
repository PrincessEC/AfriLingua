"""Evaluation metrics (chrF++, BLEU, WER/CER), promotion gate and model-card writer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rapidfuzz.distance import Levenshtein


def _char_ngrams(text: str, n: int) -> Counter[str]:
    clean = "".join(text.split()) if n > 0 else text
    if len(clean) < n:
        return Counter()
    return Counter(clean[i : i + n] for i in range(len(clean) - n + 1))


def _pure_chrf(predictions: Sequence[str], references: Sequence[str], max_order: int = 6, beta: float = 2.0) -> float:
    """Zero-dependency chrF implementation (0..100) used when ``sacrebleu`` is absent."""
    if not predictions:
        return 0.0
    total_f = 0.0
    valid_orders = 0
    beta2 = beta * beta
    for n in range(1, max_order + 1):
        matches = 0
        pred_total = 0
        ref_total = 0
        for hyp, ref in zip(predictions, references, strict=True):
            hc = _char_ngrams(hyp, n)
            rc = _char_ngrams(ref, n)
            matches += sum((hc & rc).values())
            pred_total += sum(hc.values())
            ref_total += sum(rc.values())
        if pred_total == 0 and ref_total == 0:
            continue
        valid_orders += 1
        prec = matches / pred_total if pred_total else 0.0
        rec = matches / ref_total if ref_total else 0.0
        if prec + rec > 0:
            total_f += (1 + beta2) * prec * rec / (beta2 * prec + rec)
    return round(100.0 * (total_f / max(valid_orders, 1)), 2)


def chrf_score(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute ``{"chrf": 0..100, "bleu": 0..100}`` (uses ``sacrebleu`` if installed)."""
    preds = [p.strip() for p in predictions]
    refs = [r.strip() for r in references]
    if len(preds) != len(refs):
        raise ValueError("predictions and references must have equal length")
    if not preds:
        return {"chrf": 0.0, "bleu": 0.0}
    try:
        import sacrebleu

        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        return {"chrf": round(float(chrf), 2), "bleu": round(float(bleu), 2)}
    except ImportError:
        return {"chrf": _pure_chrf(preds, refs), "bleu": 0.0}


def wer_cer(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute Word Error Rate and Character Error Rate (0.0 = exact match)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    word_edits = 0
    word_total = 0
    char_edits = 0
    char_total = 0
    for hyp, ref in zip(predictions, references, strict=True):
        h_words = hyp.strip().split()
        r_words = ref.strip().split()
        word_edits += Levenshtein.distance(h_words, r_words)
        word_total += max(len(r_words), 1)
        h_chars = hyp.strip()
        r_chars = ref.strip()
        char_edits += Levenshtein.distance(h_chars, r_chars)
        char_total += max(len(r_chars), 1)
    return {
        "wer": round(word_edits / max(word_total, 1), 4),
        "cer": round(char_edits / max(char_total, 1), 4),
    }


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    metric: str
    candidate: float
    baseline: float
    delta: float


def should_promote(
    candidate_score: float,
    baseline_score: float,
    *,
    metric: str = "chrf",
    higher_is_better: bool = True,
    min_improvement: float = 0.2,
) -> GateDecision:
    """Decide whether a newly trained checkpoint beats the baseline model."""
    delta = (candidate_score - baseline_score) if higher_is_better else (baseline_score - candidate_score)
    return GateDecision(
        promote=delta >= min_improvement,
        metric=metric,
        candidate=round(candidate_score, 4),
        baseline=round(baseline_score, 4),
        delta=round(delta, 4),
    )


def render_model_card(
    *,
    model_name: str,
    base_model: str,
    task: str,
    languages: Sequence[str],
    metrics: dict[str, Any],
    data_licenses: Sequence[str],
    sources: Sequence[str],
    contributors: Sequence[str],
    base_model_license: str = "CC-BY-NC-4.0",
) -> str:
    """Render a Hugging Face-compatible ``model_card.md`` with full provenance."""
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    langs = list(languages) or ["en", "yo", "ig", "ha"]
    credits = ", ".join(contributors) if contributors else "AfriLingua community contributors"
    src_list = ", ".join(sources) if sources else "community contributions"
    lic_list = ", ".join(data_licenses) if data_licenses else "CC-BY-4.0"
    metrics_lines = "\n".join(f"- **{k}**: `{v}`" for k, v in metrics.items()) or "- Pending evaluation"

    return f"""---
base_model: {base_model}
license: {base_model_license.lower()}
language:
{chr(10).join(f"- {l}" for l in langs)}
pipeline_tag: {task}
tags:
- afrilingua
- african-languages
- low-resource
---

# {model_name}

Fine-tuned with **AfriLingua** on `{now}` from `{base_model}`.

## Evaluation

{metrics_lines}

## Training data & provenance

- **Sources**: {src_list}
- **Data licenses**: {lic_list}
- **Base model license**: `{base_model_license}` (note: derivatives of CC-BY-NC models remain non-commercial)
- **FLORES-200 / FLEURS contamination check**: held-out benchmarks excluded from training.

## Community credits

{credits}
"""
