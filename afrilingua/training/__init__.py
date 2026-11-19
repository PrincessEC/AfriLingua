"""Community data ingestion, dataset building, LoRA/adapter fine-tuning and evaluation."""

from afrilingua.training.corpora import (
    COMMERCIAL_SAFE_LICENSES,
    CORPUS_DOMAINS,
    CORPUS_FORMATS,
    CORPUS_LICENSES,
    CorpusRecord,
    CorpusStore,
    FilterStats,
    filter_parallel_pairs,
    open_corpus_store,
)
from afrilingua.training.datasets import SplitSummary, build_training_splits
from afrilingua.training.evaluate import chrf_score, render_model_card, should_promote, wer_cer

__all__ = [
    "COMMERCIAL_SAFE_LICENSES",
    "CORPUS_DOMAINS",
    "CORPUS_FORMATS",
    "CORPUS_LICENSES",
    "CorpusRecord",
    "CorpusStore",
    "FilterStats",
    "SplitSummary",
    "build_training_splits",
    "chrf_score",
    "filter_parallel_pairs",
    "open_corpus_store",
    "render_model_card",
    "should_promote",
    "wer_cer",
]
