"""CTranslate2 NLLB implementation (int8 on CPU, float16/int8 on GPU)."""
from __future__ import annotations

import logging
from typing import Any

import ctranslate2
from transformers import AutoTokenizer

from afrilingua.backends.base import MTBackend
from afrilingua.core.hardware import resolve_runtime
from afrilingua.core.languages import get_registry

logger = logging.getLogger(__name__)


class CT2NLLBBackend(MTBackend):
    def __init__(self, model_dir: str) -> None:
        self.name = "ct2-nllb"
        self.model_dir = model_dir
        self.plan = resolve_runtime()
        self.registry = get_registry()
        self._translator: ctranslate2.Translator | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        if self.is_loaded():
            return
        
        # Determine device/compute from plan
        device = self.plan.device
        compute_type = self.plan.compute_type
        
        logger.info("Loading CT2 NLLB from %s (device=%s, compute=%s)", self.model_dir, device, compute_type)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, src_lang="eng_Latn", tgt_lang="eng_Latn")
        self._translator = ctranslate2.Translator(
            self.model_dir,
            device=device,
            compute_type=compute_type,
            intra_threads=self.plan.threads,
        )

    def is_loaded(self) -> bool:
        return self._translator is not None

    def unload(self) -> None:
        self._translator = None
        self._tokenizer = None

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        beam_size: int = 1,
        max_length: int = 256,
    ) -> list[str]:
        if not self.is_loaded():
            self.load()
        
        # Map canonical registry names to NLLB codes
        src_code = self.registry.nllb_code(src_lang)
        tgt_code = self.registry.nllb_code(tgt_lang)
        
        # NLLB requires source language prefix
        self._tokenizer.src_lang = src_code
        source_inputs = [f"{src_code} {t}" for t in texts]
        
        # Tokenize
        tokens = self._tokenizer(source_inputs, padding=True, truncation=True, return_tensors="np")
        input_ids = tokens["input_ids"]
        
        # Translate
        results = self._translator.translate_batch(
            input_ids,
            beam_size=beam_size,
            max_decoding_length=max_length,
            target_prefix=[[tgt_code]] * len(texts),
        )
        
        # Decode
        output_ids = [r.hypotheses[0] for r in results]
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
