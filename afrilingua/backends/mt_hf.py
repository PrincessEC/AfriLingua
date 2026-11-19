"""
NLLB-200 translation via Hugging Face Transformers (PyTorch fallback).

Used when CTranslate2 is unavailable, or on CUDA machines where CT2 has no GPU
build. fp16 on CUDA, fp32 on CPU, ``torch.inference_mode`` and automatic
fallback to CPU on CUDA out-of-memory.
"""

from __future__ import annotations

import logging
from typing import Any

from afrilingua.backends._common import (
    configure_torch_threads,
    dtype_kwargs,
    empty_cuda_cache,
    hf_kwargs,
    is_oom,
    resolve_mt_source,
)
from afrilingua.core.hardware import RuntimePlan, resolve_runtime
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class HFNLLBBackend:
    """:class:`~afrilingua.backends.base.MTBackend` backed by Transformers."""

    name = "transformers-nllb"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        settings: Settings | None = None,
        plan: RuntimePlan | None = None,
        registry: LanguageRegistry | None = None,
        max_batch_size: int = 16,
    ) -> None:
        self.settings = settings or get_settings()
        self.plan = plan or resolve_runtime()
        self.registry = registry or get_registry()
        self.model_id = model_id or self.plan.model_nllb
        self.max_batch_size = max(1, max_batch_size)
        self.device = self.plan.device
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load tokenizer + model onto the planned device (CPU on CUDA OOM)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        source = resolve_mt_source(self.settings, self.model_id)
        kwargs = hf_kwargs(self.settings)
        self._tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        dtype_name = self.plan.torch_dtype if self.plan.device == "cuda" else "float32"
        model = AutoModelForSeq2SeqLM.from_pretrained(source, **kwargs, **dtype_kwargs(dtype_name))
        model.eval()
        self.device = self.plan.device
        if self.device == "cuda":
            try:
                model.to("cuda")
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                logger.warning("CUDA out of memory loading NLLB; falling back to CPU.")
                empty_cuda_cache()
                model.to(device="cpu", dtype=torch.float32)
                self.device = "cpu"
        if self.device == "cpu":
            configure_torch_threads(self.plan.threads)
        self._model = model
        logger.info("Transformers NLLB ready on %s from %s", self.device, source)

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        empty_cuda_cache()

    def _generate(self, texts: list[str], tgt_code: str, beam_size: int, max_length: int) -> list[str]:
        import torch

        tok = self._tokenizer
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1000)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
                num_beams=max(1, beam_size),
                max_new_tokens=max_length,
            )
        return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        beam_size: int = 1,
        max_length: int = 256,
    ) -> list[str]:
        """Translate ``texts`` in mini-batches of ``max_batch_size``."""
        if not texts:
            return []
        self.load()
        self._tokenizer.src_lang = self.registry.nllb_code(src_lang)
        tgt_code = self.registry.nllb_code(tgt_lang)
        outputs: list[str] = []
        for i in range(0, len(texts), self.max_batch_size):
            chunk = list(texts[i : i + self.max_batch_size])
            try:
                outputs.extend(self._generate(chunk, tgt_code, beam_size, max_length))
            except RuntimeError as exc:
                if self.device != "cuda" or not is_oom(exc):
                    raise
                import torch

                logger.warning("CUDA out of memory during translation; moving NLLB to CPU.")
                empty_cuda_cache()
                self._model.to(device="cpu", dtype=torch.float32)
                self.device = "cpu"
                configure_torch_threads(self.plan.threads)
                outputs.extend(self._generate(chunk, tgt_code, beam_size, max_length))
        return outputs
