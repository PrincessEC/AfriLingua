"""Optional semantic sentence embedder for the translation memory.

Enabled only when ``settings.model_embedder`` is set (e.g.
``sentence-transformers/LaBSE``). The model is loaded lazily on first use with plain
``transformers`` (no ``sentence-transformers`` dependency): mean pooling over the
last hidden state, then L2 normalisation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TransformersEmbedder", "embedder_available"]


def embedder_available(model_id: str) -> bool:
    """True if ``model_id`` is already in the local model cache (no network access)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        from afrilingua.core.settings import get_settings

        found = try_to_load_from_cache(model_id, "config.json", cache_dir=str(get_settings().hf_cache))
        return isinstance(found, str)
    except Exception:
        return False


class TransformersEmbedder:
    """Mean-pooled, normalised sentence embeddings from a Hugging Face encoder."""

    def __init__(
        self, model_id: str, *, batch_size: int = 32, max_length: int = 128, device: str = "cpu"
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._lock = threading.Lock()
        self._tok: Any = None
        self._model: Any = None
        # E5 models expect a task prefix; "query: " is the symmetric-similarity form.
        self.prefix = "query: " if "e5" in model_id.lower() else ""

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        from afrilingua.core.settings import get_settings

        settings = get_settings()
        cache = str(settings.hf_cache)
        local = bool(settings.offline) or embedder_available(self.model_id)
        logger.info("Loading TM embedder %s", self.model_id)
        self._tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache, local_files_only=local)
        self._model = (
            AutoModel.from_pretrained(self.model_id, cache_dir=cache, local_files_only=local)
            .to(self.device)
            .eval()
        )

    def __call__(self, texts: list[str]) -> np.ndarray:
        import torch

        with self._lock:
            self._load()
            out: list[np.ndarray] = []
            for i in range(0, len(texts), self.batch_size):
                batch = [self.prefix + t for t in texts[i : i + self.batch_size]]
                enc = self._tok(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.inference_mode():
                    hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
