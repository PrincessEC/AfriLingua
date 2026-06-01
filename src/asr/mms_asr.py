"""
src/asr/mms_asr.py
Meta MMS ASR for Igbo, Yoruba, and Hausa.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.config import (
    DEVICE,
    HF_CACHE,
    MODEL_MMS_ASR,
    SAMPLE_RATE,
    TORCH_DTYPE,
)
from src.hf_compat import dtype_name_kwargs

logger = logging.getLogger(__name__)

# Lingua / internal name -> MMS language tag.
# Lingua / internal name -> MMS language tag.
MMS_LANG_MAP: dict[str, str] = {
    "igbo": "ibo",
    "igbo": "ibo",
    "yoruba": "yor",
    "hausa": "hau",
    "hausa": "hau",
}


class MMSASR:
    """
    Meta MMS speech -> text for African languages.

    MMS ASR requires one of the fine-tuned CTC checkpoints such as
    facebook/mms-1b-all. The pretrained facebook/mms-300m checkpoint does not
    include ASR adapters and cannot be used with load_adapter().

    Usage::
        asr = MMSASR()
        text = asr.transcribe(audio_np, lang="igbo")
    """

    def __init__(self) -> None:
        self._processor: Optional[Wav2Vec2Processor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None
        self._current_lang: Optional[str] = None

    def _validate_model_config(self) -> None:
        """Fail early when MODEL_MMS_ASR points at a non-ASR MMS checkpoint."""
        config = AutoConfig.from_pretrained(
            MODEL_MMS_ASR,
            cache_dir=str(HF_CACHE),
        )
        architectures = set(getattr(config, "architectures", []) or [])
        adapter_dim = getattr(config, "adapter_attn_dim", None)
        if adapter_dim is None or (
            architectures and "Wav2Vec2ForCTC" not in architectures
        ):
            raise ValueError(
                f"{MODEL_MMS_ASR!r} is not a fine-tuned MMS ASR checkpoint. "
                "Set MODEL_MMS_ASR=facebook/mms-1b-all. "
                "The facebook/mms-300m checkpoint is pretrained only and does "
                "not include the ASR adapters needed for Igbo/Yoruba/Hausa."
            )

    def _validate_model_config(self) -> None:
        """Fail early when MODEL_MMS_ASR points at a non-ASR MMS checkpoint."""
        config = AutoConfig.from_pretrained(
            MODEL_MMS_ASR,
            cache_dir=str(HF_CACHE),
        )
        architectures = set(getattr(config, "architectures", []) or [])
        adapter_dim = getattr(config, "adapter_attn_dim", None)
        if adapter_dim is None or (
            architectures and "Wav2Vec2ForCTC" not in architectures
        ):
            raise ValueError(
                f"{MODEL_MMS_ASR!r} is not a fine-tuned MMS ASR checkpoint. "
                "Set MODEL_MMS_ASR=facebook/mms-1b-all. "
                "The facebook/mms-300m checkpoint is pretrained only and does "
                "not include the ASR adapters needed for Igbo/Yoruba/Hausa."
            )

    def _load(self) -> None:
        if self._model is not None:
            return

        default_lang = MMS_LANG_MAP["igbo"]
        logger.info(
            "Loading MMS ASR processor + model on %s from %s ...",
            DEVICE,
            MODEL_MMS_ASR,
        )
        self._validate_model_config()
        self._processor = AutoProcessor.from_pretrained(
            MODEL_MMS_ASR,
            target_lang=default_lang,
            target_lang=default_lang,
            cache_dir=str(HF_CACHE),
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_MMS_ASR,
            ignore_mismatched_sizes=True,
            ignore_mismatched_sizes=True,
            cache_dir=str(HF_CACHE),
            **dtype_name_kwargs(TORCH_DTYPE),
        )
        self._model.to(DEVICE)
        self._model.eval()
        self._switch_lang(default_lang)
        logger.info("MMS ASR ready.")
        self._switch_lang(default_lang)
        logger.info("MMS ASR ready.")

    def _switch_lang(self, mms_lang: str) -> None:
        """Load a language adapter and switch the tokenizer vocabulary."""
        """Load a language adapter and switch the tokenizer vocabulary."""
        if mms_lang == self._current_lang:
            return

        assert self._model is not None
        assert self._processor is not None
        tokenizer = self._processor.tokenizer
        if not hasattr(tokenizer, "set_target_lang"):
            raise ValueError(
                f"{MODEL_MMS_ASR!r} does not expose MMS language tokenizers. "
                "Use MODEL_MMS_ASR=facebook/mms-1b-all."
            )

        tokenizer.set_target_lang(mms_lang)
        try:
            self._model.load_adapter(mms_lang, cache_dir=str(HF_CACHE))
            self._model.to(DEVICE)
        except ValueError as exc:
            raise ValueError(
                f"Could not load MMS ASR adapter {mms_lang!r} from "
                f"{MODEL_MMS_ASR!r}. Use a fine-tuned MMS ASR checkpoint such "
                "as facebook/mms-1b-all."
            ) from exc
        self._current_lang = mms_lang

    def transcribe(
        self,
        audio: np.ndarray,
        lang: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """
        Transcribe audio using MMS.

        Args:
            audio: Float32 numpy array, shape (N,), values in [-1, 1].
            lang: Language name: "igbo", "yoruba", or "hausa".
            audio: Float32 numpy array, shape (N,), values in [-1, 1].
            lang: Language name: "igbo", "yoruba", or "hausa".
            sample_rate: Source sample rate; must be 16000 Hz (auto-resampled).

        Returns:
            Transcribed text string.
        """
        self._load()

        if audio is None or len(audio) == 0:
            return ""

        mms_lang = MMS_LANG_MAP.get(lang.lower())
        if mms_lang is None:
            raise ValueError(
                f"Unsupported language for MMS ASR: {lang!r}. "
                f"Unsupported language for MMS ASR: {lang!r}. "
                f"Supported: {list(MMS_LANG_MAP.keys())}"
            )

        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            import torchaudio.functional as F


            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = F.resample(audio_t, sample_rate, 16000)
            audio = audio_t.squeeze(0).numpy()

        self._switch_lang(mms_lang)

        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1).detach().cpu()
        transcription = self._processor.decode(predicted_ids[0])
        return transcription.strip()

    def is_loaded(self) -> bool:
        return self._model is not None
