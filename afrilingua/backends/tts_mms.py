"""MMS VITS TTS implementation."""
from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer, VitsModel

from afrilingua.backends.base import SpeechAudio, TTSBackend
from afrilingua.core.hardware import resolve_runtime

logger = logging.getLogger(__name__)


class MMSTTSBackend(TTSBackend):
    def __init__(self) -> None:
        self.name = "mms-tts"
        self.plan = resolve_runtime()
        self._models: dict[str, VitsModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}

    def load_language(self, lang: str, model_id: str) -> None:
        if lang in self._models:
            return

        logger.info("Loading TTS model %s for %s", model_id, lang)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id).to(self.plan.device)
        if self.plan.torch_dtype == "float16":
            model.half()

        self._tokenizers[lang] = tokenizer
        self._models[lang] = model

    def languages(self) -> list[str]:
        return ["english", "igbo", "yoruba", "hausa"]

    def synthesize(self, text: str, lang: str) -> SpeechAudio:
        # We need the language spec to get the model ID
        from afrilingua.core.languages import get_registry
        spec = get_registry().get(lang)
        if not spec.mms_tts_model:
            raise ValueError(f"No TTS model for {lang}")

        self.load_language(lang, spec.mms_tts_model)

        inputs = self._tokenizers[lang](text, return_tensors="pt").to(self.plan.device)

        with torch.inference_mode():
            waveform = self._models[lang](**inputs).waveform[0]

        return SpeechAudio(waveform.cpu().numpy(), 16000)

    def is_loaded(self) -> bool:
        return len(self._models) > 0

    def unload(self) -> None:
        self._models.clear()
        self._tokenizers.clear()
