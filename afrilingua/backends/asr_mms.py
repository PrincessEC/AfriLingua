"""MMS ASR implementation with lazy adapter loading."""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCTC, AutoProcessor

from afrilingua.backends.base import ASRBackend, Transcript
from afrilingua.core.hardware import resolve_runtime

logger = logging.getLogger(__name__)


class MMSASRBackend(ASRBackend):
    def __init__(self, model_id: str) -> None:
        self.name = "mms-asr"
        self.model_id = model_id
        self.plan = resolve_runtime()
        self._model: AutoModelForCTC | None = None
        self._processor: AutoProcessor | None = None
        self._loaded_adapter: str | None = None

    def load(self) -> None:
        if self.is_loaded():
            return

        logger.info("Loading MMS base model %s", self.model_id)

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.plan.device)
        if self.plan.torch_dtype == "float16":
            self._model.half()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded_adapter = None

    def _switch_adapter(self, lang: str) -> None:
        if self._loaded_adapter == lang:
            return

        logger.info("Switching MMS adapter to %s", lang)
        self._model.load_adapter(lang)
        self._loaded_adapter = lang

    def languages(self) -> list[str]:
        return ["igbo", "yoruba", "hausa"]

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        if not self.is_loaded():
            self.load()

        self._switch_adapter(lang)

        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt").to(self.plan.device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._processor.batch_decode(predicted_ids)[0]

        return Transcript(text=transcription, language=lang)
