"""Silero VAD implementation."""
from __future__ import annotations
from typing import Any
from afrilingua.backends.base import Backend

class SileroVADBackend(Backend):
    def __init__(self) -> None:
        self.name = "silero-vad"
        self._model: Any = None
        self._utils: Any = None

    def load(self) -> None:
        if self.is_loaded():
            return
        # Assuming the package is installed as per coordinator
        from silero_vad import load_silero_vad
        self._model, self._utils = load_silero_vad(onnx=True)

    def is_loaded(self) -> bool:
        return self._model is not None and self._utils is not None

    def unload(self) -> None:
        self._model = None
        self._utils = None
        
    def get_speech_timestamps(self, audio: Any, **kwargs: Any) -> Any:
        if not self.is_loaded():
            self.load()
        return self._utils[0](audio, self._model, **kwargs)
