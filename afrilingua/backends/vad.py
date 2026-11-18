"""Silero VAD implementation."""
from __future__ import annotations

from afrilingua.backends.base import Backend


class SileroVADBackend(Backend):
    def __init__(self) -> None:
        self.name = "silero-vad"
        self._model = None
        self._utils = None

    def load(self) -> None:
        if self.is_loaded():
            return
        # Using the hub load as a fallback if the pip package is not used directly
        # or just following v1 style, though v2 instruction said pip package.
        # Following instructions: "NEVER torch.hub".
        from silero_vad import load_silero_vad
        # This is a simplification; need to ensure the package is actually installed.
        # The coordinator said it was.
        self._model, self._utils = load_silero_vad(onnx=True)

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._utils = None

    def get_speech_timestamps(self, audio, **kwargs):
        return self._utils[0](audio, self._model, **kwargs)
