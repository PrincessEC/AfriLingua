"""
Silero voice-activity detection via the ``silero-vad`` pip package.

The model ships inside the package (no ``torch.hub``, no GitHub, no remote
code). The ONNX build is used when onnxruntime is installed, otherwise the
TorchScript build.

Silero is *stateful* (recurrent state over the stream), so one shared model
cannot serve several concurrent audio streams. The backend registered in the
ModelManager is therefore a light factory: :meth:`SileroVADBackend.new_stream`
returns an independent :class:`VADStream` per live session.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from afrilingua.backends._common import module_available
from afrilingua.core.errors import ModelLoadError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512  # Silero's fixed window at 16 kHz (32 ms)


def _load_model(onnx: bool) -> Any:
    if not module_available("silero_vad"):
        raise ModelLoadError("silero-vad is not installed (pip install 'afrilingua[fast]').")
    from silero_vad import load_silero_vad

    try:
        return load_silero_vad(onnx=onnx)
    except Exception as exc:
        if onnx:
            logger.warning("Silero ONNX load failed (%s); using the TorchScript model.", exc)
            return load_silero_vad(onnx=False)
        raise


class VADStream:
    """Per-stream Silero state: call with 512-sample float32 windows → speech probability."""

    window_samples = WINDOW_SAMPLES

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, window: np.ndarray) -> float:
        import torch

        x = np.ascontiguousarray(window, dtype=np.float32).reshape(-1)
        if x.size != WINDOW_SAMPLES:
            x = np.pad(x, (0, max(0, WINDOW_SAMPLES - x.size)))[:WINDOW_SAMPLES]
        with torch.inference_mode():
            return float(self._model(torch.from_numpy(x), SAMPLE_RATE).item())

    def reset(self) -> None:
        """Clear the recurrent state (start of a new, unrelated stream)."""
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()


class SileroVADBackend:
    """Backend-protocol wrapper; :meth:`new_stream` gives each session its own state."""

    name = "silero-vad"

    def __init__(self, *, onnx: bool | None = None) -> None:
        self.onnx = module_available("onnxruntime") if onnx is None else onnx
        self._loaded = False

    def load(self) -> None:
        """Verify the model loads (fails fast with a clear error if not)."""
        if self._loaded:
            return
        _load_model(self.onnx)
        self._loaded = True
        logger.info("Silero VAD ready (%s)", "onnx" if self.onnx else "torchscript")

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def new_stream(self) -> VADStream:
        """A fresh, independent VAD state for one audio stream."""
        self.load()
        return VADStream(_load_model(self.onnx))

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """Per-window speech probabilities for a whole clip (fresh state)."""
        stream = self.new_stream()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = len(audio) // WINDOW_SAMPLES
        return np.array([stream(audio[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]) for i in range(n)], dtype=np.float32)
