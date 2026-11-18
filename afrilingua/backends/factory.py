"""Backend registration and factory."""
from __future__ import annotations
from typing import cast
from pathlib import Path

from afrilingua.models.manager import ModelManager
from afrilingua.backends.mt_ct2 import CT2NLLBBackend
from afrilingua.backends.asr_whisper import FasterWhisperBackend
from afrilingua.backends.asr_mms import MMSASRBackend
from afrilingua.backends.tts_mms import MMSTTSBackend
from afrilingua.backends.vad import SileroVADBackend
from afrilingua.backends.lid import HeuristicLID
from afrilingua.core.settings import get_settings

def register_default_backends(manager: ModelManager) -> None:
    settings = get_settings()
    
    # Registration
    model_dir = str(settings.finetuned_model_dir)
    manager.register("mt", lambda: CT2NLLBBackend(model_dir))
    manager.register("asr.en", lambda: FasterWhisperBackend(settings.model_whisper or "small"))
    manager.register("asr.mms", lambda: MMSASRBackend(settings.model_mms_asr))
    manager.register("tts", lambda: MMSTTSBackend())
    manager.register("vad", lambda: SileroVADBackend())

def build_detector() -> HeuristicLID:
    return HeuristicLID()
