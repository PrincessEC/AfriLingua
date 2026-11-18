"""Backend registration and factory."""
from __future__ import annotations

from afrilingua.backends.asr_mms import MMSASRBackend
from afrilingua.backends.asr_whisper import FasterWhisperBackend
from afrilingua.backends.lid import HeuristicLID
from afrilingua.backends.mt_ct2 import CT2NLLBBackend
from afrilingua.backends.tts_mms import MMSTTSBackend
from afrilingua.backends.vad import SileroVADBackend
from afrilingua.core.settings import get_settings
from afrilingua.models.manager import ModelManager


def register_default_backends(manager: ModelManager) -> None:
    settings = get_settings()

    # Registration
    manager.register("mt", lambda: CT2NLLBBackend(settings.finetuned_model_dir or "facebook/nllb-200-distilled-600M"))
    manager.register("asr.en", lambda: FasterWhisperBackend(settings.model_whisper or "small"))
    manager.register("asr.mms", lambda: MMSASRBackend(settings.model_mms_asr))
    manager.register("tts", lambda: MMSTTSBackend())
    manager.register("vad", lambda: SileroVADBackend())

def build_detector() -> HeuristicLID:
    return HeuristicLID()
