import pytest

import src.asr.mms_asr as mms_asr


class _Config:
    def __init__(self, architectures, adapter_attn_dim):
        self.architectures = architectures
        self.adapter_attn_dim = adapter_attn_dim


def test_mms_asr_rejects_pretraining_checkpoint(monkeypatch):
    monkeypatch.setattr(mms_asr, "MODEL_MMS_ASR", "facebook/mms-300m")
    monkeypatch.setattr(
        mms_asr.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _Config(["Wav2Vec2ForPreTraining"], None),
    )

    with pytest.raises(ValueError, match="fine-tuned MMS ASR checkpoint"):
        mms_asr.MMSASR()._validate_model_config()


def test_mms_asr_accepts_adapter_ctc_checkpoint(monkeypatch):
    monkeypatch.setattr(mms_asr, "MODEL_MMS_ASR", "facebook/mms-1b-all")
    monkeypatch.setattr(
        mms_asr.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _Config(["Wav2Vec2ForCTC"], 16),
    )

    mms_asr.MMSASR()._validate_model_config()
