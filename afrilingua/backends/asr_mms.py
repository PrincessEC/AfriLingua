"""
Meta MMS speech recognition (``facebook/mms-1b-all`` + per-language adapters).

* One shared wav2vec2 CTC model; switching language swaps a ~2 M-parameter
  adapter and the CTC vocabulary. Adapter weights are cached in memory after
  the first load so later switches are a cheap ``load_state_dict``.
* Adapter switching mutates global model state, so it lives entirely inside
  this backend; the :class:`~afrilingua.models.manager.ModelManager` lock
  serializes calls.
* fp16 on CUDA; on CPU optional dynamic int8 quantization of the encoder's
  Linear layers (adapters and the CTC head stay float so they can be swapped).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from afrilingua.backends._common import (
    configure_torch_threads,
    dtype_kwargs,
    empty_cuda_cache,
    hf_kwargs,
    is_oom,
)
from afrilingua.backends.base import Segment, Transcript
from afrilingua.core.errors import CapabilityUnavailableError, ModelLoadError
from afrilingua.core.hardware import RuntimePlan, resolve_runtime
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_CHUNK_S = 30.0  # long inputs are processed in windows to bound memory


def validate_mms_config(config: Any, model_id: str) -> None:
    """Fail early when ``model_id`` is not a fine-tuned MMS CTC checkpoint with adapters.

    ``facebook/mms-300m`` / ``mms-1b`` are pretraining-only checkpoints without
    the ASR adapters required for per-language recognition.
    """
    architectures = set(getattr(config, "architectures", None) or [])
    adapter_dim = getattr(config, "adapter_attn_dim", None)
    if adapter_dim is None or (architectures and "Wav2Vec2ForCTC" not in architectures):
        raise ModelLoadError(
            f"{model_id!r} is not a fine-tuned MMS ASR checkpoint. Set MODEL_MMS_ASR=facebook/mms-1b-all. "
            "Pretraining-only checkpoints such as facebook/mms-300m do not include the ASR adapters "
            "needed for Igbo/Yoruba/Hausa."
        )


class MMSASRBackend:
    """:class:`~afrilingua.backends.base.ASRBackend` for MMS-supported languages."""

    name = "mms-asr"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        settings: Settings | None = None,
        plan: RuntimePlan | None = None,
        registry: LanguageRegistry | None = None,
        quantize: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.plan = plan or resolve_runtime()
        self.registry = registry or get_registry()
        self.model_id = model_id or self.plan.model_mms_asr
        # Dynamic int8 on CPU: ~2x faster encoder; adapters/lm_head kept in float.
        self.quantize = (self.plan.device == "cpu") if quantize is None else quantize
        self.device = self.plan.device
        self.dtype_name = "float32"
        self.quantized = False
        self._model: Any = None
        self._processor: Any = None
        self._current: str | None = None
        self._adapter_cache: dict[str, dict[str, Any]] = {}

    # ── helpers ─────────────────────────────────────────────────────────────
    def languages(self) -> list[str]:
        """Canonical names of registry languages with an MMS ASR code."""
        return [s.name for s in self.registry if s.mms_asr_code]

    def _mms_code(self, lang: str) -> str:
        spec = self.registry.get(lang)
        if not spec.mms_asr_code:
            raise CapabilityUnavailableError(f"MMS ASR is not configured for {spec.name!r}.")
        return spec.mms_asr_code

    def _default_code(self) -> str:
        codes = [s.mms_asr_code for s in self.registry if s.mms_asr_code]
        if not codes:
            raise ModelLoadError("No MMS ASR languages are configured in the language registry.")
        return codes[0]

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Validate the checkpoint, load processor + model, activate the first adapter."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoProcessor, Wav2Vec2ForCTC

        kwargs = hf_kwargs(self.settings)
        config = AutoConfig.from_pretrained(self.model_id, **kwargs)
        validate_mms_config(config, self.model_id)
        default = self._default_code()
        self._processor = AutoProcessor.from_pretrained(self.model_id, target_lang=default, **kwargs)
        self.dtype_name = "float16" if self.plan.device == "cuda" and self.plan.torch_dtype == "float16" else "float32"
        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, ignore_mismatched_sizes=True, **kwargs, **dtype_kwargs(self.dtype_name)
        )
        model.eval()
        self.device = self.plan.device
        if self.device == "cuda":
            try:
                model.to("cuda")
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                logger.warning("CUDA out of memory loading MMS ASR; falling back to CPU.")
                empty_cuda_cache()
                model.to(device="cpu", dtype=torch.float32)
                self.device, self.dtype_name = "cpu", "float32"
        self._model = model
        self._current = None
        self._adapter_cache.clear()
        self._activate(default)
        if self.device == "cpu":
            configure_torch_threads(self.plan.threads)
            if self.quantize:
                self._quantize_encoder()
        logger.info(
            "MMS ASR ready on %s (%s%s) from %s",
            self.device,
            self.dtype_name,
            ", dynamic int8" if self.quantized else "",
            self.model_id,
        )

    def _quantize_encoder(self) -> None:
        """Dynamic int8 for encoder Linear layers, leaving adapters + lm_head float."""
        import torch
        from torch import nn

        try:
            spec = {
                name: torch.ao.quantization.default_dynamic_qconfig
                for name, module in self._model.named_modules()
                if isinstance(module, nn.Linear) and "adapter" not in name and not name.startswith("lm_head")
            }
            self._model = torch.ao.quantization.quantize_dynamic(self._model, spec, dtype=torch.qint8, inplace=True)
            self.quantized = True
        except Exception:
            logger.warning("Dynamic int8 quantization of MMS ASR failed; continuing in float32.", exc_info=True)
            self.quantized = False

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._current = None
        self._adapter_cache.clear()
        empty_cuda_cache()

    # ── adapters ────────────────────────────────────────────────────────────
    def _adapter_weights(self, code: str) -> dict[str, Any]:
        """Adapter + CTC-head tensors for ``code`` (read once, then cached in memory).

        Weights are read directly from ``adapter.<code>.safetensors`` instead
        of ``model.load_adapter`` because the latter uses ``load_state_dict``,
        which breaks once the encoder has been dynamically quantized.
        """
        cached = self._adapter_cache.get(code)
        if cached is not None:
            return cached
        from pathlib import Path

        import torch

        path: str | None = None
        last_exc: Exception | None = None
        for fname in (f"adapter.{code}.safetensors", f"adapter.{code}.bin"):
            local = Path(self.model_id) / fname
            if local.is_file():
                path = str(local)
                break
            try:
                from huggingface_hub import hf_hub_download

                try:
                    path = hf_hub_download(
                        self.model_id, fname, cache_dir=str(self.settings.hf_cache), local_files_only=True
                    )
                except Exception:
                    if self.settings.offline:
                        raise
                    path = hf_hub_download(self.model_id, fname, cache_dir=str(self.settings.hf_cache))
                break
            except Exception as exc:
                last_exc = exc
        if path is None:
            raise ModelLoadError(f"Could not load MMS ASR adapter {code!r} from {self.model_id!r}: {last_exc}")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            raw = load_file(path)
        else:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        dtype = self._model.lm_head.weight.dtype
        weights = {k: v.to(device=self.device, dtype=dtype) for k, v in raw.items()}
        self._adapter_cache[code] = weights
        return weights

    def _activate(self, code: str) -> None:
        """Switch the active language adapter + tokenizer vocabulary."""
        if code == self._current:
            return
        import torch
        from torch import nn

        tokenizer = self._processor.tokenizer
        if not hasattr(tokenizer, "set_target_lang"):
            raise ModelLoadError(f"{self.model_id!r} does not expose MMS language vocabularies.")
        weights = self._adapter_weights(code)
        tokenizer.set_target_lang(code)
        model = self._model
        head_w = weights.get("lm_head.weight")
        if head_w is not None and tuple(model.lm_head.weight.shape) != tuple(head_w.shape):
            model.lm_head = nn.Linear(head_w.shape[1], head_w.shape[0]).to(device=head_w.device, dtype=head_w.dtype)
            model.config.vocab_size = head_w.shape[0]
        params = dict(model.named_parameters())
        missing = [k for k in weights if k not in params]
        if missing:
            raise ModelLoadError(f"MMS adapter {code!r} does not match the model (e.g. {missing[0]!r}).")
        with torch.no_grad():
            for key, tensor in weights.items():
                params[key].copy_(tensor)
        self._current = code

    # ── inference ───────────────────────────────────────────────────────────
    def _infer(self, audio: np.ndarray) -> tuple[str, float | None]:
        import torch

        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        values = inputs["input_values"].to(self.device)
        if self.dtype_name == "float16":
            values = values.half()
        with torch.inference_mode():
            logits = self._model(values).logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        top_p, ids = probs.max(dim=-1)
        pad_id = self._processor.tokenizer.pad_token_id
        non_blank = ids != pad_id if pad_id is not None else torch.ones_like(ids, dtype=torch.bool)
        conf = float(top_p[non_blank].mean()) if bool(non_blank.any()) else None
        text = self._processor.decode(ids.cpu())
        return text.strip(), conf

    def transcribe(self, audio: np.ndarray, lang: str) -> Transcript:
        """Transcribe 16 kHz mono float32 audio in ``lang`` (canonical name)."""
        self.load()
        spec = self.registry.get(lang)
        code = self._mms_code(spec.name)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / SAMPLE_RATE
        if audio.size < SAMPLE_RATE // 20:  # < 50 ms: nothing to recognise
            return Transcript(text="", language=spec.name, duration_s=duration)
        self._activate(code)
        step = int(_CHUNK_S * SAMPLE_RATE)
        segments: list[Segment] = []
        confs: list[float] = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + step]
            if chunk.size < SAMPLE_RATE // 20:
                continue
            try:
                text, conf = self._infer(chunk)
            except RuntimeError as exc:
                if self.device != "cuda" or not is_oom(exc):
                    raise
                import torch

                logger.warning("CUDA out of memory in MMS ASR; moving model to CPU.")
                empty_cuda_cache()
                self._model.to(device="cpu", dtype=torch.float32)
                self._adapter_cache = {k: {n: t.float().cpu() for n, t in v.items()} for k, v in self._adapter_cache.items()}
                self.device, self.dtype_name = "cpu", "float32"
                text, conf = self._infer(chunk)
            if text:
                segments.append(Segment(start / SAMPLE_RATE, (start + len(chunk)) / SAMPLE_RATE, text))
                if conf is not None:
                    confs.append(conf)
        return Transcript(
            text=" ".join(s.text for s in segments).strip(),
            language=spec.name,
            segments=tuple(segments),
            confidence=float(np.mean(confs)) if confs else None,
            duration_s=duration,
        )
