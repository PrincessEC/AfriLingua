"""
NLLB-200 translation via CTranslate2 (int8 on CPU, fp16 / int8_fp16 on CUDA).

The Hugging Face checkpoint is converted once into
``settings.ct2_cache/<model-slug>-<quantization>`` and reused afterwards.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from afrilingua.backends._common import (
    checkpoint_fingerprint,
    ct2_compute_type,
    ct2_device_for,
    is_ct2_model_dir,
    module_available,
    resolve_mt_source,
    slugify,
    snapshot_path,
)
from afrilingua.core.errors import ModelLoadError
from afrilingua.core.hardware import RuntimePlan, resolve_runtime
from afrilingua.core.languages import LanguageRegistry, get_registry
from afrilingua.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_CONVERT_LOCK = threading.Lock()
_NLLB_PATTERNS = ["*.json", "*.model", "model.safetensors", "*.safetensors.index.json"]
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "added_tokens.json",
)


def local_nllb_checkpoint(model_id: str, settings: Settings) -> Path:
    """Local snapshot dir for an NLLB id; prefers safetensors, falls back to .bin."""
    try:
        return snapshot_path(
            model_id,
            settings,
            allow_patterns=_NLLB_PATTERNS,
            required_any=("model.safetensors", "model.safetensors.index.json"),
        )
    except ModelLoadError:
        return snapshot_path(
            model_id,
            settings,
            allow_patterns=["*.json", "*.model", "pytorch_model.bin*"],
            required_any=("pytorch_model.bin", "pytorch_model.bin.index.json"),
        )


# ── Script-restricted decoding ──────────────────────────────────────────────
# NLLB's output layer scores all 256k vocabulary entries at every step, and most of them
# belong to scripts our languages never use (CJK, Arabic, Devanagari ...). CTranslate2 can
# restrict decoding to a subset listed in ``vmap.txt`` inside the model directory. We keep
# every token whose characters belong to the scripts of the registered languages, so the
# chosen output cannot change for those languages; it only skips impossible candidates.
VMAP_FILE = "vmap.txt"
VMAP_TAG_FILE = "vmap.tag"
#: NLLB script code -> Unicode character-name prefixes for that script.
SCRIPT_NAMES: dict[str, tuple[str, ...]] = {
    "Latn": ("LATIN",),
    "Arab": ("ARABIC",),
    "Ethi": ("ETHIOPIC",),
    "Cyrl": ("CYRILLIC",),
    "Grek": ("GREEK",),
    "Hebr": ("HEBREW",),
    "Deva": ("DEVANAGARI",),
    "Tfng": ("TIFINAGH",),
    "Nkoo": ("NKO",),
    "Vaii": ("VAI",),
}


def nllb_script(code: str) -> str:
    """``"yor_Latn"`` -> ``"Latn"``."""
    return code.split("_", 1)[1] if "_" in code else ""


def token_in_scripts(token: str, prefixes: tuple[str, ...]) -> bool:
    """True if every character of a SentencePiece token is neutral or in an allowed script."""
    for ch in token.replace("\u2581", ""):
        if ch.isascii():
            continue
        cat = unicodedata.category(ch)
        if cat == "Mn" or (cat[0] in "PSZ" and ord(ch) < 0x3000):
            continue  # combining marks (tone marks, dots below), punctuation, symbols
        name = unicodedata.name(ch, "")
        if not any(name.startswith(pre) for pre in prefixes):
            return False
    return True


def ensure_vocab_map(model_dir: Path, tokenizer: Any, scripts: set[str]) -> bool:
    """Write ``vmap.txt`` for ``scripts`` into ``model_dir`` (cached). False if not applicable."""
    if not scripts or any(sc not in SCRIPT_NAMES for sc in scripts):
        return False
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    tag = f"{','.join(sorted(scripts))}|{len(vocab)}"
    tag_file = model_dir / VMAP_TAG_FILE
    vmap = model_dir / VMAP_FILE
    if vmap.is_file() and tag_file.is_file() and tag_file.read_text(encoding="utf-8") == tag:
        return True
    prefixes = tuple(pre for sc in sorted(scripts) for pre in SCRIPT_NAMES[sc])
    keep = set(tokenizer.all_special_tokens)
    allowed = [t for t in vocab if t in keep or token_in_scripts(t, prefixes)]
    tmp = vmap.with_suffix(".tmp")
    # One line with an empty source key: these targets are always allowed.
    tmp.write_text("\t" + " ".join(allowed) + "\n", encoding="utf-8")
    os.replace(tmp, vmap)
    tag_file.write_text(tag, encoding="utf-8")
    logger.info("Vocabulary map for %s: %d of %d tokens", ",".join(sorted(scripts)), len(allowed), len(vocab))
    return True


def ct2_target_dir(source: str, quantization: str, settings: Settings) -> Path:
    """Where the CT2 conversion of ``source`` with ``quantization`` lives."""
    src_path = Path(source)
    if src_path.is_dir():
        slug = f"local-{slugify(src_path.name)}-{checkpoint_fingerprint(src_path)}"
    else:
        slug = slugify(source)
    return settings.ct2_cache / f"{slug}-{quantization}"


def ensure_ct2_model(source: str, quantization: str, settings: Settings | None = None) -> Path:
    """Convert ``source`` (hub id or local checkpoint) to CTranslate2 if needed.

    Conversion goes to a temporary sibling directory and is renamed atomically,
    so an interrupted conversion never leaves a half-written model behind.
    Returns the converted model directory.
    """
    settings = settings or get_settings()
    src = Path(source)
    if is_ct2_model_dir(src):
        return src
    target = ct2_target_dir(source, quantization, settings)
    if is_ct2_model_dir(target):
        return target
    with _CONVERT_LOCK:
        if is_ct2_model_dir(target):
            return target
        import ctranslate2

        local = src if src.is_dir() else local_nllb_checkpoint(source, settings)
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_files = [f for f in _TOKENIZER_FILES if (local / f).is_file()]
        logger.info(
            "Converting %s to CTranslate2 (%s) — one-time, may take a few minutes ...", source, quantization
        )
        t0 = time.perf_counter()
        try:
            converter = ctranslate2.converters.TransformersConverter(str(local), copy_files=copy_files)
            converter.convert(str(tmp), quantization=quantization, force=True)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            tmp.rename(target)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ModelLoadError(f"CTranslate2 conversion of {source!r} failed: {exc}") from exc
        logger.info("Converted %s in %.1fs → %s", source, time.perf_counter() - t0, target)
        return target


class CT2NLLBBackend:
    """:class:`~afrilingua.backends.base.MTBackend` backed by CTranslate2."""

    name = "ctranslate2-nllb"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        settings: Settings | None = None,
        plan: RuntimePlan | None = None,
        registry: LanguageRegistry | None = None,
        max_batch_size: int = 32,
    ) -> None:
        self.settings = settings or get_settings()
        self.plan = plan or resolve_runtime()
        self.registry = registry or get_registry()
        self.model_id = model_id or self.plan.model_nllb
        self.max_batch_size = max(1, max_batch_size)
        self._translator: Any = None
        self._tokenizer: Any = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.model_dir: Path | None = None
        self._vmap_scripts: set[str] = set()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self) -> None:
        """Convert (first run), then load the translator and tokenizer."""
        if self._translator is not None:
            return
        if not module_available("ctranslate2"):
            raise ModelLoadError("ctranslate2 is not installed (pip install 'afrilingua[fast]').")
        import ctranslate2
        from transformers import AutoTokenizer

        self.device = ct2_device_for(self.plan.device)
        self.compute_type = ct2_compute_type(self.device, self.plan.compute_type)
        source = resolve_mt_source(self.settings, self.model_id)
        model_dir = ensure_ct2_model(source, self.compute_type, self.settings)
        threads = self.plan.threads if self.device == "cpu" else 1
        tok_source = model_dir if (model_dir / "tokenizer_config.json").is_file() else source
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tok_source),
            cache_dir=str(self.settings.hf_cache),
            local_files_only=bool(self.settings.offline) or Path(str(tok_source)).is_dir(),
        )
        self._vmap_scripts = set()
        if self.settings.mt_vocab_restrict:
            scripts = {nllb_script(spec.nllb_code) for spec in self.registry if spec.nllb_code}
            try:
                if ensure_vocab_map(model_dir, self._tokenizer, scripts):
                    self._vmap_scripts = scripts
            except Exception:  # never block translation on an optimisation
                logger.exception("Could not build the vocabulary map; decoding the full vocabulary")
        self._translator = ctranslate2.Translator(
            str(model_dir),
            device=self.device,
            compute_type=self.compute_type,
            inter_threads=1,
            intra_threads=threads,
        )
        self.model_dir = model_dir
        logger.info(
            "CT2 NLLB ready (%s, %s, %d threads) from %s", self.device, self.compute_type, threads, model_dir
        )

    def is_loaded(self) -> bool:
        return self._translator is not None

    def unload(self) -> None:
        self._translator = None
        self._tokenizer = None

    # ── inference ───────────────────────────────────────────────────────────
    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        beam_size: int = 1,
        max_length: int = 256,
    ) -> list[str]:
        """Translate ``texts`` from ``src_lang`` to ``tgt_lang`` (canonical names)."""
        if not texts:
            return []
        self.load()
        src_code = self.registry.nllb_code(src_lang)
        tgt_code = self.registry.nllb_code(tgt_lang)
        tok = self._tokenizer
        tok.src_lang = src_code
        sources = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=1000)) for t in texts]
        use_vmap = bool(self._vmap_scripts) and nllb_script(tgt_code) in self._vmap_scripts
        results = self._translator.translate_batch(
            sources,
            target_prefix=[[tgt_code]] * len(sources),
            beam_size=max(1, beam_size),
            max_decoding_length=max_length,
            max_batch_size=self.max_batch_size,
            use_vmap=use_vmap,
        )
        outputs: list[str] = []
        for res in results:
            tokens = res.hypotheses[0]
            if tokens and tokens[0] == tgt_code:
                tokens = tokens[1:]
            ids = tok.convert_tokens_to_ids(tokens)
            outputs.append(tok.decode(ids, skip_special_tokens=True).strip())
        return outputs
