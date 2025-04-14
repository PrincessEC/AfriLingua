"""
src/model_loader.py
Eager model pre-loading with progress callbacks.

Call load_all() at application startup so that model weights are downloaded
and loaded into memory *before* the user interacts with AfriLingua.

Pass the same singleton instances used by the UI so that pre-loaded models
are shared — no double-loading.

Progress callback signature:
    on_progress(step: int, total: int, label: str)

    step == 0  → loading is starting (0 % complete)
    step == total → everything is ready (100 % complete)
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Number of discrete loading steps (must match the steps inside load_all)
TOTAL_STEPS = 7


def _is_offline_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a network / connectivity failure."""
    import socket
    if isinstance(exc, (ConnectionError, TimeoutError,
                        socket.gaierror, socket.timeout, OSError)):
        msg = str(exc).lower()
        _OFFLINE_HINTS = (
            "couldn't connect", "connection error", "network is unreachable",
            "offline", "unreachable", "timed out", "huggingface.co",
            "getaddrinfo failed", "name or service not known",
            "temporary failure in name resolution",
        )
        return any(hint in msg for hint in _OFFLINE_HINTS)
    return False

ProgressCallback = Callable[[int, int, str], None]
# filename, bytes_done, bytes_total, speed_bps
DownloadCallback = Callable[[str, int, int, float], None]


@contextlib.contextmanager
def _hf_download_progress_ctx(on_download_progress: "Optional[DownloadCallback]"):
    """
    Context manager that temporarily patches huggingface_hub's file-download
    tqdm bar so every downloaded chunk calls:
        on_download_progress(filename, bytes_done, bytes_total, speed_bps)

    Safe to use when models are already cached (tqdm.total will be 0, so the
    callback is skipped and there is no visible overhead).
    """
    if on_download_progress is None:
        yield
        return

    targets: list = []
    try:
        import importlib
        mod = importlib.import_module("huggingface_hub.file_download")
        if hasattr(mod, "tqdm"):
            targets.append((mod, "tqdm", getattr(mod, "tqdm")))
    except ImportError:
        pass

    if not targets:
        yield
        return

    _cb = on_download_progress

    class _Tqdm:  # minimal tqdm-compatible shim
        def __init__(self, iterable=None, *, desc=None, total=None,
                     unit="it", unit_scale=False, unit_divisor=1000,
                     disable=False, miniters=1, mininterval=0.1, **kw):
            self._iter    = iterable
            self._desc    = str(desc or "")
            self._total   = int(total) if total else 0
            self._n       = 0
            self._start   = time.monotonic()
            self._disable = disable

        def update(self, n: int = 1) -> None:
            self._n += n
            if self._disable or not self._total:
                return
            elapsed = time.monotonic() - self._start
            _cb(self._desc, self._n, self._total,
                self._n / elapsed if elapsed > 0 else 0.0)

        def set_postfix(self, *a, **kw) -> None:
            pass

        def set_description(self, desc=None, *a, **kw) -> None:
            if desc is not None:
                self._desc = desc

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            if not self._disable and self._total > 0:
                _cb(self._desc, self._total, self._total, 0.0)

        def __iter__(self):
            if self._iter is not None:
                yield from self._iter

    for mod, attr, _ in targets:
        setattr(mod, attr, _Tqdm)
    try:
        yield
    finally:
        for mod, attr, orig in targets:
            setattr(mod, attr, orig)


def load_all(
    on_progress: Optional[ProgressCallback] = None,
    on_download_progress: Optional[DownloadCallback] = None,
    *,
    engine=None,
    whisper=None,
    mms_asr=None,
    eng_tts=None,
    mms_tts=None,
) -> None:
    """
    Eagerly load every AI model used by AfriLingua in sequence.

    Args:
        on_progress: Optional callback fired before each loading step.
                     Signature: (step: int, total: int, label: str)
        on_download_progress: Optional callback fired per downloaded chunk.
                     Signature: (filename, bytes_done, bytes_total, speed_bps)
                     Only fires when the file is being downloaded (not cached).
        engine:   Pre-created TranslationEngine singleton (or None).
        whisper:  Pre-created WhisperASR singleton (or None).
        mms_asr:  Pre-created MMSASR singleton (or None).
        eng_tts:  Pre-created EnglishTTS singleton (or None).
        mms_tts:  Pre-created MMSTTS singleton (or None).
    """
    # Lazy import — keeps startup fast when this module is imported at top level
    from src.translation.engine import TranslationEngine
    from src.asr.whisper_asr import WhisperASR
    from src.asr.mms_asr import MMSASR
    from src.tts.english_tts import EnglishTTS
    from src.tts.mms_tts import MMSTTS

    _engine  = engine  if engine  is not None else TranslationEngine()
    _whisper = whisper if whisper is not None else WhisperASR()
    _mms_asr = mms_asr if mms_asr is not None else MMSASR()
    _eng_tts = eng_tts if eng_tts is not None else EnglishTTS()
    _mms_tts = mms_tts if mms_tts is not None else MMSTTS()

    def _emit(step: int, label: str) -> None:
        logger.info("[%d/%d] %s", step, TOTAL_STEPS, label)
        if on_progress:
            on_progress(step, TOTAL_STEPS, label)

    def _step(step: int, label: str, load_fn) -> None:
        """Emit progress, then load with HF download-progress interception."""
        _emit(step, label)
        try:
            with _hf_download_progress_ctx(on_download_progress):
                load_fn()
        except Exception as exc:
            if _is_offline_error(exc):
                raise ConnectionError(
                    "No internet connection.\n\n"
                    "AfriLingua needs to download AI model weights on first run. "
                    "Please connect to the internet and restart the app.\n\n"
                    "(Once downloaded, the app works fully offline.)"
                ) from exc
            raise

    _step(1, "Loading NLLB-200 translation model …",  _engine._load)
    _step(2, "Loading Whisper English ASR …",          _whisper._load)
    _step(3, "Loading MMS African-language ASR …",     _mms_asr._load)
    _step(4, "Loading English TTS (pyttsx3) …",        _eng_tts._load)
    _step(5, "Loading MMS TTS — Igbo …",               lambda: _mms_tts._load_lang("igbo"))
    _step(6, "Loading MMS TTS — Yoruba …",             lambda: _mms_tts._load_lang("yoruba"))
    _step(7, "Loading MMS TTS — Hausa …",              lambda: _mms_tts._load_lang("hausa"))

    logger.info("All models loaded and ready.")
    if on_progress:
        on_progress(TOTAL_STEPS, TOTAL_STEPS, "All models ready ✓")
