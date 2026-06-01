"""
ui/app.py
AfriLingua — Gradio MVP UI

Three tabs:
    1. Live Conversation  — microphone → real-time text + audio output
    2. Document Translation — upload TXT/PDF/DOCX → download translated file
    3. Text Translation   — type/paste text → instant translation + TTS play

Launch:
    python ui/app.py
Then open http://localhost:7860 in your browser.
"""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from html import escape
from pathlib import Path

# ── IMPORTANT: set HuggingFace cache env-vars BEFORE importing gradio ────────
# gradio imports huggingface_hub at module level, which reads HUGGINGFACE_HUB_CACHE
# on import.  If src.config hasn't run yet, HF falls back to ~/.cache/huggingface
# and every restart re-downloads models to the wrong location.
# ── IMPORTANT: set HuggingFace cache env-vars BEFORE importing gradio ────────
# gradio imports huggingface_hub at module level, which reads HUGGINGFACE_HUB_CACHE
# on import.  If src.config hasn't run yet, HF falls back to ~/.cache/huggingface
# and every restart re-downloads models to the wrong location.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (                           # noqa: E402  (intentional early import)
    GRADIO_PORT,
    DEVICE, TORCH_DTYPE, HARDWARE_PROFILE, GPU_INFO,
    MODEL_NLLB, MODEL_WHISPER, MODEL_MMS_ASR,
)

import base64
import io

from src.config import (                           # noqa: E402  (intentional early import)
    GRADIO_PORT,
    DEVICE, TORCH_DTYPE, HARDWARE_PROFILE, GPU_INFO,
    MODEL_NLLB, MODEL_WHISPER, MODEL_MMS_ASR,
)

import base64
import io

import gradio as gr
from gradio.themes import Soft
import numpy as np
import soundfile as sf
from src.detection.lang_detector import LanguageDetector
from src.document.translator import DocumentTranslator
from src.model_loader import load_all, TOTAL_STEPS
from src.pipeline.realtime_pipeline import RealtimePipeline, TranslationResult
from src.translation.engine import TranslationEngine, SUPPORTED_LANGUAGES
from src.asr.whisper_asr import WhisperASR
from src.asr.mms_asr import MMSASR
from src.tts.english_tts import EnglishTTS
from src.tts.mms_tts import MMSTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Shared singletons — pre-loaded at startup via demo.load() ───────────────
_engine    = TranslationEngine()
_whisper   = WhisperASR()
_mms_asr   = MMSASR()
_detector  = LanguageDetector()
_doc_trans = DocumentTranslator()
_mms_tts   = MMSTTS()
_eng_tts   = EnglishTTS()

_LANGUAGE_CHOICES = ["english", "igbo", "yoruba", "hausa"]
_LIVE_PIPELINE: dict[str, RealtimePipeline] = {}   # keyed by session ID


# ── Custom CSS ─────────────────────────────────────────────────────────────────

_CSS = """
/* ── global ───────────────────────────────────────── */
.al-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px 16px;
    margin-bottom: 14px;
}
.al-title {
    font-size: 28px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #e94560 20%, #fb923c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin: 0 0 4px;
}
.al-subtitle { color: #94a3b8; font-size: 13px; margin: 0; }

/* ── hardware info bar ────────────────────────────── */
.al-hw-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 12px;
    font-size: 12px;
}
.al-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #334155;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
}
.al-chip-gpu    { color: #4ade80; border-color: #166534; background: #14532d22; }
.al-chip-cpu    { color: #fb923c; border-color: #9a3412; background: #7c2d1222; }
.al-chip-high   { color: #4ade80; border-color: #166534; background: #14532d22; }
.al-chip-mid    { color: #fbbf24; border-color: #92400e; background: #78350f22; }
.al-chip-low    { color: #fb923c; border-color: #9a3412; background: #7c2d1222; }
.al-chip-model  { color: #a5b4fc; border-color: #3730a3; background: #1e1b4b22; }
.al-chip-dtype  { color: #94a3b8; border-color: #475569; background: #1e293b22; }

/* ── tab panel improvements ───────────────────────── */
.al-tab-desc {
    background: #1e293b;
    border-left: 3px solid #e94560;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 14px;
}

/* ── transcript box ───────────────────────────────── */
.al-transcript textarea {
    font-family: 'Consolas', 'Menlo', monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

/* Modern app shell overrides */
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: #f6f7f2 !important;
    color: #17201b !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}
body, .main, .app {
    background: #f6f7f2 !important;
}
.al-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 18px;
    background: #ffffff !important;
    border: 1px solid #d9dfd2 !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    margin: 12px 0 10px !important;
    box-shadow: 0 10px 26px rgba(23, 32, 27, 0.06);
}
.al-title {
    color: #17201b !important;
    -webkit-text-fill-color: #17201b !important;
    background: none !important;
    font-size: 30px !important;
    letter-spacing: 0 !important;
}
.al-subtitle {
    color: #5e6a61 !important;
    font-size: 13px !important;
}
.al-product-mark {
    border-left: 3px solid #d95f43;
    padding-left: 12px;
}
.al-status-pill {
    color: #1d6f5f;
    background: #e8f3ee;
    border: 1px solid #b8d8c9;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.al-hw-bar {
    background: #ffffff !important;
    border: 1px solid #d9dfd2 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin-bottom: 10px !important;
}
.al-chip {
    border-radius: 999px !important;
    font-size: 11px !important;
    letter-spacing: 0 !important;
}
.al-chip-gpu, .al-chip-high { color: #16624f !important; border-color: #9bc8b5 !important; background: #e8f3ee !important; }
.al-chip-cpu, .al-chip-low { color: #9a4b22 !important; border-color: #e6b999 !important; background: #fff1e8 !important; }
.al-chip-mid { color: #7a5c00 !important; border-color: #dbc36a !important; background: #fff9d7 !important; }
.al-chip-model { color: #264653 !important; border-color: #b6c8ce !important; background: #edf4f6 !important; }
.al-chip-dtype { color: #52605a !important; border-color: #c9d1c7 !important; background: #f7f8f4 !important; }
.al-tab-desc, .al-section-title {
    background: transparent !important;
    border: 0 !important;
    border-left: 3px solid #d95f43 !important;
    border-radius: 0 !important;
    padding: 2px 0 2px 12px !important;
    color: #26332b !important;
    font-size: 15px !important;
    font-weight: 750 !important;
    margin: 8px 0 12px !important;
}
.tab-nav button {
    border-radius: 999px !important;
    font-weight: 700 !important;
}
button.primary, .primary {
    background: #1d6f5f !important;
    border-color: #1d6f5f !important;
    color: #ffffff !important;
}
button.secondary {
    border-color: #cbd5c8 !important;
}
textarea, input, select {
    border-radius: 8px !important;
}
.block {
    border-radius: 8px !important;
}
"""


def _cpu_or_cuda_unavailable_chip() -> str:
    label = "CUDA unavailable" if GPU_INFO.get("cuda_built") else "CPU only"
    reason = escape(str(GPU_INFO.get("reason") or ""))
    title = f' title="{reason}"' if reason else ""
    return f'<span class="al-chip al-chip-cpu"{title}>{label}</span>'


def _legacy_hw_info_html() -> str:
    """Generate the hardware / active-model info bar HTML."""
    # GPU chip
    if GPU_INFO["available"] and DEVICE == "cuda":
        gpu_html = (
            f'<span class="al-chip al-chip-gpu">'
            f'🖥️ {GPU_INFO["name"]} &nbsp;&middot;&nbsp; {GPU_INFO["vram_gb"]:.1f} GB VRAM'
            f'</span>'
        )
    else:
        gpu_html = _cpu_or_cuda_unavailable_chip()

    # Profile chip
    _profile_labels = {
        "high": ("⚡ High quality",  "al-chip-high"),
        "mid":  ("⚡ Balanced",       "al-chip-mid"),
        "low":  ("⚡ Lite / fastest", "al-chip-low"),
    }
    p_label, p_cls = _profile_labels.get(HARDWARE_PROFILE, ("⚡ Lite", "al-chip-low"))
    profile_html = f'<span class="al-chip {p_cls}">{p_label}</span>'

    # Model chips
    nllb_short    = MODEL_NLLB.split("/")[-1]
    whisper_short = MODEL_WHISPER.split("/")[-1]
    asr_short     = MODEL_MMS_ASR.split("/")[-1]
    models_html = (
        f'<span class="al-chip al-chip-model">🔤 {nllb_short}</span>'
        f'<span class="al-chip al-chip-model">🎤 {whisper_short}</span>'
        f'<span class="al-chip al-chip-model">🔊 {asr_short}</span>'
    )

    dtype_html = (
        f'<span class="al-chip al-chip-dtype">{TORCH_DTYPE} · {DEVICE.upper()}</span>'
    )

    return (
        f'<div class="al-hw-bar">'
        f'{gpu_html}{profile_html}{models_html}{dtype_html}'
        f'</div>'
    )


# ── Loading banner helpers ────────────────────────────────────────────────────

def _hw_info_html() -> str:
    """Generate the hardware / active-model info bar HTML."""
    if GPU_INFO["available"] and DEVICE == "cuda":
        gpu_html = (
            f'<span class="al-chip al-chip-gpu">'
            f'GPU {GPU_INFO["name"]} &nbsp;&middot;&nbsp; {GPU_INFO["vram_gb"]:.1f} GB VRAM'
            f'</span>'
        )
    else:
        gpu_html = _cpu_or_cuda_unavailable_chip()

    profile_labels = {
        "high": ("High quality", "al-chip-high"),
        "mid": ("Balanced", "al-chip-mid"),
        "low": ("Lite profile", "al-chip-low"),
    }
    p_label, p_cls = profile_labels.get(HARDWARE_PROFILE, ("Lite profile", "al-chip-low"))
    profile_html = f'<span class="al-chip {p_cls}">{p_label}</span>'

    nllb_short = MODEL_NLLB.split("/")[-1]
    whisper_short = MODEL_WHISPER.split("/")[-1]
    asr_short = MODEL_MMS_ASR.split("/")[-1]
    models_html = (
        f'<span class="al-chip al-chip-model">Translation: {nllb_short}</span>'
        f'<span class="al-chip al-chip-model">English ASR: {whisper_short}</span>'
        f'<span class="al-chip al-chip-model">African ASR: {asr_short}</span>'
    )

    dtype_html = f'<span class="al-chip al-chip-dtype">{TORCH_DTYPE} · {DEVICE.upper()}</span>'
    return f'<div class="al-hw-bar">{gpu_html}{profile_html}{models_html}{dtype_html}</div>'


def _loading_banner_html(step: int, total: int, label: str, dl_info: dict | None = None) -> str:
    """Return an HTML string for the progressive loading progress bar.

    dl_info (optional): {"filename": str, "done": int, "total": int, "speed": float}
    When supplied and a download is in progress, a secondary progress bar is rendered.
    """
    pct = int(step / total * 100)
    done = step == total
    bar_color = "#1d6f5f" if done else "#d95f43"
    bar_color = "#1d6f5f" if done else "#d95f43"
    spin_style = (
        "display:none" if done else
        "display:inline-block;width:15px;height:15px;"
        "border:2px solid #d95f43;border-top-color:transparent;"
        "border:2px solid #d95f43;border-top-color:transparent;"
        "border-radius:50%;animation:al-spin .7s linear infinite;"
        "vertical-align:middle;margin-right:8px;"
    )
    icon = "" if done else ""
    icon = "" if done else ""
    heading = (
        "AfriLingua is ready" if done else
        "AfriLingua is ready" if done else
        f"AfriLingua — loading models ({step}/{total})"
    )
    sub = (
        "All models are loaded. You can now translate text, documents, and live speech."
        if done else
        f"{label}"
        f"{label}"
    )
    note = (
        ""
        if done else
        '<p style="color:#6b746c;font-size:11px;margin:4px 0 0;">'
        "First-time setup downloads model weights (~7-8 GB). "
        '<p style="color:#6b746c;font-size:11px;margin:4px 0 0;">'
        "First-time setup downloads model weights (~7-8 GB). "
        "This only happens once — subsequent starts are much faster."
        "</p>"
    )
    # ── Per-file download sub-progress ─────────────────────────────────────
    if not done and dl_info and dl_info.get("total", 0) > 0:
        pct_dl   = min(100, int(dl_info["done"] / dl_info["total"] * 100))
        done_mb  = dl_info["done"]  / 1_048_576
        total_mb = dl_info["total"] / 1_048_576
        speed_mb = dl_info.get("speed", 0.0) / 1_048_576
        fname    = (dl_info.get("filename") or "").split("/")[-1] or "file"
        dl_section = (
            '<div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e3a5f;">'
            '<div style="display:flex;justify-content:space-between;'
            'color:#1d6f5f;font-size:11px;margin-bottom:4px;">'
            f'<span>{fname}</span>'
            'color:#1d6f5f;font-size:11px;margin-bottom:4px;">'
            f'<span>{fname}</span>'
            f'<span>{done_mb:.1f}&thinsp;/&thinsp;{total_mb:.1f}&thinsp;MB'
            f'&nbsp;&middot;&nbsp;{speed_mb:.1f}&thinsp;MB/s</span>'
            '</div>'
            '<div style="background:#dfe5db;border-radius:3px;height:4px;overflow:hidden;">'
            f'<div style="background:#4f8f80;width:{pct_dl}%;height:4px;border-radius:3px;"></div>'
            '<div style="background:#dfe5db;border-radius:3px;height:4px;overflow:hidden;">'
            f'<div style="background:#4f8f80;width:{pct_dl}%;height:4px;border-radius:3px;"></div>'
            '</div>'
            '</div>'
        )
    else:
        dl_section = ""
    visibility = "" if not done else "opacity:.6;"
    return (
        f'<div style="{visibility}background:#ffffff;border:1px solid #d9dfd2;'
        f'<div style="{visibility}background:#ffffff;border:1px solid #d9dfd2;'
        f'border-radius:8px;padding:14px 16px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
        f'<div style="{spin_style}"></div>'
        f'<b style="color:#17201b;">{heading}</b></div>'
        f'<div style="background:#dfe5db;border-radius:4px;height:6px;overflow:hidden;">'
        f'<b style="color:#17201b;">{heading}</b></div>'
        f'<div style="background:#dfe5db;border-radius:4px;height:6px;overflow:hidden;">'
        f'<div style="background:{bar_color};width:{pct}%;height:6px;border-radius:4px;"></div>'
        "</div>"
        f'<p style="color:#5e6a61;font-size:12px;margin:6px 0 0;">{sub}</p>'
        f'<p style="color:#5e6a61;font-size:12px;margin:6px 0 0;">{sub}</p>'
        f"{note}"
        f"{dl_section}"
        f"</div>"
        "<style>@keyframes al-spin{to{transform:rotate(360deg);}}</style>"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _waveform_to_gradio_audio(waveform: np.ndarray, sample_rate: int):
    """Convert numpy waveform to a (sample_rate, ndarray) tuple for gr.Audio."""
    return (sample_rate, (waveform * 32767).astype(np.int16))


def _synthesize(text: str, lang: str):
    if lang == "english":
        waveform, sr = _eng_tts.synthesize(text)
    else:
        waveform, sr = _mms_tts.synthesize(text, lang=lang)
    return _waveform_to_gradio_audio(waveform, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Text Translation (simplest, defined first so singletons warm up fast)
# ─────────────────────────────────────────────────────────────────────────────

def text_translate(input_text: str, src_lang: str, tgt_lang: str, play_tts: bool):
    if not input_text.strip():
        return "", None, "Please enter some text."

    src = src_lang if src_lang != "auto-detect" else _detector.detect_or_default(input_text, "english")

    try:
        translated = _engine.translate(input_text, src, tgt_lang)
    except ValueError as exc:
        return "", None, str(exc)

    audio = _synthesize(translated, tgt_lang) if play_tts else None
    status = f"Detected source: {src}" if src_lang == "auto-detect" else ""
    return translated, audio, status


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Document Translation
# ─────────────────────────────────────────────────────────────────────────────

def doc_translate(file_obj, src_lang: str, tgt_lang: str):
    if file_obj is None:
        return "No file uploaded.", None

    src = None if src_lang == "auto-detect" else src_lang
    try:
        out_path = _doc_trans.translate_file(file_obj.name, tgt_lang, src_lang=src)
        preview = out_path.read_text(encoding="utf-8")[:3000]
        if len(preview) == 3000:
            preview += "\n\n… (truncated — download full file)"
        return preview, str(out_path)
    except Exception as exc:
        logger.error("Document translation error: %s", exc)
        return f"Error: {exc}", None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Conversation (mic → VAD → ASR → NLLB → TTS)
# ─────────────────────────────────────────────────────────────────────────────
# Gradio's audio streaming passes raw audio chunks to a generator function.

def live_translate_chunk(audio_chunk, src_lang: str, tgt_lang: str, transcript_state: list):
    """
    Called by Gradio for each recorded audio chunk.
    audio_chunk: (sample_rate, np.ndarray) from gr.Audio(streaming=True)

    Simultaneous interpretation mode:
    - The VAD fires every ≤4 s of continuous speech AND on silence boundaries.
    - Text appears on screen as soon as translation is ready.
    - TTS audio plays in the target language concurrently.
    """
    if audio_chunk is None:
        return transcript_state, _format_transcript(transcript_state), None

    sample_rate, data = audio_chunk
    audio_np = data.astype(np.float32) / 32768.0  # int16 → float32

    results: list[TranslationResult] = []

    def capture(r: TranslationResult):
        results.append(r)

    # Reuse the cached pipeline for this (src, tgt) pair; create one if new.
    cache_key = f"{src_lang}→{tgt_lang}"
    if cache_key not in _LIVE_PIPELINE:
        _LIVE_PIPELINE[cache_key] = RealtimePipeline(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            on_result=lambda r: None,   # placeholder; overridden below
            audio_output=True,
            whisper=_whisper,
            mms_asr=_mms_asr,
            engine=_engine,
            mms_tts=_mms_tts,
            eng_tts=_eng_tts,
        )
    pipeline = _LIVE_PIPELINE[cache_key]
    pipeline.on_result = capture
    pipeline.process_audio_chunk(audio_np)

    for r in results:
        transcript_state.append({
            "speaker": src_lang.capitalize(),
            "original": r.original_text,
            "translated": r.translated_text,
            "lang": r.tgt_lang,
        })

    tts_audio = None
    if results and results[-1].audio_b64:
        raw = base64.b64decode(results[-1].audio_b64)
        buf = io.BytesIO(raw)
        waveform, sr = sf.read(buf, dtype="int16")
        tts_audio = (sr, waveform)

    return transcript_state, _format_transcript(transcript_state), tts_audio


def _format_transcript(state: list) -> str:
    if not state:
        return "Waiting for speech ..."
        return "Waiting for speech ..."
    lines = []
    for i, entry in enumerate(state):
        is_latest = (i == len(state) - 1)
        prefix = ">" if is_latest else " "
        prefix = ">" if is_latest else " "
        lines.append(
            f"{prefix} [{entry['speaker']}]  →  {entry['lang'].capitalize()}\n"
            f"   {entry['original']}\n"
            f"   ↳ {entry['translated']}"
        )
    return "\n\n".join(lines)


def clear_transcript():
    return [], "Waiting for speech ...", None
    return [], "Waiting for speech ...", None


# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────────────────────

# ── Module-level loading state ──────────────────────────────────────────────
# Shared across every Gradio client connection so the background thread that
# calls load_all() is started exactly once, no matter how many browser tabs
# connect or reload while the app is still initialising.
_loading_state: dict = {
    "started": False,
    "step":    0,
    "label":   "Initialising …",
    "done":    False,
    "error":   None,
    "dl":      None,
}
_loading_lock = threading.Lock()


# ── Module-level loading state ──────────────────────────────────────────────
# Shared across every Gradio client connection so the background thread that
# calls load_all() is started exactly once, no matter how many browser tabs
# connect or reload while the app is still initialising.
_loading_state: dict = {
    "started": False,
    "step":    0,
    "label":   "Initialising …",
    "done":    False,
    "error":   None,
    "dl":      None,
}
_loading_lock = threading.Lock()


def _startup_load():
    """
    Generator that streams loading progress to the loading banner.
    The background thread is started at most once — additional browser clients
    that connect while loading is in progress share the same _loading_state.
    Generator that streams loading progress to the loading banner.
    The background thread is started at most once — additional browser clients
    that connect while loading is in progress share the same _loading_state.
    """
    with _loading_lock:
        should_start = not _loading_state["started"]
        if should_start:
            _loading_state["started"] = True

    if should_start:
        def _on_progress(step: int, total: int, label: str) -> None:
            _loading_state.update(step=step, label=label, dl=None)
    with _loading_lock:
        should_start = not _loading_state["started"]
        if should_start:
            _loading_state["started"] = True

    if should_start:
        def _on_progress(step: int, total: int, label: str) -> None:
            _loading_state.update(step=step, label=label, dl=None)

        def _on_download(filename: str, done: int, total: int, speed: float) -> None:
            _loading_state["dl"] = {"filename": filename, "done": done,
                                    "total": total, "speed": speed}
        def _on_download(filename: str, done: int, total: int, speed: float) -> None:
            _loading_state["dl"] = {"filename": filename, "done": done,
                                    "total": total, "speed": speed}

        def _bg() -> None:
            try:
                load_all(
                    on_progress=_on_progress,
                    on_download_progress=_on_download,
                    engine=_engine,
                    whisper=_whisper,
                    mms_asr=_mms_asr,
                    eng_tts=_eng_tts,
                    mms_tts=_mms_tts,
                )
            except Exception as exc:
                logger.error("Startup load error: %s", exc, exc_info=True)
                _loading_state["error"] = str(exc)
            finally:
                _loading_state["done"] = True
        def _bg() -> None:
            try:
                load_all(
                    on_progress=_on_progress,
                    on_download_progress=_on_download,
                    engine=_engine,
                    whisper=_whisper,
                    mms_asr=_mms_asr,
                    eng_tts=_eng_tts,
                    mms_tts=_mms_tts,
                )
            except Exception as exc:
                logger.error("Startup load error: %s", exc, exc_info=True)
                _loading_state["error"] = str(exc)
            finally:
                _loading_state["done"] = True

        threading.Thread(target=_bg, daemon=True).start()
        threading.Thread(target=_bg, daemon=True).start()

    while not _loading_state["done"]:
        yield _loading_banner_html(
            _loading_state["step"], TOTAL_STEPS,
            _loading_state["label"], _loading_state["dl"],
        )
    while not _loading_state["done"]:
        yield _loading_banner_html(
            _loading_state["step"], TOTAL_STEPS,
            _loading_state["label"], _loading_state["dl"],
        )
        time.sleep(0.35)

    if _loading_state.get("error"):
        err = _loading_state["error"]
    if _loading_state.get("error"):
        err = _loading_state["error"]
        if "No internet connection" in err:
            label = (
                "No internet connection — "
                "No internet connection — "
                "AfriLingua needs to download model weights on first run. "
                "Please connect to the internet and reload this page."
            )
        else:
            label = f"Loading failed: {err}"
            label = f"Loading failed: {err}"
        yield _loading_banner_html(0, TOTAL_STEPS, label, None)
    else:
        yield _loading_banner_html(TOTAL_STEPS, TOTAL_STEPS, "All models ready", None)
        yield _loading_banner_html(TOTAL_STEPS, TOTAL_STEPS, "All models ready", None)


def build_ui() -> gr.Blocks:
    src_choices = ["auto-detect"] + _LANGUAGE_CHOICES

    with gr.Blocks(title="AfriLingua") as demo:
    with gr.Blocks(title="AfriLingua") as demo:

        # ── Styled header ──────────────────────────────────────────────────
        gr.HTML(
            '<div class="al-header">'
            '<div class="al-product-mark">'
            '<p class="al-title">AfriLingua</p>'
            '<p class="al-subtitle">Real-time translation for English, Igbo, Yoruba, and Hausa</p>'
            '</div>'
            '<span class="al-status-pill">Local models</span>'
            '</div>'
            '<div class="al-product-mark">'
            '<p class="al-title">AfriLingua</p>'
            '<p class="al-subtitle">Real-time translation for English, Igbo, Yoruba, and Hausa</p>'
            '</div>'
            '<span class="al-status-pill">Local models</span>'
            '</div>'
        )

        # ── Hardware info bar ──────────────────────────────────────────────
        gr.HTML(value=_hw_info_html())

        # Loading progress banner — visible until all models are ready
        loading_banner = gr.HTML(
            value=_loading_banner_html(0, TOTAL_STEPS, "Waiting for server …"),
            label="",
        )

        # ── Tab 1: Live Conversation ──────────────────────────────────────────
        with gr.Tab("Live"):
            gr.HTML('<div class="al-section-title">Live interpretation</div>')
        with gr.Tab("Live"):
            gr.HTML('<div class="al-section-title">Live interpretation</div>')
            with gr.Row():
                live_src = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="english", label="Source language"
                    _LANGUAGE_CHOICES, value="english", label="Source language"
                )
                live_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="igbo", label="Target language"
                    _LANGUAGE_CHOICES, value="igbo", label="Target language"
                )
            live_audio_in = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Microphone",
                label="Microphone",
            )
            live_transcript = gr.Textbox(
                label="Transcript",
                label="Transcript",
                lines=18,
                interactive=False,
                value="Waiting for speech ...",
                value="Waiting for speech ...",
            )
            live_audio_out = gr.Audio(
                label="Audio output",
                label="Audio output",
                type="numpy",
                autoplay=True,
            )
            clear_btn = gr.Button("Clear Transcript", variant="secondary")

            transcript_state = gr.State([])

            live_audio_in.stream(
                fn=live_translate_chunk,
                inputs=[live_audio_in, live_src, live_tgt, transcript_state],
                outputs=[transcript_state, live_transcript, live_audio_out],
            )
            clear_btn.click(
                fn=clear_transcript,
                inputs=[],
                outputs=[transcript_state, live_transcript, live_audio_out],
            )

        # ── Tab 2: Document Translation ───────────────────────────────────────
        with gr.Tab("Documents"):
            gr.HTML('<div class="al-section-title">Document translation</div>')
        with gr.Tab("Documents"):
            gr.HTML('<div class="al-section-title">Document translation</div>')
            with gr.Row():
                doc_src = gr.Dropdown(
                    src_choices, value="auto-detect", label="Source language"
                    src_choices, value="auto-detect", label="Source language"
                )
                doc_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="igbo", label="Target language"
                    _LANGUAGE_CHOICES, value="igbo", label="Target language"
                )
            doc_upload   = gr.File(
                label="Upload document",
                label="Upload document",
                file_types=[".txt", ".pdf", ".docx"]
            )
            doc_btn = gr.Button("Translate document", variant="primary")
            doc_btn = gr.Button("Translate document", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    doc_preview = gr.Textbox(
                        label="Translation preview",
                        label="Translation preview",
                        lines=14,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    doc_download = gr.File(label="Translated file")
                    doc_download = gr.File(label="Translated file")

            doc_btn.click(
                fn=doc_translate,
                inputs=[doc_upload, doc_src, doc_tgt],
                outputs=[doc_preview, doc_download],
            )

        # ── Tab 3: Text Translation ───────────────────────────────────────────
        with gr.Tab("Text"):
            gr.HTML('<div class="al-section-title">Text translation</div>')
        with gr.Tab("Text"):
            gr.HTML('<div class="al-section-title">Text translation</div>')
            with gr.Row():
                txt_src = gr.Dropdown(
                    src_choices, value="auto-detect", label="Source language"
                    src_choices, value="auto-detect", label="Source language"
                )
                txt_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="yoruba", label="Target language"
                    _LANGUAGE_CHOICES, value="yoruba", label="Target language"
                )
                play_tts = gr.Checkbox(label="Play audio", value=True)
                play_tts = gr.Checkbox(label="Play audio", value=True)
            with gr.Row():
                with gr.Column():
                    txt_input = gr.Textbox(
                        label="Input text",
                        label="Input text",
                        lines=8,
                        placeholder="Type or paste text here...",
                        placeholder="Type or paste text here...",
                    )
                with gr.Column():
                    txt_output = gr.Textbox(
                        label="Translation",
                        label="Translation",
                        lines=8,
                        interactive=False,
                        placeholder="Translation will appear here...",
                        placeholder="Translation will appear here...",
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    txt_btn = gr.Button("Translate", variant="primary")
                    txt_btn = gr.Button("Translate", variant="primary")
                with gr.Column(scale=3):
                    txt_status = gr.Markdown("")
            txt_audio = gr.Audio(
                label="Audio output",
                label="Audio output",
                type="numpy",
                autoplay=False
            )

            txt_btn.click(
                fn=text_translate,
                inputs=[txt_input, txt_src, txt_tgt, play_tts],
                outputs=[txt_output, txt_audio, txt_status],
            )
            txt_input.submit(
                fn=text_translate,
                inputs=[txt_input, txt_src, txt_tgt, play_tts],
                outputs=[txt_output, txt_audio, txt_status],
            )

        # Wire up the startup loader — fires once when the page loads
        demo.load(
            fn=_startup_load,
            inputs=[],
            outputs=[loading_banner],
        )

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────

def _first_available_port(start_port: int, tries: int = 20) -> int:
    """Return start_port if free, otherwise the next free local port."""
    for port in range(start_port, start_port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(
        f"Cannot find an empty Gradio port in range {start_port}-{start_port + tries - 1}."
    )


if __name__ == "__main__":
    demo = build_ui()
    port = _first_available_port(GRADIO_PORT)
    if port != GRADIO_PORT:
        logger.warning(
            "Port %s is already in use; launching AfriLingua on port %s instead.",
            GRADIO_PORT,
            port,
        )
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=Soft(),
        css=_CSS,
        theme=Soft(),
        css=_CSS,
    )
