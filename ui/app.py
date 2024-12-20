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

import base64
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from gradio.themes import Soft
import numpy as np
import soundfile as sf

from src.config import API_HOST, API_PORT, GRADIO_PORT
from src.detection.lang_detector import LanguageDetector
from src.document.translator import DocumentTranslator
from src.pipeline.realtime_pipeline import RealtimePipeline, TranslationResult
from src.translation.engine import TranslationEngine, SUPPORTED_LANGUAGES
from src.tts.english_tts import EnglishTTS
from src.tts.mms_tts import MMSTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Shared singletons (lazy-loaded on first use) ──────────────────────────────
_engine       = TranslationEngine()
_detector     = LanguageDetector()
_doc_trans    = DocumentTranslator()
_mms_tts      = MMSTTS()
_eng_tts      = EnglishTTS()

_LANGUAGE_CHOICES = ["english", "igbo", "yoruba", "hausa"]
_LIVE_PIPELINE: dict[str, RealtimePipeline] = {}   # keyed by session ID


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
    """
    if audio_chunk is None:
        return transcript_state, _format_transcript(transcript_state), None

    sample_rate, data = audio_chunk
    audio_np = data.astype(np.float32) / 32768.0  # int16 → float32

    # Build a one-shot pipeline for this chunk (no mic capture loop)
    results: list[TranslationResult] = []

    def capture(r: TranslationResult):
        results.append(r)

    pipeline = RealtimePipeline(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        on_result=capture,
        audio_output=True,
    )
    pipeline.process_audio_chunk(audio_np)

    for r in results:
        transcript_state.append({
            "speaker": f"Speaker ({r.src_lang})",
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
    lines = []
    for entry in state:
        lines.append(
            f"[{entry['speaker']}]\n"
            f"  Original:   {entry['original']}\n"
            f"  {entry['lang'].capitalize()}: {entry['translated']}\n"
        )
    return "\n".join(lines) if lines else "Waiting for speech …"


def clear_transcript():
    return [], "Waiting for speech …", None


# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    src_choices = ["auto-detect"] + _LANGUAGE_CHOICES

    with gr.Blocks(title="AfriLingua", theme=Soft()) as demo:
        gr.Markdown(
            """
            # 🌍 AfriLingua
            **Real-time AI Translation — English ↔ Igbo · Yoruba · Hausa**
            Powered by Meta NLLB-200 · MMS · Whisper
            """
        )

        # ── Tab 1: Live Conversation ──────────────────────────────────────────
        with gr.Tab("🎙️ Live Conversation"):
            gr.Markdown(
                "Speak into your microphone. Your speech is transcribed and translated in real-time."
            )
            with gr.Row():
                live_src = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="english", label="Your Language (Speaker)"
                )
                live_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="igbo", label="Translate Into"
                )
            live_audio_in = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Microphone Input",
            )
            live_transcript = gr.Textbox(
                label="Live Transcript & Translation",
                lines=14,
                interactive=False,
                value="Waiting for speech …",
            )
            live_audio_out = gr.Audio(
                label="Translated Audio Output",
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
        with gr.Tab("📄 Document Translation"):
            gr.Markdown(
                "Upload a `.txt`, `.pdf`, or `.docx` file. "
                "The document is translated sentence-by-sentence."
            )
            with gr.Row():
                doc_src = gr.Dropdown(
                    src_choices, value="auto-detect", label="Source Language"
                )
                doc_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="igbo", label="Translate Into"
                )
            doc_upload   = gr.File(label="Upload Document", file_types=[".txt", ".pdf", ".docx"])
            doc_btn      = gr.Button("Translate Document", variant="primary")
            doc_preview  = gr.Textbox(label="Translation Preview", lines=16, interactive=False)
            doc_download = gr.File(label="Download Translated File")

            doc_btn.click(
                fn=doc_translate,
                inputs=[doc_upload, doc_src, doc_tgt],
                outputs=[doc_preview, doc_download],
            )

        # ── Tab 3: Text Translation ───────────────────────────────────────────
        with gr.Tab("✍️ Text Translation"):
            gr.Markdown("Type or paste text and translate instantly.")
            with gr.Row():
                txt_src = gr.Dropdown(
                    src_choices, value="auto-detect", label="Source Language"
                )
                txt_tgt = gr.Dropdown(
                    _LANGUAGE_CHOICES, value="yoruba", label="Translate Into"
                )
            txt_input  = gr.Textbox(label="Input Text", lines=6, placeholder="Enter text here …")
            play_tts   = gr.Checkbox(label="Play translated audio", value=True)
            txt_btn    = gr.Button("Translate", variant="primary")
            txt_output = gr.Textbox(label="Translation", lines=6, interactive=False)
            txt_audio  = gr.Audio(label="Audio Output", type="numpy", autoplay=False)
            txt_status = gr.Markdown("")

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

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )
