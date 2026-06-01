"""AfriLingua Gradio web workspace."""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from html import escape
from pathlib import Path

# Configure project imports and Hugging Face cache paths before importing Gradio.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GRADIO_PORT  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from gradio.themes import Soft  # noqa: E402

from src.asr.mms_asr import MMSASR  # noqa: E402
from src.asr.whisper_asr import WhisperASR  # noqa: E402
from src.detection.lang_detector import LanguageDetector  # noqa: E402
from src.document.translator import DocumentTranslator  # noqa: E402
from src.languages import TEXT_LANGUAGE_CHOICES  # noqa: E402
from src.model_loader import TOTAL_STEPS, load_all  # noqa: E402
from src.translation.engine import TranslationEngine  # noqa: E402
from src.tts.english_tts import EnglishTTS  # noqa: E402
from src.tts.mms_tts import MMSTTS  # noqa: E402
from ui.web_runtime import WebRuntime, current_resource_strip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_engine = TranslationEngine()
_whisper = WhisperASR()
_mms_asr = MMSASR()
_detector = LanguageDetector()
_doc_trans = DocumentTranslator(engine=_engine, detector=_detector)
_mms_tts = MMSTTS()
_eng_tts = EnglishTTS()

_LANGUAGE_CHOICES = TEXT_LANGUAGE_CHOICES
_web_runtime = WebRuntime(
    engine=_engine,
    whisper=_whisper,
    mms_asr=_mms_asr,
    mms_tts=_mms_tts,
    eng_tts=_eng_tts,
    languages=_LANGUAGE_CHOICES,
)

_loading_lock = threading.Lock()
_loading_state: dict[str, object] = {
    "started": False,
    "done": False,
    "error": "",
    "step": 0,
    "total": TOTAL_STEPS,
    "label": "Preparing workspace",
}

_CSS = """
.gradio-container {
    max-width: 1180px !important;
}
.al-workspace-shell {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 18px;
    padding: 22px 4px 18px;
    border-bottom: 1px solid rgba(107, 116, 108, 0.22);
    margin-bottom: 14px;
}
.al-brand-name {
    margin: 0;
    color: #17201b;
    font-size: 30px;
    line-height: 1.1;
    font-weight: 760;
}
.al-brand-subtitle {
    margin: 6px 0 0;
    color: #5e6a61;
    font-size: 13px;
}
.al-workspace-badge {
    border: 1px solid #b8c9bd;
    color: #1d6f5f;
    background: #eef6f1;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.al-resource-strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 180px;
    align-items: center;
    gap: 16px;
    margin: 8px 0 16px;
    padding: 12px 14px;
    border: 1px solid #d9dfd2;
    border-radius: 8px;
    background: #fbfcf8;
}
.al-resource-copy {
    display: flex;
    align-items: baseline;
    gap: 12px;
    min-width: 0;
    flex-wrap: wrap;
}
.al-resource-kicker {
    color: #7b857b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
}
.al-resource-copy strong {
    color: #17201b;
    font-size: 14px;
}
.al-resource-copy span:last-child {
    color: #5e6a61;
    font-size: 12px;
}
.al-resource-meter {
    height: 8px;
    border-radius: 999px;
    background: #e5e8df;
    overflow: hidden;
}
.al-resource-meter span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #1d6f5f, #d95f43);
}
.al-loader {
    margin: 6px 0 14px;
    padding: 12px 14px;
    border: 1px solid #f0d5c9;
    border-radius: 8px;
    background: #fff7f2;
}
.al-loader-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: #17201b;
    font-size: 13px;
    font-weight: 700;
}
.al-loader-track {
    height: 7px;
    margin-top: 10px;
    background: #efd8cc;
    border-radius: 999px;
    overflow: hidden;
}
.al-loader-track span {
    display: block;
    height: 100%;
    background: #d95f43;
}
.al-live-status {
    border-radius: 8px;
    padding: 10px 12px;
    border: 1px solid #d9dfd2;
    background: #fbfcf8;
    color: #17201b;
    font-size: 13px;
    font-weight: 700;
}
.al-live-status-live {
    border-color: #9fc7b5;
    background: #eef7f2;
    color: #1d6f5f;
}
.al-live-status-warn {
    border-color: #e4bd86;
    background: #fff7e8;
    color: #7a5c00;
}
.al-live-status-error {
    border-color: #efb1a2;
    background: #fff1ed;
    color: #9a3412;
}
textarea, input, select, button {
    border-radius: 8px !important;
}
footer {
    display: none !important;
}
html[data-afri-theme="dark"] .gradio-container {
    background: #111714 !important;
    color: #edf2ec !important;
}
html[data-afri-theme="dark"] .al-brand-name {
    color: #edf2ec;
}
html[data-afri-theme="dark"] .al-brand-subtitle {
    color: #afbbaf;
}
html[data-afri-theme="dark"] .al-workspace-shell {
    border-bottom-color: rgba(197, 207, 196, 0.18);
}
html[data-afri-theme="dark"] .al-workspace-badge,
html[data-afri-theme="dark"] .al-resource-strip,
html[data-afri-theme="dark"] .al-live-status {
    background: #17201b;
    border-color: #304137;
}
html[data-afri-theme="dark"] .al-resource-copy strong,
html[data-afri-theme="dark"] .al-live-status {
    color: #edf2ec;
}
html[data-afri-theme="dark"] .al-resource-copy span:last-child,
html[data-afri-theme="dark"] .al-resource-kicker {
    color: #afbbaf;
}
html[data-afri-theme="dark"] .al-loader {
    background: #241b16;
    border-color: #5a3528;
}
html[data-afri-theme="dark"] .al-loader-top {
    color: #f6eee8;
}
@media (max-width: 720px) {
    .al-workspace-shell {
        align-items: flex-start;
        flex-direction: column;
    }
    .al-resource-strip {
        grid-template-columns: 1fr;
    }
}
"""

_THEME_HEAD = """
<script>
(function () {
  function applyTheme(choice) {
    var selected = choice || localStorage.getItem("afrilingua-theme") || "System";
    var dark = selected === "Dark" ||
      (selected === "System" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.afriTheme = dark ? "dark" : "light";
  }
  applyTheme();
  window.afriLinguaApplyTheme = function (choice) {
    localStorage.setItem("afrilingua-theme", choice || "System");
    applyTheme(choice);
    return "Theme set to " + (choice || "System") + ".";
  };
})();
</script>
"""

_THEME_JS = """
(theme) => {
  if (window.afriLinguaApplyTheme) {
    return window.afriLinguaApplyTheme(theme);
  }
  return "Theme set to " + (theme || "System") + ".";
}
"""


def _waveform_to_gradio_audio(waveform: np.ndarray, sample_rate: int):
    return (sample_rate, (waveform * 32767).astype(np.int16))


def _synthesize(text: str, lang: str):
    if lang == "english":
        waveform, sample_rate = _eng_tts.synthesize(text)
    else:
        waveform, sample_rate = _mms_tts.synthesize(text, lang=lang)
    return _waveform_to_gradio_audio(waveform, sample_rate)


def text_translate(input_text: str, src_lang: str, tgt_lang: str, play_tts: bool):
    if not input_text.strip():
        return "", None, "Please enter some text."

    src = src_lang if src_lang != "auto-detect" else _detector.detect_or_default(input_text, "english")
    try:
        translated = _engine.translate(input_text, src, tgt_lang)
    except ValueError as exc:
        return "", None, str(exc)

    audio = None
    status_parts = [f"Detected source: {src}"] if src_lang == "auto-detect" else []
    if play_tts:
        try:
            audio = _synthesize(translated, tgt_lang)
        except Exception as exc:
            logger.warning("TTS unavailable for %s: %s", tgt_lang, exc)
            status_parts.append(f"Audio unavailable for {tgt_lang}; text translation completed.")
    return translated, audio, " | ".join(status_parts)


def doc_translate(file_obj, src_lang: str, tgt_lang: str):
    if file_obj is None:
        return "No file uploaded.", None

    src = None if src_lang == "auto-detect" else src_lang
    try:
        out_path = _doc_trans.translate_file(file_obj.name, tgt_lang, src_lang=src)
        preview = out_path.read_text(encoding="utf-8", errors="replace")[:3000]
        if len(preview) == 3000:
            preview += "\n\n... (truncated - download full file)"
        return preview, str(out_path)
    except Exception as exc:
        logger.error("Document translation error: %s", exc, exc_info=True)
        return f"Error: {exc}", None


def _loader_html(step: int, total: int, label: str) -> str:
    total = max(total, 1)
    pct = max(0, min(100, int(step / total * 100)))
    safe_label = escape(label or "Preparing workspace")
    return (
        '<div class="al-loader">'
        '<div class="al-loader-top">'
        f'<span>{safe_label}</span><span>{pct}%</span>'
        '</div><div class="al-loader-track">'
        f'<span style="width:{pct}%"></span>'
        '</div></div>'
    )


def _startup_load_workspace():
    with _loading_lock:
        should_start = not bool(_loading_state["started"])
        if should_start:
            _loading_state["started"] = True

    if should_start:
        def _on_progress(step: int, total: int, label: str) -> None:
            _loading_state.update(step=step, total=total, label=label)

        def _bg() -> None:
            try:
                load_all(
                    on_progress=_on_progress,
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

    while not bool(_loading_state["done"]):
        yield _loader_html(
            int(_loading_state.get("step") or 0),
            int(_loading_state.get("total") or TOTAL_STEPS),
            str(_loading_state.get("label") or "Preparing workspace"),
        )
        time.sleep(0.35)

    if _loading_state.get("error"):
        yield _loader_html(0, TOTAL_STEPS, f"Loading failed: {_loading_state['error']}")
    else:
        yield ""


def _refresh_settings() -> tuple[str, str]:
    return current_resource_strip(), _web_runtime.settings_report()


def _theme_label(theme_name: str) -> str:
    return f"Theme set to {theme_name or 'System'}."


def build_ui() -> gr.Blocks:
    src_choices = ["auto-detect"] + _LANGUAGE_CHOICES

    with gr.Blocks(title="AfriLingua Workspace", fill_width=True) as demo:
        gr.HTML(
            '<header class="al-workspace-shell">'
            '<div>'
            '<h1 class="al-brand-name">AfriLingua</h1>'
            '<p class="al-brand-subtitle">Local interpretation workspace for English, Igbo, Yoruba, and Hausa.</p>'
            '</div>'
            '<span class="al-workspace-badge">Private local workspace</span>'
            '</header>'
        )

        resource_strip = gr.HTML(value=current_resource_strip(), label="")
        loading_banner = gr.HTML(value=_loader_html(0, TOTAL_STEPS, "Preparing workspace"), label="")

        with gr.Tab("Live"):
            live_status = gr.HTML(value=_web_runtime.refresh()[0], label="")
            with gr.Row():
                live_src = gr.Dropdown(_LANGUAGE_CHOICES, value="english", label="Source")
                live_tgt = gr.Dropdown(_LANGUAGE_CHOICES, value="igbo", label="Target")
                mic_select = gr.Dropdown(
                    choices=_web_runtime.microphone_choices(),
                    value="Default input",
                    label="Microphone",
                )
            with gr.Row():
                speak_translation = gr.Checkbox(value=True, label="Speak translation")
                start_btn = gr.Button("Start", variant="primary")
                pause_btn = gr.Button("Pause", variant="secondary")
                clear_btn = gr.Button("Clear", variant="secondary")
                reset_btn = gr.Button("Reset workspace", variant="stop")
            live_transcript = gr.Textbox(
                label="Live transcript",
                value=_web_runtime.transcript_text(),
                lines=18,
                interactive=False,
            )

        with gr.Tab("Text"):
            with gr.Row():
                txt_src = gr.Dropdown(src_choices, value="auto-detect", label="Source")
                txt_tgt = gr.Dropdown(_LANGUAGE_CHOICES, value="yoruba", label="Target")
                play_tts = gr.Checkbox(label="Play audio", value=True)
            with gr.Row():
                txt_input = gr.Textbox(label="Input", lines=9, placeholder="Type or paste text here...")
                txt_output = gr.Textbox(
                    label="Translation",
                    lines=9,
                    interactive=False,
                    placeholder="Translation will appear here...",
                )
            with gr.Row():
                txt_btn = gr.Button("Translate", variant="primary")
                txt_status = gr.Markdown("")
            txt_audio = gr.Audio(label="Audio output", type="numpy", autoplay=False)

        with gr.Tab("Documents"):
            with gr.Row():
                doc_src = gr.Dropdown(src_choices, value="auto-detect", label="Source")
                doc_tgt = gr.Dropdown(_LANGUAGE_CHOICES, value="igbo", label="Target")
            doc_upload = gr.File(label="Upload", file_types=[".txt", ".pdf", ".docx"])
            doc_btn = gr.Button("Translate document", variant="primary")
            with gr.Row():
                doc_preview = gr.Textbox(label="Preview", lines=14, interactive=False)
                doc_download = gr.File(label="Translated file")

        with gr.Tab("Settings"):
            with gr.Row():
                theme_select = gr.Dropdown(["System", "Dark", "Light"], value="System", label="Theme")
                refresh_devices_btn = gr.Button("Refresh microphones", variant="secondary")
                refresh_settings_btn = gr.Button("Refresh runtime", variant="secondary")
                clear_cache_btn = gr.Button("Clear GPU cache", variant="secondary")
            settings_status = gr.HTML(value="", label="")
            theme_status = gr.Markdown("")
            settings_report = gr.Textbox(
                label="Runtime diagnostics",
                value=_web_runtime.settings_report(),
                lines=20,
                interactive=False,
            )

        start_btn.click(
            fn=_web_runtime.start_live,
            inputs=[live_src, live_tgt, mic_select, speak_translation],
            outputs=[live_status, live_transcript, resource_strip],
        )
        pause_btn.click(fn=_web_runtime.stop_live, inputs=[], outputs=[live_status, live_transcript, resource_strip])
        clear_btn.click(fn=_web_runtime.clear_transcript, inputs=[], outputs=[live_transcript, live_status])
        reset_btn.click(
            fn=_web_runtime.reset_workspace,
            inputs=[],
            outputs=[live_status, live_transcript, resource_strip, settings_report],
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
        doc_btn.click(fn=doc_translate, inputs=[doc_upload, doc_src, doc_tgt], outputs=[doc_preview, doc_download])
        refresh_settings_btn.click(fn=_refresh_settings, inputs=[], outputs=[resource_strip, settings_report])
        clear_cache_btn.click(fn=_web_runtime.clear_cache, inputs=[], outputs=[settings_status, resource_strip, settings_report])
        refresh_devices_btn.click(fn=_web_runtime.refresh_devices, inputs=[], outputs=[mic_select])
        theme_select.change(fn=_theme_label, inputs=[theme_select], outputs=[theme_status], js=_THEME_JS)

        timer = gr.Timer(value=1.5)
        timer.tick(fn=_web_runtime.refresh, inputs=[], outputs=[live_status, live_transcript, resource_strip])
        demo.load(fn=_startup_load_workspace, inputs=[], outputs=[loading_banner], show_progress="hidden")

    return demo


def _first_available_port(start_port: int, tries: int = 20) -> int:
    for port in range(start_port, start_port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"Cannot find an empty Gradio port in range {start_port}-{start_port + tries - 1}.")


if __name__ == "__main__":
    demo = build_ui()
    port = _first_available_port(GRADIO_PORT)
    if port != GRADIO_PORT:
        logger.warning("Port %s is already in use; launching on port %s instead.", GRADIO_PORT, port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        footer_links=[],
        theme=Soft(),
        css=_CSS,
        head=_THEME_HEAD,
    )
