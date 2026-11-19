"""AfriLingua Gradio web workspace."""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from html import escape
from pathlib import Path

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
:root {
    --al-bg: #f6f5ef;
    --al-surface: #ffffff;
    --al-surface-2: #f0f2ea;
    --al-ink: #171f1b;
    --al-muted: #657066;
    --al-line: #d9ded2;
    --al-green: #1d6f5f;
    --al-rust: #c45d35;
    --al-indigo: #485286;
    --al-warn: #806018;
    --al-danger: #9a3412;
}
html[data-afri-theme="dark"] {
    --al-bg: #101411;
    --al-surface: #171d19;
    --al-surface-2: #202821;
    --al-ink: #edf2ec;
    --al-muted: #aab5aa;
    --al-line: #344038;
    --al-green: #69c3a6;
    --al-rust: #e38a62;
    --al-indigo: #9aa6e8;
    --al-warn: #dfbd6a;
    --al-danger: #ef916f;
}
body, .gradio-container {
    background: var(--al-bg) !important;
    color: var(--al-ink) !important;
}
.gradio-container {
    max-width: 1280px !important;
}
.al-shell {
    padding: 20px 4px 8px;
}
.al-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--al-line);
}
.al-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}
.al-mark {
    width: 42px;
    height: 42px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--al-green), var(--al-rust));
    box-shadow: 0 10px 24px rgba(29, 111, 95, 0.22);
}
.al-brand h1 {
    margin: 0;
    color: var(--al-ink);
    font-size: 27px;
    line-height: 1.1;
    font-weight: 760;
}
.al-brand p {
    margin: 4px 0 0;
    color: var(--al-muted);
    font-size: 12px;
}
.al-theme-switch {
    display: inline-flex;
    gap: 4px;
    padding: 4px;
    border: 1px solid var(--al-line);
    border-radius: 8px;
    background: var(--al-surface);
}
.al-theme-button {
    border: 0;
    background: transparent;
    color: var(--al-muted);
    border-radius: 6px !important;
    padding: 7px 11px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
}
.al-theme-button[data-active="true"] {
    color: #ffffff;
    background: var(--al-green);
}
.al-resource-strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 190px;
    align-items: center;
    gap: 18px;
    margin: 16px 0 14px;
    padding: 13px 15px;
    border: 1px solid var(--al-line);
    border-radius: 8px;
    background: var(--al-surface);
}
.al-resource-copy {
    display: flex;
    align-items: baseline;
    gap: 13px;
    min-width: 0;
    flex-wrap: wrap;
}
.al-resource-kicker {
    color: var(--al-muted);
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
}
.al-resource-copy strong {
    color: var(--al-ink);
    font-size: 14px;
}
.al-resource-copy span:last-child {
    color: var(--al-muted);
    font-size: 12px;
}
.al-resource-meter {
    height: 8px;
    border-radius: 999px;
    background: var(--al-surface-2);
    overflow: hidden;
}
.al-resource-meter span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--al-green), var(--al-rust));
}
.al-loader {
    margin: 0 0 14px;
    padding: 12px 14px;
    border: 1px solid #e6c8b8;
    border-radius: 8px;
    background: color-mix(in srgb, var(--al-rust) 8%, var(--al-surface));
}
.al-loader-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: var(--al-ink);
    font-size: 13px;
    font-weight: 750;
}
.al-loader-track {
    height: 7px;
    margin-top: 10px;
    background: var(--al-surface-2);
    border-radius: 999px;
    overflow: hidden;
}
.al-loader-track span {
    display: block;
    height: 100%;
    background: var(--al-rust);
}
.al-live-status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    min-height: 42px;
    border-radius: 8px;
    padding: 10px 12px;
    border: 1px solid var(--al-line);
    background: var(--al-surface);
    color: var(--al-ink);
    font-size: 13px;
    font-weight: 750;
}
.al-live-status-live {
    border-color: color-mix(in srgb, var(--al-green) 45%, var(--al-line));
    background: color-mix(in srgb, var(--al-green) 10%, var(--al-surface));
    color: var(--al-green);
}
.al-live-status-warn {
    border-color: color-mix(in srgb, var(--al-warn) 45%, var(--al-line));
    background: color-mix(in srgb, var(--al-warn) 10%, var(--al-surface));
    color: var(--al-warn);
}
.al-live-status-error {
    border-color: color-mix(in srgb, var(--al-danger) 45%, var(--al-line));
    background: color-mix(in srgb, var(--al-danger) 10%, var(--al-surface));
    color: var(--al-danger);
}
.al-live-main {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
}
.al-live-elapsed {
    color: var(--al-muted);
    font-variant-numeric: tabular-nums;
}
.al-listening-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: var(--al-green);
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--al-green) 42%, transparent);
    animation: alPulse 1.35s infinite;
    flex: 0 0 auto;
}
@keyframes alPulse {
    70% { box-shadow: 0 0 0 9px transparent; }
    100% { box-shadow: 0 0 0 0 transparent; }
}
.al-section-title {
    color: var(--al-ink);
    font-size: 16px;
    font-weight: 780;
    margin: 2px 0 12px;
}
.al-subtitle {
    color: var(--al-muted);
    font-size: 12px;
    margin: -6px 0 12px;
}
.al-mini-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}
.al-mini-stat {
    border: 1px solid var(--al-line);
    border-radius: 8px;
    padding: 11px 12px;
    background: var(--al-surface);
}
.al-mini-stat span {
    display: block;
    color: var(--al-muted);
    font-size: 11px;
    font-weight: 750;
    text-transform: uppercase;
}
.al-mini-stat strong {
    display: block;
    color: var(--al-ink);
    font-size: 15px;
    margin-top: 4px;
}
textarea, input, select, button {
    border-radius: 8px !important;
}
footer {
    display: none !important;
}
@media (max-width: 760px) {
    .al-topbar {
        align-items: flex-start;
        flex-direction: column;
    }
    .al-resource-strip,
    .al-mini-grid {
        grid-template-columns: 1fr;
    }
}
"""

_THEME_HEAD = """
<script>
(function () {
  function selectedTheme(choice) {
    return choice || localStorage.getItem("afrilingua-theme") || "System";
  }
  function applyTheme(choice) {
    var selected = selectedTheme(choice);
    var dark = selected === "Dark" ||
      (selected === "System" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.afriTheme = dark ? "dark" : "light";
    document.documentElement.dataset.afriThemeChoice = selected;
    document.querySelectorAll("[data-afri-theme-choice]").forEach(function (button) {
      button.dataset.active = button.dataset.afriThemeChoice === selected ? "true" : "false";
    });
  }
  window.afriLinguaApplyTheme = function (choice) {
    localStorage.setItem("afrilingua-theme", choice || "System");
    applyTheme(choice);
  };
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-afri-theme-choice]");
    if (button) {
      event.preventDefault();
      window.afriLinguaApplyTheme(button.dataset.afriThemeChoice);
    }
  });
  applyTheme();
  new MutationObserver(function () { applyTheme(); }).observe(document.documentElement, {
    childList: true,
    subtree: true
  });
})();
</script>
"""


def _theme_switch_html() -> str:
    return (
        '<div class="al-theme-switch" aria-label="Theme">'
        '<button class="al-theme-button" data-afri-theme-choice="System" type="button">System</button>'
        '<button class="al-theme-button" data-afri-theme-choice="Light" type="button">Light</button>'
        '<button class="al-theme-button" data-afri-theme-choice="Dark" type="button">Dark</button>'
        '</div>'
    )


def _shell_html() -> str:
    return (
        '<div class="al-shell">'
        '<div class="al-topbar">'
        '<div class="al-brand">'
        '<div class="al-mark"></div>'
        '<div><h1>AfriLingua</h1><p>Local interpretation workspace</p></div>'
        '</div>'
        f'{_theme_switch_html()}'
        '</div>'
        '</div>'
    )


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


def build_ui() -> gr.Blocks:
    src_choices = ["auto-detect"] + _LANGUAGE_CHOICES

    with gr.Blocks(title="AfriLingua Workspace", fill_width=True) as demo:
        gr.HTML(_shell_html())
        resource_strip = gr.HTML(value=current_resource_strip(), label="")
        loading_banner = gr.HTML(value=_loader_html(0, TOTAL_STEPS, "Preparing workspace"), label="")

        with gr.Tab("Live Studio"):
            gr.HTML(
                '<div class="al-mini-grid">'
                '<div class="al-mini-stat"><span>Session</span><strong>Live Studio</strong></div>'
                '<div class="al-mini-stat"><span>Input</span><strong>Server audio</strong></div>'
                '<div class="al-mini-stat"><span>Privacy</span><strong>Local runtime</strong></div>'
                '</div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    live_status = gr.HTML(value=_web_runtime.refresh()[0], label="")
                    with gr.Row():
                        live_src = gr.Dropdown(_LANGUAGE_CHOICES, value="english", label="Source")
                        live_tgt = gr.Dropdown(_LANGUAGE_CHOICES, value="igbo", label="Target")
                    with gr.Row():
                        mic_select = gr.Dropdown(
                            choices=_web_runtime.microphone_choices(),
                            value="Default microphone",
                            label="Microphone",
                        )
                        output_select = gr.Dropdown(
                            choices=_web_runtime.output_choices(),
                            value="Default speakers",
                            label="Output",
                        )
                    with gr.Row():
                        speak_translation = gr.Checkbox(value=True, label="Speak translation")
                        start_btn = gr.Button("Start listening", variant="primary")
                        pause_btn = gr.Button("Pause", variant="secondary")
                        reset_btn = gr.Button("Reset", variant="stop")
                    live_transcript = gr.Textbox(
                        label="Transcript",
                        value=_web_runtime.transcript_text(),
                        lines=18,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    gr.HTML('<div class="al-section-title">Session</div>')
                    session_title = gr.Textbox(label="Title", value=_web_runtime.session_title())
                    saved_sessions = gr.Dropdown(
                        choices=_web_runtime.session_choices(),
                        label="Saved sessions",
                    )
                    with gr.Row():
                        save_session_btn = gr.Button("Save", variant="primary")
                        load_session_btn = gr.Button("Load", variant="secondary")
                        new_session_btn = gr.Button("New", variant="secondary")
                    clear_btn = gr.Button("Clear transcript", variant="secondary")
                    session_status = gr.HTML(value="", label="")

        with gr.Tab("Text"):
            with gr.Row():
                txt_src = gr.Dropdown(src_choices, value="auto-detect", label="Source")
                txt_tgt = gr.Dropdown(_LANGUAGE_CHOICES, value="yoruba", label="Target")
                play_tts = gr.Checkbox(label="Play audio", value=True)
            with gr.Row():
                txt_input = gr.Textbox(label="Input", lines=10, placeholder="Type or paste text here...")
                txt_output = gr.Textbox(label="Translation", lines=10, interactive=False)
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
                with gr.Column(scale=1):
                    gr.HTML('<div class="al-section-title">Appearance</div>' + _theme_switch_html())
                    gr.HTML('<div class="al-section-title">Audio Routing</div>')
                    refresh_devices_btn = gr.Button("Refresh audio devices", variant="secondary")
                    clear_cache_btn = gr.Button("Clear GPU cache", variant="secondary")
                    settings_status = gr.HTML(value="", label="")
                with gr.Column(scale=2):
                    settings_report = gr.Textbox(
                        label="Runtime diagnostics",
                        value=_web_runtime.settings_report(),
                        lines=14,
                        interactive=False,
                    )
                    refresh_settings_btn = gr.Button("Refresh runtime", variant="secondary")

            with gr.Accordion("Languages", open=False):
                language_status = gr.HTML(value="", label="")
                language_report = gr.Textbox(
                    label="Configured languages",
                    value=_web_runtime.language_summary(),
                    lines=8,
                    interactive=False,
                )
                with gr.Row():
                    lang_name = gr.Textbox(label="Name", placeholder="swahili")
                    lang_nllb = gr.Textbox(label="NLLB code", placeholder="swh_Latn")
                    lang_asr = gr.Textbox(label="MMS ASR code", placeholder="swh")
                with gr.Row():
                    lang_tts = gr.Textbox(label="MMS TTS model", placeholder="facebook/mms-tts-swh")
                    lang_whisper = gr.Textbox(label="Whisper code", placeholder="en")
                lang_aliases = gr.Textbox(label="Aliases", placeholder="kiswahili, sw")
                lang_words = gr.Textbox(label="Detection words", placeholder="habari, asante")
                add_language_btn = gr.Button("Add language", variant="primary")

            with gr.Accordion("Models", open=False):
                model_status = gr.HTML(value="", label="")
                with gr.Row():
                    model_role = gr.Dropdown(
                        ["translation", "english-asr", "mms-asr", "mms-tts"],
                        value="translation",
                        label="Model role",
                    )
                    model_id = gr.Textbox(label="Hugging Face model ID", placeholder="facebook/nllb-200-distilled-600M")
                cache_model_btn = gr.Button("Cache model", variant="primary")

        start_btn.click(
            fn=_web_runtime.start_live,
            inputs=[live_src, live_tgt, mic_select, output_select, speak_translation],
            outputs=[live_status, live_transcript, resource_strip],
        )
        pause_btn.click(fn=_web_runtime.stop_live, inputs=[], outputs=[live_status, live_transcript, resource_strip])
        clear_btn.click(fn=_web_runtime.clear_transcript, inputs=[], outputs=[live_transcript, session_status])
        reset_btn.click(
            fn=_web_runtime.reset_workspace,
            inputs=[],
            outputs=[live_status, live_transcript, resource_strip, settings_report],
        )
        save_session_btn.click(
            fn=_web_runtime.save_current_session,
            inputs=[session_title],
            outputs=[session_status, saved_sessions],
        )
        load_session_btn.click(
            fn=_web_runtime.load_saved_session,
            inputs=[saved_sessions],
            outputs=[live_status, live_transcript, session_title, saved_sessions],
        )
        new_session_btn.click(
            fn=_web_runtime.new_session,
            inputs=[],
            outputs=[live_status, live_transcript, session_title, saved_sessions],
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
        refresh_devices_btn.click(fn=_web_runtime.refresh_devices, inputs=[], outputs=[mic_select, output_select])
        add_language_btn.click(
            fn=_web_runtime.add_language,
            inputs=[lang_name, lang_nllb, lang_asr, lang_tts, lang_whisper, lang_aliases, lang_words],
            outputs=[language_status, language_report, live_src, live_tgt, txt_src, txt_tgt, doc_src, doc_tgt],
        )
        cache_model_btn.click(fn=_web_runtime.cache_model, inputs=[model_id, model_role], outputs=[model_status, settings_report])

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
