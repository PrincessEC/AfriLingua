"""Stateful Gradio handlers for the AfriLingua web workspace."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from html import escape
from typing import Iterable

import gradio as gr

from src.pipeline.realtime_pipeline import RealtimePipeline, TranslationResult
from src.runtime.resources import (
    clear_accelerator_cache,
    format_resource_strip,
    format_settings_report,
    get_runtime_snapshot,
    input_device_choices,
    output_device_choices,
    parse_input_device,
    parse_output_device,
)
from src.runtime.sessions import list_sessions, load_session, save_session

logger = logging.getLogger(__name__)


@dataclass
class LiveEntry:
    index: int
    source: str
    target: str
    original: str
    translated: str
    created_at: float


class WebRuntime:
    """Owns the live microphone session and operational UI callbacks."""

    def __init__(
        self,
        *,
        engine,
        whisper,
        mms_asr,
        mms_tts,
        eng_tts,
        languages: Iterable[str],
    ) -> None:
        self._engine = engine
        self._whisper = whisper
        self._mms_asr = mms_asr
        self._mms_tts = mms_tts
        self._eng_tts = eng_tts
        self._languages = list(languages)
        self._lock = threading.RLock()
        self._pipeline: RealtimePipeline | None = None
        self._entries: list[LiveEntry] = []
        self._status = "Ready"
        self._started_at: float | None = None
        self._session_title = "Untitled session"

    def microphone_choices(self) -> list[str]:
        return input_device_choices()

    def output_choices(self) -> list[str]:
        return output_device_choices()

    def session_choices(self) -> list[str]:
        return list_sessions()

    def start_live(
        self,
        src_lang: str,
        tgt_lang: str,
        mic_choice: str | None,
        output_choice: str | None,
        speak_translation: bool,
    ) -> tuple[str, str, str]:
        src = (src_lang or "").lower()
        tgt = (tgt_lang or "").lower()
        if not src or not tgt:
            return self._status_html("Choose source and target languages.", "warn"), self.transcript_text(), format_resource_strip()
        if src == tgt:
            return self._status_html("Source and target languages must be different.", "warn"), self.transcript_text(), format_resource_strip()

        input_device = parse_input_device(mic_choice)
        output_device = parse_output_device(output_choice)
        with self._lock:
            self._stop_locked()
            bridge = self

            def _on_result(result: TranslationResult) -> None:
                bridge._append_result(result)

            self._pipeline = RealtimePipeline(
                src_lang=src,
                tgt_lang=tgt,
                on_result=_on_result,
                audio_output=bool(speak_translation),
                whisper=self._whisper,
                mms_asr=self._mms_asr,
                engine=self._engine,
                mms_tts=self._mms_tts,
                eng_tts=self._eng_tts,
                local_playback=bool(speak_translation),
                input_device=input_device,
                output_device=output_device,
            )
            try:
                self._pipeline.start_microphone()
            except Exception as exc:
                logger.error("Could not start microphone: %s", exc, exc_info=True)
                self._pipeline = None
                self._status = f"Microphone could not start: {exc}"
                return self._status_html(self._status, "error"), self.transcript_text(), format_resource_strip()

            mic_label = mic_choice or "Default input"
            self._started_at = time.time()
            self._status = f"Listening: {src.title()} -> {tgt.title()} on {mic_label}"
            return self._status_html(self._status, "live", self._elapsed()), self.transcript_text(), format_resource_strip()

    def stop_live(self) -> tuple[str, str, str]:
        with self._lock:
            self._stop_locked()
            self._started_at = None
            self._status = "Paused"
        return self._status_html("Paused. Microphone capture is off.", "idle"), self.transcript_text(), format_resource_strip()

    def reset_workspace(self) -> tuple[str, str, str, str]:
        with self._lock:
            self._stop_locked()
            self._entries.clear()
            self._started_at = None
            self._status = "Workspace reset"
        cleanup = clear_accelerator_cache()
        status = self._status_html(f"Workspace reset. {cleanup}", "idle")
        return status, self.transcript_text(), format_resource_strip(), self.settings_report()

    def clear_transcript(self) -> tuple[str, str]:
        with self._lock:
            self._entries.clear()
        return self.transcript_text(), self._status_html("Transcript cleared.", "idle")

    def refresh(self) -> tuple[str, str, str]:
        with self._lock:
            running = bool(self._pipeline and self._pipeline.is_running())
            status = self._status
        tone = "live" if running else "idle"
        return self._status_html(status, tone, self._elapsed() if running else None), self.transcript_text(), format_resource_strip()

    def refresh_devices(self):
        return (
            gr.update(choices=self.microphone_choices(), value="Default microphone"),
            gr.update(choices=self.output_choices(), value="Default speakers"),
        )

    def clear_cache(self) -> tuple[str, str, str]:
        message = clear_accelerator_cache()
        return self._status_html(message, "idle"), format_resource_strip(), self.settings_report()

    def settings_report(self) -> str:
        return format_settings_report(self._loaded_model_state())

    def session_title(self) -> str:
        return self._session_title

    def save_current_session(self, title: str) -> tuple[str, gr.Dropdown]:
        with self._lock:
            entries = [asdict(entry) for entry in self._entries]
            self._session_title = title.strip() or self._session_title
        if not entries:
            return self._status_html("There is no transcript to save yet.", "warn"), gr.update(choices=self.session_choices())
        session_id = save_session(
            self._session_title,
            entries,
            metadata={"entry_count": len(entries)},
        )
        return self._status_html(f"Saved session: {self._session_title}", "idle"), gr.update(
            choices=self.session_choices(),
            value=next((choice for choice in self.session_choices() if f"[{session_id}]" in choice), None),
        )

    def load_saved_session(self, selection: str | None) -> tuple[str, str, str, gr.Dropdown]:
        if not selection:
            return self._status_html("Choose a saved session first.", "warn"), self.transcript_text(), self._session_title, gr.update()
        payload = load_session(selection)
        raw_entries = payload.get("entries") or []
        entries: list[LiveEntry] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            entries.append(
                LiveEntry(
                    index=int(item.get("index") or len(entries) + 1),
                    source=str(item.get("source") or ""),
                    target=str(item.get("target") or ""),
                    original=str(item.get("original") or ""),
                    translated=str(item.get("translated") or ""),
                    created_at=float(item.get("created_at") or time.time()),
                )
            )
        with self._lock:
            self._stop_locked()
            self._entries = entries
            self._session_title = str(payload.get("title") or "Untitled session")
            self._status = f"Loaded session: {self._session_title}"
            self._started_at = None
        return (
            self._status_html(self._status, "idle"),
            self.transcript_text(),
            self._session_title,
            gr.update(choices=self.session_choices(), value=selection),
        )

    def new_session(self) -> tuple[str, str, str, gr.Dropdown]:
        with self._lock:
            self._stop_locked()
            self._entries.clear()
            self._session_title = "Untitled session"
            self._started_at = None
            self._status = "New session"
        return (
            self._status_html("New session ready.", "idle"),
            self.transcript_text(),
            self._session_title,
            gr.update(choices=self.session_choices(), value=None),
        )

    def language_summary(self) -> str:
        from src.languages import LANGUAGE_REGISTRY

        lines = ["Language", "--------"]
        for name, spec in LANGUAGE_REGISTRY.items():
            speech = []
            if spec.whisper_lang:
                speech.append(f"Whisper {spec.whisper_lang}")
            if spec.mms_asr_code:
                speech.append(f"MMS ASR {spec.mms_asr_code}")
            if spec.mms_tts_model:
                speech.append("TTS")
            lines.append(f"{name}: {spec.nllb_code} | {', '.join(speech) or 'text only'}")
        return "\n".join(lines)

    def add_language(
        self,
        name: str,
        nllb_code: str,
        mms_asr_code: str,
        mms_tts_model: str,
        whisper_lang: str,
        aliases: str,
        detection_words: str,
    ):
        from src import languages as language_module
        from src.asr import mms_asr as mms_asr_module
        from src.detection import lang_detector as detector_module
        from src.tts import mms_tts as mms_tts_module

        spec = language_module.register_language_from_values(
            name=name,
            nllb_code=nllb_code,
            mms_asr_code=mms_asr_code,
            mms_tts_model=mms_tts_model,
            whisper_lang=whisper_lang,
            aliases=aliases,
            detection_words=detection_words,
        )
        if spec.mms_asr_code:
            mms_asr_module.MMS_LANG_MAP[spec.name] = spec.mms_asr_code
        if spec.mms_tts_model:
            mms_tts_module._LANG_TO_MODEL_ID[spec.name] = spec.mms_tts_model
            if spec.name not in mms_tts_module.SUPPORTED_LANGUAGES:
                mms_tts_module.SUPPORTED_LANGUAGES.append(spec.name)
        if spec.detection_words:
            detector_module._HEURISTIC_WORDS[spec.name] = {word.lower() for word in spec.detection_words}
        self._languages = list(language_module.TEXT_LANGUAGE_CHOICES)

        src_choices = ["auto-detect"] + self._languages
        status = self._status_html(f"Added language: {spec.name}", "idle")
        return (
            status,
            self.language_summary(),
            gr.update(choices=self._languages, value=spec.name),
            gr.update(choices=self._languages, value=spec.name),
            gr.update(choices=src_choices, value="auto-detect"),
            gr.update(choices=self._languages),
            gr.update(choices=src_choices, value="auto-detect"),
            gr.update(choices=self._languages),
        )

    def cache_model(self, model_id: str, model_role: str) -> tuple[str, str]:
        if not model_id.strip():
            return self._status_html("Enter a Hugging Face model ID.", "warn"), self.settings_report()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model_id.strip())
        except Exception as exc:
            logger.error("Model cache failed: %s", exc, exc_info=True)
            return self._status_html(f"Model cache failed: {exc}", "error"), self.settings_report()
        return self._status_html(f"Cached {model_role}: {model_id.strip()}", "idle"), self.settings_report()

    def transcript_text(self) -> str:
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return "Start a live session and speak into the selected microphone."

        lines: list[str] = []
        for entry in entries[-40:]:
            lines.extend(
                [
                    f"#{entry.index}  {entry.source.title()} -> {entry.target.title()}",
                    entry.original,
                    f"=> {entry.translated}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _append_result(self, result: TranslationResult) -> None:
        with self._lock:
            self._entries.append(
                LiveEntry(
                    index=len(self._entries) + 1,
                    source=result.src_lang,
                    target=result.tgt_lang,
                    original=result.original_text,
                    translated=result.translated_text,
                    created_at=time.time(),
                )
            )
            self._status = f"Listening: {result.src_lang.title()} -> {result.tgt_lang.title()}"

    def _stop_locked(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop_microphone()
            finally:
                self._pipeline = None
                self._started_at = None

    def _loaded_model_state(self) -> dict[str, object]:
        tts_loaded = ", ".join(self._mms_tts.loaded_languages()) or "none"
        return {
            "Translation": "loaded" if self._engine.is_loaded() else "not loaded",
            "Whisper ASR": "loaded" if self._whisper.is_loaded() else "lazy",
            "MMS ASR": "loaded" if self._mms_asr.is_loaded() else "lazy",
            "MMS TTS": tts_loaded,
            "Languages": ", ".join(self._languages),
        }

    def _elapsed(self) -> int | None:
        if self._started_at is None:
            return None
        return max(0, int(time.time() - self._started_at))

    @staticmethod
    def _status_html(message: str, tone: str = "idle", elapsed_seconds: int | None = None) -> str:
        tone = tone if tone in {"idle", "live", "warn", "error"} else "idle"
        elapsed = ""
        if elapsed_seconds is not None:
            minutes, seconds = divmod(elapsed_seconds, 60)
            elapsed = f'<span class="al-live-elapsed">{minutes:02d}:{seconds:02d}</span>'
        dot = '<span class="al-listening-dot"></span>' if tone == "live" else ""
        return (
            f'<div class="al-live-status al-live-status-{tone}">'
            f'<span class="al-live-main">{dot}{escape(message)}</span>{elapsed}'
            '</div>'
        )


def current_resource_strip() -> str:
    return format_resource_strip(get_runtime_snapshot())
