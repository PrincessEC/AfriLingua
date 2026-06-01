"""Stateful Gradio handlers for the AfriLingua web workspace."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
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
    parse_input_device,
)

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

    def microphone_choices(self) -> list[str]:
        return input_device_choices()

    def start_live(
        self,
        src_lang: str,
        tgt_lang: str,
        mic_choice: str | None,
        speak_translation: bool,
    ) -> tuple[str, str, str]:
        src = (src_lang or "").lower()
        tgt = (tgt_lang or "").lower()
        if not src or not tgt:
            return self._status_html("Choose source and target languages.", "warn"), self.transcript_text(), format_resource_strip()
        if src == tgt:
            return self._status_html("Source and target languages must be different.", "warn"), self.transcript_text(), format_resource_strip()

        input_device = parse_input_device(mic_choice)
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
            )
            try:
                self._pipeline.start_microphone()
            except Exception as exc:
                logger.error("Could not start microphone: %s", exc, exc_info=True)
                self._pipeline = None
                self._status = f"Microphone could not start: {exc}"
                return self._status_html(self._status, "error"), self.transcript_text(), format_resource_strip()

            mic_label = mic_choice or "Default input"
            self._status = f"Listening: {src.title()} -> {tgt.title()} on {mic_label}"
            return self._status_html(self._status, "live"), self.transcript_text(), format_resource_strip()

    def stop_live(self) -> tuple[str, str, str]:
        with self._lock:
            self._stop_locked()
            self._status = "Paused"
        return self._status_html("Paused. Microphone capture is off.", "idle"), self.transcript_text(), format_resource_strip()

    def reset_workspace(self) -> tuple[str, str, str, str]:
        with self._lock:
            self._stop_locked()
            self._entries.clear()
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
        return self._status_html(status, tone), self.transcript_text(), format_resource_strip()

    def refresh_devices(self):
        return gr.update(choices=self.microphone_choices(), value="Default input")

    def clear_cache(self) -> tuple[str, str, str]:
        message = clear_accelerator_cache()
        return self._status_html(message, "idle"), format_resource_strip(), self.settings_report()

    def settings_report(self) -> str:
        return format_settings_report(self._loaded_model_state())

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
            self._status = f"Last translation: {result.src_lang.title()} -> {result.tgt_lang.title()}"

    def _stop_locked(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop_microphone()
            finally:
                self._pipeline = None

    def _loaded_model_state(self) -> dict[str, object]:
        tts_loaded = ", ".join(self._mms_tts.loaded_languages()) or "none"
        return {
            "Translation": "loaded" if self._engine.is_loaded() else "not loaded",
            "Whisper ASR": "loaded" if self._whisper.is_loaded() else "lazy",
            "MMS ASR": "loaded" if self._mms_asr.is_loaded() else "lazy",
            "MMS TTS": tts_loaded,
            "Languages": ", ".join(self._languages),
        }

    @staticmethod
    def _status_html(message: str, tone: str = "idle") -> str:
        tone = tone if tone in {"idle", "live", "warn", "error"} else "idle"
        return f'<div class="al-live-status al-live-status-{tone}">{escape(message)}</div>'


def current_resource_strip() -> str:
    return format_resource_strip(get_runtime_snapshot())
