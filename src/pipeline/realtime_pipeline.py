"""
src/pipeline/realtime_pipeline.py
Orchestrates the full real-time translation pipeline:
    Mic → VAD → ASR → LangDetect → NLLB → TTS → Speaker

Each pipeline instance is associated with one WebSocket session and holds
its own AudioCapture + AudioPlayer.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

from src.asr.audio_capture import AudioCapture
from src.asr.mms_asr import MMSASR
from src.asr.whisper_asr import WhisperASR
from src.detection.lang_detector import LanguageDetector
from src.translation.engine import TranslationEngine
from src.tts.audio_player import AudioPlayer
from src.tts.english_tts import EnglishTTS
from src.tts.mms_tts import MMSTTS

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    original_text:    str
    translated_text:  str
    src_lang:         str
    tgt_lang:         str
    audio_b64:        Optional[str] = None   # base64-encoded WAV bytes


class RealtimePipeline:
    """
    One pipeline per speaker / session.

    Args:
        src_lang:       Assigned source language for this speaker
                        ("english", "igbo", "yoruba", "hausa").
        tgt_lang:       Language to translate into.
        on_result:      Callback fired with a TranslationResult when
                        an utterance has been fully processed.
        audio_output:   Whether to synthesise TTS and queue audio playback.
    """

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        on_result: Callable[[TranslationResult], None],
        audio_output: bool = True,
    ) -> None:
        self.src_lang = src_lang.lower()
        self.tgt_lang = tgt_lang.lower()
        self.on_result = on_result
        self.audio_output = audio_output

        # Shared, lazily initialised singletons (created once per process)
        self._whisper  = WhisperASR()
        self._mms_asr  = MMSASR()
        self._detector = LanguageDetector()
        self._engine   = TranslationEngine()
        self._mms_tts  = MMSTTS()
        self._eng_tts  = EnglishTTS()
        self._player   = AudioPlayer()

        self._capture: Optional[AudioCapture] = None
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # ASR dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        if self.src_lang == "english":
            return self._whisper.transcribe(audio, language="en")
        return self._mms_asr.transcribe(audio, lang=self.src_lang)

    # ──────────────────────────────────────────────────────────────────────────
    # TTS dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        if lang == "english":
            return self._eng_tts.synthesize(text)
        return self._mms_tts.synthesize(text, lang=lang)

    @staticmethod
    def _to_wav_b64(waveform: np.ndarray, sample_rate: int) -> str:
        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ──────────────────────────────────────────────────────────────────────────
    # Core utterance processor (called from AudioCapture callback thread)
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_utterance(self, audio: np.ndarray) -> None:
        try:
            # 1. ASR
            original_text = self._transcribe(audio)
            if not original_text.strip():
                return

            logger.info("[ASR %s] %s", self.src_lang, original_text)

            # 2. Translation
            translated_text = self._engine.translate(
                original_text, self.src_lang, self.tgt_lang
            )
            logger.info("[NLLB %s→%s] %s", self.src_lang, self.tgt_lang, translated_text)

            # 3. TTS
            audio_b64: Optional[str] = None
            if self.audio_output and translated_text.strip():
                waveform, sr = self._synthesize(translated_text, self.tgt_lang)
                self._player.play(waveform, sr)
                audio_b64 = self._to_wav_b64(waveform, sr)

            # 4. Fire result callback
            result = TranslationResult(
                original_text=original_text,
                translated_text=translated_text,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                audio_b64=audio_b64,
            )
            self.on_result(result)

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Process a pre-recorded audio chunk (used by WebSocket endpoint)
    # ──────────────────────────────────────────────────────────────────────────

    def process_audio_chunk(self, audio: np.ndarray) -> Optional[TranslationResult]:
        """
        Synchronously process a single audio array and return the result.
        Use this when audio arrives over WebSocket (already chunked by client).
        """
        results: list[TranslationResult] = []

        def _capture(r: TranslationResult) -> None:
            results.append(r)

        original_cb = self.on_result
        self.on_result = _capture
        self._handle_utterance(audio)
        self.on_result = original_cb

        return results[0] if results else None

    # ──────────────────────────────────────────────────────────────────────────
    # Live mic mode
    # ──────────────────────────────────────────────────────────────────────────

    def start_microphone(self) -> None:
        """Start continuous mic capture + real-time translation."""
        with self._lock:
            if self._capture is not None:
                return
            self._player.start()
            self._capture = AudioCapture(on_utterance=self._handle_utterance)
            self._capture.start()
            logger.info("Live pipeline started: %s → %s", self.src_lang, self.tgt_lang)

    def stop_microphone(self) -> None:
        """Stop mic capture."""
        with self._lock:
            if self._capture is None:
                return
            self._capture.stop()
            self._capture = None
            self._player.stop()
            logger.info("Live pipeline stopped.")

    def is_running(self) -> bool:
        return self._capture is not None
