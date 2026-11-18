"""Streaming interpretation pipeline."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import numpy as np

from afrilingua.engine import AfriLinguaEngine
from afrilingua.pipeline import EventCallback

logger = logging.getLogger(__name__)

class StreamingSession:
    def __init__(self, engine: AfriLinguaEngine, src_lang: str, targets: list[str],
                 synthesize_audio: bool, on_event: EventCallback) -> None:
        self.engine = engine
        self.src_lang = src_lang
        self.targets = targets
        self.synthesize_audio = synthesize_audio
        self.on_event = on_event
        self._running = False
        self._queue = queue.Queue()
        self._thread = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed_pcm16(self, data: bytes, sample_rate: int = 16_000) -> None:
        if not self._running: return
        # Convert bytes to numpy float32
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self._queue.put(audio)

    def feed_float32(self, samples: Any, sample_rate: int = 16_000) -> None:
        if not self._running: return
        self._queue.put(samples)

    def set_targets(self, targets: list[str]) -> None:
        self.targets = targets

    def close(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        while self._running:
            try:
                audio = self._queue.get(timeout=1.0)
                if audio is None: break

                # Simplified pipeline:
                # 1. ASR
                transcript = self.engine.transcribe(audio, self.src_lang)
                if not transcript.text: continue

                self.on_event({"type": "final", "text": transcript.text})

                # 2. MT
                for target in self.targets:
                    result = self.engine.translate_one(transcript.text, self.src_lang, target)
                    self.on_event({"type": "translation", "lang": target, "text": result.text})

                    # 3. TTS
                    if self.synthesize_audio:
                        audio_out = self.engine.synthesize(result.text, target)
                        self.on_event({"type": "audio", "lang": target, "data": audio_out.waveform.tobytes()})

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Pipeline error")
                self.on_event({"type": "error", "message": str(e)})
