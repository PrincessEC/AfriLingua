"""
Local (desktop / CLI) interpreter: microphone → StreamingSession → speakers.

Usage::

    engine = build_engine()
    interp = LocalInterpreter(engine, "english", ["yoruba"], speak=True,
                              on_event=print)
    interp.start()
    ...
    interp.stop()
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from afrilingua.audio.capture import MicrophoneCapture
from afrilingua.audio.pcm import decode_audio
from afrilingua.audio.player import AudioPlayer
from afrilingua.pipeline import Event, EventCallback
from afrilingua.pipeline.streaming import StreamingSession

if TYPE_CHECKING:
    from afrilingua.engine import AfriLinguaEngine

logger = logging.getLogger(__name__)


class LocalInterpreter:
    """Wire the host microphone and speakers to a :class:`StreamingSession`."""

    def __init__(
        self,
        engine: AfriLinguaEngine,
        src_lang: str,
        targets: Sequence[str],
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        speak: bool = True,
        speak_lang: str | None = None,
        on_event: EventCallback | None = None,
        player: AudioPlayer | None = None,
        partials: bool | None = None,
    ) -> None:
        self.engine = engine
        self.on_event = on_event
        self.speak = speak
        self.session = StreamingSession(
            engine,
            src_lang,
            list(targets),
            synthesize_audio=speak,
            on_event=self._handle,
            partials=partials,
        )
        tgts = self.session.targets
        self.speak_lang = engine.registry.normalize(speak_lang) if speak_lang else (tgts[0] if tgts else None)
        self.player = player or AudioPlayer(device=output_device)
        self.capture = MicrophoneCapture(self.session.feed_float32, device=input_device)
        self._running = False

    def _handle(self, event: Event) -> None:
        if event.get("type") == "audio" and self.speak and event.get("lang") == self.speak_lang:
            try:
                audio, sr = decode_audio(base64.b64decode(event["data"]))
                self.player.play(audio, sr)
            except Exception:
                logger.exception("Could not play synthesized audio")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("on_event callback raised")

    def set_targets(self, targets: Sequence[str], *, speak_lang: str | None = None) -> None:
        """Change target languages (and optionally the spoken one) while running."""
        self.session.set_targets(list(targets))
        if speak_lang:
            self.speak_lang = self.engine.registry.normalize(speak_lang)
            self.player.clear()

    def start(self) -> LocalInterpreter:
        """Start pipeline workers, the player and the microphone."""
        if self._running:
            return self
        self.session.start()
        if self.speak:
            self.player.start()
        self.capture.start()
        self._running = True
        return self

    def stop(self) -> None:
        """Stop the microphone, flush the pipeline and stop playback."""
        if not self._running:
            return
        self._running = False
        self.capture.stop()
        self.session.close()
        if self.speak:
            self.player.wait_idle(timeout=15)
            self.player.stop()

    @property
    def running(self) -> bool:
        return self._running

    def __enter__(self) -> LocalInterpreter:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
