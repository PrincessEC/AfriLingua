"""
ModelManager — the single owner of every loaded model in the process.

* Lazy: a backend is constructed and loaded on first use.
* Shared: every session / request / WebSocket uses the same instances
  (fixes v1 bug where each WebSocket loaded ~5 GB of models).
* Serialized: one inference lock per backend instance, so non-thread-safe
  libraries are safe to call from worker threads.
* Observable: ``status()`` reports load state, load time and call counts.

Usage::

    manager = ModelManager()
    manager.register("mt", lambda: CT2NLLBBackend(...))
    with manager.use("mt") as mt:          # loads on first use, holds lock
        mt.translate_batch([...], "english", "yoruba")
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from afrilingua.backends.base import Backend
from afrilingua.core.errors import ModelLoadError

logger = logging.getLogger(__name__)

B = TypeVar("B", bound=Backend)


@dataclass
class _Slot:
    factory: Callable[[], Backend]
    description: str = ""
    instance: Backend | None = None
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    use_lock: threading.RLock = field(default_factory=threading.RLock)
    load_seconds: float | None = None
    calls: int = 0
    last_used: float = 0.0
    error: str | None = None


class ModelManager:
    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}
        self._registry_lock = threading.Lock()

    # ── registration ────────────────────────────────────────────────────────
    def register(self, key: str, factory: Callable[[], Backend], *, description: str = "", replace: bool = False) -> None:
        with self._registry_lock:
            if key in self._slots and not replace:
                return
            old = self._slots.get(key)
            if old and old.instance is not None:
                with contextlib.suppress(Exception):
                    old.instance.unload()
            self._slots[key] = _Slot(factory=factory, description=description)

    def has(self, key: str) -> bool:
        return key in self._slots

    def keys(self) -> list[str]:
        return list(self._slots)

    # ── access ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> Backend:
        """Return the loaded backend (loading it if necessary) WITHOUT locking."""
        slot = self._slot(key)
        if slot.instance is not None and slot.instance.is_loaded():
            return slot.instance
        with slot.load_lock:
            if slot.instance is None:
                slot.instance = slot.factory()
            if not slot.instance.is_loaded():
                t0 = time.perf_counter()
                try:
                    slot.instance.load()
                except Exception as exc:
                    slot.error = f"{type(exc).__name__}: {exc}"
                    logger.exception("Failed to load model %r", key)
                    if isinstance(exc, ModelLoadError):
                        raise
                    raise ModelLoadError(f"Could not load {key!r}: {exc}") from exc
                slot.load_seconds = round(time.perf_counter() - t0, 2)
                slot.error = None
                logger.info("Loaded %s (%s) in %.2fs", key, slot.instance.name, slot.load_seconds)
        return slot.instance

    @contextlib.contextmanager
    def use(self, key: str) -> Iterator[Any]:
        """Load-on-demand and hold the backend's inference lock for the block."""
        backend = self.get(key)
        slot = self._slots[key]
        with slot.use_lock:
            slot.calls += 1
            slot.last_used = time.time()
            yield backend

    def preload(self, keys: list[str], on_progress: Callable[[int, int, str], None] | None = None) -> None:
        for i, key in enumerate(keys, start=1):
            if on_progress:
                on_progress(i - 1, len(keys), f"Loading {key}")
            self.get(key)
        if on_progress:
            on_progress(len(keys), len(keys), "Ready")

    def unload(self, key: str) -> None:
        slot = self._slot(key)
        with slot.load_lock, slot.use_lock:
            if slot.instance is not None:
                slot.instance.unload()
            slot.instance = None
            slot.load_seconds = None

    def unload_idle(self, max_idle_s: float) -> list[str]:
        now = time.time()
        evicted = []
        for key, slot in list(self._slots.items()):
            if slot.instance is not None and slot.instance.is_loaded() and now - slot.last_used > max_idle_s:
                self.unload(key)
                evicted.append(key)
        return evicted

    def status(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, slot in self._slots.items():
            inst = slot.instance
            out[key] = {
                "description": slot.description,
                "backend": getattr(inst, "name", None),
                "loaded": bool(inst is not None and inst.is_loaded()),
                "load_seconds": slot.load_seconds,
                "calls": slot.calls,
                "error": slot.error,
            }
        return out

    def _slot(self, key: str) -> _Slot:
        try:
            return self._slots[key]
        except KeyError as exc:
            raise KeyError(f"No model registered under {key!r}. Registered: {sorted(self._slots)}") from exc
