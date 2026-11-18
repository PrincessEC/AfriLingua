"""Typed error hierarchy. API layers map these to HTTP / MCP errors."""
from __future__ import annotations


class AfriLinguaError(Exception):
    """Base class for all AfriLingua errors."""

    code: str = "internal_error"
    status: int = 500


class UnsupportedLanguageError(AfriLinguaError, ValueError):
    code = "unsupported_language"
    status = 400

    def __init__(self, lang: str, supported: list[str] | None = None) -> None:
        self.lang = lang
        self.supported = supported or []
        hint = f" Choose from: {', '.join(self.supported)}" if self.supported else ""
        super().__init__(f"Unsupported language: {lang!r}.{hint}")


class CapabilityUnavailableError(AfriLinguaError):
    """A language/model does not support the requested capability (e.g. no TTS)."""

    code = "capability_unavailable"
    status = 422


class ModelLoadError(AfriLinguaError):
    code = "model_load_failed"
    status = 503


class OfflineModelMissingError(ModelLoadError):
    code = "model_not_downloaded"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f"Model {model_id!r} is not in the local cache and the machine is offline. "
            f"Run `afrilingua models pull` while connected."
        )


class InvalidInputError(AfriLinguaError, ValueError):
    code = "invalid_input"
    status = 400
