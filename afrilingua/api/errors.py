"""Error envelope + exception handlers.

Every JSON endpoint returns errors as ``{"error": {"code": "...", "message": "..."}}``
(docs/api-contract.md). Under ``/v1`` an OpenAI-style ``type`` field is added so the
official OpenAI SDKs surface the message nicely.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from afrilingua.core.errors import AfriLinguaError

logger = logging.getLogger(__name__)

_STATUS_CODES: dict[int, str] = {
    400: "invalid_input",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_input",
    429: "rate_limited",
    503: "unavailable",
}


class APIError(AfriLinguaError):
    """Ad-hoc API error with an explicit code/status (e.g. ``not_found``)."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.extra = extra


def not_found(what: str) -> APIError:
    return APIError(404, "not_found", f"{what} not found.")


def error_body(code: str, message: str, *, openai: bool = False, **extra: Any) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if openai:
        err["type"] = (
            "invalid_request_error" if code not in {"internal_error", "model_load_failed"} else "server_error"
        )
    err.update({k: v for k, v in extra.items() if v is not None})
    return {"error": err}


def error_response(
    status: int, code: str, message: str, request: Request | None = None, **extra: Any
) -> JSONResponse:
    openai = bool(request is not None and request.url.path.startswith("/v1"))
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(
        error_body(code, message, openai=openai, **extra), status_code=status, headers=headers
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AfriLinguaError)
    async def _afrilingua_error(request: Request, exc: AfriLinguaError) -> JSONResponse:
        if exc.status >= 500:
            logger.warning("%s on %s: %s", exc.code, request.url.path, exc)
        extra = getattr(exc, "extra", {}) or {}
        supported = getattr(exc, "supported", None)
        if supported:
            extra = {**extra, "supported": supported}
        return error_response(exc.status, exc.code, str(exc), request, **extra)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        resp = error_response(exc.status_code, code, message, request)
        if exc.headers:
            resp.headers.update(exc.headers)
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p not in {"body", "query", "path", "form"})
            details.append({"field": loc, "message": err.get("msg", "")})
        summary = "; ".join(f"{d['field'] or 'body'}: {d['message']}" for d in details) or "Invalid request."
        return error_response(400, "invalid_input", summary, request, details=details)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", f"{type(exc).__name__}: {exc}", request)
