"""Parallel datasets / corpora upload & management router (``/api/v1/datasets``)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from afrilingua.core.errors import CapabilityUnavailableError, PayloadTooLargeError
from afrilingua.core.settings import Settings, get_settings

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


def _require_datasets(request: Request) -> Any:
    store = getattr(request.app.state, "datasets", None)
    if store is None:
        raise CapabilityUnavailableError("Corpus store is disabled on this server.")
    return store


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.get("")
async def list_datasets(request: Request) -> dict[str, Any]:
    store = _require_datasets(request)
    items = await run_in_threadpool(store.list)
    return {"items": [r.to_dict() for r in items]}


@router.get("/{corpus_id}")
async def get_dataset(corpus_id: str, request: Request) -> dict[str, Any]:
    store = _require_datasets(request)
    rec = await run_in_threadpool(store.get, corpus_id)
    return rec.to_dict()


@router.post("")
async def upload_dataset(
    request: Request,
    files: list[UploadFile] = File(...),
    format: str = Form(..., description="ebible|usfm|srt|vtt|lines|tsv|csv|jsonl|tmx|unaligned"),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    source_name: str = Form(..., description="Provenance title, e.g. 'Yoruba Public Domain Bible (eBible)'."),
    license: str = Form(..., description="public-domain|CC0-1.0|CC-BY-4.0|CC-BY-SA-4.0|research-only|proprietary-permission"),
    domain: str = Form("general"),
    add_to_tm: bool = Form(False),
) -> dict[str, Any]:
    store = _require_datasets(request)
    max_bytes = _settings(request).max_upload_mb * 1024 * 1024
    raw_files: list[tuple[str, bytes]] = []
    total_bytes = 0
    for up in files:
        chunk = await up.read(max_bytes - total_bytes + 1)
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise PayloadTooLargeError(f"Total dataset upload exceeds {_settings(request).max_upload_mb} MB.")
        raw_files.append((Path(up.filename or "upload.txt").name, chunk))

    def _ingest() -> dict[str, Any]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="afrilingua-corpus-"))
        try:
            paths: list[Path] = []
            for idx, (fname, data) in enumerate(raw_files):
                p = tmp_dir / f"{idx:02d}_{fname}"
                p.write_bytes(data)
                paths.append(p)
            rec = store.import_corpus(
                fmt=format,
                files=paths,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                source_name=source_name,
                license=license,
                domain=domain,
                add_to_tm=add_to_tm,
            )
            return rec.to_dict()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return await run_in_threadpool(_ingest)


@router.delete("/{corpus_id}")
async def delete_dataset(corpus_id: str, request: Request) -> dict[str, Any]:
    store = _require_datasets(request)
    return await run_in_threadpool(store.delete, corpus_id)
