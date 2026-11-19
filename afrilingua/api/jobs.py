"""Background document-translation jobs.

Small thread pool; state in memory + ``<data_dir>/jobs/<id>/job.json`` so finished
jobs survive a restart. Jobs (and their files) are removed after 24 h.
"""
from __future__ import annotations

import contextlib
import json
import logging
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from afrilingua.core.errors import AfriLinguaError

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "error"]
DocumentTranslator = Callable[..., Path]

JOB_TTL_S = 24 * 3600


@dataclass
class Job:
    id: str
    filename: str
    tgt_lang: str
    src_lang: str | None
    src_path: str
    status: JobState = "queued"
    progress: float = 0.0
    error: str | None = None
    error_code: str | None = None
    result_path: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "progress": round(min(max(self.progress, 0.0), 1.0), 4),
            "filename": self.filename,
            "tgt_lang": self.tgt_lang,
            "src_lang": self.src_lang,
            "created_at": self.created_at,
        }
        if self.error:
            out["error"] = self.error
        if self.status == "done":
            out["download_url"] = f"/api/v1/documents/{self.id}/download"
            out["result_filename"] = Path(self.result_path or "").name
        return out


def _default_translator() -> DocumentTranslator:
    from afrilingua.documents import translate_document

    return translate_document


class JobManager:
    def __init__(
        self,
        engine: Any,
        root: Path,
        *,
        max_workers: int = 2,
        ttl_s: float = JOB_TTL_S,
        translator: DocumentTranslator | None = None,
    ) -> None:
        self.engine = engine
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s
        self._translator = translator
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="afrilingua-doc")
        self._load_existing()

    # ── persistence ─────────────────────────────────────────────────────────
    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        try:
            d = self._job_dir(job.id)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "job.json.tmp"
            tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
            tmp.replace(d / "job.json")
        except OSError:
            logger.exception("Could not persist job %s", job.id)

    def _load_existing(self) -> None:
        for meta in self.root.glob("*/job.json"):
            try:
                job = Job(**json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Skipping unreadable job file %s", meta)
                continue
            if job.status in {"queued", "running"}:
                job.status, job.error, job.error_code = "error", "Interrupted by a server restart.", "interrupted"
                self._save(job)
            self._jobs[job.id] = job
        self.cleanup()

    # ── API ─────────────────────────────────────────────────────────────────
    @property
    def translator(self) -> DocumentTranslator:
        if self._translator is None:
            self._translator = _default_translator()
        return self._translator

    def ensure_available(self) -> None:
        """Raise CapabilityUnavailableError if the documents module is missing."""
        from afrilingua.core.errors import CapabilityUnavailableError

        try:
            _ = self.translator
        except ImportError as exc:
            raise CapabilityUnavailableError(f"Document translation is not available in this build: {exc}") from exc

    def submit(self, filename: str, data: bytes, tgt_lang: str, src_lang: str | None = None) -> Job:
        self.ensure_available()
        job_id = secrets.token_hex(8)
        safe_name = Path(filename).name or "document.txt"
        d = self._job_dir(job_id)
        (d / "in").mkdir(parents=True, exist_ok=True)
        src_path = d / "in" / safe_name
        src_path.write_bytes(data)
        job = Job(id=job_id, filename=safe_name, tgt_lang=tgt_lang, src_lang=src_lang, src_path=str(src_path))
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def summary(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {"total": len(self._jobs)}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return counts

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.progress = max(job.progress, 0.01)
        self._save(job)
        last_save = [0.0]

        def on_progress(*args: Any, **_kw: Any) -> None:
            value: float | None = None
            nums = [a for a in args if isinstance(a, int | float) and not isinstance(a, bool)]
            if len(nums) >= 2 and nums[1]:
                value = float(nums[0]) / float(nums[1])
            elif len(nums) == 1:
                value = float(nums[0])
                if value > 1.0:  # percent
                    value /= 100.0
            if value is not None:
                job.progress = min(max(value, job.progress), 0.99)
                now = time.monotonic()
                if now - last_save[0] > 1.0:
                    last_save[0] = now
                    self._save(job)

        try:
            out_dir = self._job_dir(job.id) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            result = self.translator(
                self.engine, Path(job.src_path), out_dir, job.tgt_lang, src_lang=job.src_lang, on_progress=on_progress
            )
            job.result_path = str(result)
            job.status, job.progress = "done", 1.0
        except AfriLinguaError as exc:
            job.status, job.error, job.error_code = "error", str(exc), exc.code
        except Exception as exc:
            logger.exception("Document job %s failed", job.id)
            job.status, job.error, job.error_code = "error", f"{type(exc).__name__}: {exc}", "internal_error"
        job.finished_at = time.time()
        self._save(job)

    def cleanup(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        removed: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status in {"done", "error"} and now - job.created_at > self.ttl_s:
                    self._jobs.pop(job_id, None)
                    removed.append(job_id)
        for job_id in removed:
            shutil.rmtree(self._job_dir(job_id), ignore_errors=True)
        # orphan folders without job.json
        for d in self.root.iterdir() if self.root.exists() else []:
            if d.is_dir() and not (d / "job.json").exists():
                with contextlib.suppress(OSError):
                    if now - d.stat().st_mtime > self.ttl_s:
                        shutil.rmtree(d, ignore_errors=True)
        return removed

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
