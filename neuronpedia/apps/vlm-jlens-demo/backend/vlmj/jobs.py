# SPDX-License-Identifier: Apache-2.0
"""Single-worker job queue for the demo backend.

Every model-touching request becomes a job: one ``ThreadPoolExecutor`` with
``max_workers=1`` runs them in submission order, a dictionary keeps their
status, and ``GET /api/jobs/{id}`` reports ``stage``/``progress`` while the
worker is busy. The executor itself does not serialize against the synchronous
session-creation path — the engine's global lock does that, because both run
model forwards on the same process-wide model.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

#: Finished jobs kept for polling before the oldest are dropped.
MAX_HISTORY = 128

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Job functions are always built with ``functools.partial`` at the app's call sites, and
# pyright cannot narrow a ParamSpec-based ``partial[...]`` to a one-argument protocol, so
# the signature stays permissive; the worker calls ``fn(ctx)`` in every case and a
# mismatched signature surfaces as that job's error status.
JobFunction = Callable[..., Any]
JobWrapper = Callable[["JobContext", JobFunction], Any]


@dataclass
class JobRecord:
    """Mutable job state; read through :meth:`JobManager.snapshot`."""

    job_id: str
    kind: str
    status: str = STATUS_QUEUED
    stage: str = STATUS_QUEUED
    progress: float = 0.0
    t_submit: float = field(default_factory=time.time)
    t_start: float | None = None
    t_end: float | None = None
    error: str | None = None
    result: Any = None

    def snapshot(self) -> dict[str, Any]:
        """Wire shape of ``GET /api/jobs/{job_id}``."""
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "t_submit": self.t_submit,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "error": self.error,
            "result": self.result,
        }


class JobContext:
    """Progress handle passed to a job function (single writer: the worker thread)."""

    def __init__(self, manager: JobManager, record: JobRecord) -> None:
        self._manager = manager
        self._record = record

    @property
    def job_id(self) -> str:
        return self._record.job_id

    def update(self, stage: str, progress: float) -> None:
        """Set the human-readable stage and a monotonic 0..1 progress value."""
        self._manager.set_progress(self._record, stage, progress)


class JobManager:
    """Queue + status store around one worker thread.

    Args:
        wrapper: Optional callable running ``fn(ctx)`` under the engine's global
            lock / ``torch.inference_mode``; when omitted jobs run bare (tests).
    """

    def __init__(self, wrapper: JobWrapper | None = None) -> None:
        self._wrapper = wrapper
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlmj-job")
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ submit
    def submit(self, kind: str, fn: JobFunction) -> str:
        """Queue ``fn`` and return its job id."""
        record = JobRecord(job_id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[record.job_id] = record
            self._prune_locked()
        self._executor.submit(self._run, record, fn)
        return record.job_id

    def _prune_locked(self) -> None:
        if len(self._jobs) <= MAX_HISTORY:
            return
        for job_id, record in list(self._jobs.items()):
            if len(self._jobs) <= MAX_HISTORY:
                break
            if record.status in (STATUS_DONE, STATUS_ERROR):
                del self._jobs[job_id]

    # ------------------------------------------------------------------ worker
    def _run(self, record: JobRecord, fn: JobFunction) -> None:
        with self._lock:
            record.status = STATUS_RUNNING
            record.stage = "starting"
            record.t_start = time.time()
        ctx = JobContext(self, record)
        try:
            result = self._wrapper(ctx, fn) if self._wrapper is not None else fn(ctx)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the client verbatim
            with self._lock:
                record.status = STATUS_ERROR
                record.stage = "error"
                record.error = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                record.status = STATUS_DONE
                record.stage = "done"
                record.progress = 1.0
                record.result = result
        finally:
            with self._lock:
                record.t_end = time.time()

    # ------------------------------------------------------------------ status
    def set_progress(self, record: JobRecord, stage: str, progress: float) -> None:
        with self._lock:
            record.stage = str(stage)
            record.progress = max(record.progress, min(1.0, max(0.0, float(progress))))

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return None if record is None else record.snapshot()

    def counts(self) -> dict[str, int]:
        """``{"active": running, "queued": waiting}`` for ``/api/meta``."""
        with self._lock:
            active = sum(1 for job in self._jobs.values() if job.status == STATUS_RUNNING)
            queued = sum(1 for job in self._jobs.values() if job.status == STATUS_QUEUED)
        return {"active": active, "queued": queued}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def job_status_counts(jobs: Mapping[str, JobRecord]) -> dict[str, int]:
    """Counts over a raw job mapping (helper for tests/CLI)."""
    active = sum(1 for job in jobs.values() if job.status == STATUS_RUNNING)
    queued = sum(1 for job in jobs.values() if job.status == STATUS_QUEUED)
    return {"active": active, "queued": queued}


__all__ = [
    "MAX_HISTORY",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "JobContext",
    "JobManager",
    "JobRecord",
    "job_status_counts",
]
