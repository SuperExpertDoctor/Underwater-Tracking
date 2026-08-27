"""Thread-safe operational frame publication and human-input queueing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from underwater_tracking.api.dependencies import RuntimePort
from underwater_tracking.domain.ui_models import OperationalFrame


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[OperationalFrame]


class OperationalHub:
    """Broadcast validated frames without coupling the engine to FastAPI.

    ``publish`` is synchronous and safe to call from the simulation thread.
    Each WebSocket gets a bounded queue; if a slow browser falls behind, its
    oldest queued frame is discarded and the newest frame remains available.
    This preserves live continuity without allowing an operator connection
    to back-pressure the high-frequency tracking loop.
    """

    def __init__(self, *, queue_size: int = 32) -> None:
        if queue_size < 2:
            raise ValueError("queue_size must be at least 2")
        self._queue_size = queue_size
        self._latest: OperationalFrame | None = None
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_subscriber = 0
        self._lock = Lock()

    def publish(self, frame: OperationalFrame) -> None:
        """Store and fan out one already-validated operational frame."""
        with self._lock:
            self._latest = frame
            subscribers = tuple(self._subscribers.items())
        for subscriber_id, subscriber in subscribers:
            def deliver(
                subscriber: _Subscriber = subscriber,
                subscriber_id: int = subscriber_id,
            ) -> None:
                try:
                    if subscriber.queue.full():
                        subscriber.queue.get_nowait()
                    subscriber.queue.put_nowait(frame)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    # A concurrent disconnect can race the bounded queue
                    # operation; the next frame will retry delivery.
                    return
                except RuntimeError:
                    with self._lock:
                        self._subscribers.pop(subscriber_id, None)

            try:
                subscriber.loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                with self._lock:
                    self._subscribers.pop(subscriber_id, None)

    def snapshot(self) -> OperationalFrame | None:
        """Return the latest frame, or ``None`` before the first publish."""
        with self._lock:
            return self._latest

    @property
    def subscriber_count(self) -> int:
        """Number of active async subscribers, useful for lifecycle checks."""
        with self._lock:
            return len(self._subscribers)

    async def stream(self) -> AsyncIterator[OperationalFrame]:
        """Yield the latest frame first, then every subsequent live frame."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[OperationalFrame] = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            subscriber_id = self._next_subscriber
            self._next_subscriber += 1
            self._subscribers[subscriber_id] = _Subscriber(loop, queue)
            latest = self._latest
        try:
            if latest is not None:
                yield latest
            while True:
                yield await queue.get()
        finally:
            with self._lock:
                self._subscribers.pop(subscriber_id, None)


@dataclass
class _DirectiveJob:
    request_id: str
    text: str
    author: str
    expected_plan_version: int
    target_ids: tuple[str, ...]
    execution_revision: int | None = None
    frame_id: int | None = None
    assignment_target_id: str | None = None
    assignment_uuv_ids: tuple[str, ...] = ()
    status: str = "queued"
    directive: dict[str, object] | None = None
    error: str | None = None


class DirectiveQueueFull(RuntimeError):
    """The bounded operator-input queue has no available job slot."""


class RuntimeDirectiveQueue:
    """Run directive parsing/apply work away from the event loop.

    Preview and apply are separate operations.  The HTTP handler only
    creates a job; the worker calls the runtime after the request has
    returned, so a slow provider cannot pause WebSocket frame delivery.
    """

    _TERMINAL_STATUSES = frozenset({"preview", "applied", "rejected", "error"})

    def __init__(self, runtime: RuntimePort, *, workers: int = 2, max_jobs: int = 256) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        self._runtime = runtime
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._max_jobs = max_jobs
        self._jobs: dict[str, _DirectiveJob] = {}
        self._lock = Lock()
        self._closed = False

    def submit(
        self,
        *,
        text: str,
        author: str,
        expected_plan_version: int,
        target_ids: Sequence[str],
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> str:
        request_id = f"directive:{uuid4().hex[:12]}"
        job = _DirectiveJob(
            request_id=request_id,
            text=text,
            author=author,
            expected_plan_version=expected_plan_version,
            target_ids=tuple(sorted(set(target_ids))),
            execution_revision=execution_revision,
            frame_id=frame_id,
        )
        with self._lock:
            self._reserve_job_locked(job)
        try:
            self._executor.submit(self._run_preview, request_id)
        except RuntimeError:
            with self._lock:
                self._jobs.pop(request_id, None)
            raise DirectiveQueueFull("directive queue is closed")
        return request_id

    def submit_assignment(
        self,
        *,
        uuv_ids: Sequence[str],
        target_id: str,
        expected_plan_version: int,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> str:
        request_id = f"assignment:{uuid4().hex[:12]}"
        job = _DirectiveJob(
            request_id=request_id,
            text="",
            author="operator",
            expected_plan_version=expected_plan_version,
            target_ids=(target_id,),
            execution_revision=execution_revision,
            frame_id=frame_id,
            assignment_target_id=target_id,
            assignment_uuv_ids=tuple(sorted(set(uuv_ids))),
        )
        with self._lock:
            self._reserve_job_locked(job)
        try:
            self._executor.submit(self._run_preview, request_id)
        except RuntimeError:
            with self._lock:
                self._jobs.pop(request_id, None)
            raise DirectiveQueueFull("directive queue is closed")
        return request_id

    def _reserve_job_locked(self, job: _DirectiveJob) -> None:
        if self._closed:
            raise DirectiveQueueFull("directive queue is closed")
        while len(self._jobs) >= self._max_jobs:
            evicted = next(
                (
                    request_id
                    for request_id, existing in self._jobs.items()
                    if existing.status in self._TERMINAL_STATUSES
                ),
                None,
            )
            if evicted is None:
                raise DirectiveQueueFull("directive queue is full")
            self._jobs.pop(evicted)
        self._jobs[job.request_id] = job

    def _run_preview(self, request_id: str) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                return
            job.status = "processing"
        try:
            assert job is not None
            if job.assignment_target_id is not None:
                kwargs: dict[str, object] = {
                    "uuv_ids": job.assignment_uuv_ids,
                    "target_id": job.assignment_target_id,
                }
                if job.execution_revision is not None:
                    kwargs.update(
                        execution_revision=job.execution_revision,
                        frame_id=job.frame_id,
                    )
                directive = self._runtime.preview_assignment(**kwargs)
            else:
                if job.execution_revision is None:
                    directive = self._runtime.preview_directive(job.text)
                else:
                    directive = self._runtime.preview_directive(
                        job.text,
                        execution_revision=job.execution_revision,
                        frame_id=job.frame_id,
                    )
            with self._lock:
                job.status = directive.status
                job.directive = directive.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - status is surfaced to the UI
            with self._lock:
                job.status = "error"
                job.error = str(exc)

    def status(self, request_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                return {"request_id": request_id, "status": "unknown"}
            result: dict[str, object] = {
                "request_id": job.request_id,
                "status": job.status,
                "author": job.author,
                "expected_plan_version": job.expected_plan_version,
                "target_ids": list(job.target_ids),
            }
            if job.execution_revision is not None:
                result["execution_revision"] = job.execution_revision
                result["frame_id"] = job.frame_id
            if job.directive is not None:
                result["directive"] = job.directive
            if job.error is not None:
                result["error"] = job.error
            return result

    def apply(self, request_id: str) -> None:
        """Apply a completed preview only if its plan version is still current."""
        with self._lock:
            if self._closed:
                raise DirectiveQueueFull("directive queue is closed")
            job = self._jobs.get(request_id)
            if job is None or job.directive is None:
                raise ValueError(f"directive request {request_id!r} has no preview")
            if job.status != "preview":
                raise ValueError(
                    f"directive request {request_id!r} is not ready to apply"
                )
            directive_id = str(job.directive["directive_id"])
            expected_plan_version = job.expected_plan_version
        current_plan_version = _runtime_plan_version(self._runtime)
        if current_plan_version != expected_plan_version:
            raise ValueError(
                "the operational plan changed after preview; "
                f"expected {expected_plan_version}, current {current_plan_version}"
            )
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.directive is None:
                raise ValueError(f"directive request {request_id!r} has no preview")
            job.status = "applying"
        try:
            self._executor.submit(self._run_apply, request_id, directive_id)
        except RuntimeError as exc:
            with self._lock:
                job = self._jobs.get(request_id)
                if job is not None:
                    job.status = "error"
                    job.error = "directive queue is closed"
            raise DirectiveQueueFull("directive queue is closed") from exc

    def _run_apply(self, request_id: str, directive_id: str) -> None:
        try:
            directive = self._runtime.apply_directive(directive_id)
            with self._lock:
                job = self._jobs[request_id]
                job.status = directive.status
                job.directive = directive.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - status is surfaced to the UI
            with self._lock:
                job = self._jobs[request_id]
                job.status = "error"
                job.error = str(exc)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def abort(self) -> None:
        """Cancel queued work without waiting for an in-flight provider call."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


def _runtime_plan_version(runtime: RuntimePort) -> int:
    """Read the authoritative execution revision before legacy plan revision."""
    reader = getattr(runtime, "current_plan_version", None)
    if callable(reader):
        return int(reader())
    execution_reader = getattr(runtime, "current_execution_snapshot", None)
    execution = execution_reader() if callable(execution_reader) else execution_reader
    if execution is not None:
        return int(execution.execution_revision)
    active_plan = runtime.active_plan()
    return active_plan.revision if active_plan is not None else 0
