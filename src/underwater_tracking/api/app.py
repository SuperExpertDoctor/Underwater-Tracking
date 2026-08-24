"""FastAPI transport for the operational command center."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
import re
from threading import Lock
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.encoders import jsonable_encoder
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Literal, cast

from underwater_tracking.agent.nodes.questions import QuestionAnswer, QuestionEvidenceError
from underwater_tracking.domain.conversation_models import AssistantMode, ConversationMessage
from underwater_tracking.api.dependencies import (
    DirectiveQueuePort,
    MemoryPort,
    QuestionPort,
    ReplayPort,
    RuntimePort,
)
from underwater_tracking.api.evaluation import EvaluationPort
from underwater_tracking.api.frame_builder import operational_frame_json, operational_frame_payload
from underwater_tracking.api.hub import (
    DirectiveQueueFull,
    OperationalHub,
    RuntimeDirectiveQueue,
)
from underwater_tracking.api.replay import ReplayIndexError
from underwater_tracking.domain.models import IntelligenceReport, OperationalScheme
from underwater_tracking.domain.memory_models import MemoryType
from underwater_tracking.domain.ui_models import PlanningHealthView
from underwater_tracking.runtime.run_catalog import RunCatalog, RunNotFoundError
from underwater_tracking.runtime.models import RunRequest


_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"


class DirectiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    author: str = Field(min_length=1, max_length=120)
    expected_plan_version: int = Field(ge=0)
    target_ids: tuple[str, ...] = ()


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    counterfactual: dict[str, object] | None = None


class ConversationMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(default="operator", min_length=1, max_length=120)
    assistant_mode: AssistantMode = "auto"
    text: str = Field(min_length=1, max_length=4000)
    expected_plan_version: int = Field(ge=0)
    target_scope: tuple[str, ...] = Field(
        default=(), validation_alias=AliasChoices("target_scope", "target_ids")
    )
    region_scope: tuple[str, ...] = Field(
        default=(), validation_alias=AliasChoices("region_scope", "region_ids")
    )
    evidence_ids: tuple[str, ...] = ()


class ConversationApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="operator", min_length=1, max_length=120)
    turn_id: str = Field(min_length=1, max_length=240)
    expected_plan_version: int = Field(ge=0)


class AssignmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1, max_length=120)
    uuv_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    expected_plan_version: int = Field(ge=0)


class SensorModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuv_id: str = Field(min_length=1, max_length=120)
    mode: Literal["passive", "active"]
    target_id: str | None = Field(default=None, max_length=120)
    expected_plan_version: int = Field(ge=0)


class PlanningRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_epoch_id: str | None = Field(default=None, max_length=240)


def _valid_identifier(value: str) -> bool:
    return bool(len(value) <= 240 and re.fullmatch(_IDENTIFIER_PATTERN, value))


class MemorySnapshotQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="operator", min_length=1, max_length=120, pattern=_IDENTIFIER_PATTERN)
    conversation_id: str = Field(min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)
    query: str = Field(default="", max_length=4000)
    memory_type: MemoryType | None = None
    min_importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=100, ge=1, le=128)


class MemoryVersionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="operator", min_length=1, max_length=120, pattern=_IDENTIFIER_PATTERN)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)


class MemoryDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="operator", min_length=1, max_length=120, pattern=_IDENTIFIER_PATTERN)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)


class MemoryStreamQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="operator", min_length=1, max_length=120, pattern=_IDENTIFIER_PATTERN)
    conversation_id: str = Field(min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN)
    after_cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=128)
    include_scenario_events: bool = True


def create_app(
    *,
    runtime: RuntimePort | None = None,
    memory_port: MemoryPort | None = None,
    replay: ReplayPort | None = None,
    controller: object | None = None,
    catalog: RunCatalog | None = None,
    directive_queue: DirectiveQueuePort | None = None,
    question_service: QuestionPort | None = None,
    hub: OperationalHub | None = None,
    evaluation: EvaluationPort | None = None,
    evaluation_enabled: bool = False,
    directive_job_limit: int = 256,
    web_ui_url: str | None = None,
    verification_audit: bool = False,
) -> FastAPI:
    """Create the transport app over injected runtime ports.

    ``evaluation_enabled`` is intentionally retained as an explicit gate for
    the later truth-only evaluation routes.  This operational app never
    mounts those routes, and its frames are always ``OperationalFrame``
    instances.
    """
    if controller is None and (runtime is None or replay is None):
        raise ValueError("runtime and replay are required without a controller")
    frame_hub = hub or OperationalHub()
    snapshot_cache_lock = Lock()
    snapshot_cache_frame: object | None = None
    snapshot_cache_body: bytes | None = None

    def current_runtime() -> RuntimePort:
        if controller is not None:
            return cast(RuntimePort, getattr(controller, "runtime"))
        assert runtime is not None
        return runtime

    def current_replay() -> ReplayPort:
        if controller is not None:
            return cast(ReplayPort, getattr(controller, "replay"))
        assert replay is not None
        return replay

    def current_hub() -> OperationalHub:
        if controller is not None:
            return cast(OperationalHub, getattr(controller, "hub"))
        return frame_hub

    def current_operational_snapshot_response() -> Response:
        """Return the cached JSON body for the immutable latest frame."""
        nonlocal snapshot_cache_frame, snapshot_cache_body
        frame = current_hub().snapshot()
        if frame is None:
            raise HTTPException(status_code=503, detail="operational frame is not ready")
        with snapshot_cache_lock:
            if frame is not snapshot_cache_frame or snapshot_cache_body is None:
                snapshot_cache_frame = frame
                snapshot_cache_body = operational_frame_json(frame).encode("utf-8")
            body = snapshot_cache_body
        return Response(content=body, media_type="application/json")

    def current_memory_port() -> MemoryPort:
        if memory_port is not None:
            return memory_port
        candidate = getattr(current_runtime(), "memory_port", None)
        if candidate is None:
            candidate = getattr(current_runtime(), "memory", None)
        if candidate is None:
            raise HTTPException(status_code=501, detail="memory service is unavailable")
        return cast(MemoryPort, candidate)

    class _ResolvedRuntime:
        def __getattr__(self, name: str) -> object:
            return getattr(current_runtime(), name)

    resolved_runtime = _ResolvedRuntime()
    resolved_runtime_port = cast(RuntimePort, resolved_runtime)
    queue = directive_queue or RuntimeDirectiveQueue(
        resolved_runtime_port,
        max_jobs=directive_job_limit,
    )
    questions: QuestionPort = question_service or cast(QuestionPort, resolved_runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        cancelled = False
        try:
            yield
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            aborted = bool(getattr(controller, "aborted", False))
            if cancelled or aborted:
                abort_queue = getattr(queue, "abort", None)
                if callable(abort_queue):
                    abort_queue()
                if controller is not None:
                    controller_abort = getattr(controller, "abort", None)
                    if callable(controller_abort):
                        controller_abort()
            else:
                try:
                    close = getattr(queue, "close", None)
                    if callable(close):
                        close()
                finally:
                    if controller is not None:
                        controller_close = getattr(controller, "close", None)
                        if callable(controller_close):
                            controller_close()

    app = FastAPI(
        title="Underwater Tracking Command Center", version="1.0", lifespan=lifespan
    )
    app.state.operational_hub = frame_hub
    app.state.directive_queue = queue

    @app.get("/", include_in_schema=False, response_model=None)
    async def service_root() -> JSONResponse | RedirectResponse:
        """Make the API port useful when opened from the one-command banner."""
        if web_ui_url:
            return RedirectResponse(url=web_ui_url, status_code=307)
        return JSONResponse(
            {
                "service": "underwater-tracking-api",
                "web_ui_url": None,
                "api_docs_url": "/docs",
                "message": "The command center UI is served separately; use /docs for the API.",
            }
        )

    def current_plan_version() -> int:
        frame = current_hub().snapshot()
        if frame is not None:
            return frame.plan_version
        plan = current_runtime().active_plan()
        return plan.revision if plan is not None else 0

    def current_sim_time_s() -> int | None:
        current = getattr(current_runtime(), "current_sim_time_s", None)
        return int(current()) if callable(current) else None

    def reject_expired_input(valid_until_s: int, input_name: str) -> None:
        current = current_sim_time_s()
        if current is not None and valid_until_s <= current:
            raise HTTPException(
                status_code=422,
                detail=f"{input_name} is already expired at simulation time {current}",
            )

    def current_planning_health(
        *, fallback_degraded: bool = False
    ) -> PlanningHealthView:
        source: object = controller if controller is not None else current_runtime()
        reader = getattr(source, "planning_health", None)
        if callable(reader):
            value = reader()
            if isinstance(value, PlanningHealthView):
                return value
            return PlanningHealthView.model_validate(value)
        return PlanningHealthView(status="degraded" if fallback_degraded else "idle")

    def current_run_phase() -> str | None:
        if controller is None:
            return None
        reader = getattr(controller, "current", None)
        if not callable(reader):
            return None
        try:
            summary = reader()
        except RuntimeError:
            return None
        phase = getattr(summary, "phase", None)
        return getattr(phase, "value", phase) if phase is not None else None

    def reject_completed_run_mutation() -> None:
        if current_run_phase() == "completed":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "run_completed",
                    "message": "the live run is completed; mutation endpoints are closed",
                },
            )

    def require_open_run_mutation() -> Iterator[None]:
        """Hold the controller lock across a mutating request when available."""
        guard = getattr(controller, "mutation_guard", None)
        if not callable(guard):
            reject_completed_run_mutation()
            yield
            return
        try:
            with guard():
                yield
        except RuntimeError as exc:
            if "completed" in str(exc).lower():
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "run_completed",
                        "message": "the live run is completed; mutation endpoints are closed",
                    },
                ) from exc
            raise

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        active_runtime = current_runtime()
        active_hub = current_hub()
        llm_paused = bool(getattr(active_runtime, "llm_paused", False))
        pause_reason = getattr(active_runtime, "llm_pause_reason", None)
        llm_reconnectable = bool(getattr(active_runtime, "llm_reconnectable", False))
        try:
            active_memory_port = current_memory_port()
        except HTTPException:
            active_memory_port = None
        memory_reason = getattr(active_memory_port, "degraded_reason", None)
        planning = current_planning_health(fallback_degraded=llm_paused)
        return {
            "status": "paused" if llm_paused else "ok",
            "stream_subscribers": active_hub.subscriber_count,
            "plan_version": current_plan_version(),
            "llm_paused": llm_paused,
            "llm_pause_reason": str(pause_reason) if pause_reason else None,
            "llm_reconnectable": llm_reconnectable,
            "planning_status": "degraded" if planning.status == "degraded" else "ready",
            "planning": planning.model_dump(mode="json"),
            "chat_status": "degraded" if llm_paused else "ready",
            "chat_degraded_reason": str(pause_reason) if pause_reason else None,
            "memory_status": "degraded" if memory_reason else "ready",
            "memory_degraded_reason": str(memory_reason) if memory_reason else None,
        }

    @app.get("/api/verification/physics")
    async def verification_physics() -> dict[str, object]:
        if not verification_audit:
            raise HTTPException(status_code=404, detail="verification audit is disabled")
        reader = getattr(controller, "verification_physics", None)
        if not callable(reader):
            raise HTTPException(status_code=501, detail="verification audit is unavailable")
        try:
            return cast(dict[str, object], await asyncio.to_thread(reader))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/verification/evidence")
    async def verification_evidence() -> dict[str, object]:
        if not verification_audit:
            raise HTTPException(status_code=404, detail="verification audit is disabled")
        reader = getattr(controller, "verification_evidence", None)
        if not callable(reader):
            raise HTTPException(status_code=501, detail="verification evidence is unavailable")
        try:
            return cast(dict[str, object], await asyncio.to_thread(reader))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/runs/current/planning/retry",
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def retry_initial_planning(request: PlanningRetryRequest) -> dict[str, object]:
        reject_completed_run_mutation()
        if controller is None:
            raise HTTPException(status_code=501, detail="run controller is unavailable")
        retry = getattr(controller, "retry_initial_planning", None)
        if not callable(retry):
            raise HTTPException(status_code=501, detail="planning retry is unavailable")
        try:
            epoch_id = retry(expected_epoch_id=request.expected_epoch_id)
        except ValueError as exc:
            message = str(exc)
            status_code = 409 if "stale" in message or "unavailable" in message else 422
            raise HTTPException(status_code=status_code, detail=message) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"epoch_id": str(epoch_id), "planning_status": "bootstrap_planning"}

    @app.get("/api/operational/snapshot")
    async def operational_snapshot() -> Response:
        return current_operational_snapshot_response()

    @app.get("/api/replay")
    async def replay_frames(
        run_id: str | None = Query(default=None),
        start_s: float = Query(default=0.0),
        end_s: float | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=10_000),
    ) -> dict[str, object]:
        selected_run_id = run_id
        replay_reader = current_replay()
        if run_id is not None:
            if catalog is None:
                raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
            try:
                replay_reader = catalog.replay(run_id)
            except RunNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from exc
        elif controller is not None:
            current_reader = getattr(controller, "current", None)
            if callable(current_reader):
                selected_run_id = getattr(current_reader(), "run_id", None)
        try:
            frames = replay_reader.range(
                start_s=start_s,
                end_s=end_s,
                offset=offset,
                limit=limit,
            )
        except ReplayIndexError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        count_reader = getattr(replay_reader, "count", None)
        total_count = (
            count_reader(start_s=start_s, end_s=end_s)
            if callable(count_reader)
            else len(replay_reader.range(start_s=start_s, end_s=end_s))
        )
        return {
            "frames": [operational_frame_payload(frame) for frame in frames],
            "count": len(frames),
            "total_count": total_count,
            "run_id": selected_run_id,
            "start_s": start_s,
            "end_s": end_s,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/runs")
    async def list_runs() -> dict[str, object]:
        if catalog is None:
            raise HTTPException(status_code=501, detail="run catalog is unavailable")
        return {"runs": [summary.model_dump(mode="json") for summary in catalog.list_runs()]}

    @app.post("/api/runs", status_code=202)
    async def start_run(request: RunRequest) -> JSONResponse:
        if controller is None:
            raise HTTPException(status_code=501, detail="run controller is unavailable")
        try:
            summary = controller.start_run(request.target_count, request.seed)  # type: ignore[attr-defined]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(status_code=202, content=summary.model_dump(mode="json"))

    @app.post(
        "/api/directives",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def queue_directive(request: DirectiveRequest) -> JSONResponse:
        reject_completed_run_mutation()
        current = current_plan_version()
        if request.expected_plan_version != current:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        "The operational plan changed while this directive was being "
                        "composed; review the newer plan before resubmitting."
                    ),
                    "current_plan_version": current,
                    "expected_plan_version": request.expected_plan_version,
                },
            )
        try:
            request_id = queue.submit(
                text=request.text,
                author=request.author,
                expected_plan_version=request.expected_plan_version,
                target_ids=request.target_ids,
            )
        except DirectiveQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "queued"},
        )

    @app.post(
        "/api/intelligence",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def submit_intelligence(report: IntelligenceReport) -> JSONResponse:
        reject_completed_run_mutation()
        submit = getattr(current_runtime(), "submit_intelligence", None)
        if not callable(submit):
            raise HTTPException(
                status_code=501, detail="intelligence input port is unavailable"
            )
        reject_expired_input(report.valid_until_s, "intelligence report")
        try:
            submit(report)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={"report_id": report.report_id, "status": "queued"},
        )

    @app.put(
        "/api/operational-scheme",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def set_operational_scheme(scheme: OperationalScheme) -> JSONResponse:
        reject_completed_run_mutation()
        setter = getattr(current_runtime(), "set_operational_scheme", None)
        if not callable(setter):
            raise HTTPException(
                status_code=501, detail="operational scheme input port is unavailable"
            )
        reject_expired_input(scheme.valid_until_s, "operational scheme")
        try:
            setter(scheme)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={
                "scheme_id": scheme.scheme_id,
                "version": scheme.version,
                "status": "queued",
            },
        )

    @app.post(
        "/api/assignments",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def queue_assignment(request: AssignmentRequest) -> JSONResponse:
        reject_completed_run_mutation()
        current = current_plan_version()
        if request.expected_plan_version != current:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "方案已更新，请确认最新方案后重新提交指派。",
                    "current_plan_version": current,
                    "expected_plan_version": request.expected_plan_version,
                },
            )
        submit_assignment = getattr(queue, "submit_assignment", None)
        if not callable(submit_assignment):
            raise HTTPException(status_code=501, detail="assignment queue is unavailable")
        try:
            request_id = submit_assignment(
                uuv_ids=sorted(set(request.uuv_ids)),
                target_id=request.target_id,
                expected_plan_version=request.expected_plan_version,
            )
        except DirectiveQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "queued"},
        )

    @app.post(
        "/api/sensor-modes",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def queue_sensor_mode(request: SensorModeRequest) -> JSONResponse:
        reject_completed_run_mutation()
        current = current_plan_version()
        if request.expected_plan_version != current:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "方案已更新，请确认最新方案后重新调整声纳模式。",
                    "current_plan_version": current,
                    "expected_plan_version": request.expected_plan_version,
                },
            )
        setter = getattr(current_runtime(), "submit_sensor_mode", None)
        if not callable(setter):
            raise HTTPException(status_code=501, detail="sensor mode input port is unavailable")
        try:
            await asyncio.to_thread(
                setter,
                uuv_id=request.uuv_id,
                mode=request.mode,
                target_id=request.target_id,
                expected_plan_version=request.expected_plan_version,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={
                "uuv_id": request.uuv_id,
                "mode": request.mode,
                "target_id": request.target_id,
                "passive_continuous": True,
                "status": "queued",
            },
        )

    @app.get("/api/directives/{request_id}")
    async def directive_status(request_id: str) -> dict[str, object]:
        return queue.status(request_id)

    @app.post(
        "/api/directives/{request_id}/apply",
        status_code=202,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def apply_directive(request_id: str) -> JSONResponse:
        reject_completed_run_mutation()
        apply_method = getattr(queue, "apply", None)
        if not callable(apply_method):
            raise HTTPException(status_code=501, detail="directive apply queue is unavailable")
        try:
            apply_method(request_id)
        except DirectiveQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "applying"},
        )

    @app.post(
        "/api/questions",
        response_model=None,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def answer_question(request: QuestionRequest) -> JSONResponse | dict[str, object]:
        try:
            answer: QuestionAnswer = await asyncio.to_thread(
                questions.ask, request.text, request.counterfactual
            )
        except QuestionEvidenceError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "insufficient_evidence",
                    "message": str(exc),
                    "evidence_ids": [],
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return cast(dict[str, object], answer.model_dump(mode="json"))

    @app.post(
        "/api/conversation/messages",
        response_model=None,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def conversation_message(
        request: ConversationMessageRequest,
    ) -> JSONResponse | dict[str, object]:
        reject_completed_run_mutation()
        submit = getattr(current_runtime(), "conversation_message", None)
        if not callable(submit):
            raise HTTPException(status_code=501, detail="conversation service is unavailable")
        current = current_plan_version()
        if request.expected_plan_version != current:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "方案已更新，请确认最新方案后重新提交对话。",
                    "current_plan_version": current,
                    "expected_plan_version": request.expected_plan_version,
                },
            )
        message = ConversationMessage(
            conversation_id=request.conversation_id,
            message_id=f"{request.conversation_id}:message:{uuid4().hex}",
            user_id=request.user_id,
            assistant_mode=request.assistant_mode,
            role="expert",
            text=request.text,
            target_scope=request.target_scope,
            region_scope=request.region_scope,
            evidence_ids=request.evidence_ids,
            expected_plan_version=request.expected_plan_version,
        )
        try:
            result = await asyncio.to_thread(submit, message)
        except QuestionEvidenceError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "insufficient_evidence",
                    "message": str(exc),
                    "evidence_ids": [],
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return cast(dict[str, object], result.model_dump(mode="json"))

    @app.get("/api/assistant/memory", response_model=None)
    async def memory_snapshot(query: MemorySnapshotQuery = Depends()) -> dict[str, object]:
        try:
            result = await asyncio.to_thread(
                current_memory_port().snapshot,
                user_id=query.user_id,
                conversation_id=query.conversation_id,
                scenario_id=query.scenario_id,
                query=query.query,
                memory_type=query.memory_type,
                min_importance_score=query.min_importance_score,
                limit=query.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        payload = dict(result)
        if payload.get("user_id") != query.user_id:
            raise HTTPException(status_code=403, detail="memory snapshot user scope mismatch")
        if payload.get("conversation_id") not in {None, query.conversation_id}:
            raise HTTPException(status_code=403, detail="memory snapshot conversation scope mismatch")
        if query.scenario_id is not None and payload.get("scenario_id") != query.scenario_id:
            raise HTTPException(status_code=403, detail="memory snapshot scenario scope mismatch")
        return cast(dict[str, object], jsonable_encoder(payload))

    @app.get("/api/assistant/memory/{memory_family_id}/versions", response_model=None)
    async def memory_versions(
        memory_family_id: str,
        query: MemoryVersionQuery = Depends(),
    ) -> dict[str, object]:
        if not _valid_identifier(memory_family_id):
            raise HTTPException(status_code=422, detail="invalid memory_family_id")
        try:
            versions = await asyncio.to_thread(
                current_memory_port().versions,
                user_id=query.user_id,
                memory_family_id=memory_family_id,
                scenario_id=query.scenario_id,
            )
        except (ValueError,) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (LookupError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="memory family was not found") from exc
        if any(
            version.user_id != query.user_id
            or version.memory_family_id != memory_family_id
            or (query.scenario_id is not None and version.scenario_id != query.scenario_id)
            for version in versions
        ):
            raise HTTPException(status_code=404, detail="memory family was not found")
        return {"user_id": query.user_id, "memory_family_id": memory_family_id, "versions": jsonable_encoder(versions)}

    @app.delete(
        "/api/assistant/memory/{memory_id}",
        response_model=None,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def delete_memory(
        memory_id: str,
        request: MemoryDeleteRequest | None = Body(default=None),
        user_id: str | None = Query(default=None, min_length=1, max_length=120, pattern=_IDENTIFIER_PATTERN),
        scenario_id: str | None = Query(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN),
        conversation_id: str | None = Query(default=None, min_length=1, max_length=240, pattern=_IDENTIFIER_PATTERN),
    ) -> dict[str, object]:
        reject_completed_run_mutation()
        if not _valid_identifier(memory_id):
            raise HTTPException(status_code=422, detail="invalid memory_id")
        selected_user_id = request.user_id if request is not None else (user_id or "operator")
        selected_scenario_id = request.scenario_id if request is not None else scenario_id
        selected_conversation_id = request.conversation_id if request is not None else conversation_id
        if user_id is not None and user_id != selected_user_id:
            raise HTTPException(status_code=422, detail="conflicting user_id values")
        if scenario_id is not None and scenario_id != selected_scenario_id:
            raise HTTPException(status_code=422, detail="conflicting scenario_id values")
        if conversation_id is not None and conversation_id != selected_conversation_id:
            raise HTTPException(status_code=422, detail="conflicting conversation_id values")
        try:
            deleted = await asyncio.to_thread(
                current_memory_port().delete,
                user_id=selected_user_id,
                memory_id=memory_id,
                scenario_id=selected_scenario_id,
                conversation_id=selected_conversation_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="memory was not found for this user")
        return {"status": "deleted", "memory_id": memory_id, "user_id": selected_user_id}

    @app.get("/api/assistant/memory/stream", response_model=None)
    async def memory_stream(query: MemoryStreamQuery = Depends()) -> dict[str, object]:
        memory_port = current_memory_port()
        try:
            events = await asyncio.to_thread(
                memory_port.stream,
                user_id=query.user_id,
                conversation_id=query.conversation_id,
                scenario_id=query.scenario_id,
                after_cursor=query.after_cursor,
                limit=query.limit,
                include_scenario_events=query.include_scenario_events,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if any(
            event.user_id != query.user_id
            or (
                event.conversation_id != query.conversation_id
                and (not query.include_scenario_events or event.conversation_id is not None)
            )
            or (query.scenario_id is not None and event.scenario_id != query.scenario_id)
            or event.cursor <= query.after_cursor
            for event in events
        ):
            raise HTTPException(status_code=403, detail="memory stream scope or cursor mismatch")
        next_cursor = max((event.cursor for event in events), default=query.after_cursor)
        adapter_reason = getattr(memory_port, "degraded_reason", None)
        stream_status = events[-1].status.value if events else ("degraded" if adapter_reason else "completed")
        degraded_reason = next(
            (
                event.payload.reason_code.value
                for event in events
                if event.status.value in {"degraded", "failed"}
                and event.payload.reason_code is not None
            ),
            adapter_reason,
        )
        return {
            "user_id": query.user_id,
            "conversation_id": query.conversation_id,
            "scenario_id": query.scenario_id,
            "include_scenario_events": query.include_scenario_events,
            "events": jsonable_encoder(events),
            "after_cursor": query.after_cursor,
            "next_cursor": next_cursor,
            "memory_status": stream_status,
            "degraded_reason": degraded_reason,
        }

    @app.post(
        "/api/conversation/{conversation_id}/apply",
        response_model=None,
        dependencies=[Depends(require_open_run_mutation)],
    )
    async def apply_conversation(
        conversation_id: str,
        request: ConversationApplyRequest,
    ) -> JSONResponse | dict[str, object]:
        reject_completed_run_mutation()
        apply_method = getattr(current_runtime(), "apply_conversation", None)
        if not callable(apply_method):
            raise HTTPException(status_code=501, detail="conversation apply is unavailable")
        current = current_plan_version()
        if request.expected_plan_version != current:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "方案已更新，请确认最新方案后再应用对话预览。",
                    "current_plan_version": current,
                    "expected_plan_version": request.expected_plan_version,
                },
            )
        try:
            result = await asyncio.to_thread(
                apply_method,
                conversation_id,
                request.turn_id,
                request.expected_plan_version,
                user_id=request.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return cast(dict[str, object], result.model_dump(mode="json"))

    if evaluation_enabled:
        @app.get("/api/evaluation/frames")
        async def evaluation_frames(
            start_s: float = Query(default=0.0),
            end_s: float | None = Query(default=None),
        ) -> dict[str, object]:
            if evaluation is None:
                raise HTTPException(status_code=503, detail="evaluation store is unavailable")
            try:
                frames = evaluation.range(start_s=start_s, end_s=end_s)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {
                "frames": [
                    cast(dict[str, object], frame.model_dump(mode="json"))
                    for frame in frames
                ],
                "count": len(frames),
                "start_s": start_s,
                "end_s": end_s,
            }

    @app.websocket("/ws/operational")
    async def operational_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send_json(payload: object) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def send_frames() -> None:
            async for frame in current_hub().stream():
                await send_json(operational_frame_payload(frame))

        async def receive_commands() -> None:
            while True:
                message = await websocket.receive_text()
                if message.strip().lower() == "ping":
                    async with send_lock:
                        await websocket.send_text("pong")

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(15.0)
                frame = current_hub().snapshot()
                await send_json(
                    {
                        "type": "heartbeat",
                        "sim_time_s": frame.sim_time_s if frame is not None else None,
                    }
                )

        tasks = {
            asyncio.create_task(send_frames()),
            asyncio.create_task(receive_commands()),
            asyncio.create_task(heartbeat()),
        }
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task

    return app
