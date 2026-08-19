"""FastAPI transport for the operational command center."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Literal

from underwater_tracking.agent.nodes.questions import QuestionEvidenceError
from underwater_tracking.domain.conversation_models import ConversationMessage
from underwater_tracking.api.dependencies import (
    DirectiveQueuePort,
    QuestionPort,
    ReplayPort,
    RuntimePort,
)
from underwater_tracking.api.evaluation import EvaluationPort
from underwater_tracking.api.hub import OperationalHub, RuntimeDirectiveQueue
from underwater_tracking.api.replay import ReplayIndexError
from underwater_tracking.domain.models import IntelligenceReport, OperationalScheme


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


def create_app(
    *,
    runtime: RuntimePort,
    replay: ReplayPort,
    directive_queue: DirectiveQueuePort | None = None,
    question_service: QuestionPort | None = None,
    hub: OperationalHub | None = None,
    evaluation: EvaluationPort | None = None,
    evaluation_enabled: bool = False,
) -> FastAPI:
    """Create the transport app over injected runtime ports.

    ``evaluation_enabled`` is intentionally retained as an explicit gate for
    the later truth-only evaluation routes.  This operational app never
    mounts those routes, and its frames are always ``OperationalFrame``
    instances.
    """
    frame_hub = hub or OperationalHub()
    queue = directive_queue or RuntimeDirectiveQueue(runtime)
    questions = question_service or runtime

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close = getattr(queue, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="Underwater Tracking Command Center", version="1.0", lifespan=lifespan
    )
    app.state.operational_hub = frame_hub
    app.state.directive_queue = queue

    def current_plan_version() -> int:
        frame = frame_hub.snapshot()
        if frame is not None:
            return frame.plan_version
        plan = runtime.active_plan()
        return plan.revision if plan is not None else 0

    def current_sim_time_s() -> int | None:
        current = getattr(runtime, "current_sim_time_s", None)
        return int(current()) if callable(current) else None

    def reject_expired_input(valid_until_s: int, input_name: str) -> None:
        current = current_sim_time_s()
        if current is not None and valid_until_s <= current:
            raise HTTPException(
                status_code=422,
                detail=f"{input_name} is already expired at simulation time {current}",
            )

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        llm_paused = bool(getattr(runtime, "llm_paused", False))
        pause_reason = getattr(runtime, "llm_pause_reason", None)
        llm_reconnectable = bool(getattr(runtime, "llm_reconnectable", False))
        return {
            "status": "paused" if llm_paused else "ok",
            "stream_subscribers": frame_hub.subscriber_count,
            "plan_version": current_plan_version(),
            "llm_paused": llm_paused,
            "llm_pause_reason": str(pause_reason) if pause_reason else None,
            "llm_reconnectable": llm_reconnectable,
        }

    @app.get("/api/operational/snapshot")
    async def operational_snapshot() -> dict[str, object]:
        frame = frame_hub.snapshot()
        if frame is None:
            raise HTTPException(status_code=503, detail="operational frame is not ready")
        return frame.model_dump(mode="json")

    @app.get("/api/replay")
    async def replay_frames(
        start_s: float = Query(default=0.0),
        end_s: float | None = Query(default=None),
    ) -> dict[str, object]:
        try:
            frames = replay.range(start_s=start_s, end_s=end_s)
        except ReplayIndexError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "frames": [frame.model_dump(mode="json") for frame in frames],
            "count": len(frames),
            "start_s": start_s,
            "end_s": end_s,
        }

    @app.post("/api/directives", status_code=202)
    async def queue_directive(request: DirectiveRequest) -> JSONResponse:
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
        request_id = queue.submit(
            text=request.text,
            author=request.author,
            expected_plan_version=request.expected_plan_version,
            target_ids=request.target_ids,
        )
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "queued"},
        )

    @app.post("/api/intelligence", status_code=202)
    async def submit_intelligence(report: IntelligenceReport) -> JSONResponse:
        submit = getattr(runtime, "submit_intelligence", None)
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

    @app.put("/api/operational-scheme", status_code=202)
    async def set_operational_scheme(scheme: OperationalScheme) -> JSONResponse:
        setter = getattr(runtime, "set_operational_scheme", None)
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

    @app.post("/api/assignments", status_code=202)
    async def queue_assignment(request: AssignmentRequest) -> JSONResponse:
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
        request_id = submit_assignment(
            uuv_ids=sorted(set(request.uuv_ids)),
            target_id=request.target_id,
            expected_plan_version=request.expected_plan_version,
        )
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "queued"},
        )

    @app.post("/api/sensor-modes", status_code=202)
    async def queue_sensor_mode(request: SensorModeRequest) -> JSONResponse:
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
        setter = getattr(runtime, "submit_sensor_mode", None)
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

    @app.post("/api/directives/{request_id}/apply", status_code=202)
    async def apply_directive(request_id: str) -> JSONResponse:
        apply_method = getattr(queue, "apply", None)
        if not callable(apply_method):
            raise HTTPException(status_code=501, detail="directive apply queue is unavailable")
        try:
            apply_method(request_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": "applying"},
        )

    @app.post("/api/questions", response_model=None)
    async def answer_question(request: QuestionRequest) -> JSONResponse | dict[str, object]:
        try:
            answer = await asyncio.to_thread(
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
        return answer.model_dump(mode="json")

    @app.post("/api/conversation/messages", response_model=None)
    async def conversation_message(
        request: ConversationMessageRequest,
    ) -> JSONResponse | dict[str, object]:
        submit = getattr(runtime, "conversation_message", None)
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
            message_id="",
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
        return result.model_dump(mode="json")

    @app.post("/api/conversation/{conversation_id}/apply", response_model=None)
    async def apply_conversation(
        conversation_id: str,
        request: ConversationApplyRequest,
    ) -> JSONResponse | dict[str, object]:
        apply_method = getattr(runtime, "apply_conversation", None)
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
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.model_dump(mode="json")

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
                "frames": [frame.model_dump(mode="json") for frame in frames],
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
            async for frame in frame_hub.stream():
                await send_json(frame.model_dump(mode="json"))

        async def receive_commands() -> None:
            while True:
                message = await websocket.receive_text()
                if message.strip().lower() == "ping":
                    async with send_lock:
                        await websocket.send_text("pong")

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(15.0)
                frame = frame_hub.snapshot()
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
