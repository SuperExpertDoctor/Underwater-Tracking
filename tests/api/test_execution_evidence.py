from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from tests.domain.test_execution_models import _snapshot as execution_snapshot
from underwater_tracking.api.app import create_app
from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.conversation_models import (
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryStreamEvent,
    MemoryStreamStatus,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.agent.nodes.questions import QuestionAnswer


def _frame():
    snapshot = execution_snapshot(frame_id=42)
    situation = SituationSnapshot(
        scenario_id=snapshot.scenario_id,
        snapshot_revision=snapshot.source_snapshot_revision,
        sim_time_s=int(snapshot.source_sim_time_s),
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    return build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        frame_id=42,
        uuv_only=True,
        execution_snapshot=snapshot,
    )


class _Replay:
    def range(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        return []

    def count(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        return 0


@dataclass
class _MemoryPort:
    snapshot_value: dict[str, object] = field(
        default_factory=lambda: {
            "user_id": "operator",
            "scenario_id": "S1",
            "conversation_id": "conversation-1",
            "short_term": None,
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "retrieved_hits": [],
            "versions": [],
            "memory_status": "completed",
            "degraded_reason": None,
        }
    )
    events: list[MemoryStreamEvent] = field(default_factory=list)

    def snapshot(self, **kwargs: object) -> dict[str, object]:
        self.last_snapshot_kwargs = kwargs
        return dict(self.snapshot_value)

    def stream(self, **kwargs: object) -> list[MemoryStreamEvent]:
        self.last_stream_kwargs = kwargs
        return list(self.events)


class _Queue:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def submit(self, **kwargs: object) -> str:
        self.submissions.append(kwargs)
        return "directive-1"

    def submit_assignment(self, **kwargs: object) -> str:
        self.submissions.append(kwargs)
        return "assignment-1"

    def status(self, request_id: str) -> dict[str, object]:
        return {"request_id": request_id, "status": "queued"}

    def apply(self, request_id: str) -> None:
        del request_id


class _Runtime:
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot
        self.received_messages: list[ConversationMessage] = []
        self.sensor_modes: list[dict[str, object]] = []
        self.questions: list[dict[str, object]] = []
        self.applied: list[dict[str, object]] = []

    def active_plan(self) -> None:
        return None

    @property
    def current_execution_snapshot(self) -> object:
        return self.snapshot

    def ask(
        self,
        raw_text: str,
        counterfactual: object = None,
        *,
        evidence_ids: tuple[str, ...] = (),
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> QuestionAnswer:
        self.questions.append(
            {
                "raw_text": raw_text,
                "counterfactual": counterfactual,
                "evidence_ids": evidence_ids,
                "execution_revision": execution_revision,
                "frame_id": frame_id,
            }
        )
        return QuestionAnswer(
            answer="目标轨迹与 IMM 支持当前区域编组。",
            evidence_ids=("execution-9",),
        )

    def conversation_message(self, message: ConversationMessage) -> ConversationTurnResult:
        self.received_messages.append(message)
        return ConversationTurnResult(
            conversation_id=message.conversation_id,
            turn_id=f"{message.conversation_id}:turn:1",
            user_id=message.user_id,
            assistant_mode=message.assistant_mode,
            classification={"classification": "clarification"},
            messages=(message,),
            expected_plan_version=0,
            memory_context=MemoryContext(
                user_id=message.user_id,
                memory_status=MemoryStreamStatus.DEGRADED,
                degraded_reason="memory provider unavailable",
            ),
        )

    def submit_sensor_mode(self, **kwargs: object) -> None:
        self.sensor_modes.append(kwargs)

    def apply_conversation(
        self,
        conversation_id: str,
        turn_id: str,
        expected_plan_version: int,
        *,
        user_id: str = "operator",
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> ConversationTurnResult:
        self.applied.append(
            {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "expected_plan_version": expected_plan_version,
                "user_id": user_id,
                "execution_revision": execution_revision,
                "frame_id": frame_id,
            }
        )
        return ConversationTurnResult(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            classification={"classification": "clarification"},
            messages=(),
            expected_plan_version=expected_plan_version,
        )


def _app() -> tuple[Any, _Runtime, _MemoryPort, _Queue]:
    runtime = _Runtime(execution_snapshot())
    memory = _MemoryPort(
        events=[
            MemoryStreamEvent(
                cursor=1,
                event_id="memory-event-1",
                user_id="operator",
                scenario_id="S1",
                conversation_id="conversation-1",
                status=MemoryStreamStatus.COMPLETED,
                type="context_loaded",
            )
        ]
    )
    queue = _Queue()
    hub = OperationalHub()
    hub.publish(_frame())
    return create_app(
        runtime=runtime,
        replay=_Replay(),
        memory_port=memory,
        directive_queue=queue,
        hub=hub,
    ), runtime, memory, queue


def test_execution_context_is_shared_by_assistant_memory_stream_and_controls() -> None:
    app, runtime, memory, queue = _app()
    with TestClient(app) as client:
        framed_question = client.post(
            "/api/questions",
            json={
                "text": "current evidence",
                "evidence_ids": ["execution-9"],
                "execution_revision": 9,
                "frame_id": 42,
            },
        )
        question = client.post("/api/questions", json={"text": "为何这样制定方案？"})
        conversation = client.post(
            "/api/conversation/messages",
            json={
                "conversation_id": "conversation-1",
                "text": "请说明当前方案",
                "expected_plan_version": 0,
            },
        )
        snapshot = client.get(
            "/api/assistant/memory",
            params={
                "conversation_id": "conversation-1",
                "scenario_id": "S1",
            },
        )
        stream = client.get(
            "/api/assistant/memory/stream",
            params={
                "conversation_id": "conversation-1",
                "scenario_id": "S1",
            },
        )
        directive = client.post(
            "/api/directives",
            json={
                "text": "保留当前编组",
                "author": "operator",
                "expected_plan_version": 0,
            },
        )
        assignment = client.post(
            "/api/assignments",
            json={
                "target_id": "target_00",
                "uuv_ids": ["uuv_00", "uuv_01"],
                "expected_plan_version": 0,
            },
        )
        sensor = client.post(
            "/api/sensor-modes",
            json={
                "uuv_id": "uuv_00",
                "mode": "passive",
                "expected_plan_version": 0,
            },
        )

    applied = client.post(
        "/api/conversation/conversation-1/apply",
        json={
            "turn_id": "conversation-1:turn:1",
            "expected_plan_version": 0,
            "execution_revision": 9,
            "frame_id": 42,
        },
    )
    for response in (
        question,
        framed_question,
        conversation,
        snapshot,
        stream,
        directive,
        assignment,
        sensor,
        applied,
    ):
        assert response.status_code in {200, 202}
        assert response.json()["execution_revision"] == 9
        assert response.json()["frame_id"] == 42
    assert runtime.received_messages[0].execution_revision == 9
    assert runtime.received_messages[0].frame_id == 42
    assert memory.last_snapshot_kwargs["execution_revision"] == 9
    assert memory.last_stream_kwargs["frame_id"] == 42
    assert queue.submissions[0]["execution_revision"] == 9
    assert queue.submissions[1]["frame_id"] == 42
    assert runtime.sensor_modes[0]["execution_revision"] == 9
    assert runtime.questions[0]["execution_revision"] == 9
    assert runtime.questions[0]["frame_id"] == 42
    assert runtime.applied[0]["execution_revision"] == 9
    assert runtime.applied[0]["frame_id"] == 42


def test_evidence_query_is_read_only_and_reports_unresolved_ids() -> None:
    app, runtime, _, _ = _app()
    before = runtime.snapshot.execution_revision
    with TestClient(app) as client:
        response = client.get(
            "/api/evidence",
            params=[
                ("evidence_ids", "execution-9"),
                ("evidence_ids", "missing-evidence"),
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_revision"] == 9
    assert payload["frame_id"] == 42
    assert payload["unresolved_evidence"] == ["missing-evidence"]
    assert payload["read_only"] is True
    assert runtime.snapshot.execution_revision == before


def test_execution_explanation_contains_required_decision_chain_and_missing_evidence() -> None:
    app, _, _, _ = _app()
    with TestClient(app) as client:
        response = client.post(
            "/api/questions",
            json={
                "text": "为何这样制定方案？",
                "evidence_ids": ["execution-9", "not-stored"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_revision"] == 9
    assert payload["frame_id"] == 42
    for phrase in ("目标", "轨迹", "IMM", "意图", "四区域", "task group", "确定性算法"):
        assert phrase in payload["answer"]
    assert "execution-9" in payload["evidence_ids"]
    assert payload["unresolved_evidence"] == ["not-stored"]


def test_memory_degradation_keeps_execution_context() -> None:
    app, _, memory, _ = _app()
    memory.snapshot_value["memory_status"] = "degraded"
    memory.snapshot_value["degraded_reason"] = "LLM unavailable"
    with TestClient(app) as client:
        response = client.get(
            "/api/assistant/memory",
            params={"conversation_id": "conversation-1", "scenario_id": "S1"},
        )

    assert response.status_code == 200
    assert response.json()["memory_status"] == "degraded"
    assert response.json()["degraded_reason"] == "LLM unavailable"
    assert response.json()["execution_revision"] == 9
    assert response.json()["frame_id"] == 42
