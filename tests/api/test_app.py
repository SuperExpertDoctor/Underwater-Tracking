from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.agent.nodes.questions import QuestionAnswer
from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain import EvaluationFrame, OperationalFrame
from underwater_tracking.domain.truth import TargetTruth

from tests.api.test_frame_contracts import _full_frame


class FakeReplay:
    def __init__(self, frames: list[OperationalFrame]) -> None:
        self.frames = frames

    def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[OperationalFrame]:
        return [
            frame
            for frame in self.frames
            if frame.sim_time_s >= start_s
            and (end_s is None or frame.sim_time_s <= end_s)
        ]


class FakeEvaluation:
    def __init__(self, frames: list[EvaluationFrame]) -> None:
        self.frames = frames

    def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[EvaluationFrame]:
        return [
            frame for frame in self.frames
            if frame.sim_time_s >= start_s and (end_s is None or frame.sim_time_s <= end_s)
        ]


@dataclass
class FakeRuntime:
    answer: QuestionAnswer = field(
        default_factory=lambda: QuestionAnswer(
            answer="保持 T1 的当前编组。",
            evidence_ids=("obs-1",),
        )
    )

    def active_plan(self) -> None:
        return None

    def ask(
        self,
        raw_text: str,
        counterfactual: dict[str, object] | None = None,
    ) -> QuestionAnswer:
        del raw_text, counterfactual
        return self.answer


class FakeDirectiveQueue:
    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []
        self.applied: list[str] = []

    def submit(self, **payload: Any) -> str:
        self.submissions.append(payload)
        return "directive-job-1"

    def status(self, request_id: str) -> dict[str, object]:
        return {"request_id": request_id, "status": "queued"}

    def submit_assignment(self, **payload: Any) -> str:
        self.submissions.append(payload)
        return "assignment-job-1"

    def apply(self, request_id: str) -> None:
        self.applied.append(request_id)


def _client(frame: OperationalFrame | None = None) -> tuple[TestClient, FakeDirectiveQueue, OperationalHub]:
    hub = OperationalHub()
    if frame is not None:
        hub.publish(frame)
    queue = FakeDirectiveQueue()
    app = create_app(
        runtime=FakeRuntime(),
        replay=FakeReplay([_full_frame()]),
        directive_queue=queue,
        hub=hub,
    )
    return TestClient(app), queue, hub


def test_health_and_operational_snapshot_are_truth_safe() -> None:
    client, _, _ = _client(_full_frame())

    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/api/operational/snapshot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert "target_truth" not in str(payload).lower()


def test_evaluation_routes_are_absent_when_the_gate_is_disabled() -> None:
    client, _, _ = _client(_full_frame())
    assert client.get("/api/evaluation/frames").status_code == 404


def test_evaluation_routes_are_explicitly_gated_and_separate() -> None:
    evaluation = EvaluationFrame(
        frame_id=1,
        sim_time_s=30,
        scenario_id="S1",
        run_id="run-1",
        plan_version=4,
        targets=(TargetTruth("T1", (1.0, 2.0), (0.1, 0.2), "transit"),),
    )
    hub = OperationalHub()
    hub.publish(_full_frame())
    app = create_app(
        runtime=FakeRuntime(),
        replay=FakeReplay([_full_frame()]),
        directive_queue=FakeDirectiveQueue(),
        hub=hub,
        evaluation=FakeEvaluation([evaluation]),
        evaluation_enabled=True,
    )
    response = TestClient(app).get("/api/evaluation/frames", params={"start_s": 0, "end_s": 30})
    assert response.status_code == 200
    assert response.json()["frames"][0]["targets"][0]["target_id"] == "T1"


def test_replay_returns_a_validated_time_range() -> None:
    client, _, _ = _client(_full_frame())

    response = client.get("/api/replay", params={"start_s": 0, "end_s": 30})

    assert response.status_code == 200
    assert [frame["frame_id"] for frame in response.json()["frames"]] == [1]


def test_replay_route_serializes_legacy_carrierless_deploymentless_jsonl_for_frontend() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "legacy-carrierless-deploymentless.jsonl"
    app = create_app(
        runtime=FakeRuntime(),
        replay=ReplayService(fixture),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )

    response = TestClient(app).get("/api/replay")

    assert response.status_code == 200
    frame = response.json()["frames"][0]
    assert frame["carrier"] is None
    assert {uuv["uuv_id"]: uuv["deployment_state"] for uuv in frame["uuvs"]} == {
        "UUV-legacy-deployed": "deployed",
        "UUV-legacy-returning": "returning",
    }


def test_directive_is_queued_without_running_the_graph() -> None:
    client, queue, _ = _client(_full_frame())

    response = client.post(
        "/api/directives",
        json={
            "text": "优先保证 T1 的观测质量",
            "author": "expert-1",
            "expected_plan_version": 4,
        },
    )

    assert response.status_code == 202
    assert response.json() == {"request_id": "directive-job-1", "status": "queued"}
    assert queue.submissions[0]["text"] == "优先保证 T1 的观测质量"


def test_stale_directive_is_rejected_with_current_plan_version() -> None:
    client, queue, _ = _client(_full_frame())

    response = client.post(
        "/api/directives",
        json={
            "text": "切换 T1 的编组",
            "author": "expert-1",
            "expected_plan_version": 3,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["current_plan_version"] == 4
    assert queue.submissions == []


def test_assignment_is_queued_with_the_current_plan_version() -> None:
    client, queue, _ = _client(_full_frame())

    response = client.post(
        "/api/assignments",
        json={
            "target_id": "T1",
            "uuv_ids": ["uuv_02", "uuv_01"],
            "expected_plan_version": 4,
        },
    )

    assert response.status_code == 202
    assert response.json() == {"request_id": "assignment-job-1", "status": "queued"}
    assert queue.submissions[-1]["uuv_ids"] == ["uuv_01", "uuv_02"]


def test_directive_apply_is_a_separate_explicit_request() -> None:
    client, queue, _ = _client(_full_frame())

    response = client.post("/api/directives/directive-job-1/apply")

    assert response.status_code == 202
    assert response.json() == {"request_id": "directive-job-1", "status": "applying"}
    assert queue.applied == ["directive-job-1"]


def test_question_returns_cited_answer_without_mutating_the_graph() -> None:
    client, _, _ = _client(_full_frame())

    response = client.post("/api/questions", json={"text": "为什么保持 T1？"})

    assert response.status_code == 200
    assert response.json()["evidence_ids"] == ["obs-1"]
    assert "target_truth" not in str(response.json()).lower()


def test_websocket_streams_latest_then_continuous_frames_and_pong() -> None:
    client, _, hub = _client(_full_frame())

    with client.websocket_connect("/ws/operational") as socket:
        first = socket.receive_json()
        assert first["frame_id"] == 1

        hub.publish(_full_frame(plan_version=4).model_copy(update={"frame_id": 2, "sim_time_s": 30}))
        second = socket.receive_json()
        assert second["frame_id"] == 2

        socket.send_text("ping")
        assert socket.receive_text() == "pong"
