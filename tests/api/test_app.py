from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from underwater_tracking.agent.nodes.questions import QuestionAnswer
from underwater_tracking.api.app import create_app
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.runtime.run_catalog import RunCatalog
from underwater_tracking.domain import EvaluationFrame, OperationalFrame
from underwater_tracking.domain.models import IntelligenceReport, OperationalScheme
from underwater_tracking.domain.truth import TargetTruth

from tests.api.test_frame_contracts import _full_frame


class FakeReplay:
    def __init__(self, frames: list[OperationalFrame]) -> None:
        self.frames = frames

    def range(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[OperationalFrame]:
        frames = [
            frame
            for frame in self.frames
            if frame.sim_time_s >= start_s
            and (end_s is None or frame.sim_time_s <= end_s)
        ]
        end = None if limit is None else offset + limit
        return frames[offset:end]

    def count(self, start_s: float = 0.0, end_s: float | None = None) -> int:
        return len(self.range(start_s=start_s, end_s=end_s))


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
    llm_paused = False
    llm_pause_reason: str | None = None
    llm_reconnectable = False

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


@dataclass
class FakeInputRuntime(FakeRuntime):
    sim_time_s: int = 30
    intelligence: list[IntelligenceReport] = field(default_factory=list)
    schemes: list[OperationalScheme] = field(default_factory=list)

    def current_sim_time_s(self) -> int:
        return self.sim_time_s

    def submit_intelligence(self, report: IntelligenceReport) -> None:
        self.intelligence.append(report)

    def set_operational_scheme(self, scheme: OperationalScheme) -> None:
        self.schemes.append(scheme)


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


class FakeCompletedController:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()
        self.replay = FakeReplay([])
        self.hub = OperationalHub()

    def current(self) -> SimpleNamespace:
        return SimpleNamespace(phase="completed")


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


def test_api_root_redirects_to_the_actual_web_ui_when_configured() -> None:
    app = create_app(
        runtime=FakeRuntime(),
        replay=FakeReplay([_full_frame()]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
        web_ui_url="http://127.0.0.1:5181",
    )

    response = TestClient(app).get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:5181"


def test_health_exposes_a_paused_reconnectable_llm_runtime() -> None:
    runtime = FakeRuntime()
    runtime.llm_paused = True
    runtime.llm_pause_reason = "server error (503) while calling adversary_escape"
    runtime.llm_reconnectable = True
    app = create_app(
        runtime=runtime,
        replay=FakeReplay([_full_frame()]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )

    health = TestClient(app).get("/api/health").json()

    assert health["status"] == "paused"
    assert health["llm_reconnectable"] is True
    assert health["llm_pause_reason"] == runtime.llm_pause_reason


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


def test_replay_supports_bounded_pagination() -> None:
    frames = [
        _full_frame().model_copy(update={"frame_id": frame_id, "sim_time_s": frame_id})
        for frame_id in (1, 2, 3)
    ]
    client = TestClient(
        create_app(
            runtime=FakeRuntime(),
            replay=FakeReplay(frames),
            directive_queue=FakeDirectiveQueue(),
            hub=OperationalHub(),
        )
    )

    response = client.get("/api/replay", params={"offset": 1, "limit": 1})

    assert response.status_code == 200
    assert [frame["frame_id"] for frame in response.json()["frames"]] == [2]
    assert response.json()["count"] == 1
    assert response.json()["total_count"] == 3


def test_catalog_routes_list_runs_and_isolate_explicit_replay(tmp_path: Path) -> None:
    run_a = tmp_path / "outputs" / "serve-a"
    run_b = tmp_path / "outputs" / "serve-b"
    run_a.mkdir(parents=True)
    run_b.mkdir()
    frame_a = _full_frame().model_copy(update={"frame_id": 11, "sim_time_s": 10})
    frame_b = _full_frame().model_copy(update={"frame_id": 22, "sim_time_s": 20})
    for path, frame in ((run_a, frame_a), (run_b, frame_b)):
        (path / "operational_frames.jsonl").write_text(
            frame.model_dump_json() + "\n", encoding="utf-8"
        )
    (run_a / "manifest.json").write_text(
        '{"scenario_id":"scenario-a","target_count":1,"seed":1}', encoding="utf-8"
    )
    (run_b / "manifest.json").write_text(
        '{"scenario_id":"scenario-b","target_count":1,"seed":2}', encoding="utf-8"
    )

    client = TestClient(
        create_app(
            runtime=FakeRuntime(),
            replay=FakeReplay([_full_frame()]),
            catalog=RunCatalog(tmp_path / "outputs"),
            directive_queue=FakeDirectiveQueue(),
            hub=OperationalHub(),
        )
    )

    runs = client.get("/api/runs")
    replay = client.get("/api/replay", params={"run_id": "serve-a"})
    missing = client.get("/api/replay", params={"run_id": "../serve-a"})

    assert runs.status_code == 200
    assert [item["run_id"] for item in runs.json()["runs"]] == ["serve-a", "serve-b"]
    assert [item["frame_id"] for item in replay.json()["frames"]] == [11]
    assert replay.json()["run_id"] == "serve-a"
    assert missing.status_code == 404


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


def test_intelligence_input_is_queued_when_runtime_exposes_the_port() -> None:
    runtime = FakeInputRuntime()
    app = create_app(
        runtime=runtime,
        replay=FakeReplay([]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )

    response = TestClient(app).post(
        "/api/intelligence",
        json={
            "report_id": "intel-1",
            "source": "technical_reconnaissance",
            "target_id": "T1",
            "confidence": 0.8,
            "issued_at_s": 10,
            "valid_until_s": 100,
            "content_summary": "Propulsion signature changed.",
            "assessment": {"intent": "evade"},
        },
    )

    assert response.status_code == 202
    assert response.json() == {"report_id": "intel-1", "status": "queued"}
    assert runtime.intelligence[0].source.value == "technical_reconnaissance"


def test_operational_scheme_input_is_queued_when_runtime_exposes_the_port() -> None:
    runtime = FakeInputRuntime()
    app = create_app(
        runtime=runtime,
        replay=FakeReplay([]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )

    response = TestClient(app).put(
        "/api/operational-scheme",
        json={
            "scheme_id": "scheme-2",
            "version": 2,
            "target_priorities": {"T1": 1.0},
            "minimum_quality": {"T1": 0.8},
            "valid_from_s": 0,
            "valid_until_s": 1000,
            "constraints": ["keep-passive"],
        },
    )

    assert response.status_code == 202
    assert response.json() == {"scheme_id": "scheme-2", "version": 2, "status": "queued"}
    assert runtime.schemes[0].minimum_quality == {"T1": 0.8}


def test_expired_adaptive_inputs_are_rejected_at_the_current_simulation_time() -> None:
    runtime = FakeInputRuntime(sim_time_s=30)
    app = create_app(
        runtime=runtime,
        replay=FakeReplay([]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )
    client = TestClient(app)

    scheme = client.put(
        "/api/operational-scheme",
        json={
            "scheme_id": "scheme-expired",
            "version": 2,
            "valid_from_s": 0,
            "valid_until_s": 30,
        },
    )
    intelligence = client.post(
        "/api/intelligence",
        json={
            "report_id": "intel-expired",
            "source": "sonar",
            "target_id": "T1",
            "confidence": 0.8,
            "issued_at_s": 0,
            "valid_until_s": 30,
        },
    )

    assert scheme.status_code == 422
    assert intelligence.status_code == 422
    assert runtime.schemes == []
    assert runtime.intelligence == []


def test_adaptive_input_routes_return_501_without_optional_runtime_ports() -> None:
    app = create_app(
        runtime=FakeRuntime(),
        replay=FakeReplay([]),
        directive_queue=FakeDirectiveQueue(),
        hub=OperationalHub(),
    )

    intelligence = TestClient(app).post(
        "/api/intelligence",
        json={
            "report_id": "intel-1",
            "source": "sonar",
            "target_id": "T1",
            "confidence": 0.8,
            "issued_at_s": 10,
            "valid_until_s": 100,
        },
    )
    scheme = TestClient(app).put(
        "/api/operational-scheme",
        json={
            "scheme_id": "scheme-2",
            "version": 2,
            "valid_from_s": 0,
            "valid_until_s": 1000,
        },
    )

    assert intelligence.status_code == 501
    assert scheme.status_code == 501


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


def test_completed_run_rejects_live_mutation_endpoints() -> None:
    queue = FakeDirectiveQueue()
    app = create_app(controller=FakeCompletedController(), directive_queue=queue)
    client = TestClient(app)

    directive = client.post(
        "/api/directives",
        json={
            "text": "更新跟踪编组",
            "author": "operator",
            "expected_plan_version": 0,
        },
    )
    conversation = client.post(
        "/api/conversation/messages",
        json={
            "conversation_id": "completed-run",
            "text": "请重新规划",
            "expected_plan_version": 0,
        },
    )

    assert directive.status_code == 409
    assert conversation.status_code == 409
    assert directive.json()["detail"]["code"] == "run_completed"
    assert conversation.json()["detail"]["code"] == "run_completed"
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
