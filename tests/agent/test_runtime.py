from __future__ import annotations

from collections import deque
from threading import Event, RLock, Thread
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.planning.coverage import coverage_gap_area_m2
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.runtime.mission_controller import execution_snapshot_to_mission_plan
from tests.domain.test_execution_models import _snapshot as _execution_snapshot


def _accepted_prediction(prediction: PredictedTrackRef) -> AcceptedPrediction:
    return AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="degraded",
            regime=prediction.prediction_regime,
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=max(prediction.corridor_radius_m, default=0.0),
            raw_prediction_id=prediction.prediction_id,
        ),
    )


def test_conversation_does_not_wait_for_the_planning_graph_lock() -> None:
    entered = Event()
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = "S1"
    runtime._lock = RLock()
    runtime._assistant_lock = RLock()
    runtime._conversation_turns = {}
    runtime._dependencies = SimpleNamespace(
        plans=SimpleNamespace(get_active=lambda _scenario_id: None),
        situation_provider=lambda _snapshot_ref: SimpleNamespace(),
        ledger=object(),
        events=object(),
        llm=object(),
        memory_service=object(),
        short_term_repository=object(),
        model_id="test-model",
        optimizer=object(),
        retention=SimpleNamespace(conversation_turn_limit=10),
    )
    message = SimpleNamespace(
        conversation_id="conversation-1",
        user_id="operator",
        assistant_mode="plan_revision",
        expected_plan_version=0,
    )
    result = SimpleNamespace(
        conversation_id="conversation-1",
        turn_id="turn-1",
    )

    def process(_message: object, _context: object) -> object:
        entered.set()
        return result

    worker = Thread(target=runtime.conversation_message, args=(message,))
    with patch("underwater_tracking.agent.runtime.process_conversation_message", process):
        with runtime._lock:
            worker.start()
            entered_while_planning_locked = entered.wait(timeout=0.25)
        worker.join(timeout=1.0)

    assert entered_while_planning_locked
    assert not worker.is_alive()
    assert runtime._conversation_turns[("conversation-1", "turn-1")] is result


class _Closable:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def close(self) -> None:
        self.calls.append(self.name)


class _Checkpointer:
    def __init__(self, calls: list[str]) -> None:
        self.conn = _Closable(calls, "checkpointer")


def test_runtime_close_is_idempotent_and_closes_runtime_resources_once() -> None:
    calls: list[str] = []
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._closed = False
    runtime._payload_store = _Closable(calls, "payload")
    runtime._checkpointer = _Checkpointer(calls)
    runtime._dependencies = Any  # type: ignore[assignment]
    runtime._pre_close_hooks = [lambda: calls.append("worker")]

    runtime.close()
    runtime.close()

    assert calls == ["worker", "payload", "checkpointer"]


def test_runtime_passes_uuv_only_mode_into_each_carrier_graph_cycle() -> None:
    class Graph:
        def __init__(self) -> None:
            self.inputs: list[dict[str, object]] = []

        def invoke(self, value: dict[str, object], *, config: dict[str, object]) -> dict[str, object]:
            del config
            self.inputs.append(value)
            return {}

    runtime = CarrierRuntime.__new__(CarrierRuntime)
    graph = Graph()
    runtime._graph = graph
    runtime._config = {}
    runtime._scenario_id = "S1"
    runtime._pending = []
    runtime._processed_event_ids = set()
    runtime._processed_event_order = deque()
    runtime._regional_replan_latches = set()
    runtime._state_cache = {}
    runtime._dependencies = SimpleNamespace(
        uuv_only=True,
        retention=SimpleNamespace(processed_event_limit=16),
    )

    runtime._run_cycle()

    assert graph.inputs[0]["uuv_only"] is True


def test_runtime_refresh_predictions_publishes_live_diff_before_graph_finishes() -> None:
    class EventStore:
        def __init__(self) -> None:
            self.appended: list[str] = []

        def append_if_absent(self, **payload: object) -> int:
            self.appended.append(str(payload["event_id"]))
            return 1

    class Predictor:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, situation: Any, target_id: str) -> AcceptedPrediction:
            self.calls += 1
            offset = 500.0 * (self.calls - 1)
            return _accepted_prediction(
                PredictedTrackRef(
                    prediction_id=f"prediction-{self.calls}",
                    target_id=target_id,
                    sim_time_s=situation.sim_time_s,
                    horizon_s=300.0,
                    sample_step_s=100.0,
                    times_s=(100.0, 200.0, 300.0, 400.0),
                    points_xy=((offset, 0.0),) * 4,
                    corridor_radius_m=(1.0,) * 4,
                    source_belief_history_ids=(f"belief-{self.calls}",),
                )
            )

    predictor = Predictor()
    event_store = EventStore()
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = "S1"
    runtime._dependencies = SimpleNamespace(
        predictor=predictor,
        uuv_only=False,
        trajectory_diff_config=TrajectoryDiffConfig(
            minimum_overlap_s=100.0,
            minimum_samples=3,
        ),
        events=event_store,
    )
    runtime._state_cache = {}
    runtime._cycle_running = True
    runtime._live_prediction_state = {}
    runtime._live_prediction_events = ()
    runtime._live_prediction_event_ids = set()
    runtime._live_prediction_pending_events = deque()
    runtime._live_prediction_snapshot_revision = -1
    runtime._live_prediction_lock = __import__("threading").RLock()
    runtime._pending = []
    runtime._processed_event_ids = set()

    def situation(revision: int, sim_time_s: int) -> Any:
        return SimpleNamespace(
            scenario_id="S1",
            snapshot_revision=revision,
            sim_time_s=sim_time_s,
            group_reports=(SimpleNamespace(target_id="T1"),),
            target_search_priors=(),
        )

    runtime.refresh_predictions(situation(1, 30))
    runtime.refresh_predictions(situation(2, 60))
    runtime.refresh_predictions(situation(3, 90))

    state = runtime.get_state()
    assert predictor.calls == 3
    assert state["predictions"]["T1"].prediction_id == "prediction-3"
    assert state["accepted_predictions"]["T1"].prediction.prediction_id == "prediction-3"
    assert state["accepted_predictions"]["T1"].health.status == "degraded"
    assert state["prediction_diffs"]["T1"].exceeded is True
    assert state["prediction_diff_gates"]["T1"].latched is True
    assert state["prediction_intent_verification_target_ids"] == ("T1",)
    assert event_store.appended == ["S1:target_intent_change_suspected:T1:90"]
    runtime._drain_live_prediction_events()
    assert [event.event_type for event in runtime._pending] == [
        "target_intent_change_suspected"
    ]


def test_runtime_builds_world_model_from_the_fresh_prediction_fragment() -> None:
    class Predictor:
        def __call__(self, situation: Any, target_id: str) -> AcceptedPrediction:
            return _accepted_prediction(
                PredictedTrackRef(
                    prediction_id="prediction-1",
                    target_id=target_id,
                    sim_time_s=situation.sim_time_s,
                    horizon_s=300.0,
                    sample_step_s=100.0,
                    times_s=(130.0, 230.0, 330.0),
                    points_xy=((100.0, 0.0), (200.0, 0.0), (300.0, 0.0)),
                    corridor_radius_m=(10.0, 20.0, 30.0),
                )
            )

    active_plan = object()
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = "S1"
    runtime._dependencies = SimpleNamespace(
        predictor=Predictor(),
        uuv_only=False,
        trajectory_diff_config=TrajectoryDiffConfig(),
        events=SimpleNamespace(append_if_absent=lambda **_payload: 1),
        plans=SimpleNamespace(get_active=lambda _scenario_id: active_plan),
        world_model_config=SimpleNamespace(enabled=True),
    )
    runtime._state_cache = {}
    runtime._live_prediction_state = {}
    runtime._live_prediction_events = ()
    runtime._live_prediction_event_ids = set()
    runtime._live_prediction_pending_events = deque()
    runtime._live_prediction_snapshot_revision = -1
    runtime._live_prediction_lock = __import__("threading").RLock()
    runtime._world_model_tracking_history = {}
    runtime._execution_coordinator = ExecutionCoordinator(
        snapshot=_execution_snapshot()
    )
    situation = SimpleNamespace(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        group_reports=(SimpleNamespace(target_id="T1"),),
        target_search_priors=(),
    )
    expected = {"T1": object()}

    with patch(
        "underwater_tracking.agent.runtime.build_world_model_forecasts",
        return_value=expected,
    ) as builder:
        state = runtime.refresh_predictions(situation)

    assert state["world_model_forecasts"] == expected
    builder.assert_called_once()
    _, predictions = builder.call_args.args
    assert predictions["T1"].prediction_id == "prediction-1"
    assert builder.call_args.kwargs["active_plan"] is active_plan
    assert builder.call_args.kwargs["source_plan_revision"] == 9
    assert builder.call_args.kwargs["previous_tracking_by_target"] == {}


def test_runtime_does_not_fall_back_to_cached_plan_after_natural_expiry() -> None:
    snapshot = _execution_snapshot(
        valid_from_s=0.0,
        valid_until_s=450.0,
    )
    cached = execution_snapshot_to_mission_plan(snapshot)
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._execution_coordinator = ExecutionCoordinator(snapshot=snapshot)
    runtime._dependencies = SimpleNamespace(execution_hard_stale_s=900.0)
    runtime._baseline_executable_mission_plan = cached
    runtime.current_sim_time_s = lambda: 901  # type: ignore[method-assign]
    runtime.get_state = lambda: {"executable_mission_plan": cached}  # type: ignore[method-assign]

    assert runtime.active_mission_plan() is None


def test_execution_snapshot_assigns_distinct_scan_lanes_to_each_group_member() -> None:
    snapshot = _execution_snapshot()

    plan = execution_snapshot_to_mission_plan(snapshot)

    assert len(plan.region_assignments) == 4
    for region in plan.region_assignments:
        member_ids = (*region.active_scan_uuv_ids, *region.passive_track_uuv_ids)
        assert len(member_ids) == 2
        assert set(region.scan_waypoints_by_uuv) == set(member_ids)
        assert region.scan_waypoints_by_uuv[member_ids[0]]
        assert region.scan_waypoints_by_uuv[member_ids[1]]
        assert (
            region.scan_waypoints_by_uuv[member_ids[0]]
            != region.scan_waypoints_by_uuv[member_ids[1]]
        )


def test_execution_snapshot_uses_detection_radius_for_complete_coverage() -> None:
    plan = execution_snapshot_to_mission_plan(
        _execution_snapshot(),
        detection_radius_m=5.0,
    )

    active_region = plan.region_assignments[0]
    assert (
        coverage_gap_area_m2(
            active_region.region_polygon,
            active_region.scan_waypoints_by_uuv,
            5.0,
        )
        <= 1e-6
    )
    assert active_region.coverage == 1.0
    assert "coverage_path_incomplete" not in active_region.degraded_reasons


def test_execution_snapshot_marks_incomplete_coverage_as_degraded() -> None:
    def incomplete_routes(polygon, uuv_ids, **_kwargs):
        return {
            uuv_id: (polygon[0], polygon[1])
            for uuv_id in uuv_ids
        }

    with patch(
        "underwater_tracking.runtime.mission_controller."
        "serpentine_coverage_waypoints_by_uuv",
        incomplete_routes,
    ):
        plan = execution_snapshot_to_mission_plan(
            _execution_snapshot(),
            detection_radius_m=1.0,
        )

    active_region = plan.region_assignments[0]
    assert (
        coverage_gap_area_m2(
            active_region.region_polygon,
            active_region.scan_waypoints_by_uuv,
            1.0,
        )
        > 0.0
    )
    assert active_region.coverage < 1.0
    assert "coverage_path_incomplete" in active_region.degraded_reasons
