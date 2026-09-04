from __future__ import annotations

from argparse import Namespace
from collections import deque
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from tests.domain.test_execution_models import _snapshot as _execution_snapshot
from underwater_tracking import cli
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.mission_models import (
    RegionLifecycle,
    RegionMissionState,
    UUVResourceState,
)
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.runtime.mission_controller import execution_snapshot_to_mission_plan
from underwater_tracking.simulation.engine import SimulationEngine


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/uuv_only_single_target.yaml"


def test_parse_args_uses_shared_api_port_environment_for_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_PORT", "8001")
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_app_config", lambda _path: object())

    def capture_serve(_config: object, args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_serve", capture_serve)

    assert cli.main(
        [
            "serve",
            "--config",
            str(CONFIG_PATH),
            "--seed",
            "42",
        ]
    ) == 0

    assert captured["args"].port == 8001


def test_local_memory_embedding_provider_verifies_readiness_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH).memory
    assert config is not None
    calls: list[str] = []
    received_paths: list[str | None] = []

    class Provider:
        def __init__(self, received_config, **kwargs: object) -> None:
            assert received_config is config
            received_paths.append(received_config.embedding_model_path)
            del kwargs

        def verify_ready(self) -> None:
            calls.append("verify_ready")

    monkeypatch.setattr(cli, "SentenceTransformerEmbeddingProvider", Provider)

    provider = cli._build_memory_embedding_provider(config)

    assert isinstance(provider, Provider)
    assert calls == ["verify_ready"]
    assert received_paths == [config.embedding_model_path]


def test_deterministic_startup_proposals_are_square_corner_contracts() -> None:
    proposals = cli._deterministic_region_proposals(
        ((-6_800.0, -6_800.0), (-5_000.0, -6_800.0)),
        (-12_000.0, 12_000.0, -12_000.0, 12_000.0),
    )

    assert len(proposals.regions) == 4
    for proposal in proposals.regions:
        width = proposal.bottom_right_xy[0] - proposal.top_left_xy[0]
        height = proposal.top_left_xy[1] - proposal.bottom_right_xy[1]
        assert width == pytest.approx(4_000.0)
        assert height == pytest.approx(width)
        assert set(proposal.model_dump(mode="json")) == {
            "top_left_xy",
            "bottom_right_xy",
            "rationale",
        }


def test_serve_leaves_configured_demo_speed_in_control_when_speed_is_omitted(
    monkeypatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            captured["steps"] = steps
            captured["speed"] = speed

        def start_run(self, _target_count, *, seed) -> None:
            captured["seed"] = seed

        def abort(self) -> None:
            captured["aborted"] = True

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

    result = cli._serve(
        config,
        Namespace(
            steps=0,
            speed=None,
            seed=42,
            host="127.0.0.1",
            port=8000,
        ),
    )

    assert result == 0
    assert captured == {"steps": 0, "speed": None, "seed": 42, "closed": True}


def test_serve_passes_explicit_speed_as_an_override(monkeypatch) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            captured["speed"] = speed

        def start_run(self, _target_count, *, seed) -> None:
            del seed

        def abort(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

    assert cli._serve(
        config,
        Namespace(
            steps=0,
            speed=4.0,
            seed=42,
            host="127.0.0.1",
            port=8000,
        ),
    ) == 0
    assert captured["speed"] == 4.0


def test_serve_passes_one_static_ui_directory_to_fastapi(monkeypatch, tmp_path: Path) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    class Controller:
        def __init__(self, _config, *, steps, speed, output_root=None) -> None:
            captured["steps"] = steps
            captured["speed"] = speed
            captured["output_root"] = output_root

        def start_run(self, _target_count, *, seed) -> None:
            captured["seed"] = seed

        def abort(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(
        cli,
        "create_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "_run_api_server", lambda *_args, **_kwargs: None)

    static_dir = tmp_path / "dist"
    assert cli._serve(
        config,
        Namespace(
            steps=1,
            speed=0.0,
            seed=42,
            host="127.0.0.1",
            port=8000,
            static_ui_dir=static_dir,
            output_root=tmp_path / "outputs",
        ),
    ) == 0

    assert captured["static_ui_dir"] == static_dir
    assert captured["output_root"] == tmp_path / "outputs"


def test_serve_aborts_controller_on_first_keyboard_interrupt(monkeypatch) -> None:
    config = load_app_config(CONFIG_PATH)
    captured: list[str] = []

    class Controller:
        def __init__(self, _config, *, steps, speed) -> None:
            del steps, speed

        def start_run(self, _target_count, *, seed) -> None:
            del seed

        def abort(self) -> None:
            captured.append("abort")

        def close(self) -> None:
            captured.append("close")

    monkeypatch.setattr(cli, "RunController", Controller)
    monkeypatch.setattr(cli, "RunCatalog", lambda _root: object())
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: object())

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_api_server", interrupt)

    with pytest.raises(KeyboardInterrupt):
        cli._serve(
            config,
            Namespace(
                steps=0,
                speed=None,
                seed=42,
                host="127.0.0.1",
                port=8000,
            ),
        )

    assert captured == ["abort", "close"]


def test_agent_dependencies_use_configured_prediction_history_and_target_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    predictor_args: dict[str, object] = {}

    def capture_predictor(**kwargs: object) -> object:
        predictor_args.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "make_snapshot_predictor", capture_predictor)
    monkeypatch.setattr(
        cli,
        "CarrierDependencies",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop.plans = object()
    loop.events = object()
    loop.ledger = object()
    loop.llm = object()
    loop._clock = object()
    loop.scenario_id = config.scenario.scenario_id
    loop._memory_service = object()
    loop._memory_short_term = object()
    loop._memory_port = object()
    loop._active_epoch = None
    loop._epoch_commit_port = None
    loop._role_model = lambda _role: "model"  # type: ignore[method-assign]

    dependencies = loop._deps()

    assert predictor_args["belief_history"].__self__ is loop  # type: ignore[union-attr]
    assert predictor_args["belief_history"].__name__ == "_belief_history"  # type: ignore[union-attr]
    assert predictor_args["max_speed_mps"] == config.tracking.submarine_sprint_speed_mps
    assert (
        predictor_args["max_turn_rate_rad_s"]
        == config.tracking.submarine_turn_rate_rad_s
    )
    assert predictor_args["health_config"] is config.tracking.prediction_health
    assert dependencies.belief_history.__self__ is loop
    assert dependencies.belief_history.__name__ == "_belief_history"
    assert (
        dependencies.task_region_side_m
        == config.scenario.tracking_policy.task_region_side_m
    )
    assert dependencies.world_model_config is config.world_model


def test_agent_dependencies_include_memory_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    monkeypatch.setattr(cli, "make_snapshot_predictor", lambda **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "CarrierDependencies",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop.plans = object()
    loop.events = object()
    loop.ledger = object()
    loop.llm = object()
    loop._clock = object()
    loop.scenario_id = config.scenario.scenario_id
    loop._memory_service = object()
    loop._memory_short_term = object()
    loop._memory_port = object()
    loop._active_epoch = None
    loop._epoch_commit_port = None
    loop._role_model = lambda _role: "model"  # type: ignore[method-assign]

    dependencies = loop._deps()

    assert dependencies.memory_service is loop._memory_service


def test_uuv_only_prediction_history_uses_estimated_belief_history() -> None:
    config = load_app_config(CONFIG_PATH)
    calls: list[str] = []

    class Engine:
        def belief_history(self, target_id: str):
            calls.append(f"belief:{target_id}")
            return ((0, 1.0, 2.0), (30, 4.0, 6.0))

    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._engine = Engine()

    history = loop._belief_history(SimpleNamespace(), "target_00")

    assert history == ((0, 1.0, 2.0), (30, 4.0, 6.0))
    assert calls == ["belief:target_00"]


def test_observation_checks_deterministic_region_rollover_after_prediction_refresh() -> None:
    prediction_state = {"predictions": {"target_00": object()}}
    calls: list[tuple[object, object]] = []
    order: list[str] = []
    situation = SimpleNamespace(sim_time_s=30)
    loop = object.__new__(cli._AgentLoop)
    loop._runtime = SimpleNamespace(
        refresh_predictions=lambda current: prediction_state,
    )
    def refresh_mission(current: object, state: object) -> None:
        calls.append((current, state))
        order.append("deterministic")

    loop._refresh_deterministic_mission = refresh_mission  # type: ignore[method-assign]
    loop._epoch_coordinator = None
    loop._background_carrier = True
    loop._submit_due_periodic_summary = lambda _current: None  # type: ignore[method-assign]
    loop._start_background_cycle = lambda _current: order.append("async_llm")  # type: ignore[method-assign]
    loop._record_carrier_error = lambda *_args: None  # type: ignore[method-assign]

    loop.on_situation(situation)

    assert calls == [(situation, prediction_state)]
    assert order == ["deterministic", "async_llm"]


def test_real_llm_mode_still_commits_deterministic_execution_first() -> None:
    config = load_app_config(CONFIG_PATH)
    situation = SimpleNamespace(sim_time_s=450)
    prediction_state = {"accepted_predictions": {"target_00": object()}}
    committed: list[tuple[object, object]] = []
    loop = object.__new__(cli._AgentLoop)
    loop._llm_execution_required = True
    loop._config = config
    loop._engine = SimpleNamespace(
        mission_snapshot=lambda: SimpleNamespace(plan_revision=1),
    )
    loop._runtime = object()
    loop._execution_coordinator = SimpleNamespace(
        rolling_check_due=lambda _sim_time_s: True,
        mark_rolling_check=lambda _sim_time_s: None,
    )
    loop._ensure_uuv_only_execution_snapshot = (  # type: ignore[method-assign]
        lambda current, *, prediction_state=None, **_kwargs: committed.append(
            (current, prediction_state)
        )
        or SimpleNamespace(execution_revision=1)
    )
    loop.publish_latest = lambda: None  # type: ignore[method-assign]

    loop._refresh_deterministic_mission(situation, prediction_state)

    assert committed == [(situation, prediction_state)]


def _execution_gate_loop(
    *, runtime_state: dict[str, object] | None = None
) -> tuple[object, list[str], list[str]]:
    config = load_app_config(CONFIG_PATH)
    failures: list[str] = []
    publishes: list[str] = []
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._engine = object()
    loop._runtime = SimpleNamespace(get_state=lambda: runtime_state or {})
    loop._execution_coordinator = SimpleNamespace(
        active_mission_plan=lambda: None,
        mark_failed=lambda reason: failures.append(reason),
    )
    loop.publish_latest = lambda: publishes.append("publish")  # type: ignore[method-assign]
    return loop, failures, publishes


def _accepted_prediction() -> AcceptedPrediction:
    prediction = PredictedTrackRef(
        prediction_id="prediction:target_00:450",
        target_id="target_00",
        sim_time_s=450,
        horizon_s=1_800.0,
        sample_step_s=30.0,
        times_s=tuple(float(450 + index * 30) for index in range(61)),
        points_xy=tuple((float(index * 100), 0.0) for index in range(61)),
        corridor_radius_m=(100.0,) * 61,
        prediction_regime="short_history",
    )
    return AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="degraded",
            regime="short_history",
            reason_codes=("short_history_fallback",),
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=100.0,
            raw_prediction_id=prediction.prediction_id,
        ),
    )


def test_execution_snapshot_build_rejects_raw_prediction_refs() -> None:
    loop, failures, publishes = _execution_gate_loop()
    situation = SimpleNamespace(sim_time_s=450)

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        situation,
        prediction_state={
            "predictions": {"target_00": object()},
            "accepted_predictions": {"target_00": object()},
        },
    )

    assert result is None
    assert failures == ["accepted_prediction_missing"]
    assert publishes == ["publish"]


def test_execution_snapshot_build_marks_failed_when_prediction_is_unavailable() -> None:
    loop, failures, publishes = _execution_gate_loop()
    situation = SimpleNamespace(sim_time_s=450)
    rejected = AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="boundary_recovery",
            reason_codes=("all_fallbacks_failed",),
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=0.0,
            raw_prediction_id=None,
        ),
    )

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        situation,
        prediction_state={"accepted_predictions": {"target_00": rejected}},
    )

    assert result is None
    assert failures == ["accepted_prediction_unavailable"]
    assert publishes == ["publish"]


def test_execution_snapshot_build_marks_failed_when_track_sources_are_missing() -> None:
    loop, failures, publishes = _execution_gate_loop()
    situation = SimpleNamespace(
        sim_time_s=450,
        snapshot_revision=1,
        group_reports=(),
        target_search_priors=(),
    )

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        situation,
        prediction_state={
            "accepted_predictions": {"target_00": _accepted_prediction()}
        },
    )

    assert result is None
    assert failures == ["execution_track_source_missing"]
    assert publishes == ["publish"]


def test_uuv_execution_admission_blocks_semantic_plan_without_public_source() -> None:
    config = load_app_config(CONFIG_PATH)
    baseline = _execution_snapshot()
    coordinator = ExecutionCoordinator(snapshot=baseline)
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._execution_coordinator = coordinator
    loop.plans = SimpleNamespace(
        get_active=lambda _scenario_id: SimpleNamespace(
            revision=baseline.execution_revision
        )
    )
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=300,
        group_reports=(),
        target_search_priors=(),
    )

    result = cli._AgentLoop._uuv_execution_admission(
        loop,
        situation,
        object(),
    )

    assert result == "execution_track_source_missing"


def test_uuv_execution_admission_allows_bootstrap_without_snapshot() -> None:
    config = load_app_config(CONFIG_PATH)
    coordinator = ExecutionCoordinator("S1")
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._execution_coordinator = coordinator
    loop.plans = SimpleNamespace(get_active=lambda _scenario_id: None)
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=0,
        group_reports=(),
        target_search_priors=(),
    )

    result = cli._AgentLoop._uuv_execution_admission(
        loop,
        situation,
        object(),
    )

    assert result is None


def test_prior_only_track_reaches_mission_health_gate() -> None:
    loop, failures, publishes = _execution_gate_loop()
    loop._baseline_intent_hypotheses = {}
    situation = SimpleNamespace(
        sim_time_s=450,
        snapshot_revision=1,
        group_reports=(),
        target_search_priors=(
            SimpleNamespace(
                target_id="target_00",
                prior_id="prior:target_00:1",
                issued_at_s=0,
                valid_until_s=900,
                center_xy=(100.0, 200.0),
            ),
        ),
    )

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        situation,
        prediction_state={
            "accepted_predictions": {"target_00": _accepted_prediction()}
        },
    )

    assert result is None
    assert failures == ["mission_snapshot_missing"]
    assert publishes == ["publish"]


def test_cli_builds_commits_and_publishes_real_baseline_before_install() -> None:
    config = load_app_config(CONFIG_PATH)
    calls: list[str] = []
    situation = SimulationEngine(config, seed=7).publication_situation()
    resources = {
        f"uuv_{index:02d}": UUVResourceState(
            uuv_id=f"uuv_{index:02d}",
            mileage_m=0.0,
            energy_fraction=1.0,
            deployment_state="deployed",
        )
        for index in range(12)
    }
    coordinator = ExecutionCoordinator(scenario_id=config.scenario.scenario_id)

    class Engine:
        _mission_controller = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(uuv_resources=resources, regions=())
        )

        @staticmethod
        def apply_verified_mission_plan(_plan: object) -> bool:
            assert coordinator.current is None
            calls.append("apply")
            return True

    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = config.scenario.scenario_id
    runtime._dependencies = SimpleNamespace(
        predictor=lambda *_args: pytest.fail("prior-only refresh must not call predictor"),
        uuv_only=True,
        trajectory_diff_config=None,
        events=SimpleNamespace(append_if_absent=lambda **_kwargs: 1),
        world_model_config=None,
    )
    runtime._state_cache = {}
    runtime._cycle_running = True
    runtime._live_prediction_state = {}
    runtime._live_prediction_events = ()
    runtime._live_prediction_event_ids = set()
    runtime._live_prediction_pending_events = deque()
    runtime._live_prediction_snapshot_revision = -1
    runtime._live_prediction_lock = RLock()
    runtime._pending = []
    runtime._processed_event_ids = set()
    runtime._baseline_executable_mission_plan = None
    prediction_state = runtime.refresh_predictions(situation)

    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._engine = Engine()
    loop._runtime = runtime
    loop._execution_coordinator = coordinator
    loop._baseline_intent_hypotheses = {}
    loop._last_mission_revision = 0
    loop.events = SimpleNamespace(append_if_absent=lambda **_kwargs: None)
    loop.plans = SimpleNamespace(get_active=lambda _scenario_id: None)
    loop._record_carrier_error = lambda *_args: None  # type: ignore[method-assign]

    def publish() -> None:
        calls.append("publish" if coordinator.current is not None else "failed_publish")

    loop.publish_latest = publish  # type: ignore[method-assign]
    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        situation,
        prediction_state=prediction_state,
    )

    assert result is not None
    assert result.plan_source == "deterministic"
    assert len(result.regions) == 4
    assert len(result.task_groups) == 4
    assert runtime._baseline_executable_mission_plan is not None
    assert calls == ["apply", "publish"]


def test_committed_graph_result_can_commit_semantic_execution_revision() -> None:
    config = load_app_config(CONFIG_PATH)
    baseline = _execution_snapshot()
    coordinator = ExecutionCoordinator(snapshot=baseline)
    executable = execution_snapshot_to_mission_plan(baseline)
    graph_result = {
        "epoch_commit_result": EpochCommitResult(
            epoch_id="epoch-1",
            status="committed",
            plan_id="plan-9",
            plan_version=executable.revision,
            validation_report_id="validation-9",
            executable_plan=executable,
        )
    }
    committed_plan = cli._committed_epoch_plan(graph_result)
    installed: list[int] = []
    published: list[int] = []
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._execution_coordinator = coordinator
    loop._engine = SimpleNamespace(apply_verified_mission_plan=lambda _plan: True)
    loop._runtime = SimpleNamespace(
        install_executable_baseline=lambda plan: installed.append(plan.revision)
    )
    loop._last_mission_revision = baseline.execution_revision
    loop.publish_latest = lambda: published.append(  # type: ignore[method-assign]
        coordinator.execution_revision
    )

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        SimpleNamespace(),
        plan=committed_plan,
        base_execution_revision=baseline.execution_revision,
    )

    assert result is not None
    assert result.plan_source == "llm_optimized"
    assert result.execution_revision == baseline.execution_revision + 1
    assert coordinator.current == result
    assert installed == [result.execution_revision]
    assert published == [result.execution_revision]


def test_semantic_commit_preserves_authoritative_region_lifecycle() -> None:
    config = load_app_config(CONFIG_PATH)
    baseline = _execution_snapshot()
    coordinator = ExecutionCoordinator(snapshot=baseline)
    executable = execution_snapshot_to_mission_plan(baseline)
    controller_region = RegionMissionState(
        region_id=baseline.regions[0].region_id,
        target_id=baseline.target_id,
        lifecycle=RegionLifecycle.TRACKING_COMPLETED,
        active_scan_uuv_ids=baseline.task_groups[0].member_uuv_ids,
        passive_track_uuv_ids=(),
        plan_revision=baseline.execution_revision,
    )
    controller_successor = RegionMissionState(
        region_id=baseline.regions[1].region_id,
        target_id=baseline.target_id,
        lifecycle=RegionLifecycle.ACTIVE_SCAN,
        active_scan_uuv_ids=baseline.task_groups[1].member_uuv_ids,
        passive_track_uuv_ids=(),
        plan_revision=baseline.execution_revision,
    )
    loop = object.__new__(cli._AgentLoop)
    loop._config = config
    loop._execution_coordinator = coordinator
    loop._engine = SimpleNamespace(
        _mission_controller=SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                regions=(controller_region, controller_successor)
            )
        ),
        apply_verified_mission_plan=lambda _plan: True,
    )
    loop._runtime = SimpleNamespace(
        install_executable_baseline=lambda _plan: None
    )
    loop._last_mission_revision = baseline.execution_revision
    loop.publish_latest = lambda: None  # type: ignore[method-assign]

    result = cli._AgentLoop._ensure_uuv_only_execution_snapshot(
        loop,
        SimpleNamespace(),
        plan=executable,
        base_execution_revision=baseline.execution_revision,
    )

    assert result is not None
    assert result.regions[0].status == "monitoring_complete"
    assert result.regions[1].status == "active"
    assert result.task_groups[0].status == "complete"
    assert result.task_groups[1].status == "active"
    assert result.current_region_id == baseline.regions[1].region_id
    assert result.next_region_id == baseline.regions[2].region_id
