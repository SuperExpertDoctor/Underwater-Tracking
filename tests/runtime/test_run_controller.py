from __future__ import annotations

from pathlib import Path
import sqlite3
from threading import Event, RLock, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import TrajectoryDiffResult
from underwater_tracking.runtime.run_controller import (
    RunAlreadyStartedError,
    RunController,
    _RunBundle,
    _stored_verification_event_projection,
    _target_wall_deadline,
)
from underwater_tracking.runtime.models import RunPhase


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/scenario/default.yaml"
)
EXPLICIT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/scenario/segmented_single_target.yaml"
)


class FakeLLM:
    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del operation, payload, prompt_version
        raise AssertionError(f"unexpected LLM call for {response_model!r}")


def _controller(tmp_path: Path) -> RunController:
    return RunController(
        load_app_config(CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )


def test_controller_lifetime_creates_exactly_one_run_directory(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    try:
        first = controller.start_run(1, seed=7)

        with pytest.raises(RuntimeError, match="already started"):
            controller.start_run(2, seed=8)

        run_directories = tuple((tmp_path / "outputs").glob("run-*"))
        assert first.target_count == 1
        assert first.path.name.startswith("run-")
        assert run_directories == (first.path,)
        assert not tuple((tmp_path / "outputs").glob("serve-*"))
        assert controller.current() == first
    finally:
        controller.close()


def test_failed_valid_start_does_not_allow_a_second_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    calls = 0

    def fail_build(_config: object, _seed: int) -> None:
        nonlocal calls
        calls += 1
        (tmp_path / "outputs" / "run-failed").mkdir(parents=True)
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(controller, "_build_bundle", fail_build)

    with pytest.raises(RuntimeError, match="injected startup failure"):
        controller.start_run(1, seed=7)
    with pytest.raises(RunAlreadyStartedError, match="already started"):
        controller.start_run(1, seed=7)

    assert calls == 1
    assert [path.name for path in (tmp_path / "outputs").iterdir()] == ["run-failed"]


@pytest.mark.parametrize("target_count", [0, 5])
def test_invalid_target_count_preserves_current_bundle(
    tmp_path: Path, target_count: int
) -> None:
    controller = _controller(tmp_path)
    try:
        current = controller.start_run(1, seed=7)

        with pytest.raises(ValueError, match="target_count"):
            controller.start_run(target_count, seed=8)

        assert controller.current() == current
        assert controller.runtime is not None
        assert controller.replay is not None
        assert controller.hub is not None
    finally:
        controller.close()


def test_explicit_roster_does_not_invent_additional_targets(tmp_path: Path) -> None:
    controller = RunController(
        load_app_config(EXPLICIT_CONFIG_PATH),
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )
    try:
        current = controller.start_run(1, seed=7)

        with pytest.raises(ValueError, match="platform-core target roster"):
            controller.start_run(2, seed=8)

        assert controller.current() == current
    finally:
        controller.close()


def test_cli_speed_zero_remains_an_unthrottled_override_not_a_config_value(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )

    assert config.timing.demo_time_scale == 60.0
    assert controller._speed == 0.0


def test_missing_speed_inherits_configured_demo_time_scale(tmp_path: Path) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=None,
    )

    assert controller._speed is None
    assert controller._effective_speed(config) == 60.0


def test_default_interactive_run_starts_without_bootstrap_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config(CONFIG_PATH)
    bootstrap_calls: list[object] = []

    class Loop:
        hub = object()
        events = object()
        on_situation = None
        _transition_coordinator = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def attach(self, _engine: object) -> None:
            return None

        def begin_bootstrap_planning(self, situation: object) -> None:
            bootstrap_calls.append(situation)

        def close(self) -> None:
            return None

    class Engine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def publication_situation(self) -> object:
            return "initial-situation"

    from underwater_tracking import cli
    from underwater_tracking.runtime import run_controller

    monkeypatch.setattr(cli, "_AgentLoop", Loop)
    monkeypatch.setattr(cli, "_create_public_run_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(cli, "_mission_controller_for", lambda _config: None)
    monkeypatch.setattr(run_controller, "SimulationEngine", Engine)

    controller = RunController(
        config,
        output_root=tmp_path,
        llm={"master": FakeLLM()},
        steps=0,
    )

    bundle = controller._build_bundle(config, seed=42)

    assert bundle.phase is RunPhase.RUNNING
    assert bootstrap_calls == []


def test_explicit_bootstrap_planning_installs_baseline_and_runs_in_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config(CONFIG_PATH)
    bootstrap_calls: list[object] = []
    baseline_calls: list[object] = []

    class Loop:
        hub = object()
        events = object()
        on_situation = None
        _transition_coordinator = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def attach(self, _engine: object) -> None:
            return None

        def install_deterministic_baseline(self, situation: object) -> None:
            baseline_calls.append(situation)

        def begin_bootstrap_planning(self, situation: object) -> None:
            bootstrap_calls.append(situation)

        def close(self) -> None:
            return None

    class Engine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def publication_situation(self) -> object:
            return "initial-situation"

    from underwater_tracking import cli
    from underwater_tracking.runtime import run_controller

    monkeypatch.setattr(cli, "_AgentLoop", Loop)
    monkeypatch.setattr(cli, "_create_public_run_dir", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(cli, "_mission_controller_for", lambda _config: None)
    monkeypatch.setattr(run_controller, "SimulationEngine", Engine)

    controller = RunController(
        config,
        output_root=tmp_path,
        llm={"master": FakeLLM()},
        steps=0,
        bootstrap_planning=True,
    )

    bundle = controller._build_bundle(config, seed=42)

    assert bundle.phase is RunPhase.RUNNING
    assert baseline_calls == ["initial-situation"]
    assert bootstrap_calls == ["initial-situation"]


def test_demo_deadline_maps_eight_simulation_hours_to_about_eight_minutes() -> None:
    assert _target_wall_deadline(
        wall_origin=10.0,
        sim_origin=0.0,
        sim_time_s=28_800,
        effective_speed=60.0,
    ) == pytest.approx(490.0)


def test_worker_uses_deadline_pacing_after_each_simulation_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=2,
        speed=None,
    )

    class Clock:
        sim_time_s = 0

    class Engine:
        _clock = Clock()

    class Stop:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            return False

        def set(self) -> None:
            return None

    stop = Stop()
    monotonic_values = iter((0.0, 0.0, 0.1))
    monkeypatch.setattr(
        "underwater_tracking.runtime.run_controller.time.monotonic",
        lambda: next(monotonic_values),
    )

    def fake_step(_engine: Engine, _loop: object, _config: object, *, stop: Stop) -> bool:
        del _loop, _config, stop
        _engine._clock.sim_time_s += 5
        return True

    monkeypatch.setattr("underwater_tracking.cli._step_with_llm_retries", fake_step)
    bundle = _RunBundle(
        config=config,
        run_dir=tmp_path / "run",
        loop=object(),
        engine=Engine(),
        replay=object(),
        hub=object(),
        stop=stop,  # type: ignore[arg-type]
        worker_errors=[],
    )

    worker = controller._start_worker(bundle)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stop.waits == pytest.approx([5 / 60, 10 / 60 - 0.1])


def test_worker_does_not_poll_bootstrap_before_advancing_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=None,
    )

    class Clock:
        sim_time_s = 0

    class Engine:
        _clock = Clock()

    now = [0.0]

    class Loop:
        def bootstrap_result(self) -> object | None:
            raise AssertionError("physics must not wait for bootstrap planning")

        def publish_latest(self) -> None:
            return None

    class Stop:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            return False

        def set(self) -> None:
            return None

    stop = Stop()
    monkeypatch.setattr(
        "underwater_tracking.runtime.run_controller.time.monotonic",
        lambda: now[0],
    )

    def fake_step(_engine: Engine, _loop: object, _config: object, *, stop: Stop) -> bool:
        del _loop, _config, stop
        _engine._clock.sim_time_s += 5
        now[0] = 0.01
        return True

    monkeypatch.setattr("underwater_tracking.cli._step_with_llm_retries", fake_step)
    bundle = _RunBundle(
        config=config,
        run_dir=tmp_path / "run",
        loop=Loop(),
        engine=Engine(),
        replay=object(),
        hub=object(),
        stop=stop,  # type: ignore[arg-type]
        worker_errors=[],
        phase=RunPhase.BOOTSTRAP_PLANNING,
    )

    worker = controller._start_worker(bundle)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stop.waits == pytest.approx([5 / 60 - 0.01])


def test_worker_yields_after_each_unpaced_simulation_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=2,
        speed=0.0,
    )

    class Clock:
        sim_time_s = 0

    class Engine:
        _clock = Clock()

    class Stop:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            return False

        def set(self) -> None:
            return None

    stop = Stop()

    def fake_step(_engine: Engine, _loop: object, _config: object, *, stop: Stop) -> bool:
        del _loop, _config, stop
        _engine._clock.sim_time_s += 5
        return True

    monkeypatch.setattr("underwater_tracking.cli._step_with_llm_retries", fake_step)
    bundle = _RunBundle(
        config=config,
        run_dir=tmp_path / "run",
        loop=object(),
        engine=Engine(),
        replay=object(),
        hub=object(),
        stop=stop,  # type: ignore[arg-type]
        worker_errors=[],
    )

    worker = controller._start_worker(bundle)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stop.waits == [0.001, 0.001]


def test_worker_drains_background_cycles_before_marking_run_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": FakeLLM()},
        steps=1,
        speed=0.0,
    )

    class Clock:
        sim_time_s = 0

    class Engine:
        _clock = Clock()

    class Loop:
        def __init__(self) -> None:
            self.drain_timeouts: list[float] = []

        def drain_background_cycle(self, *, timeout_s: float) -> bool:
            self.drain_timeouts.append(timeout_s)
            return True

        def publish_latest(self) -> None:
            return None

    class Stop:
        def is_set(self) -> bool:
            return False

        def wait(self, _timeout: float) -> bool:
            return False

        def set(self) -> None:
            return None

    def fake_step(_engine: Engine, _loop: object, _config: object, *, stop: Stop) -> bool:
        del _loop, _config, stop
        _engine._clock.sim_time_s += 5
        return True

    monkeypatch.setattr("underwater_tracking.cli._step_with_llm_retries", fake_step)
    loop = Loop()
    bundle = _RunBundle(
        config=config,
        run_dir=tmp_path / "run",
        loop=loop,
        engine=Engine(),
        replay=object(),
        hub=object(),
        stop=Stop(),  # type: ignore[arg-type]
        worker_errors=[],
    )

    worker = controller._start_worker(bundle)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert bundle.phase is RunPhase.COMPLETED
    assert loop.drain_timeouts == [180.0]


def test_close_keeps_bundle_installed_when_agent_loop_reports_incomplete() -> None:
    class Loop:
        def __init__(self) -> None:
            self.close_results = iter((False, True))
            self.close_calls = 0
            self.manifest_calls = 0

        def write_manifest(self, _run_dir: Path) -> None:
            self.manifest_calls += 1

        def close(self) -> bool:
            self.close_calls += 1
            return next(self.close_results)

    loop = Loop()
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=loop,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    controller.close()
    assert controller._bundle is bundle
    assert loop.close_calls == 1

    controller.close()
    assert controller._bundle is None
    assert loop.close_calls == 2
    assert loop.manifest_calls == 1


def test_completed_run_drains_background_cycles_before_abort() -> None:
    class Loop:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def drain_background_cycle(self, *, timeout_s: float) -> bool:
            assert timeout_s > 0
            self.calls.append("drain")
            return True

        def abort(self) -> None:
            self.calls.append("abort")

        def write_manifest(self, _run_dir: Path) -> None:
            self.calls.append("manifest")

        def close(self, *, timeout_s: float) -> bool:
            assert timeout_s > 0
            self.calls.append("close")
            return True

    loop = Loop()
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=loop,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        phase=RunPhase.COMPLETED,
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    assert controller.close() is True
    assert loop.calls == ["drain", "abort", "manifest", "close"]


def test_completed_evidence_drains_background_cycles_before_reading_state() -> None:
    calls: list[float] = []

    class Loop:
        def drain_background_cycle(self, *, timeout_s: float) -> bool:
            calls.append(timeout_s)
            return True

    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=Loop(),
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        phase=RunPhase.COMPLETED,
    )

    assert RunController._drain_completed_background_for_evidence(bundle) is True
    assert calls == [4.0]


def test_mutation_guard_serializes_terminal_phase_transition() -> None:
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=object(),
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        phase=RunPhase.RUNNING,
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle
    started = Event()

    def complete() -> None:
        started.set()
        controller._set_phase(bundle, RunPhase.COMPLETED)

    worker = Thread(target=complete)
    with controller.mutation_guard():
        worker.start()
        assert started.wait(timeout=1.0)
        assert bundle.phase is RunPhase.RUNNING
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert bundle.phase is RunPhase.COMPLETED
    with pytest.raises(RuntimeError, match="completed"):
        with controller.mutation_guard():
            pass


def test_close_keeps_bundle_when_simulation_worker_does_not_stop() -> None:
    class Worker:
        alive = True

        def join(self, timeout: float) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

    class Loop:
        close_calls = 0
        manifest_calls = 0

        def write_manifest(self, _run_dir: Path) -> None:
            self.manifest_calls += 1

        def close(self) -> bool:
            self.close_calls += 1
            return True

    worker = Worker()
    loop = Loop()
    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=loop,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        worker=worker,
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    controller.close()
    assert controller._bundle is bundle
    assert loop.close_calls == 0
    assert loop.manifest_calls == 0

    worker.alive = False
    controller.close()
    assert controller._bundle is None
    assert loop.close_calls == 1
    assert loop.manifest_calls == 1


def test_abort_detaches_active_bundle_without_waiting_for_blocked_workers() -> None:
    class Worker:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            raise AssertionError(f"abort must not wait for the worker ({timeout=})")

    class Loop:
        def close(self) -> bool:
            raise AssertionError("abort must not close resources synchronously")

    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=Loop(),
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        worker=Worker(),
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    controller.abort()

    assert bundle.stop.is_set()
    assert controller._bundle is None


def test_close_returns_false_when_bounded_timeout_expires() -> None:
    class Worker:
        def join(self, timeout: float) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=Any,
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
        worker=Worker(),
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    assert controller.close(timeout_s=0.0) is False
    assert controller._bundle is bundle


def test_close_exposes_remaining_resource_report() -> None:
    class Loop:
        def write_manifest(self, _run_dir: Path) -> None:
            return None

        def close(self) -> bool:
            return False

        def shutdown_report(self) -> dict[str, object]:
            return {
                "completed": False,
                "remaining_resources": ("carrier-llm",),
            }

    bundle = _RunBundle(
        config=Any,
        run_dir=Path("run"),
        loop=Loop(),
        engine=Any,
        replay=Any,
        hub=Any,
        stop=Event(),
        worker_errors=[],
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle

    assert controller.close() is False
    report = controller.shutdown_report()
    assert report.completed is False
    assert report.remaining_resources == ("carrier-llm",)


def test_verification_evidence_projects_durable_prediction_intent_chain() -> None:
    call_metadata = (
        {
            "operation": "intent",
            "model": "LongCat-Flash-Chat",
            "prompt_version": "intent-v2",
            "request_hash": "request-1",
            "response_hash": "response-1",
            "sim_time_s": 61,
            "scenario_id": "S1",
        },
        {
            "operation": "intent",
            "model": "LongCat-Flash-Chat",
            "prompt_version": "intent-v2",
            "request_hash": "request-2",
            "response_hash": "response-2",
            "sim_time_s": 62,
            "scenario_id": "S1",
        },
    )
    stored_events = [
        SimpleNamespace(
            id=1,
            event_id="E-suspect",
            event_type="target_intent_change_suspected",
            target_id="target_00",
            sim_time_s=60,
            payload={"diff_id": "D1"},
        ),
        SimpleNamespace(
            id=2,
            event_id="E-confirmed",
            event_type="target_intent_changed",
            target_id="target_00",
            sim_time_s=62,
            payload={
                "diff_id": "D1",
                "suspicion_event_id": "E-suspect",
                "intent_llm_calls": call_metadata,
                "source": "real_intent_llm",
            },
        ),
    ]
    llm_rows = [
        SimpleNamespace(id=index, error_category="", **metadata)
        for index, metadata in enumerate(call_metadata, start=1)
    ]
    decision = SimpleNamespace(
        decision_id="decision-3",
        sim_time_s=63,
        trigger_event_ids=("E-confirmed",),
        final_plan_id="plan-3",
    )
    plan = SimpleNamespace(
        plan_id="plan-3",
        revision=3,
        status="active",
        trigger_event_ids=("E-confirmed",),
        regional_plans={"target_00": object()},
    )

    class Engine:
        def verification_evidence(self) -> dict[str, object]:
            return {
                "events": (
                    {
                        "event_id": "E-blue-response",
                        "event_type": "state_changed",
                        "entity_id": "target_00",
                        "sim_time_s": 70,
                        "payload": {
                            "phase": "blue_response",
                            "plan_revision": 3,
                        },
                    },
                )
            }

    class EventStore:
        def list_events(self, **kwargs):
            return stored_events if kwargs.get("since_id") == 0 else []

    loop = SimpleNamespace(
        events=EventStore(),
        ledger=SimpleNamespace(
            list_llm_calls=lambda **_kwargs: llm_rows,
            list_decisions=lambda *_args, **_kwargs: [decision],
        ),
        plans=SimpleNamespace(get_plan=lambda plan_id: plan if plan_id == "plan-3" else None),
        runtime=SimpleNamespace(
            get_state=lambda: {
                "prediction_diffs": {
                    "target_00": TrajectoryDiffResult(
                        diff_id="D1",
                        target_id="target_00",
                        previous_prediction_id="P1",
                        current_prediction_id="P2",
                        previous_sim_time_s=30,
                        current_sim_time_s=60,
                        status="comparable",
                        overlap_start_s=60,
                        overlap_end_s=660,
                        absolute_rms_m=300,
                        normalized_rms=3,
                        absolute_floor_m=250,
                        normalized_threshold=2.45,
                        reset_normalized_threshold=1.75,
                        reset_absolute_floor_m=150,
                        threshold_schema_version="trajectory-diff-v1",
                        confirmation_cycles=2,
                    )
                }
            }
        ),
        _epoch_repository=None,
    )
    bundle = _RunBundle(
        config=SimpleNamespace(scenario=SimpleNamespace(scenario_id="S1")),
        run_dir=Path("run"),
        loop=loop,
        engine=Engine(),
        replay=object(),
        hub=object(),
        stop=Event(),
        worker_errors=[],
    )
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._bundle = bundle
    controller._verification_audit = True

    evidence = controller.verification_evidence()

    assert evidence["events"][1]["payload"]["intent_llm_call_ids"] == (
        "LLM-1",
        "LLM-2",
    )
    assert evidence["prediction_diffs"][0]["diff_id"] == "D1"
    assert evidence["llm_calls"][0]["call_id"] == "LLM-1"
    assert evidence["decisions"][0]["final_plan_id"] == "plan-3"
    assert evidence["committed_plans"][0]["revision"] == 3


def test_verification_evidence_retries_transient_sqlite_lock() -> None:
    controller = RunController.__new__(RunController)
    controller._lock = RLock()
    controller._verification_evidence_attempts = 0

    def flaky_reader() -> dict[str, object]:
        controller._verification_evidence_attempts += 1
        if controller._verification_evidence_attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return {"events": ()}

    controller._verification_evidence_once = flaky_reader

    assert controller.verification_evidence() == {"events": ()}
    assert controller._verification_evidence_attempts == 2


def test_stored_verification_event_projection_keeps_causal_fields_public() -> None:
    projection = _stored_verification_event_projection(
        SimpleNamespace(
            event_id="chain-1:motion_effect",
            event_type="state_changed",
            target_id="target_00",
            sim_time_s=25,
            payload={
                "phase": "adversary_motion_effect",
                "chain_id": "chain-1",
                "decision_id": "decision-1",
                "motion_effect_event_id": "chain-1:motion_effect",
                "speed_delta_mps": 1.0,
                "heading_delta_rad": 0.1,
                "depth_delta_m": 0.0,
            },
        )
    )

    assert projection["phase"] == "adversary_motion_effect"
    assert projection["chain_id"] == "chain-1"
    assert projection["decision_id"] == "decision-1"
    assert projection["speed_delta_mps"] == 1.0
    assert projection["depth_delta_m"] == 0.0
