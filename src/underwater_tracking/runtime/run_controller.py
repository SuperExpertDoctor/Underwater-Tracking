"""Own the replaceable live-simulation bundle used by ``serve``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
import time
from typing import Any

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.config.models import AppConfig
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.runtime.models import RunPhase, RunRequest, RunSummary, ShutdownReport
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.domain.ui_models import PlanningHealthView


def _all_verification_events(repository: Any, scenario_id: str) -> tuple[Any, ...]:
    """Read the append-only event store without truncating an eight-hour run."""
    events: list[Any] = []
    since_id = 0
    while True:
        batch = repository.list_events(
            scenario_id=scenario_id,
            since_id=since_id,
            limit=1_000,
        )
        if not batch:
            break
        events.extend(batch)
        next_id = max(int(event.id) for event in batch)
        if next_id <= since_id:
            raise RuntimeError("verification event cursor did not advance")
        since_id = next_id
        if len(batch) < 1_000:
            break
    return tuple(events)


def _llm_call_projection(call: Any) -> dict[str, object]:
    return {
        "call_id": f"LLM-{call.id}",
        "operation": call.operation,
        "model": call.model,
        "prompt_version": call.prompt_version,
        "request_hash": call.request_hash,
        "response_hash": call.response_hash,
        "error_category": call.error_category,
        "sim_time_s": call.sim_time_s,
        "scenario_id": call.scenario_id,
    }


def _llm_call_ref(value: Any) -> tuple[object, ...]:
    def field(name: str) -> object:
        return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

    return tuple(
        field(name)
        for name in (
            "operation",
            "model",
            "prompt_version",
            "request_hash",
            "response_hash",
            "sim_time_s",
            "scenario_id",
        )
    )


def _engine_verification_event_projection(
    event: Mapping[str, object],
) -> dict[str, object]:
    projection = dict(event)
    raw_payload = event.get("payload")
    payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
    if isinstance(event.get("phase"), str):
        payload.setdefault("phase", event["phase"])
    plan_version = event.get("plan_version")
    if isinstance(plan_version, int) and not isinstance(plan_version, bool):
        payload.setdefault("plan_revision", plan_version)
    projection["payload"] = payload
    return projection


def _target_wall_deadline(
    *,
    wall_origin: float,
    sim_origin: float,
    sim_time_s: float,
    effective_speed: float,
) -> float:
    """Map a simulation timestamp to its monotonic wall-clock deadline."""
    return wall_origin + max(0.0, float(sim_time_s) - sim_origin) / effective_speed


@dataclass(slots=True)
class _RunBundle:
    config: AppConfig
    run_dir: Path
    loop: Any
    engine: SimulationEngine
    replay: ReplayService
    hub: OperationalHub
    stop: Event
    worker_errors: list[BaseException]
    mission_controller: MissionController | None = None
    worker: Thread | None = None
    manifest_written: bool = False
    effective_demo_speed: float | None = None
    phase: RunPhase = RunPhase.RUNNING


class RunController:
    """Create, replace, and close a single running simulation bundle.

    A request is completely validated and constructed before the installed
    bundle is touched. This makes an invalid target count a no-op for the
    active run, which is important once the command center exposes this as
    an operator control.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        output_root: Path = Path("outputs"),
        llm: Mapping[str, StructuredLLM[Any]] | None = None,
        steps: int = 0,
        speed: float | None = None,
        synthetic_max_target_count: int | None = None,
        continuous: bool = False,
        verification_audit: bool = False,
    ) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if speed is not None and speed < 0:
            raise ValueError("speed must be non-negative")
        if synthetic_max_target_count is not None and synthetic_max_target_count < 1:
            raise ValueError("synthetic_max_target_count must be positive")
        self._config = config
        self._output_root = output_root
        self._llm = llm
        self._steps = steps
        self._speed = speed
        self._synthetic_max_target_count = synthetic_max_target_count
        self._continuous = continuous
        self._verification_audit = verification_audit
        self._lock = RLock()
        self._bundle: _RunBundle | None = None
        self._aborted_bundle: _RunBundle | None = None
        self._aborted = False
        self._last_shutdown_report = ShutdownReport(completed=True)

    def _effective_speed(self, config: AppConfig) -> float:
        return config.timing.demo_time_scale if self._speed is None else self._speed

    def start_run(self, target_count: int, seed: int | None = None) -> RunSummary:
        """Build and atomically install a new bundle for ``target_count``."""
        request = RunRequest(target_count=target_count, seed=seed)
        config = self._config_for(request.target_count)
        selected_seed = config.scenario.seed if request.seed is None else request.seed
        candidate: _RunBundle | None = None
        try:
            candidate = self._build_bundle(config, selected_seed)
            candidate.worker = self._start_worker(candidate)
            with self._lock:
                previous = self._bundle
                if previous is not None and not self._close_bundle(previous, timeout_s=10.0):
                    raise RuntimeError("the active run is still shutting down")
                self._bundle = candidate
                return self._summary(candidate)
        except BaseException:
            if candidate is not None:
                self._close_bundle(candidate)
            raise

    def current(self) -> RunSummary:
        """Return the installed run's latest public state."""
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._summary(self._bundle)

    def planning_health(self) -> PlanningHealthView:
        """Read planning status without holding the live engine lock."""
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            return PlanningHealthView(status="idle")
        reader = getattr(bundle.loop, "planning_health", None)
        if not callable(reader):
            return PlanningHealthView(status="idle")
        try:
            value = reader()
        except Exception as exc:  # noqa: BLE001 - health must remain available
            return PlanningHealthView(
                status="degraded",
                last_error=f"{type(exc).__name__}: {exc}"[:2000],
            )
        if isinstance(value, PlanningHealthView):
            return value
        return PlanningHealthView.model_validate(value)

    @contextmanager
    def mutation_guard(self) -> Iterator[None]:
        """Serialize a mutating request with the terminal phase transition."""
        with self._lock:
            bundle = self._bundle
            if bundle is not None and bundle.phase is RunPhase.COMPLETED:
                raise RuntimeError(
                    "the live run is completed; mutation endpoints are closed"
                )
            yield

    def verification_physics(self) -> dict[str, object]:
        """Return redacted aggregate physics audit data when explicitly enabled."""
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise RuntimeError("no live run has been started")
        reader = getattr(bundle.engine, "verification_audit", None)
        if not self._verification_audit or not callable(reader):
            raise RuntimeError("verification audit is disabled")
        return dict(reader())

    @staticmethod
    def _drain_completed_background_for_evidence(bundle: _RunBundle) -> bool:
        """Finish a completed finite run before exposing its audit snapshot."""
        if bundle.phase is not RunPhase.COMPLETED:
            return True
        drain = getattr(bundle.loop, "drain_background_cycle", None)
        if not callable(drain):
            return True
        try:
            return bool(drain(timeout_s=4.0))
        except TypeError:
            return bool(drain())
        except BaseException:  # noqa: BLE001 - a failed drain must remain visible
            return False

    def verification_evidence(self) -> dict[str, object]:
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise RuntimeError("no live run has been started")
        reader = getattr(bundle.engine, "verification_evidence", None)
        if not self._verification_audit or not callable(reader):
            raise RuntimeError("verification audit is disabled")
        background_drain_completed = self._drain_completed_background_for_evidence(bundle)
        evidence = dict(reader())
        evidence["background_drain_completed"] = background_drain_completed
        scenario_id = bundle.config.scenario.scenario_id
        stored_events = _all_verification_events(bundle.loop.events, scenario_id)
        llm_calls = tuple(
            _llm_call_projection(call)
            for call in sorted(
                bundle.loop.ledger.list_llm_calls(
                    scenario_id=scenario_id,
                    operation="intent",
                    limit=10_000,
                ),
                key=lambda call: call.id,
            )
        )
        llm_call_ids_by_ref = {
            _llm_call_ref(call): str(call["call_id"])
            for call in llm_calls
        }
        event_by_id: dict[str, dict[str, object]] = {}
        raw_engine_events = evidence.get("events", ())
        if isinstance(raw_engine_events, (list, tuple)):
            for value in raw_engine_events:
                if isinstance(value, Mapping):
                    projection = _engine_verification_event_projection(value)
                    event_id = projection.get("event_id")
                    if isinstance(event_id, str) and event_id:
                        event_by_id[event_id] = projection
        for event in stored_events:
            projection = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "entity_id": event.target_id,
                "sim_time_s": event.sim_time_s,
                "payload": dict(event.payload),
            }
            if event.event_type == "target_intent_changed":
                raw_refs = event.payload.get("intent_llm_calls", ())
                call_ids = tuple(
                    call_id
                    for raw_ref in raw_refs
                    if isinstance(raw_ref, Mapping)
                    and (call_id := llm_call_ids_by_ref.get(_llm_call_ref(raw_ref)))
                    is not None
                )
                projection["payload"]["intent_llm_call_ids"] = call_ids
            event_by_id[event.event_id] = projection
        evidence["events"] = tuple(
            sorted(
                event_by_id.values(),
                key=lambda event: (
                    int(event.get("sim_time_s", 0)),
                    str(event.get("event_id", "")),
                ),
            )
        )
        evidence["llm_calls"] = llm_calls

        state = bundle.loop.runtime.get_state()
        raw_diffs = state.get("prediction_diffs", {}) if isinstance(state, Mapping) else {}
        diff_by_id: dict[str, dict[str, object]] = {}
        if isinstance(raw_diffs, Mapping):
            for value in raw_diffs.values():
                model_dump = getattr(value, "model_dump", None)
                projection = (
                    model_dump(mode="json")
                    if callable(model_dump)
                    else dict(value)
                    if isinstance(value, Mapping)
                    else None
                )
                if isinstance(projection, dict):
                    diff_id = projection.get("diff_id")
                    if isinstance(diff_id, str) and diff_id:
                        diff_by_id[diff_id] = projection
        for event in stored_events:
            if event.event_type != "target_intent_change_suspected":
                continue
            diff_id = event.payload.get("diff_id")
            if not isinstance(diff_id, str) or diff_id in diff_by_id:
                continue
            diff_by_id[diff_id] = {
                "target_id": event.target_id,
                **dict(event.payload),
            }
        evidence["prediction_diffs"] = tuple(diff_by_id.values())

        decisions = bundle.loop.ledger.list_decisions(scenario_id, limit=10_000)
        evidence["decisions"] = tuple(
            {
                "decision_id": decision.decision_id,
                "sim_time_s": decision.sim_time_s,
                "trigger_event_ids": decision.trigger_event_ids,
                "final_plan_id": decision.final_plan_id,
            }
            for decision in reversed(decisions)
        )
        committed_plans: dict[str, dict[str, object]] = {}
        for decision in decisions:
            if not decision.final_plan_id or decision.final_plan_id in committed_plans:
                continue
            plan = bundle.loop.plans.get_plan(decision.final_plan_id)
            if plan is None:
                continue
            committed_plans[plan.plan_id] = {
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "status": plan.status,
                "target_ids": tuple(sorted(plan.regional_plans)),
                "trigger_event_ids": plan.trigger_event_ids,
            }
        evidence["committed_plans"] = tuple(committed_plans.values())
        epoch_repository = getattr(bundle.loop, "_epoch_repository", None)
        latest = getattr(epoch_repository, "latest", None)
        if callable(latest):
            try:
                value = latest(bundle.config.scenario.scenario_id)
            except (KeyError, OSError, RuntimeError, ValueError):
                value = None
            if value is not None:
                epoch, result = value
                evidence["blue_epoch_id"] = epoch.epoch_id
                evidence["blue_plan_version"] = (
                    result.plan_version if result is not None else None
                )
                evidence["blue_estimate_ids"] = epoch.public_target_estimate_ids
        return evidence

    def retry_initial_planning(self, *, expected_epoch_id: str | None) -> str:
        """Start one explicit bootstrap retry without advancing physics."""
        with self._lock:
            bundle = self._bundle
            if bundle is None:
                raise RuntimeError("no live run has been started")
            if bundle.phase is not RunPhase.AWAITING_RETRY:
                raise ValueError(
                    f"planning retry is unavailable during {bundle.phase.value}"
                )
            retry = getattr(bundle.loop, "retry_initial_planning", None)
            if not callable(retry):
                raise RuntimeError("the active run does not support bootstrap retry")
            self._set_phase(bundle, RunPhase.BOOTSTRAP_PLANNING)
            epoch_id = str(retry(expected_epoch_id=expected_epoch_id))
            bundle.worker = self._start_worker(bundle)
            return epoch_id

    @property
    def runtime(self) -> Any:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.loop.runtime

    @property
    def replay(self) -> ReplayService:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.replay

    @property
    def hub(self) -> OperationalHub:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.hub

    @property
    def mission_controller(self) -> MissionController | None:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("no live run has been started")
            return self._bundle.mission_controller

    def close(self, *, timeout_s: float = 10.0) -> bool:
        """Stop and release the current bundle within a bounded timeout."""
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        with self._lock:
            bundle = self._bundle or getattr(self, "_aborted_bundle", None)
            if bundle is None:
                self._last_shutdown_report = ShutdownReport(completed=True)
                return True
            closed = self._close_bundle(bundle, timeout_s=timeout_s)
            self._last_shutdown_report = self._bundle_shutdown_report(bundle, closed)
            if not closed:
                return False
            if self._bundle is bundle:
                self._bundle = None
            if getattr(self, "_aborted_bundle", None) is bundle:
                self._aborted_bundle = None
            return True

    def shutdown_report(self) -> ShutdownReport:
        """Return the latest redacted shutdown outcome without mutating state."""
        with self._lock:
            report = getattr(self, "_last_shutdown_report", ShutdownReport(completed=True))
            return report.model_copy(deep=True)

    def abort(self) -> None:
        """Request immediate shutdown without waiting for provider calls.

        Ctrl+C must not wait for an in-flight remote LLM request or a worker
        that is currently inside provider code. The process owner performs
        final child-process cleanup; this method only detaches the live bundle
        and signals its cooperative workers before the interpreter exits.
        """
        self._aborted = True
        if not self._lock.acquire(blocking=False):
            return
        try:
            bundle = self._bundle
            if bundle is None:
                return
            self._bundle = None
            self._aborted_bundle = bundle
        finally:
            self._lock.release()

        bundle.stop.set()
        abort = getattr(bundle.loop, "abort", None)
        if callable(abort):
            abort()

    @property
    def aborted(self) -> bool:
        """Whether the active run has entered the non-blocking abort path."""
        return bool(getattr(self, "_aborted", False))

    def _config_for(self, target_count: int) -> AppConfig:
        scenario = self._config.scenario
        if self._config.environment is not None:
            roster_size = len(self._config.environment.submarines)
            if target_count > roster_size:
                raise ValueError(
                    "target_count exceeds the loaded platform-core target roster"
                )
            return self._config

        capacity = (
            self._synthetic_max_target_count
            if self._synthetic_max_target_count is not None
            else scenario.max_target_count
        )
        if target_count > capacity:
            raise ValueError(
                f"target_count must be between 1 and {capacity} for this scenario"
            )
        return self._config.model_copy(
            update={
                "scenario": scenario.model_copy(
                    update={
                        "initial_target_count": target_count,
                        "max_target_count": target_count,
                    }
                )
            }
        )

    def _build_bundle(self, config: AppConfig, seed: int) -> _RunBundle:
        """Construct all resources before replacing the active bundle."""
        # Kept lazy to avoid a module cycle while ``cli`` owns _AgentLoop.
        from underwater_tracking.cli import (
            _AgentLoop,
            _create_public_run_dir,
            _mission_controller_for,
        )

        run_dir = _create_public_run_dir("serve", output_root=self._output_root)
        loop: Any | None = None
        try:
            mission_controller = _mission_controller_for(config)
            loop = _AgentLoop(
                config,
                database_path=run_dir / "agent.db",
                llm=self._llm,
                run_id=run_dir.name,
                steps=self._steps,
                seed=seed,
                background_carrier=True,
            )
            effective_demo_speed = self._effective_speed(config)
            loop._effective_demo_speed = effective_demo_speed
            engine = SimulationEngine(
                config,
                seed=seed,
                output_dir=run_dir,
                carrier=loop.on_situation,
                mission_controller=mission_controller,
                transition_coordinator=loop._transition_coordinator,
                event_repository=loop.events,
                verification_audit=self._verification_audit,
            )
            initial_phase = (
                RunPhase.BOOTSTRAP_PLANNING
                if self._steps == 0
                else RunPhase.RUNNING
            )
            loop._run_phase = initial_phase.value
            loop.attach(engine)
            if self._steps == 0:
                loop.begin_bootstrap_planning(engine.publication_situation())
            return _RunBundle(
                config=config,
                run_dir=run_dir,
                loop=loop,
                engine=engine,
                replay=ReplayService(run_dir / "operational_frames.jsonl"),
                hub=loop.hub,
                stop=Event(),
                worker_errors=[],
                mission_controller=mission_controller,
                effective_demo_speed=effective_demo_speed,
                phase=initial_phase,
            )
        except BaseException:
            if loop is not None:
                loop.close()
            raise

    def _start_worker(self, bundle: _RunBundle) -> Thread:
        """Start a completed bundle's simulation worker."""
        from underwater_tracking.cli import _step_with_llm_retries

        def drive() -> None:
            completed = 0
            effective_speed = self._effective_speed(bundle.config)
            wall_origin = time.monotonic()
            sim_origin = float(bundle.engine._clock.sim_time_s)
            try:
                if bundle.phase is RunPhase.BOOTSTRAP_PLANNING:
                    while not bundle.stop.is_set():
                        outcome = bundle.loop.bootstrap_result()
                        if outcome is not None:
                            self._set_phase(bundle, (
                                RunPhase.RUNNING
                                if outcome.status == "committed"
                                else RunPhase.AWAITING_RETRY
                            ))
                            break
                        if bundle.stop.wait(0.05):
                            self._set_phase(bundle, RunPhase.STOPPED)
                            return
                    if bundle.phase == RunPhase.AWAITING_RETRY:
                        return
                while (
                    not bundle.stop.is_set()
                    and (self._steps == 0 or completed < self._steps)
                    and (
                        self._continuous
                        or bundle.engine._clock.sim_time_s
                        < bundle.config.scenario.duration_s
                    )
                ):
                    if not _step_with_llm_retries(
                        bundle.engine, bundle.loop, bundle.config, stop=bundle.stop
                    ):
                        if bundle.stop.wait(1.0):
                            break
                        continue
                    completed += 1
                    if effective_speed > 0:
                        deadline = _target_wall_deadline(
                            wall_origin=wall_origin,
                            sim_origin=sim_origin,
                            sim_time_s=bundle.engine._clock.sim_time_s,
                            effective_speed=effective_speed,
                        )
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            bundle.stop.wait(remaining)
                    else:
                        bundle.stop.wait(0.001)
                if not bundle.stop.is_set() and not bundle.worker_errors:
                    self._set_phase(bundle, RunPhase.COMPLETED)
            except BaseException as exc:  # noqa: BLE001 - reported via RunSummary
                bundle.worker_errors.append(exc)
                self._set_phase(bundle, RunPhase.FAILED)
                bundle.stop.set()

        worker = Thread(target=drive, name="underwater-simulation", daemon=True)
        worker.start()
        return worker

    def _summary(self, bundle: _RunBundle) -> RunSummary:
        if bundle.worker_errors:
            status = "failed"
        elif bundle.phase is RunPhase.AWAITING_RETRY:
            status = "awaiting_retry"
        elif bundle.stop.is_set():
            status = "stopped"
        elif bundle.worker is not None and not bundle.worker.is_alive():
            status = "completed"
        else:
            status = "running"
        publisher = getattr(bundle.loop, "_publisher", None)
        return RunSummary(
            run_id=bundle.run_dir.name,
            scenario_id=bundle.config.scenario.scenario_id,
            target_count=bundle.config.scenario.initial_target_count,
            seed=bundle.engine._seed,
            sim_time_s=bundle.engine._clock.sim_time_s,
            frame_count=publisher.frame_count if publisher is not None else 0,
            status=status,
            path=bundle.run_dir,
            effective_demo_speed=(
                bundle.effective_demo_speed
                if bundle.effective_demo_speed is not None
                else self._effective_speed(bundle.config)
            ),
            phase=bundle.phase,
        )

    def _set_phase(self, bundle: _RunBundle, phase: RunPhase) -> None:
        with self._lock:
            bundle.phase = phase
            try:
                setattr(bundle.loop, "_run_phase", phase.value)
                if phase in {
                    RunPhase.COMPLETED,
                    RunPhase.FAILED,
                    RunPhase.AWAITING_RETRY,
                    RunPhase.STOPPED,
                }:
                    setattr(bundle.loop, "_manifest_status", phase.value)
            except AttributeError:
                # Keep legacy injected loop fakes usable; real AgentLoop instances
                # always expose the publisher phase projection.
                return
        if phase is RunPhase.STOPPED:
            return
        publish = getattr(bundle.loop, "publish_latest", None)
        if callable(publish):
            try:
                publish()
            except Exception:  # noqa: BLE001 - phase telemetry cannot stop shutdown
                pass

    @staticmethod
    def _bundle_shutdown_report(
        bundle: _RunBundle,
        completed: bool,
    ) -> ShutdownReport:
        reader = getattr(bundle.loop, "shutdown_report", None)
        if callable(reader):
            try:
                report = reader()
                if isinstance(report, ShutdownReport):
                    return report
                return ShutdownReport.model_validate(report)
            except Exception:  # noqa: BLE001 - reporting must not mask shutdown state
                pass
        return ShutdownReport(completed=completed)

    def _close_bundle(self, bundle: _RunBundle, *, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        completed_run = bundle.phase is RunPhase.COMPLETED
        drain_ok = True
        manifest_status = {
            RunPhase.COMPLETED: "completed",
            RunPhase.AWAITING_RETRY: "awaiting_retry",
            RunPhase.FAILED: "failed",
        }.get(bundle.phase, "stopped")
        try:
            setattr(bundle.loop, "_manifest_status", manifest_status)
        except AttributeError:
            pass
        self._set_phase(bundle, RunPhase.STOPPING)
        bundle.stop.set()
        abort = getattr(bundle.loop, "abort", None)
        if callable(abort) and not completed_run:
            abort()
        if bundle.worker is not None:
            bundle.worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if bundle.worker.is_alive():
                return False
        if completed_run:
            drain = getattr(bundle.loop, "drain_background_cycle", None)
            if callable(drain):
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    drain_ok = bool(drain(timeout_s=remaining))
                except TypeError:
                    drain_ok = bool(drain())
                except BaseException as exc:  # noqa: BLE001 - release gate stays truthful
                    bundle.worker_errors.append(exc)
                    drain_ok = False
                if not drain_ok:
                    bundle.worker_errors.append(
                        RuntimeError("background carrier drain did not complete")
                    )
                    try:
                        setattr(bundle.loop, "_manifest_status", "failed")
                    except AttributeError:
                        pass
        if callable(abort):
            abort()
        if not bundle.manifest_written:
            bundle.loop.write_manifest(bundle.run_dir)
            bundle.manifest_written = True
        close = getattr(bundle.loop, "close")
        remaining = max(0.0, deadline - time.monotonic())
        try:
            closed = close(timeout_s=remaining)
        except TypeError:
            # Keep injected legacy loop fakes source-compatible.
            closed = close()
        if closed is False or not drain_ok:
            return False
        self._set_phase(bundle, RunPhase.STOPPED)
        return True
