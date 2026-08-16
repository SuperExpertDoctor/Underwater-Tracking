# src/underwater_tracking/cli.py
"""Command-line entry points for the underwater tracking assistant.

``simulate`` runs the deterministic headless simulation and writes frames.
``agent-run`` runs the same scenario through the resilient LangGraph
carrier: it loads the config, creates the SQLite repositories and
checkpointer, builds the real LongCat HTTP provider (the API key is read at
call time from the configured api_key or environment variable (env wins);
``agent-run`` fails with a message naming both sources when neither exists),
wires the engine's
group reports into ``CarrierRuntime`` (the carrier hook is called at the
end of every observation cycle), applies the carrier's committed plan
commands back to the group manager at the next observation cycle, and
writes a run manifest (``manifest.json``) plus the frame log
(``frames.jsonl``) into ``outputs/run-<uuid>/``.
``serve`` uses the same loop in a background simulation thread and exposes
the runtime's truth-safe operational frames, replay, WebSocket, directive,
assignment, and question ports through FastAPI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from threading import Event, Thread
import uuid
from pathlib import Path
from typing import Any, cast

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.api.app import create_app
from underwater_tracking.api.frame_logger import FrameLogger as OperationalFrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.domain.agent_models import VerificationCommand
from underwater_tracking.domain.models import (
    DeploymentState,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.sqlite import now_ms
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_OBSERVATION_STEP_S = 30
_BATTERY_ROTATION_THRESHOLD = 0.3


def _create_public_run_dir(prefix: str, *, output_root: Path = Path("outputs")) -> Path:
    """Create a public run directory without exposing deterministic state."""
    run_dir = output_root / f"{prefix}-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="underwater-tracking")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--steps", type=int, required=True)
    simulate.add_argument("--seed", type=int, required=True)
    simulate.set_defaults(handler=_simulate)

    agent_run = sub.add_parser("agent-run")
    agent_run.add_argument("--config", required=True)
    agent_run.add_argument("--steps", type=int, required=True)
    agent_run.add_argument("--seed", type=int, required=True)
    agent_run.set_defaults(handler=_agent_run)

    serve = sub.add_parser("serve")
    serve.add_argument("--config", required=True)
    serve.add_argument("--seed", type=int, required=True)
    serve.add_argument("--steps", type=int, default=0, help="0 runs until shutdown")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulation speed relative to wall time; 0 runs without pacing",
    )
    serve.set_defaults(handler=_serve)

    args = parser.parse_args(argv)
    return cast(int, args.handler(load_app_config(args.config), args))


def _simulate(config: AppConfig, args: argparse.Namespace) -> int:
    engine = SimulationEngine(config, seed=args.seed)
    for _ in range(args.steps):
        engine.step()
    return 0


def _agent_run(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the agent-coupled scenario and write manifest plus JSONL."""
    run_dir = _create_public_run_dir("run")
    database_path = run_dir / "agent.db"
    loop = _AgentLoop(
        config,
        database_path=database_path,
        llm=_build_llm(config),
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
    )
    engine = SimulationEngine(
        config, seed=args.seed, output_dir=run_dir, carrier=loop.on_situation
    )
    loop.attach(engine)
    try:
        for _ in range(args.steps):
            engine.step()
    except Exception as exc:  # noqa: BLE001 - surface as a CLI failure
        print(f"agent-run failed: {exc}", file=sys.stderr)
        loop.close()
        return 1
    loop.write_manifest(run_dir)
    loop.close()
    return 0


def _serve(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the LangGraph simulation beside the FastAPI command-center API."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - packaging failure path
        print("serve requires the 'uvicorn' package", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.speed < 0:
        raise SystemExit("--speed must be non-negative")

    run_dir = _create_public_run_dir("serve")
    loop = _AgentLoop(
        config,
        database_path=run_dir / "agent.db",
        llm=_build_llm(config),
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
    )
    engine = SimulationEngine(
        config, seed=args.seed, output_dir=run_dir, carrier=loop.on_situation
    )
    loop.attach(engine)
    replay = ReplayService(run_dir / "operational_frames.jsonl")
    runtime = loop.runtime
    app = create_app(runtime=runtime, replay=replay, hub=loop.hub)
    stop = Event()
    worker_errors: list[BaseException] = []

    def drive() -> None:
        completed = 0
        try:
            while not stop.is_set() and (args.steps == 0 or completed < args.steps):
                engine.step()
                completed += 1
                if args.speed > 0:
                    stop.wait(config.timing.physics_step_s / args.speed)
        except BaseException as exc:  # noqa: BLE001 - surfaced after server shutdown
            worker_errors.append(exc)
            stop.set()

    worker = Thread(target=drive, name="underwater-simulation", daemon=True)
    worker.start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop.set()
        worker.join(timeout=30.0)
        loop.write_manifest(run_dir)
        loop.close()
    if worker_errors:
        print(f"serve simulation failed: {worker_errors[0]}", file=sys.stderr)
        return 1
    return 0


def _build_llm(config: AppConfig) -> HTTPStructuredLLM:
    """The real LongCat HTTP client, failing clearly when it cannot run.

    ``agent-run`` has no mock fallback: the bearer token is read at call
    time from the configured api_key (``configs/.env``, git-ignored) or the
    configured environment variable (env wins), so ``agent-run`` fails up
    front, naming the two sources, only when neither exists.
    """
    llm_config = config.llm
    if llm_config is None:
        print(
            "agent-run requires an 'llm' section in the config file",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if os.environ.get(llm_config.api_key_env) is None and llm_config.api_key is None:
        print(
            f"agent-run requires the {llm_config.api_key_env} environment variable"
            " or a configured api_key in the llm config",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return HTTPStructuredLLM(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key_env=llm_config.api_key_env,
        api_key=llm_config.api_key,
        request_timeout_s=llm_config.request_timeout_s,
        connect_timeout_s=llm_config.connect_timeout_s,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=llm_config.max_retries,
        backoff_base_s=llm_config.backoff_base_s,
        backoff_max_s=llm_config.backoff_max_s,
    )


class _AgentLoop:
    """Wires the engine's group reports into CarrierRuntime and back.

    ``on_situation`` is the engine hook called at the end of every
    observation cycle: it submits the initialization event once the belief
    history is warm, runs one carrier tick (deferring any carrier error so
    the group loop keeps running), and applies newly committed plan
    commands back to the engine (translated into group commands at the next
    observation cycle).
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        database_path: Path,
        llm: HTTPStructuredLLM,
        run_id: str,
        steps: int,
        seed: int,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self.database_path = database_path
        self.scenario_id = _SCENARIO_ID
        self.run_id = run_id
        self.steps = steps
        self._seed = seed
        self.plans = PlanRepository(database_path)
        self.events = EventRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self.llm = llm
        self.situation: SituationSnapshot | None = None
        self.carrier_error_count = 0
        self._runtime: CarrierRuntime | None = None
        self._engine: SimulationEngine | None = None
        self._clock = SimulationClock(step_s=_OBSERVATION_STEP_S)
        self._initialization_submitted = False
        self._last_plan_id: str | None = None
        self._last_strategic_review_s = 0
        self._last_battery_rotation_s: dict[str, int] = {}
        self.hub = OperationalHub()
        self._publisher: OperationalFramePublisher | None = None

    def attach(self, engine: SimulationEngine) -> None:
        """Create the carrier runtime over the same SQLite database."""
        self._engine = engine
        self._runtime = CarrierRuntime(
            self._deps(), scenario_id=self.scenario_id, database_path=self.database_path
        )
        self._publisher = OperationalFramePublisher(
            runtime=self._runtime,
            ledger=self.ledger,
            events=self.events,
            hub=self.hub,
            logger=OperationalFrameLogger(
                self.database_path.parent / "operational_frames.jsonl"
            ),
        )

    @property
    def runtime(self) -> CarrierRuntime:
        """The live runtime exposed to the API after ``attach``."""
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("agent loop is not attached to an engine")
        return runtime

    def _deps(self) -> CarrierDependencies:
        config = self._config
        agent = config.agent
        return CarrierDependencies(
            plans=self.plans,
            events=self.events,
            ledger=self.ledger,
            llm=self.llm,
            predictor=make_snapshot_predictor(
                belief_history=self._belief_history,
                horizon_s=config.timing.prediction_horizon_s,
                sample_step_s=config.timing.observation_step_s,
                max_speed_mps=config.tracking.uuv_max_speed_mps,
                max_turn_rate_rad_s=config.tracking.uuv_max_turn_rate_rad_s,
            ),
            situation_provider=self._live_situation,
            belief_history=self._belief_history,
            clock=self._clock,
            monitor=EventMonitor(
                scenario_id=self.scenario_id,
                warning_threshold=config.tracking.quality_warning,
                warning_hold_s=agent.quality_warning_persist_s if agent else 120,
                critical_threshold=config.tracking.quality_critical,
                cooldown_s=agent.event_cooldown_s if agent else 300,
                critical_hold_s=agent.quality_critical_persist_s if agent else 30,
                group_min_size=config.tracking.group_min_size,
            ),
            semantic_repairs=agent.semantic_repairs if agent else 2,
            model_id=config.llm.model if config.llm else "http",
        )

    def _live_situation(self, ref: str) -> SituationSnapshot:
        situation = self.situation
        if situation is None:
            raise RuntimeError(f"no live situation recorded for {ref!r}")
        return situation

    def _belief_history(
        self, snapshot: SituationSnapshot, target_id: str
    ) -> tuple[tuple[int, float, float], ...]:
        del snapshot
        engine = self._engine
        assert engine is not None
        return engine.belief_history(target_id)

    def _initialization_ready(self, situation: SituationSnapshot) -> bool:
        engine = self._engine
        assert engine is not None
        return all(
            len(engine.belief_history(report.target_id)) >= 3
            for report in situation.group_reports
        )

    def on_situation(self, situation: SituationSnapshot) -> None:
        """Engine hook: run one carrier cycle over the latest situation."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        engine.set_reservations(runtime.reservations())
        runtime.submit_events((*situation.pending_events, *self._feedback_events(situation)))
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            runtime.submit_event(
                event_type="initialization",
                entity_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
            )
        result: dict[str, Any] = {}
        try:
            result = runtime.tick()
            if result.get("commit_status") == "committed":
                self._apply_new_commands()
            self._apply_verification_commands(result)
        except Exception:  # noqa: BLE001 - the group loop must keep running
            self.carrier_error_count += 1
        finally:
            publisher = self._publisher
            if publisher is not None:
                try:
                    publisher.publish(situation)
                except Exception:  # noqa: BLE001 - telemetry cannot stop tracking
                    self.carrier_error_count += 1

    def _feedback_events(self, situation: SituationSnapshot) -> tuple[RuntimeEvent, ...]:
        """Generate deterministic review and low-energy rotation events."""
        events: list[RuntimeEvent] = []
        sim_time_s = situation.sim_time_s
        review_interval_s = self._config.timing.strategic_review_s
        last_review_s = self._last_strategic_review_s
        if review_interval_s > 0 and sim_time_s - last_review_s >= review_interval_s:
            events.append(
                RuntimeEvent(
                    event_id=f"{self.scenario_id}:strategic_review:{sim_time_s}",
                    scenario_id=self.scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="strategic_review",
                    entity_id=self.scenario_id,
                    level=EventLevel.STRATEGIC,
                    payload={"interval_s": review_interval_s},
                )
            )
            self._last_strategic_review_s = sim_time_s
        cooldown_s = self._config.agent.event_cooldown_s if self._config.agent else 300
        for uuv in sorted(situation.uuvs, key=lambda state: state.uuv_id):
            if (
                uuv.deployment_state is not DeploymentState.DEPLOYED
                or uuv.energy_fraction >= _BATTERY_ROTATION_THRESHOLD
                or uuv.group_id is None
            ):
                continue
            last_emitted_s = self._last_battery_rotation_s.get(uuv.uuv_id)
            if last_emitted_s is not None and sim_time_s - last_emitted_s < cooldown_s:
                continue
            events.append(
                RuntimeEvent(
                    event_id=f"{self.scenario_id}:battery_rotation:{uuv.uuv_id}:{sim_time_s}",
                    scenario_id=self.scenario_id,
                    sim_time_s=sim_time_s,
                    event_type="battery_rotation",
                    entity_id=uuv.uuv_id,
                    level=EventLevel.TACTICAL,
                    payload={
                        "energy_fraction": uuv.energy_fraction,
                        "rotation_threshold": _BATTERY_ROTATION_THRESHOLD,
                        "target_id": uuv.group_id,
                    },
                )
            )
            self._last_battery_rotation_s[uuv.uuv_id] = sim_time_s
        return tuple(events)

    def _apply_new_commands(self) -> None:
        """Apply newly committed plan commands back to the group manager."""
        engine = self._engine
        assert engine is not None
        active = self.plans.get_active(self.scenario_id)
        if active is None or active.plan_id == self._last_plan_id:
            return
        self._last_plan_id = active.plan_id
        for command in self.plans.list_commands(active.plan_id):
            engine.apply_plan_command(command)

    def _apply_verification_commands(self, result: dict[str, Any]) -> None:
        """Apply the deterministic verification protocol commands to the engine.

        Runs after the plan-command gate: the protocol's sensor-mode writes
        win over any plan command from the same cycle.
        """
        engine = self._engine
        assert engine is not None
        for command in result.get("verification_commands") or ():
            assert isinstance(command, VerificationCommand)
            engine.apply_verification_command(command)
        # Re-arm the pingers every cycle: a plan command's sensor-mode write
        # resets ``_ping_targets`` and would otherwise kill a live ping
        # mid-protocol, and the node only re-emits ping commands on new ping
        # events. Pingers are popped when the protocol closes, so this stops
        # exactly then.
        for contact_id, pinger in (result.get("verification_pingers") or {}).items():
            engine.set_sensor_mode(pinger, "active", ping_contact_id=contact_id)

    def write_manifest(self, run_dir: Path) -> None:
        """Write the run manifest summarizing the finished agent run."""
        active = self.plans.get_active(self.scenario_id)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "steps": self.steps,
            "llm": self._config.llm.model if self._config.llm else "http",
            "created_at_ms": now_ms(),
            "carrier_error_count": self.carrier_error_count,
            "decision_count": len(self.ledger.list_decisions(self.scenario_id)),
            "llm_call_count": len(self.ledger.list_llm_calls()),
            "active_plan_id": active.plan_id if active is not None else None,
            "active_plan_revision": active.revision if active is not None else None,
            "operational_frame_count": (
                self._publisher.frame_count
                if self._publisher is not None
                else 0
            ),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
        if self._runtime is not None:
            self._runtime.close()
        self.plans.close()
        self.events.close()
        self.ledger.close()


if __name__ == "__main__":
    sys.exit(main())
