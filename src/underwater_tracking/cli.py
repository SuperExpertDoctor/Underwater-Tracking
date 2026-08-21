# src/underwater_tracking/cli.py
"""Command-line entry points for the underwater tracking assistant.

``simulate`` runs the deterministic headless simulation and writes frames.
``agent-run`` runs the same scenario through the resilient LangGraph
carrier: it loads the config, creates the SQLite repositories and
checkpointer, builds the real LongCat HTTP provider (the API key is read at
call time from the configured api_key or environment variable (env wins);
``agent-run`` exposes a visible degraded state when chat configuration or
credentials are unavailable),
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
import signal
import sys
import time
from threading import Condition, Event, RLock, Thread
import uuid
from pathlib import Path
from typing import Any, cast

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.graphs.adversary import build_adversary_graph
from underwater_tracking.agent.graphs.slave import build_slave_graph
from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    LLMConfigError,
    LLMError,
    StructuredLLM,
    UnavailableStructuredLLM,
)
from underwater_tracking.agent.llm_factory import build_role_llm
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.optimize import PlanningConfig
from underwater_tracking.agent.runtime import CarrierRuntime, SensorModeControl
from underwater_tracking.api.app import create_app
from underwater_tracking.api.dependencies import MemoryServiceAdapter
from underwater_tracking.api.frame_logger import FrameLogger as OperationalFrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig, MemoryConfig, RuntimeRetentionConfig
from underwater_tracking.domain.agent_models import VerificationCommand
from underwater_tracking.domain.adversary_models import (
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
)
from underwater_tracking.domain.models import (
    DeploymentState,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.slave_models import SlaveSonarContext, SlaveSonarDecision
from underwater_tracking.knowledge.client import OntologyKnowledgeClient
from underwater_tracking.memory.embeddings import (
    EmbeddingProvider,
    HTTPEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from underwater_tracking.memory.reasoner import MemoryReasoner
from underwater_tracking.memory.retriever import DegradedMemoryRetriever, MemoryRetriever
from underwater_tracking.memory.service import MemoryService
from underwater_tracking.memory.source_reader import MemorySourceReader
from underwater_tracking.memory.worker import MemoryWorker
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.sqlite import now_ms
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.runtime.run_controller import RunController
from underwater_tracking.runtime.run_catalog import RunCatalog
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_BATTERY_ROTATION_THRESHOLD = 0.3
_DEFAULT_API_PORT = 8000
_API_PORT_ENV = "UNDERWATER_TRACKING_API_PORT"


def _configured_api_port() -> int:
    """Return the shared API port used by standalone backend and UI commands."""
    raw_port = os.environ.get(_API_PORT_ENV)
    if raw_port is None or not raw_port.strip():
        return _DEFAULT_API_PORT
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise SystemExit(f"{_API_PORT_ENV} must be an integer port") from exc
    if not 1 <= port <= 65_535:
        raise SystemExit(f"{_API_PORT_ENV} must be between 1 and 65535")
    return port


def _is_uuv_only_config(config: AppConfig | None) -> bool:
    """Use one strict UUV-only boundary for every production entry point."""
    if config is None:
        return False
    return bool(
        getattr(getattr(config, "scenario", None), "uuv_only", False)
        or getattr(getattr(config, "environment", None), "uuv_only", False)
    )


def _require_uuv_only_live_config(config: AppConfig) -> None:
    """Reject legacy or mixed rosters at every live runtime boundary."""
    if not _is_uuv_only_config(config) or config.environment is None or config.environment.usvs:
        raise SystemExit("live runtime requires an explicit UUV-only scenario")


def _build_memory_embedding_provider(
    config: MemoryConfig,
    *,
    ledger: DecisionLedger | None = None,
    scenario_id: str = "",
) -> EmbeddingProvider:
    """Build the configured real embedding provider without implicit fallback."""
    if config.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            config,
            ledger=ledger,
            scenario_id=scenario_id,
        )
    if config.embedding_provider == "http":
        return HTTPEmbeddingProvider(
            config,
            ledger=ledger,
            scenario_id=scenario_id,
        )
    raise LLMConfigError(
        f"unsupported memory embedding provider {config.embedding_provider!r}"
    )


@dataclass(slots=True)
class _BackgroundCarrierCycle:
    """One LLM cycle whose result is applied by the physics thread."""

    situation: SituationSnapshot
    adversary_contexts: tuple[AdversaryEscapeInput, ...]
    slave_contexts: tuple[SlaveSonarContext, ...]
    sensor_controls: tuple[SensorModeControl, ...] = ()
    slave_decisions: tuple[SlaveSonarDecision, ...] = ()
    adversary_decisions: tuple[AdversaryEscapeDecision, ...] = ()
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    done: bool = False


def _create_public_run_dir(prefix: str, *, output_root: Path = Path("outputs")) -> Path:
    """Create a public run directory without exposing deterministic state."""
    run_dir = output_root / f"{prefix}-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mission_controller_for(config: AppConfig) -> MissionController | None:
    """Create the controller shared by every UUV-only production entry point."""
    if not _is_uuv_only_config(config):
        return None
    return MissionController(
        scenario_id=config.scenario.scenario_id,
        region_entry_probability_threshold=config.scenario.region_entry_probability_threshold,
        region_transition_confirm_cycles=config.scenario.region_transition_confirm_cycles,
        event_history_limit=(
            config.agent.retention.mission_event_history_limit
            if config.agent is not None
            else 2048
        ),
    )


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
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument(
        "--speed",
        type=float,
        default=None,
        help="override simulation speed; default uses timing.demo_time_scale, 0 runs without pacing",
    )
    serve.set_defaults(handler=_serve)

    args = parser.parse_args(argv)
    if args.command == "serve" and args.port is None:
        args.port = _configured_api_port()
    return cast(int, args.handler(load_app_config(args.config), args))


def _simulate(config: AppConfig, args: argparse.Namespace) -> int:
    _require_uuv_only_live_config(config)
    engine = SimulationEngine(
        config,
        seed=args.seed,
        mission_controller=_mission_controller_for(config),
    )
    for _ in range(args.steps):
        engine.step()
    return 0


def _agent_run(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the agent-coupled scenario and write manifest plus JSONL."""
    _require_uuv_only_live_config(config)
    run_dir = _create_public_run_dir("run")
    database_path = run_dir / "agent.db"
    loop = _AgentLoop(
        config,
        database_path=database_path,
        llm=None,
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
    )
    mission_controller = _mission_controller_for(config)
    engine = SimulationEngine(
        config,
        seed=args.seed,
        output_dir=run_dir,
        carrier=loop.on_situation,
        mission_controller=mission_controller,
    )
    loop.attach(engine)
    try:
        for _ in range(args.steps):
            if not _step_with_llm_retries(engine, loop, config):
                print(
                    "agent-run paused after bounded LLM retries; "
                    "the current simulation cycle was not advanced",
                    file=sys.stderr,
                )
                loop.write_manifest(run_dir)
                loop.close()
                return 1
    except Exception as exc:  # noqa: BLE001 - surface as a CLI failure
        print(f"agent-run failed: {exc}", file=sys.stderr)
        loop.close()
        return 1
    loop.write_manifest(run_dir)
    loop.close()
    return 0


def _serve(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the LangGraph simulation beside the FastAPI command-center API."""
    _require_uuv_only_live_config(config)
    from importlib.util import find_spec

    if find_spec("uvicorn") is None:  # pragma: no cover - packaging failure path
        print("serve requires the 'uvicorn' package", file=sys.stderr)
        raise SystemExit(2)
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.speed is not None and args.speed < 0:
        raise SystemExit("--speed must be non-negative")

    controller: RunController | None = None
    interrupted = False
    try:
        controller = RunController(config, steps=args.steps, speed=args.speed)
        controller.start_run(config.scenario.initial_target_count, seed=args.seed)
        app = create_app(
            controller=controller,
            catalog=RunCatalog(Path("outputs")),
            directive_job_limit=(
                config.agent.retention.directive_job_limit
                if config.agent is not None
                else 256
            ),
        )
        assert controller is not None
        _run_api_server(
            app,
            host=args.host,
            port=args.port,
            on_interrupt=controller.abort,
        )
    except KeyboardInterrupt:
        interrupted = True
        if controller is not None:
            controller.abort()
        raise
    finally:
        if controller is not None and not interrupted:
            controller.close()
    return 0


def _run_api_server(
    app: Any,
    *,
    host: str,
    port: int,
    on_interrupt: Callable[[], None] | None = None,
) -> None:
    """Run Uvicorn while making the first signal an immediate interruption."""
    import uvicorn

    class ImmediateShutdownServer(uvicorn.Server):
        interrupted = False

        def handle_exit(self, sig: int, frame: object | None) -> None:
            del sig, frame
            if on_interrupt is not None:
                on_interrupt()
            self.interrupted = True
            self.should_exit = True

        async def serve(self, sockets: Any = None) -> None:
            # Uvicorn's capture_signals replays the signal after shutdown,
            # which would invoke the entry point's raising handler again.
            # Keep the same installation/restoration behavior without replay.
            handled_signals = (signal.SIGINT, signal.SIGTERM)
            original_handlers = {
                sig: signal.signal(sig, self.handle_exit) for sig in handled_signals
            }
            try:
                await self._serve(sockets=sockets)
            finally:
                for sig, handler in original_handlers.items():
                    signal.signal(sig, handler)

    server = ImmediateShutdownServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            timeout_graceful_shutdown=1.0,
        )
    )
    server.run()
    if server.interrupted:
        raise KeyboardInterrupt


def _build_llm(
    config: AppConfig,
    *,
    ledger: DecisionLedger | None = None,
    scenario_id: str = "",
) -> dict[str, HTTPStructuredLLM]:
    """Build role-specific HTTP clients or explicit degraded ports.

    The bearer token is read at call time from the configured api_key
    (``configs/.env``, git-ignored) or the configured environment variable
    (env wins). Missing credentials or legacy flat role settings must not
    construct a role client: the unavailable ports make the degraded state
    explicit and reject calls instead of producing synthetic output.
    """
    reason = _chat_credentials_reason(config)
    if reason is not None:
        return {
            role: cast(HTTPStructuredLLM, UnavailableStructuredLLM(reason))
            for role in ("master", "slave", "adversary")
        }
    llm_config = config.llm
    assert llm_config is not None
    clients: dict[str, HTTPStructuredLLM] = {}
    try:
        for role in ("master", "slave", "adversary"):
            clients[role] = build_role_llm(
                llm_config,
                role,
                ledger=ledger,
                scenario_id=scenario_id,
            )
    except Exception:
        for client in clients.values():
            client.close()
        raise
    return clients


def _chat_credentials_reason(config: AppConfig) -> str | None:
    """Return an operator-facing reason when the real chat provider is unavailable."""
    llm_config = config.llm
    if llm_config is None:
        return "chat LLM configuration is unavailable"
    if llm_config.roles is None:
        role_reason = "role-specific chat configuration is unavailable (legacy flat LLM config)"
    else:
        role_reason = None
    if not (os.environ.get(llm_config.api_key_env) or llm_config.api_key):
        credential_reason = (
            f"chat credentials are unavailable: neither {llm_config.api_key_env} "
            "nor a configured chat api_key is available"
        )
    else:
        credential_reason = None
    if role_reason and credential_reason:
        return f"{role_reason}; {credential_reason}"
    return role_reason or credential_reason


def _llm_reconnect_policy(config: AppConfig) -> tuple[float, float]:
    """Resolve the bounded reconnect backoff from the configured roles."""
    llm_config = config.llm
    if llm_config is None or llm_config.roles is None:
        return (0.0, 0.0)
    roles = tuple(llm_config.roles.values())
    return (
        min(role.backoff_base_s for role in roles),
        max(role.backoff_max_s for role in roles),
    )


def _llm_max_reconnect_attempts(config: AppConfig) -> int:
    """Bound outer-cycle reconnects by the strictest role configuration."""
    llm_config = config.llm
    if llm_config is None or llm_config.roles is None:
        return 1
    return max(1, min(role.max_retries for role in llm_config.roles.values()) + 1)


def _step_with_llm_retries(
    engine: SimulationEngine,
    loop: _AgentLoop,
    config: AppConfig,
    *,
    stop: Event | None = None,
) -> bool:
    """Advance one physical step without repeating a failed LLM cycle."""
    del config, stop
    loop.apply_background_cycle()
    try:
        engine.step()
    except LLMError as exc:
        loop.mark_llm_paused(exc)
        return False
    loop.publish_latest()
    return True


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
        llm: HTTPStructuredLLM | Mapping[str, StructuredLLM[Any]] | None,
        run_id: str,
        steps: int,
        seed: int,
        background_carrier: bool = False,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._effective_demo_speed: float | None = None
        self.database_path = database_path
        self.scenario_id = config.scenario.scenario_id or _SCENARIO_ID
        self.run_id = run_id
        self.steps = steps
        self._seed = seed
        self._background_carrier = background_carrier
        self.plans = PlanRepository(database_path)
        self.events = EventRepository(database_path)
        self.ledger = DecisionLedger(database_path)
        self._knowledge_client = self._build_knowledge_client()
        clients: dict[str, StructuredLLM[Any]]
        if llm is None:
            clients = dict(
                _build_llm(
                    config,
                    ledger=self.ledger,
                    scenario_id=self.scenario_id,
                )
            )
        elif isinstance(llm, Mapping):
            clients = dict(llm)
        else:
            clients = {"master": llm}
        self._clients = clients
        master_llm = self._clients.get("master")
        if master_llm is None:
            raise ValueError("agent loop requires a master LLM client")
        self.llm = master_llm
        self._memory_short_term = ShortTermContextRepository(database_path)
        self._memory_long_term = LongTermMemoryRepository(database_path)
        self._memory_embedding_provider: EmbeddingProvider | None = None
        self._memory_worker: MemoryWorker | None = None
        self._memory_worker_short_term: ShortTermContextRepository | None = None
        self._memory_worker_long_term: LongTermMemoryRepository | None = None
        self._memory_worker_events: EventRepository | None = None
        self._memory_worker_ledger: DecisionLedger | None = None
        self._memory_worker_plans: PlanRepository | None = None
        self._memory_worker_embedding_provider: EmbeddingProvider | None = None
        self._memory_worker_llm: StructuredLLM[Any] | None = None
        self._memory_degraded_reason: str | None = None
        self._memory_service = self._build_memory_service()
        self._memory_port = MemoryServiceAdapter(
            self._memory_service, scenario_id=self.scenario_id
        )
        self._slave_graph: Any | None = None
        self._adversary_graph: Any | None = None
        if "slave" in self._clients:
            self._slave_graph = build_slave_graph(
                self._clients["slave"],
                model_id=self._role_model("slave"),
            )
        if "adversary" in self._clients:
            self._adversary_graph = build_adversary_graph(self._clients["adversary"])
        self.situation: SituationSnapshot | None = None
        self.carrier_error_count = 0
        self.paused = False
        self.reconnectable = True
        self.llm_pause_reason: str | None = None
        self._chat_degraded_reason = (
            _chat_credentials_reason(config) if llm is None else None
        )
        if self._chat_degraded_reason is not None:
            self.paused = True
            self.reconnectable = False
            self.llm_pause_reason = self._chat_degraded_reason
        self._llm_failure_count = 0
        self._next_llm_retry_at = 0.0
        self._runtime: CarrierRuntime | None = None
        self._engine: SimulationEngine | None = None
        self._clock = SimulationClock(step_s=config.timing.observation_step_s)
        self._initialization_submitted = False
        self._last_plan_id: str | None = None
        self._last_mission_revision = 0
        self._last_strategic_review_s = 0
        self._last_battery_rotation_s: dict[str, int] = {}
        self.hub = OperationalHub()
        self._publisher: OperationalFramePublisher | None = None
        self._carrier_cycle_lock = RLock()
        self._background_cycle: _BackgroundCarrierCycle | None = None
        self._background_thread: Thread | None = None
        self._background_mailbox: SituationSnapshot | None = None
        self._active_cycle_situation: SituationSnapshot | None = None
        self._closing = False
        self._closed = False
        self._close_condition = Condition(RLock())
        self._close_in_progress = False
        self._close_completed: set[int] = set()

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
            mission_snapshot_provider=engine.mission_snapshot,
            physics_step_s=self._config.timing.physics_step_s,
            mission_event_history_limit=(
                self._config.agent.retention.mission_event_history_limit
                if self._config.agent is not None
                else 2048
            ),
        )
        self._runtime.bind_simulation_time(lambda: engine._clock.sim_time_s)
        if self._chat_degraded_reason is not None:
            self._runtime._llm_paused = True
            self._runtime._llm_pause_reason = self._chat_degraded_reason
            self._runtime._llm_reconnectable = False
        if self._memory_worker is not None:
            self._memory_worker.start()

    def mark_llm_paused(self, error: LLMError) -> None:
        """Expose local-brain failures through the runtime API status."""
        if not self.paused:
            self._llm_failure_count = 0
        self._llm_failure_count += 1
        self.paused = True
        max_attempts = _llm_max_reconnect_attempts(self._config)
        if self._llm_failure_count > max_attempts:
            self._next_llm_retry_at = float("inf")
            self.reconnectable = False
            self.llm_pause_reason = (
                f"{error}; bounded LLM reconnect attempts exhausted "
                f"({max_attempts})"
            )
        else:
            base_s, max_s = _llm_reconnect_policy(self._config)
            delay_s = min(max_s, base_s * (2 ** (self._llm_failure_count - 1)))
            self._next_llm_retry_at = time.monotonic() + delay_s
            self.reconnectable = True
            self.llm_pause_reason = str(error)
        runtime = self._runtime
        if runtime is None:
            return
        lock = getattr(runtime, "_lock", None)
        if lock is None:
            return
        with lock:
            runtime._llm_paused = True
            runtime._llm_pause_reason = str(error)
            runtime._llm_reconnectable = self.reconnectable

    def mark_llm_recovered(self) -> None:
        """Clear the operator-visible pause after a successful cycle."""
        self.paused = False
        self.reconnectable = True
        self.llm_pause_reason = None
        self._llm_failure_count = 0
        self._next_llm_retry_at = 0.0
        runtime = self._runtime
        if runtime is None:
            return
        lock = getattr(runtime, "_lock", None)
        if lock is None:
            return
        with lock:
            runtime._llm_paused = False
            runtime._llm_pause_reason = None
            runtime._llm_reconnectable = False

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
        planning_config = PlanningConfig(
            bounds=(
                config.environment.map_bounds_xy
                if config.environment is not None
                else PlanningConfig().bounds
            ),
            quality_warning=config.tracking.quality_warning,
            quality_release=config.tracking.quality_release,
            release_hold_s=float(config.tracking.release_hold_s),
        )
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
            optimizer=planning_config,
            semantic_repairs=agent.semantic_repairs if agent else 2,
            model_id=self._role_model("master"),
            knowledge_client=self._knowledge_client,
            uuv_only=_is_uuv_only_config(config),
            retention=(agent.retention if agent is not None else RuntimeRetentionConfig()),
            current_snapshot_revision=self._current_snapshot_revision,
            memory_service=self._memory_service,
            short_term_repository=self._memory_short_term,
            memory_port=self._memory_port,
        )

    def _build_memory_service(self) -> MemoryService:
        """Build the real memory provider chain, or an explicit degraded port."""
        memory_config = self._config.memory
        if memory_config is None or not memory_config.enabled:
            reason = "memory configuration is disabled"
            self._memory_degraded_reason = reason
            return MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                DegradedMemoryRetriever(reason),
                degraded_reason=reason,
            )
        if (
            memory_config.embedding_provider == "http"
            and not os.environ.get(memory_config.embedding_api_key_env)
        ):
            reason = (
                "memory embedding credentials are unavailable: "
                f"{memory_config.embedding_api_key_env}"
            )
            self._memory_degraded_reason = reason
            return MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                DegradedMemoryRetriever(reason),
                degraded_reason=reason,
            )
        chat_reason = _chat_credentials_reason(self._config)
        if chat_reason is not None:
            reason = f"memory LLM credentials are unavailable: {chat_reason}"
            self._memory_degraded_reason = reason
            return MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                DegradedMemoryRetriever(reason),
                degraded_reason=reason,
            )
        try:
            provider = _build_memory_embedding_provider(
                memory_config,
                ledger=self.ledger,
                scenario_id=self.scenario_id,
            )
            self._memory_embedding_provider = provider
            retriever = MemoryRetriever(
                embedding_provider=provider,
                repository=self._memory_long_term,
                config=memory_config,
            )
            service = MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                retriever,
            )
            worker_short_term = ShortTermContextRepository(self.database_path)
            worker_long_term = LongTermMemoryRepository(self.database_path)
            worker_events = EventRepository(self.database_path)
            worker_ledger = DecisionLedger(self.database_path)
            worker_plans = PlanRepository(self.database_path)
            self._memory_worker_short_term = worker_short_term
            self._memory_worker_long_term = worker_long_term
            self._memory_worker_events = worker_events
            self._memory_worker_ledger = worker_ledger
            self._memory_worker_plans = worker_plans
            worker_provider = _build_memory_embedding_provider(
                memory_config,
                ledger=worker_ledger,
                scenario_id=self.scenario_id,
            )
            self._memory_worker_embedding_provider = worker_provider
            if self._config.llm is None:
                raise LLMConfigError("memory worker requires chat LLM configuration")
            worker_llm = build_role_llm(
                self._config.llm,
                "master",
                ledger=worker_ledger,
                scenario_id=self.scenario_id,
            )
            self._memory_worker_llm = worker_llm
            worker_service = MemoryService(
                worker_short_term,
                worker_long_term,
                MemoryRetriever(
                    embedding_provider=worker_provider,
                    repository=worker_long_term,
                    config=memory_config,
                ),
            )
            reasoner = MemoryReasoner(
                llm=worker_llm,
                repository=worker_long_term,
                config=memory_config,
            )
            source_reader = MemorySourceReader(
                worker_long_term,
                event_repository=worker_events,
                decision_ledger=worker_ledger,
                plan_repository=worker_plans,
                short_term_repository=worker_short_term,
            )
            self._memory_worker = MemoryWorker(
                worker_long_term,
                worker_service,
                cast(Any, reasoner),
                source_reader,
                memory_config,
                f"{self.run_id}:memory",
                embedding_provider=worker_provider,
            )
            return service
        except Exception as exc:  # noqa: BLE001 - expose unavailable wiring as degraded state
            self._memory_degraded_reason = f"memory provider unavailable: {type(exc).__name__}"
            active_provider = self._memory_embedding_provider
            self._memory_embedding_provider = None
            if active_provider is not None:
                close = getattr(active_provider, "close", None)
                if callable(close):
                    close()
            for worker_resource in (
                self._memory_worker_embedding_provider,
                self._memory_worker_llm,
            ):
                close = getattr(worker_resource, "close", None)
                if callable(close):
                    close()
            for owned_resource in (
                self._memory_worker_short_term,
                self._memory_worker_long_term,
                self._memory_worker_events,
                self._memory_worker_ledger,
                self._memory_worker_plans,
            ):
                close = getattr(owned_resource, "close", None)
                if callable(close):
                    close()
            self._memory_worker_short_term = None
            self._memory_worker_long_term = None
            self._memory_worker_events = None
            self._memory_worker_ledger = None
            self._memory_worker_plans = None
            self._memory_worker_embedding_provider = None
            self._memory_worker_llm = None
            return MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                DegradedMemoryRetriever(self._memory_degraded_reason),
                degraded_reason=self._memory_degraded_reason,
            )

    def _current_snapshot_revision(self) -> int:
        situation = self.situation
        return situation.snapshot_revision if situation is not None else 0

    def _build_knowledge_client(self) -> OntologyKnowledgeClient | None:
        knowledge = self._config.knowledge
        if knowledge is None or not knowledge.enabled:
            return None
        return OntologyKnowledgeClient(
            base_url=knowledge.base_url,
            query_path=knowledge.query_path,
            mode=knowledge.mode,
            include_trace=knowledge.include_trace,
            request_timeout_s=knowledge.request_timeout_s,
            max_retries=knowledge.max_retries,
            backoff_base_s=knowledge.backoff_base_s,
            backoff_max_s=knowledge.backoff_max_s,
            ledger=self.ledger,
        )

    def _role_model(self, role: str) -> str:
        llm_config = self._config.llm
        if llm_config is None:
            return "http"
        if llm_config.roles is not None:
            return llm_config.for_role(role).model
        return llm_config.model

    def _live_situation(self, ref: str) -> SituationSnapshot:
        situation = self._active_cycle_situation or self.situation
        if situation is None:
            raise RuntimeError(f"no live situation recorded for {ref!r}")
        return situation

    @staticmethod
    def _merge_pending_events(
        latest: SituationSnapshot,
        earlier: SituationSnapshot | None,
    ) -> SituationSnapshot:
        """Keep the newest physical state while carrying forward unseen events."""
        if earlier is None or not earlier.pending_events:
            return latest
        events = {
            event.event_id: event
            for event in (*earlier.pending_events, *latest.pending_events)
        }
        return latest.model_copy(
            update={
                "pending_events": tuple(
                    sorted(
                        events.values(),
                        key=lambda event: (event.sim_time_s, event.event_id),
                    )
                )
            }
        )

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

    def _local_brain_decisions(
        self, situation: SituationSnapshot
    ) -> tuple[tuple[SlaveSonarDecision, ...], tuple[AdversaryEscapeDecision, ...]]:
        """Run independent local brains before mutating the engine.

        The engine gives each graph a typed, truth-safe packet. A failure in
        one local role is isolated so a healthy role can still contribute to
        the same master cycle; target-side belief state remains visible while
        an unavailable adversary provider is being recovered.
        """
        engine = self._engine
        assert engine is not None
        build_adversary_inputs = getattr(engine, "build_adversary_inputs", None)
        build_slave_contexts = getattr(engine, "build_slave_contexts", None)
        if not callable(build_adversary_inputs) or not callable(build_slave_contexts):
            # Keep lightweight engine doubles usable at this boundary.  The
            # production engine always exposes both typed context builders.
            return (), ()
        return self._local_brain_decisions_from_contexts(
            situation,
            tuple(build_adversary_inputs(situation)),
            tuple(build_slave_contexts(situation)),
        )

    def _local_brain_decisions_from_contexts(
        self,
        situation: SituationSnapshot,
        adversary_contexts: tuple[AdversaryEscapeInput, ...],
        slave_contexts: tuple[SlaveSonarContext, ...],
    ) -> tuple[tuple[SlaveSonarDecision, ...], tuple[AdversaryEscapeDecision, ...]]:
        """Invoke local brains over contexts captured at one physics boundary."""
        self._set_llm_sim_time(situation.sim_time_s)
        adversary_decisions: list[AdversaryEscapeDecision] = []
        slave_decisions: list[SlaveSonarDecision] = []
        # Keep the target-side decision path independent from a single
        # group-slave provider outage. The master runtime still owns the
        # transactional cycle boundary, while successful local decisions can
        # reach the engine in the same observation cycle.
        adversary_graph = getattr(self, "_adversary_graph", None)
        if adversary_graph is not None:
            for adversary_context in adversary_contexts:
                try:
                    result = adversary_graph.invoke({"context": adversary_context})
                    adversary_decision = result.get("decision")
                    if not isinstance(adversary_decision, AdversaryEscapeDecision):
                        raise TypeError("adversary graph returned no typed decision")
                    adversary_decisions.append(adversary_decision)
                except LLMError:
                    # Local brain failures are isolated; the public summary
                    # continues to expose target-owned belief motion.
                    continue
                except Exception as exc:  # LLM semantic output is a content failure
                    del exc
                    continue
        slave_graph = getattr(self, "_slave_graph", None)
        if slave_graph is not None:
            for context in slave_contexts:
                try:
                    result = slave_graph.invoke({"context": context})
                    slave_decision = result.get("decision")
                    if not isinstance(slave_decision, SlaveSonarDecision):
                        raise TypeError("slave graph returned no typed decision")
                    slave_decisions.append(slave_decision)
                except LLMError:
                    continue
                except Exception as exc:  # LLM semantic output is a content failure
                    del exc
                    continue
        return tuple(slave_decisions), tuple(adversary_decisions)

    def _set_llm_sim_time(self, sim_time_s: int) -> None:
        """Advance observability metadata without changing decision inputs."""
        for client in getattr(self, "_clients", {}).values():
            setter = getattr(client, "set_simulation_time", None)
            if callable(setter):
                setter(sim_time_s)

    def publish_latest(self) -> None:
        """Publish the completed physical step, including paused state."""
        engine = self._engine
        publisher = self._publisher
        if engine is None or publisher is None:
            return
        try:
            publisher.publish(engine.publication_situation())
        except Exception:  # noqa: BLE001 - telemetry cannot stop tracking
            self.carrier_error_count += 1

    def _waiting_for_llm_reconnect(self) -> bool:
        if not bool(getattr(self, "paused", False)):
            return False
        if not bool(getattr(self, "reconnectable", True)):
            return True
        return time.monotonic() < getattr(self, "_next_llm_retry_at", 0.0)

    def on_situation(self, situation: SituationSnapshot) -> None:
        """Queue or run one carrier cycle at an observation boundary."""
        if getattr(self, "_background_carrier", False):
            self._start_background_cycle(situation)
            return
        self._run_synchronous_carrier_cycle(situation)

    def _run_synchronous_carrier_cycle(self, situation: SituationSnapshot) -> None:
        """Run a carrier cycle inline for deterministic finite/test runs."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        if self._waiting_for_llm_reconnect():
            return
        active_plan_reader = getattr(runtime, "active_plan", None)
        active_plan = active_plan_reader() if callable(active_plan_reader) else None
        if _is_uuv_only_config(self._config):
            self._apply_uuv_only_mission_plan()
        elif active_plan is not None:
            engine.apply_tracking_plan(active_plan)
        sensor_controls: tuple[Any, ...] = ()
        drain_sensor_controls = getattr(runtime, "drain_sensor_controls", None)
        try:
            local_slave_decisions, adversary_decisions = self._local_brain_decisions(
                situation
            )
            sensor_controls = (
                drain_sensor_controls() if callable(drain_sensor_controls) else ()
            )
            for control in sensor_controls:
                engine.set_sensor_mode(
                    control.uuv_id,
                    control.mode,
                    ping_contact_id=control.target_id,
                )
            commit_inputs = getattr(runtime, "commit_operational_inputs", None)
            if callable(commit_inputs):
                try:
                    commit_inputs(
                        current_sim_time_s=situation.sim_time_s,
                        apply_scheme=engine.set_operational_scheme,
                        apply_intelligence=engine.submit_intelligence,
                    )
                except Exception:  # noqa: BLE001 - bad boundary input cannot stop the loop
                    self.carrier_error_count += 1
            engine.set_reservations(runtime.reservations())
            runtime.submit_events((*situation.pending_events, *self._feedback_events(situation)))
            if not self._initialization_submitted and self._initialization_ready(situation):
                self._initialization_submitted = True
                runtime.submit_event(
                    event_type="initialization",
                    entity_id=situation.scenario_id,
                    sim_time_s=situation.sim_time_s,
                )
            self._set_llm_sim_time(situation.sim_time_s)
            result = runtime.tick()
            if result.get("commit_status") == "committed":
                self._apply_new_commands()
            self._apply_verification_commands(result)
            for slave_decision in local_slave_decisions:
                engine.apply_slave_sonar_decision(slave_decision)
            for adversary_decision in adversary_decisions:
                engine.apply_adversary_decision(adversary_decision)
            self.mark_llm_recovered()
        except LLMError as exc:
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(sensor_controls)
            self.mark_llm_paused(exc)
            return
        except Exception:  # noqa: BLE001 - execution errors must roll back the cycle
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(sensor_controls)
            self.carrier_error_count += 1
            raise

    def _start_background_cycle(self, situation: SituationSnapshot) -> None:
        """Start an LLM cycle without holding up the physical simulation."""
        with self._carrier_cycle_lock:
            if getattr(self, "_closing", False):
                return
            current = self.situation
            if current is None or situation.snapshot_revision >= current.snapshot_revision:
                self.situation = situation
            if self._background_cycle is not None:
                latest = self.situation
                active_situation = getattr(self._background_cycle, "situation", None)
                if latest is not None and (
                    active_situation is None
                    or latest.snapshot_revision > active_situation.snapshot_revision
                ) and (
                    self._background_mailbox is None
                    or latest.snapshot_revision
                    > self._background_mailbox.snapshot_revision
                ):
                    self._background_mailbox = self._merge_pending_events(
                        latest, self._background_mailbox
                    )
                return
            if self._waiting_for_llm_reconnect():
                latest = self.situation or situation
                self._background_mailbox = self._merge_pending_events(
                    latest, self._background_mailbox
                )
                return
            engine = self._engine
            if engine is None:
                return
            cycle_situation = self.situation or situation
            cycle_situation = self._merge_pending_events(
                cycle_situation, self._background_mailbox
            )
            self._background_mailbox = None
            cycle = _BackgroundCarrierCycle(
                situation=cycle_situation,
                adversary_contexts=tuple(engine.build_adversary_inputs(cycle_situation)),
                slave_contexts=tuple(engine.build_slave_contexts(cycle_situation)),
            )
            self._background_cycle = cycle
            thread = Thread(
                target=self._run_background_cycle,
                args=(cycle,),
                name="underwater-carrier-llm",
                daemon=True,
            )
            self._background_thread = thread
            thread.start()

    def _run_background_cycle(self, cycle: _BackgroundCarrierCycle) -> None:
        """Run provider and graph work; engine writes wait for the next step."""
        runtime = self._runtime
        assert runtime is not None
        drain_sensor_controls = getattr(runtime, "drain_sensor_controls", None)
        self._active_cycle_situation = cycle.situation
        try:
            cycle.slave_decisions, cycle.adversary_decisions = (
                self._local_brain_decisions_from_contexts(
                    cycle.situation,
                    cycle.adversary_contexts,
                    cycle.slave_contexts,
                )
            )
            cycle.sensor_controls = (
                drain_sensor_controls() if callable(drain_sensor_controls) else ()
            )
            runtime.submit_events(
                (*cycle.situation.pending_events, *self._feedback_events(cycle.situation))
            )
            if not self._initialization_submitted and self._initialization_ready(
                cycle.situation
            ):
                self._initialization_submitted = True
                runtime.submit_event(
                    event_type="initialization",
                    entity_id=cycle.situation.scenario_id,
                    sim_time_s=cycle.situation.sim_time_s,
                )
            self._set_llm_sim_time(cycle.situation.sim_time_s)
            cycle.result = runtime.tick()
        except LLMError as exc:
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(cycle.sensor_controls)
            cycle.sensor_controls = ()
            cycle.error = exc
        except BaseException as exc:  # noqa: BLE001 - surface on the physics thread
            cycle.error = exc
        finally:
            with self._carrier_cycle_lock:
                self._active_cycle_situation = None
                cycle.done = True

    def _schedule_latest_background_cycle(
        self, completed: _BackgroundCarrierCycle
    ) -> None:
        with self._carrier_cycle_lock:
            next_situation = self._background_mailbox
            self._background_mailbox = None
            latest = self.situation
            if (
                next_situation is None
                and latest is not None
                and latest.snapshot_revision > completed.situation.snapshot_revision
            ):
                next_situation = latest
        if next_situation is not None:
            self._start_background_cycle(next_situation)

    def apply_background_cycle(self) -> None:
        """Apply one completed carrier result at a safe physics boundary."""
        if not self._background_carrier:
            return
        with self._carrier_cycle_lock:
            cycle = self._background_cycle
            if cycle is None or not cycle.done:
                return
            self._background_cycle = None
            self._background_thread = None
            latest = self.situation
        if (
            latest is not None
            and latest.snapshot_revision > cycle.situation.snapshot_revision
        ):
            self._schedule_latest_background_cycle(cycle)
            return
        if cycle.error is not None:
            if isinstance(cycle.error, LLMError):
                self.mark_llm_paused(cycle.error)
            else:
                self.carrier_error_count += 1
            self._schedule_latest_background_cycle(cycle)
            return
        runtime = self._runtime
        engine = self._engine
        if runtime is None or engine is None or cycle.result is None:
            self.carrier_error_count += 1
            self._schedule_latest_background_cycle(cycle)
            return
        active_plan_reader = getattr(runtime, "active_plan", None)
        active_plan = active_plan_reader() if callable(active_plan_reader) else None
        if _is_uuv_only_config(self._config):
            self._apply_uuv_only_mission_plan()
        elif active_plan is not None:
            engine.apply_tracking_plan(active_plan)
        for control in cycle.sensor_controls:
            engine.set_sensor_mode(
                control.uuv_id,
                control.mode,
                ping_contact_id=control.target_id,
            )
        commit_inputs = getattr(runtime, "commit_operational_inputs", None)
        if callable(commit_inputs):
            try:
                commit_inputs(
                    current_sim_time_s=engine._clock.sim_time_s,
                    apply_scheme=engine.set_operational_scheme,
                    apply_intelligence=engine.submit_intelligence,
                )
            except Exception:  # noqa: BLE001 - bad input cannot stop tracking
                self.carrier_error_count += 1
        engine.set_reservations(runtime.reservations())
        if cycle.result.get("commit_status") == "committed":
            self._apply_new_commands()
        self._apply_verification_commands(cycle.result)
        for slave_decision in cycle.slave_decisions:
            engine.apply_slave_sonar_decision(slave_decision)
        for adversary_decision in cycle.adversary_decisions:
            engine.apply_adversary_decision(adversary_decision)
        self.mark_llm_recovered()
        self._schedule_latest_background_cycle(cycle)

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
        if _is_uuv_only_config(self._config):
            self._apply_uuv_only_mission_plan()
            return
        active = self.plans.get_active(self.scenario_id)
        if active is None:
            return
        engine.apply_tracking_plan(active)
        current_uuvs = {uuv.uuv_id: uuv for uuv in (self.situation.uuvs if self.situation else ())}
        for uuv_id in active.returning_uuv_ids:
            uuv = current_uuvs.get(uuv_id)
            if uuv is not None and uuv.deployment_state is DeploymentState.DEPLOYED:
                engine.request_uuv_recovery(uuv_id, reason=f"plan:{active.plan_id}:return")
        if active.plan_id == self._last_plan_id:
            return
        self._last_plan_id = active.plan_id
        for command in self.plans.list_commands(active.plan_id):
            engine.apply_plan_command(command)

    def _apply_uuv_only_mission_plan(self) -> bool:
        """Apply only the latest verified executable plan in UUV-only mode."""
        engine = self._engine
        runtime = self._runtime
        if engine is None or runtime is None:
            return False
        reader = getattr(runtime, "active_mission_plan", None)
        plan = reader() if callable(reader) else None
        if plan is None or plan.revision <= getattr(self, "_last_mission_revision", 0):
            return False
        applied = engine.apply_verified_mission_plan(plan)
        if applied:
            self._last_mission_revision = plan.revision
        return applied

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
            "target_count": len(getattr(self._engine, "_targets", {})),
            "sim_time_s": (
                self._engine._clock.sim_time_s if self._engine is not None else 0
            ),
            "effective_demo_speed": getattr(self, "_effective_demo_speed", None),
            "status": "completed",
            "llm": self._config.llm.model if self._config.llm else "http",
            "llm_roles": sorted(self._clients),
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

    def abort(self) -> None:
        """Signal daemon workers without waiting for an in-flight LLM call."""
        if self._carrier_cycle_lock.acquire(blocking=False):
            try:
                self._closing = True
                self._background_mailbox = None
            finally:
                self._carrier_cycle_lock.release()
        else:
            self._closing = True
            self._background_mailbox = None
        if self._memory_worker is not None:
            self._memory_worker.stop(timeout=0.0)

    def close(self) -> bool:
        condition = getattr(self, "_close_condition", None)
        if condition is None:
            condition = Condition(RLock())
            self._close_condition = condition
        while True:
            with condition:
                if getattr(self, "_closed", False):
                    return True
                if getattr(self, "_close_in_progress", False):
                    condition.wait()
                    continue
                self._close_in_progress = True
            try:
                result = self._close_once()
            except BaseException:
                with condition:
                    self._close_in_progress = False
                    condition.notify_all()
                raise
            with condition:
                self._close_in_progress = False
                if result:
                    self._closed = True
                condition.notify_all()
            return result

    def _close_once(self) -> bool:
        with self._carrier_cycle_lock:
            self._closing = True
            self._background_mailbox = None
        if self._memory_worker is not None:
            if not self._memory_worker.stop(timeout=5.0):
                return False
        background_thread = self._background_thread
        if background_thread is not None and background_thread.is_alive():
            background_thread.join(timeout=30.0)
        if background_thread is not None and background_thread.is_alive():
            return False

        completed: set[int] = getattr(self, "_close_completed", set())
        self._close_completed = completed
        errors: list[BaseException] = []

        def close_resource(resource: object | None) -> None:
            if resource is None:
                return
            identity = id(resource)
            if identity in completed:
                return
            close = getattr(resource, "close", None)
            if not callable(close):
                completed.add(identity)
                return
            try:
                close()
            except BaseException as error:
                errors.append(error)
            else:
                completed.add(identity)

        close_resource(self._memory_embedding_provider)
        close_resource(getattr(self, "_memory_worker_embedding_provider", None))
        close_resource(getattr(self, "_memory_worker_llm", None))
        for client in self._clients.values():
            close_resource(client)
        close_resource(self._runtime)
        close_resource(self._publisher)
        close_resource(getattr(self, "_memory_worker_short_term", None))
        close_resource(getattr(self, "_memory_worker_long_term", None))
        close_resource(getattr(self, "_memory_worker_events", None))
        close_resource(getattr(self, "_memory_worker_ledger", None))
        close_resource(getattr(self, "_memory_worker_plans", None))
        close_resource(self._memory_short_term)
        close_resource(self._memory_long_term)
        close_resource(self._knowledge_client)
        close_resource(self.plans)
        close_resource(self.events)
        close_resource(self.ledger)
        if errors:
            raise errors[0]
        return True


if __name__ == "__main__":
    sys.exit(main())
