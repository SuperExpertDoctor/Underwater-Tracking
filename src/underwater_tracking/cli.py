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
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import json
import os
import signal
import sys
import time
from threading import Condition, Event, RLock, Thread
import uuid
from pathlib import Path
from typing import Any, Literal, cast

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
from underwater_tracking.api.live import (
    FramePersistencePolicy,
    OperationalFramePublisher,
    compact_operational_frame,
)
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import (
    AppConfig,
    IntentChangeConfirmation,
    MemoryConfig,
    RuntimeRetentionConfig,
    TrajectoryDiffConfig,
)
from underwater_tracking.domain.agent_models import VerificationCommand
from underwater_tracking.domain.adversary_models import (
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
    AdversaryIntentDecision,
)
from underwater_tracking.domain.event_registry import EVENT_REGISTRY
from underwater_tracking.domain.models import (
    DeploymentState,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.mission_models import UUVResourceState
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch
from underwater_tracking.domain.slave_models import SlaveSonarContext, SlaveSonarDecision
from underwater_tracking.domain.ui_models import PlanningHealthView
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
from underwater_tracking.memory.situation_summary import (
    PeriodicSituationSummary,
    PeriodicSituationSummaryWriter,
    build_periodic_situation_summary,
)
from underwater_tracking.memory.worker import MemoryWorker
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.memory import LongTermMemoryRepository, ShortTermContextRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository
from underwater_tracking.persistence.sqlite import now_ms
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.runtime.run_controller import RunController
from underwater_tracking.runtime.models import ShutdownReport
from underwater_tracking.runtime.run_catalog import RunCatalog
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.runtime.mission_epoch_commit import MissionEpochCommitPort
from underwater_tracking.runtime.planning_epoch import EpochTrigger, PlanningEpochCoordinator
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_BATTERY_ROTATION_THRESHOLD = 0.3
_DEFAULT_API_PORT = 8000
_API_PORT_ENV = "UNDERWATER_TRACKING_API_PORT"


class _LedgerBoundStructuredLLM:
    """Scope an explicitly injected provider to the memory worker ledger."""

    def __init__(self, delegate: StructuredLLM[Any], ledger: DecisionLedger) -> None:
        self._delegate = delegate
        self._ledger = ledger

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        return self._delegate.invoke_structured(
            operation,
            payload,
            response_model,
            prompt_version=prompt_version,
        )

    def cancel(self) -> None:
        cancel = getattr(self._delegate, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self) -> None:
        # The owning agent loop closes the injected provider after the worker.
        return None


def _epoch_event_priority(event: RuntimeEvent) -> int:
    """Map public event severity to deterministic epoch mailbox priority."""
    if event.event_type == "initialization":
        return 100
    return {
        EventLevel.CRITICAL: 4,
        EventLevel.STRATEGIC: 3,
        EventLevel.TACTICAL: 2,
        EventLevel.INFORMATIONAL: 1,
    }[event.level]


_EPOCH_ALWAYS_IMPACT_TYPES = frozenset(
    {
        "initialization",
        "expert_confirmation",
        "expert_confirmed",
    }
    | {
        event_type
        for event_type, definition in EVENT_REGISTRY.items()
        if definition.plan_impact_policy == "always" and event_type != "target_added"
    }
)

_PREDICTION_REFRESH_EVENT_TYPES = frozenset(
    {
        "target_estimate_updated",
        "target_maneuver_observed",
        "target_speed_regime_changed",
        "target_depth_regime_changed",
        "imm_confidence_shifted",
    }
)


def _event_requests_planning_epoch(event: RuntimeEvent) -> bool:
    """Reserve an epoch for registered triggers or explicit plan impact."""
    if event.event_type in _EPOCH_ALWAYS_IMPACT_TYPES:
        return True
    if event.level not in {
        EventLevel.TACTICAL,
        EventLevel.STRATEGIC,
        EventLevel.CRITICAL,
    }:
        return False
    return event.payload.get("plan_impact") is True


def _committed_epoch_plan(result: Mapping[str, Any]) -> Any:
    """Return only the executable payload of a committed epoch result."""
    epoch_result = result.get("epoch_commit_result")
    if isinstance(epoch_result, EpochCommitResult) and epoch_result.status == "committed":
        return epoch_result.executable_plan
    return None


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
    epoch: PlanningEpoch | None = None
    trigger_events: tuple[RuntimeEvent, ...] = ()
    sensor_controls: tuple[SensorModeControl, ...] = ()
    slave_decisions: tuple[SlaveSonarDecision, ...] = ()
    adversary_decisions: tuple[AdversaryIntentDecision | AdversaryEscapeDecision, ...] = ()
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    local_error: BaseException | None = None
    planning_done: bool = False
    planning_applied: bool = False
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
    if config.environment is None:
        raise ValueError("uuv-only mission controller requires an environment roster")
    owner_by_id = {
        uuv.platform_id: uuv.home_carrier_id
        for uuv in config.environment.uuvs
        if uuv.home_carrier_id is not None
    }
    initial_resources = {
        uuv.platform_id: UUVResourceState(
            uuv_id=uuv.platform_id,
            carrier_id=owner_by_id[uuv.platform_id],
            mileage_m=0.0,
            energy_fraction=uuv.energy_fraction,
            healthy=True,
            capability_active=True,
            deployment_state=DeploymentState.ONBOARD.value,
            resource_episode=0,
        )
        for uuv in config.environment.uuvs
    }
    return MissionController(
        scenario_id=config.scenario.scenario_id,
        initial_uuv_resources=initial_resources,
        uuv_owner_by_id=owner_by_id,
        region_entry_probability_threshold=config.scenario.region_entry_probability_threshold,
        region_transition_confirm_cycles=config.scenario.region_transition_confirm_cycles,
        group_min_size=config.tracking.group_min_size,
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
    serve.add_argument(
        "--continuous",
        action="store_true",
        help="continue past scenario duration instead of completing at the duration boundary",
    )
    serve.add_argument(
        "--verification-audit",
        action="store_true",
        help="enable the redacted in-process physics verification endpoint",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument(
        "--web-ui-url",
        default=None,
        help="URL to open when the API root is requested",
    )
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
        transition_coordinator=loop._transition_coordinator,
        event_repository=loop.events,
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
                loop._run_phase = "awaiting_retry"
                loop._manifest_status = "awaiting_retry"
                loop.write_manifest(run_dir)
                loop.close()
                return 1
    except Exception as exc:  # noqa: BLE001 - surface as a CLI failure
        print(f"agent-run failed: {exc}", file=sys.stderr)
        loop._run_phase = "failed"
        loop._manifest_status = "failed"
        loop.close()
        return 1
    loop._run_phase = "completed"
    loop._manifest_status = "completed"
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
    try:
        controller_kwargs: dict[str, object] = {
            "steps": args.steps,
            "speed": args.speed,
        }
        try:
            controller_parameters = inspect.signature(RunController).parameters
        except (TypeError, ValueError):
            controller_parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in controller_parameters.values()
        )
        if accepts_kwargs or "continuous" in controller_parameters:
            controller_kwargs["continuous"] = bool(getattr(args, "continuous", False))
        if accepts_kwargs or "verification_audit" in controller_parameters:
            controller_kwargs["verification_audit"] = bool(
                getattr(args, "verification_audit", False)
            )
        controller = RunController(config, **controller_kwargs)
        controller.start_run(config.scenario.initial_target_count, seed=args.seed)
        app = create_app(
            controller=controller,
            catalog=RunCatalog(Path("outputs")),
            directive_job_limit=(
                config.agent.retention.directive_job_limit
                if config.agent is not None
                else 256
            ),
            web_ui_url=getattr(args, "web_ui_url", None),
            verification_audit=bool(getattr(args, "verification_audit", False)),
        )
        assert controller is not None
        _run_api_server(
            app,
            host=args.host,
            port=args.port,
            on_interrupt=controller.abort,
        )
    except KeyboardInterrupt:
        if controller is not None:
            controller.abort()
        raise
    finally:
        if controller is not None:
            close = controller.close
            try:
                parameters: Mapping[str, inspect.Parameter] = inspect.signature(close).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "timeout_s" in parameters:
                closed = close(timeout_s=10.0)
            else:
                # Keep injected controller fakes and older integrations
                # source-compatible while production uses bounded close.
                result = close()
                closed = True if result is None else bool(result)
            if not closed:
                print(
                    "serve shutdown timed out; owned resources remain active",
                    file=sys.stderr,
                )
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
            timeout_graceful_shutdown=1,
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
        self._epoch_repository = PlanningEpochRepository(self.plans.connection)
        self._epoch_coordinator = PlanningEpochCoordinator(
            self.scenario_id,
            repository=self._epoch_repository,
        )
        self._transition_coordinator = ScenarioTransitionCoordinator(self.scenario_id)
        self._epoch_commit_port: MissionEpochCommitPort | None = None
        self._active_epoch: PlanningEpoch | None = None
        self._epoch_seen_event_ids: set[str] = set()
        self.events = EventRepository(database_path)
        self._periodic_summary_writer = PeriodicSituationSummaryWriter(database_path)
        self.ledger = DecisionLedger(database_path)
        self._knowledge_client = self._build_knowledge_client()
        self._llm_injected = llm is not None
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
        self.carrier_error_details: list[str] = []
        self.planning_epoch_invariant_failures = 0
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
        self._periodic_summary_source_ids: set[str] = set()
        self._periodic_summary_source_events: dict[str, RuntimeEvent] = {}
        self._last_built_periodic_summary: PeriodicSituationSummary | None = None
        self._pending_periodic_summaries: deque[
            tuple[PeriodicSituationSummary, RuntimeEvent]
        ] = deque()
        self._periodic_summary_next_boundary_s = config.timing.progress_report_s
        self._periodic_summary_backlog_overflow = 0
        self._periodic_summary_degradation_events: list[RuntimeEvent] = []
        self.hub = OperationalHub()
        self._publisher: OperationalFramePublisher | None = None
        self._carrier_cycle_lock = RLock()
        self._background_cycle: _BackgroundCarrierCycle | None = None
        self._background_thread: Thread | None = None
        self._background_mailbox: SituationSnapshot | None = None
        self._background_local_thread: Thread | None = None
        self._background_local_mailbox: _BackgroundCarrierCycle | None = None
        self._background_local_results: deque[_BackgroundCarrierCycle] = deque()
        self._active_cycle_situation: SituationSnapshot | None = None
        self._bootstrap_epoch_id: str | None = None
        self._bootstrap_result: EpochCommitResult | None = None
        self._bootstrap_started_monotonic: float | None = None
        self._bootstrap_timeout_requested = False
        self._run_phase = "running"
        self._manifest_status = "running"
        self._closing = False
        self._closed = False
        self._close_condition = Condition(RLock())
        self._close_in_progress = False
        self._close_completed: set[int] = set()
        self._shutdown_report = ShutdownReport(completed=False)

    def attach(self, engine: SimulationEngine) -> None:
        """Create the carrier runtime over the same SQLite database."""
        self._engine = engine
        mission_controller = getattr(engine, "_mission_controller", None)
        if isinstance(mission_controller, MissionController):
            self._epoch_commit_port = MissionEpochCommitPort(
                plans=self.plans,
                epochs=self._epoch_repository,
                mission_controller=mission_controller,
                transition_coordinator=self._transition_coordinator,
                situation_provider=self._current_commit_situation,
            )
            self._restore_latest_committed_epoch(mission_controller)
        self._runtime = CarrierRuntime(
            self._deps(), scenario_id=self.scenario_id, database_path=self.database_path
        )
        self._publisher = OperationalFramePublisher(
            runtime=self._runtime,
            ledger=self.ledger,
            events=self.events,
            hub=self.hub,
            logger=OperationalFrameLogger(
                self.database_path.parent / "operational_frames.jsonl",
                max_run_bytes=self._config.frame_log.max_run_bytes,
            ),
            mission_snapshot_provider=engine.mission_snapshot,
            physics_step_s=self._config.timing.physics_step_s,
            history_limit=64,
            mission_event_history_limit=(
                self._config.agent.retention.mission_event_history_limit
                if self._config.agent is not None
                else 2048
            ),
            configured_roles=tuple(
                cast(Literal["master", "slave", "adversary"], role)
                for role in ("master", "slave", "adversary")
                if role in self._clients
            ),
            planning_health_provider=self.planning_health,
            run_phase_provider=lambda: str(getattr(self, "_run_phase", "running")),
            persistence_policy=(
                FramePersistencePolicy(self._config.frame_log.sample_interval_s)
                if self._background_carrier and self.steps == 0
                else FramePersistencePolicy(None)
            ),
            persistence_projection=compact_operational_frame,
        )
        self._runtime.bind_simulation_time(lambda: engine._clock.sim_time_s)
        if self._chat_degraded_reason is not None:
            self._runtime._llm_paused = True
            self._runtime._llm_pause_reason = self._chat_degraded_reason
            self._runtime._llm_reconnectable = False
        # The first frame is the bootstrap contract: publish configured
        # inventory and brain readiness before any worker can mutate state.
        self._publisher.publish(engine.publication_situation())
        engine.prime_adversary_mission_triggers()
        self._periodic_summary_writer.start()
        if self._memory_worker is not None:
            self._memory_worker.start()

    def begin_bootstrap_planning(self, situation: SituationSnapshot) -> None:
        """Start the initial planning epoch while physics remains at time zero."""
        if not self._background_carrier:
            raise RuntimeError("bootstrap planning requires the background carrier")
        self._bootstrap_epoch_id = None
        self._bootstrap_result = None
        self._bootstrap_started_monotonic = time.monotonic()
        self._bootstrap_timeout_requested = False
        self.situation = situation
        self._epoch_coordinator.observe(situation)
        self._start_background_cycle(situation, allow_paused=True)
        cycle = self._background_cycle
        if cycle is not None and cycle.epoch is not None:
            self._bootstrap_epoch_id = cycle.epoch.epoch_id

    def bootstrap_result(self) -> EpochCommitResult | None:
        """Apply completed bootstrap work and return its authoritative result."""
        if self._bootstrap_result is None:
            started = self._bootstrap_started_monotonic
            timeout_s = self._config.planning.initial_plan_timeout_s
            if (
                started is not None
                and not self._bootstrap_timeout_requested
                and time.monotonic() - started >= timeout_s
            ):
                self._bootstrap_timeout_requested = True
                for client in self._clients.values():
                    cancel = getattr(client, "cancel", None)
                    if callable(cancel):
                        cancel()
        self.apply_background_cycle()
        return self._bootstrap_result

    def retry_initial_planning(self, *, expected_epoch_id: str | None) -> str:
        """Explicitly release a failed bootstrap trigger for a new epoch."""
        result = self._bootstrap_result
        if result is None or result.status == "committed":
            raise ValueError("no failed bootstrap epoch is awaiting retry")
        if expected_epoch_id is not None and result.epoch_id != expected_epoch_id:
            raise ValueError(
                f"stale bootstrap epoch id {expected_epoch_id!r}; current is {result.epoch_id!r}"
            )
        capture = self._epoch_repository.get_capture(result.epoch_id)
        event_ids = capture.epoch.critical_event_ids
        if not event_ids:
            raise ValueError("failed bootstrap epoch has no retryable trigger")
        for event_id in event_ids:
            self._epoch_coordinator.force_retry_event(event_id)
            self._epoch_seen_event_ids.discard(event_id)
        self._bootstrap_result = None
        self._bootstrap_epoch_id = None
        self._bootstrap_started_monotonic = time.monotonic()
        self._bootstrap_timeout_requested = False
        latest = self.situation
        if latest is None:
            raise RuntimeError("cannot retry bootstrap planning without a situation")
        self._start_background_cycle(latest, allow_paused=True)
        cycle = self._background_cycle
        if cycle is None or cycle.epoch is None:
            raise RuntimeError("bootstrap retry did not reserve a new planning epoch")
        self._bootstrap_epoch_id = cycle.epoch.epoch_id
        return cycle.epoch.epoch_id

    def _current_commit_situation(self) -> SituationSnapshot:
        """Return the newest public situation available to the commit port."""
        situation = self.situation
        if situation is None:
            raise RuntimeError("cannot revalidate an epoch before the first situation")
        return situation

    def _restore_latest_committed_epoch(
        self, mission_controller: MissionController
    ) -> None:
        """Reconcile the durable epoch result before accepting new work."""
        latest = self._epoch_repository.latest(self.scenario_id)
        if latest is None:
            return
        _, result = latest
        if result is None or result.status != "committed":
            return
        plan = result.executable_plan
        plan_version = result.plan_version
        if plan is None or plan_version is None or plan_version != plan.revision:
            raise RuntimeError("persisted committed epoch has an invalid plan revision")
        active = self.plans.get_active(self.scenario_id)
        if active is None or active.revision != plan_version:
            raise RuntimeError("persisted epoch and active audit plan disagree")
        current_revision = mission_controller.snapshot().plan_revision
        if current_revision > plan_version:
            raise RuntimeError("mission controller is newer than the latest committed epoch")
        if current_revision == 0:
            if not mission_controller.apply_verified_plan(plan):
                raise RuntimeError("persisted executable plan could not restore mission state")
        elif current_revision != plan_version:
            raise RuntimeError("mission controller revision does not match committed epoch")
        self._last_mission_revision = plan_version
        self._initialization_submitted = True

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

    def planning_health(self) -> PlanningHealthView:
        """Return coordinator health without entering the engine mutation path."""
        coordinator = getattr(self, "_epoch_coordinator", None)
        if coordinator is None:
            return PlanningHealthView(status="idle")
        health = coordinator.health()
        latest = coordinator.latest_situation()
        current_revision = latest.snapshot_revision if latest is not None else None
        base_revision = health.base_physics_revision
        epoch_id = health.epoch_id
        if epoch_id is not None and base_revision is None:
            repository = getattr(self, "_epoch_repository", None)
            if repository is not None:
                try:
                    base_revision = repository.get_capture(epoch_id).epoch.base_physics_revision
                except (KeyError, ValueError):
                    base_revision = None
        allowed: set[Literal[
            "idle", "queued", "running", "committed", "invalidated", "rejected", "failed", "awaiting_retry", "degraded"
        ]] = {
            "idle",
            "queued",
            "running",
            "committed",
            "invalidated",
            "rejected",
            "failed",
            "awaiting_retry",
            "degraded",
        }
        raw_status = getattr(health, "status", "degraded")
        bootstrap_result = getattr(self, "_bootstrap_result", None)
        if bootstrap_result is not None and bootstrap_result.status != "committed":
            raw_status = "awaiting_retry"
        status = cast(
            Literal[
                "idle", "queued", "running", "committed", "invalidated",
                "rejected", "failed", "awaiting_retry", "degraded",
            ],
            raw_status if raw_status in allowed else "degraded",
        )
        if self.paused and status in {"idle", "committed"}:
            status = "degraded"
        planning_config = getattr(getattr(self, "_config", None), "planning", None)
        initial_plan_timeout_s = float(
            getattr(planning_config, "initial_plan_timeout_s", 180.0)
        )
        return PlanningHealthView(
            status=status,
            epoch_id=epoch_id,
            base_physics_revision=base_revision,
            current_physics_revision=current_revision,
            latest_physics_revision=health.latest_physics_revision,
            base_sim_time_s=health.base_sim_time_s,
            current_sim_time_s=health.latest_sim_time_s,
            latest_sim_time_s=health.latest_sim_time_s,
            data_age_s=health.data_age_s,
            deadline_utc_ms=(
                health.started_at_ms
                + int(initial_plan_timeout_s * 1000)
                if health.started_at_ms is not None
                and status in {"queued", "running"}
                else None
            ),
            node="planning_epoch" if status != "idle" else None,
            attempt=health.retry_attempt,
            planning_epoch_invariant_failures=getattr(
                self, "planning_epoch_invariant_failures", 0
            ),
            queued_event_count=health.queued_event_count,
            last_result_status=health.last_result_status,
            last_error=health.last_error or self.llm_pause_reason,
        )

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
                intent_confirmation=(
                    agent.intent_change_confirmation if agent is not None else None
                ),
            ),
            prediction_intent_monitor=EventMonitor(
                scenario_id=self.scenario_id,
                intent_confirmation=(
                    agent.intent_change_confirmation if agent is not None else None
                ),
            ),
            optimizer=planning_config,
            trajectory_diff_config=(
                agent.trajectory_diff if agent is not None else TrajectoryDiffConfig()
            ),
            intent_change_confirmation=(
                agent.intent_change_confirmation
                if agent is not None
                else IntentChangeConfirmation()
            ),
            semantic_repairs=agent.semantic_repairs if agent else 2,
            regional_batch_size=config.planning.regional_batch_size,
            regional_max_concurrency=config.planning.regional_max_concurrency,
            semantic_correction_attempts=config.planning.semantic_correction_attempts,
            model_id=self._role_model("master"),
            knowledge_client=self._knowledge_client,
            uuv_only=_is_uuv_only_config(config),
            retention=(agent.retention if agent is not None else RuntimeRetentionConfig()),
            current_snapshot_revision=self._current_snapshot_revision,
            memory_service=self._memory_service,
            short_term_repository=self._memory_short_term,
            memory_port=self._memory_port,
            planning_epoch_provider=lambda: self._active_epoch,
            epoch_commit_port=self._epoch_commit_port,
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
            worker_llm: StructuredLLM[Any]
            if self._llm_injected:
                worker_llm = _LedgerBoundStructuredLLM(
                    self._clients["master"], worker_ledger
                )
            else:
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
        if _is_uuv_only_config(self._config):
            return bool(situation.target_search_priors)
        return all(
            len(engine.belief_history(report.target_id)) >= 3
            for report in situation.group_reports
        )

    def _local_brain_decisions(
        self, situation: SituationSnapshot
    ) -> tuple[
        tuple[SlaveSonarDecision, ...],
        tuple[AdversaryIntentDecision | AdversaryEscapeDecision, ...],
    ]:
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
    ) -> tuple[
        tuple[SlaveSonarDecision, ...],
        tuple[AdversaryIntentDecision | AdversaryEscapeDecision, ...],
    ]:
        """Invoke local brains over contexts captured at one physics boundary."""
        self._set_llm_sim_time(situation.sim_time_s)
        adversary_decisions: list[AdversaryIntentDecision | AdversaryEscapeDecision] = []
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
                    if not isinstance(
                        adversary_decision,
                        (AdversaryIntentDecision, AdversaryEscapeDecision),
                    ):
                        raise TypeError("adversary graph returned no typed decision")
                    adversary_decisions.append(adversary_decision)
                except LLMError:
                    # Local brain failures are isolated; the public summary
                    # continues to expose target-owned belief motion.
                    recorder = getattr(self._engine, "record_adversary_degraded", None)
                    if callable(recorder):
                        recorder(adversary_context.target_id, "llm provider failure")
                    continue
                except Exception as exc:  # LLM semantic output is a content failure
                    recorder = getattr(self._engine, "record_adversary_degraded", None)
                    if callable(recorder):
                        recorder(adversary_context.target_id, type(exc).__name__)
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
            # CarrierRuntime.get_state() serves its last completed cache while
            # a graph cycle is active.  Taking the graph's outer lock here
            # would freeze physics and the HTTP frame stream for the complete
            # planning/LLM latency window.
            publisher.publish(engine.publication_situation())
        except Exception as exc:  # noqa: BLE001 - telemetry cannot stop tracking
            self._record_carrier_error("publish_latest", exc)

    def _record_carrier_error(
        self, source: str, error: BaseException | None = None
    ) -> None:
        """Count and retain a redacted source for a deferred carrier failure."""
        self.carrier_error_count = getattr(self, "carrier_error_count", 0) + 1
        details = getattr(self, "carrier_error_details", None)
        if details is None:
            details = []
            self.carrier_error_details = details
        detail = source
        if error is not None:
            detail += f":{type(error).__name__}: {str(error)[:240]}"
        details.append(detail)

    def _waiting_for_llm_reconnect(self) -> bool:
        if not bool(getattr(self, "paused", False)):
            return False
        if not bool(getattr(self, "reconnectable", True)):
            return True
        return time.monotonic() < getattr(self, "_next_llm_retry_at", 0.0)

    def on_situation(self, situation: SituationSnapshot) -> None:
        """Queue or run one carrier cycle at an observation boundary."""
        runtime = getattr(self, "_runtime", None)
        refresh_predictions = getattr(runtime, "refresh_predictions", None)
        if callable(refresh_predictions):
            try:
                refresh_predictions(situation)
            except Exception as exc:  # noqa: BLE001 - keep physics moving; fail audit
                self._record_carrier_error("prediction_refresh", exc)
        coordinator = getattr(self, "_epoch_coordinator", None)
        if coordinator is not None:
            coordinator.observe(situation)
        self._submit_due_periodic_summary(situation)
        if getattr(self, "_background_carrier", False):
            self._start_background_cycle(situation)
            return
        self._run_synchronous_carrier_cycle(situation)

    def _prepare_epoch(
        self,
        situation: SituationSnapshot,
        feedback_events: tuple[RuntimeEvent, ...],
    ) -> tuple[PlanningEpoch | None, tuple[RuntimeEvent, ...]]:
        """Observe the latest public frame and reserve at most one epoch."""
        events = list(situation.pending_events)
        events.extend(feedback_events)
        runtime = self._runtime
        pending_event_reader = getattr(runtime, "pending_events", None)
        if callable(pending_event_reader):
            events.extend(pending_event_reader())
        if not self._initialization_submitted and self._initialization_ready(situation):
            self._initialization_submitted = True
            events.append(
                RuntimeEvent(
                    event_id=(
                        f"{self.scenario_id}:initialization:{self.scenario_id}:"
                        f"{situation.sim_time_s}"
                    ),
                    scenario_id=self.scenario_id,
                    sim_time_s=situation.sim_time_s,
                    event_type="initialization",
                    entity_id=self.scenario_id,
                    level=EventLevel.STRATEGIC,
                    payload={},
                )
            )
        coordinator = getattr(self, "_epoch_coordinator", None)
        if coordinator is None:
            return None, tuple(events)
        seen_ids: set[str] = getattr(self, "_epoch_seen_event_ids", set())
        self._epoch_seen_event_ids = seen_ids
        for event in events:
            if event.event_id in seen_ids:
                continue
            seen_ids.add(event.event_id)
            if _event_requests_planning_epoch(event):
                self._epoch_coordinator.request(
                    (
                        EpochTrigger(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            sim_time_s=event.sim_time_s,
                            priority=_epoch_event_priority(event),
                            entity_id=event.entity_id,
                            resource_episode=(
                                event.payload.get("resource_episode")
                                if isinstance(event.payload.get("resource_episode"), int)
                                else None
                            ),
                        ),
                    )
                )
        engine = self._engine
        mission_provider = getattr(engine, "mission_snapshot", None)
        mission = mission_provider() if callable(mission_provider) else None
        if mission is None:
            return None, tuple(events)
        capture = coordinator.next_epoch(mission)
        if capture is None:
            return None, tuple(events)
        coordinator.mark_running(capture.epoch.epoch_id)
        return capture.epoch, tuple(events)

    def _submit_due_periodic_summary(self, situation: SituationSnapshot) -> None:
        """Build public summaries at progress boundaries without waiting on storage."""
        source_ids = getattr(self, "_periodic_summary_source_ids", None)
        if source_ids is None:
            source_ids = set()
            self._periodic_summary_source_ids = source_ids
        source_events = getattr(self, "_periodic_summary_source_events", None)
        if source_events is None:
            source_events = {}
            self._periodic_summary_source_events = source_events
        for event in situation.pending_events:
            if event.scenario_id != situation.scenario_id:
                continue
            source_ids.add(event.event_id)
            source_events[event.event_id] = event
        if len(source_ids) > 64:
            retained_ids = set(sorted(source_ids)[-64:])
            retained_events = {
                event_id: event
                for event_id, event in source_events.items()
                if event_id in retained_ids
            }
            source_ids.intersection_update(retained_ids)
            source_events.clear()
            source_events.update(retained_events)
        config = getattr(self, "_config", None)
        timing = getattr(config, "timing", None)
        interval_s = int(getattr(timing, "progress_report_s", 0))
        if interval_s <= 0 or situation.sim_time_s < getattr(
            self, "_periodic_summary_next_boundary_s", interval_s
        ):
            self._flush_periodic_summary_backlog()
            return
        engine = getattr(self, "_engine", None)
        mission_snapshot = getattr(engine, "mission_snapshot", None)
        mission = mission_snapshot() if callable(mission_snapshot) else None
        if mission is None:
            self._flush_periodic_summary_backlog()
            return
        previous = getattr(self, "_last_built_periodic_summary", None)
        accumulated_events = tuple(
            source_events[event_id] for event_id in sorted(source_ids) if event_id in source_events
        )
        summary, event = build_periodic_situation_summary(
            situation,
            mission,
            accumulated_events,
            previous,
        )
        pending = getattr(self, "_pending_periodic_summaries", None)
        if pending is None:
            pending = deque()
            self._pending_periodic_summaries = pending
        if len(pending) >= 64:
            self._record_periodic_summary_degradation(situation.sim_time_s)
        else:
            pending.append((summary, event))
            self._last_built_periodic_summary = summary
        source_ids.clear()
        source_events.clear()
        next_boundary = getattr(
            self, "_periodic_summary_next_boundary_s", interval_s
        )
        self._periodic_summary_next_boundary_s = (
            ((situation.sim_time_s // interval_s) + 1) * interval_s
            if situation.sim_time_s >= next_boundary
            else next_boundary + interval_s
        )
        self._flush_periodic_summary_backlog()

    def _flush_periodic_summary_backlog(self) -> None:
        pending = getattr(self, "_pending_periodic_summaries", None)
        writer = getattr(self, "_periodic_summary_writer", None)
        if pending is None or writer is None:
            return
        while pending:
            try:
                accepted = writer.submit(pending[0][1])
            except Exception as error:  # noqa: BLE001 - telemetry cannot stop tracking
                self._record_periodic_summary_degradation(
                    pending[0][1].sim_time_s,
                    reason=type(error).__name__,
                )
                return
            if not accepted:
                return
            pending.popleft()

    def _record_periodic_summary_degradation(
        self, sim_time_s: int, *, reason: str = "backlog_full"
    ) -> None:
        self._periodic_summary_backlog_overflow = (
            getattr(self, "_periodic_summary_backlog_overflow", 0) + 1
        )
        degradation_events = getattr(
            self, "_periodic_summary_degradation_events", None
        )
        if degradation_events is None:
            degradation_events = []
            self._periodic_summary_degradation_events = degradation_events
        degradation_events.append(
            RuntimeEvent(
                event_id=(
                    f"periodic_summary_backlog_overflow:{self.scenario_id}:"
                    f"{sim_time_s}:{self._periodic_summary_backlog_overflow}"
                ),
                scenario_id=self.scenario_id,
                sim_time_s=sim_time_s,
                event_type="periodic_summary_backlog_overflow",
                entity_id=self.scenario_id,
                level=EventLevel.TACTICAL,
                payload={
                    "backlog_limit": 64,
                    "overflow_count": self._periodic_summary_backlog_overflow,
                    "reason": reason,
                },
            )
        )

    def _run_synchronous_carrier_cycle(self, situation: SituationSnapshot) -> None:
        """Run a carrier cycle inline for deterministic finite/test runs."""
        runtime = self._runtime
        assert runtime is not None
        engine = self._engine
        assert engine is not None
        self.situation = situation
        feedback_events = self._feedback_events(situation)
        epoch, trigger_events = self._prepare_epoch(situation, feedback_events)
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
                except Exception as exc:  # noqa: BLE001 - bad boundary input cannot stop the loop
                    self._record_carrier_error("sync_commit_operational_inputs", exc)
            engine.set_reservations(runtime.reservations())
            if epoch is not None or getattr(self, "_epoch_coordinator", None) is None:
                runtime.submit_events(trigger_events)
                self._set_llm_sim_time(situation.sim_time_s)
                result = runtime.tick(epoch=epoch) if epoch is not None else runtime.tick()
            else:
                result = {"commit_status": None}
            self._finish_epoch(epoch, result)
            if result.get("commit_status") == "committed":
                committed_plan = _committed_epoch_plan(result)
                if committed_plan is not None:
                    self._apply_uuv_only_mission_plan(committed_plan)
                self._apply_new_commands()
            self._apply_verification_commands(result)
            for slave_decision in local_slave_decisions:
                engine.apply_slave_sonar_decision(slave_decision)
            for adversary_decision in adversary_decisions:
                engine.apply_adversary_decision(adversary_decision)
            self.mark_llm_recovered()
        except LLMError as exc:
            self._finish_epoch(epoch, {}, exc)
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(sensor_controls)
            self.mark_llm_paused(exc)
            return
        except Exception as exc:  # noqa: BLE001 - execution errors must roll back the cycle
            self._finish_epoch(epoch, {}, None)
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(sensor_controls)
            self._record_carrier_error("sync_carrier_cycle", exc)
            raise

    def _start_background_cycle(
        self,
        situation: SituationSnapshot,
        *,
        allow_paused: bool = False,
    ) -> None:
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
            if not allow_paused and self._waiting_for_llm_reconnect():
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
            feedback_events = self._feedback_events(cycle_situation)
            epoch, trigger_events = self._prepare_epoch(
                cycle_situation, feedback_events
            )
            cycle = _BackgroundCarrierCycle(
                situation=cycle_situation,
                adversary_contexts=tuple(engine.build_adversary_inputs(cycle_situation)),
                slave_contexts=tuple(engine.build_slave_contexts(cycle_situation)),
                epoch=epoch,
                trigger_events=trigger_events,
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
        """Run master and local brains concurrently, exposing master first."""
        runtime = self._runtime
        assert runtime is not None
        drain_sensor_controls = getattr(runtime, "drain_sensor_controls", None)
        self._active_cycle_situation = cycle.situation
        self._active_epoch = cycle.epoch
        self._queue_local_brain_cycle(cycle)
        try:
            cycle.sensor_controls = (
                drain_sensor_controls() if callable(drain_sensor_controls) else ()
            )
            if cycle.epoch is not None or getattr(self, "_epoch_coordinator", None) is None:
                runtime.submit_events(cycle.trigger_events)
                self._set_llm_sim_time(cycle.situation.sim_time_s)
                cycle.result = (
                    runtime.tick(epoch=cycle.epoch)
                    if cycle.epoch is not None
                    else runtime.tick()
                )
            else:
                cycle.result = {"commit_status": None}
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
                cycle.planning_done = True
                cycle.done = True
                self._active_cycle_situation = None
                self._active_epoch = None

    def _queue_local_brain_cycle(self, cycle: _BackgroundCarrierCycle) -> None:
        """Serialize local LLM work without occupying the master planning slot."""
        if not cycle.adversary_contexts and not cycle.slave_contexts:
            return
        with self._carrier_cycle_lock:
            if getattr(self, "_closing", False):
                return
            active = getattr(self, "_background_local_thread", None)
            if active is not None and active.is_alive():
                mailbox = getattr(self, "_background_local_mailbox", None)
                if (
                    mailbox is None
                    or cycle.situation.snapshot_revision
                    >= mailbox.situation.snapshot_revision
                ):
                    self._background_local_mailbox = cycle
                return
            thread = Thread(
                target=self._run_local_brain_cycle,
                args=(cycle,),
                name="underwater-local-brains",
                daemon=True,
            )
            self._background_local_thread = thread
        thread.start()

    def _run_local_brain_cycle(self, cycle: _BackgroundCarrierCycle) -> None:
        try:
            cycle.slave_decisions, cycle.adversary_decisions = (
                self._local_brain_decisions_from_contexts(
                    cycle.situation,
                    cycle.adversary_contexts,
                    cycle.slave_contexts,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - local roles are isolated
            cycle.local_error = exc
        finally:
            with self._carrier_cycle_lock:
                results = getattr(self, "_background_local_results", None)
                if results is None:
                    results = deque()
                    self._background_local_results = results
                results.append(cycle)
                self._background_local_thread = None
                next_cycle = getattr(self, "_background_local_mailbox", None)
                self._background_local_mailbox = None
            if next_cycle is not None:
                self._queue_local_brain_cycle(next_cycle)

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
        """Apply master and local phases independently at physics boundaries."""
        if not self._background_carrier:
            return
        self._apply_completed_local_brain_cycles()
        with self._carrier_cycle_lock:
            cycle = self._background_cycle
            if cycle is None:
                return
            planning_ready = (
                (cycle.planning_done or cycle.done) and not cycle.planning_applied
            )
            if planning_ready:
                cycle.planning_applied = True
            cycle_done = cycle.done
            if cycle_done:
                self._background_cycle = None
                self._background_thread = None
            latest = self.situation
        stale_informational_cycle = cycle.epoch is None and (
            latest is not None
            and latest.snapshot_revision > cycle.situation.snapshot_revision
        )
        if planning_ready and not stale_informational_cycle:
            self._apply_background_planning_phase(cycle)
        if not cycle_done:
            return
        if stale_informational_cycle:
            self._schedule_latest_background_cycle(cycle)
            return
        is_bootstrap_cycle = (
            cycle.epoch is not None
            and cycle.epoch.epoch_id == getattr(self, "_bootstrap_epoch_id", None)
        )
        if is_bootstrap_cycle:
            return
        self._schedule_latest_background_cycle(cycle)

    def _apply_completed_local_brain_cycles(self) -> None:
        with self._carrier_cycle_lock:
            results = getattr(self, "_background_local_results", None)
            completed = tuple(results) if results is not None else ()
            if results is not None:
                results.clear()
        engine = self._engine
        if engine is None:
            return
        for cycle in completed:
            if cycle.local_error is not None:
                self._record_carrier_error(
                    "background_local_brain_cycle", cycle.local_error
                )
                continue
            for slave_decision in cycle.slave_decisions:
                engine.apply_slave_sonar_decision(slave_decision)
            for adversary_decision in cycle.adversary_decisions:
                engine.apply_adversary_decision(adversary_decision)

    def _apply_background_planning_phase(
        self, cycle: _BackgroundCarrierCycle
    ) -> None:
        """Commit one finished master phase without waiting for local LLMs."""
        if cycle.error is not None:
            self._finish_epoch(cycle.epoch, {}, cycle.error)
            if isinstance(cycle.error, LLMError):
                self.mark_llm_paused(cycle.error)
            else:
                self._record_carrier_error("background_carrier_cycle", cycle.error)
            return
        runtime = self._runtime
        engine = self._engine
        if runtime is None or engine is None or cycle.result is None:
            self._finish_epoch(cycle.epoch, {}, None)
            self._record_carrier_error("background_carrier_cycle_missing_result")
            return
        active_plan_reader = getattr(runtime, "active_plan", None)
        active_plan = active_plan_reader() if callable(active_plan_reader) else None
        self._finish_epoch(cycle.epoch, cycle.result)
        if _is_uuv_only_config(self._config):
            committed_plan = _committed_epoch_plan(cycle.result)
            if committed_plan is not None:
                self._apply_uuv_only_mission_plan(committed_plan)
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
            except Exception as exc:  # noqa: BLE001 - bad input cannot stop tracking
                self._record_carrier_error("background_commit_operational_inputs", exc)
        engine.set_reservations(runtime.reservations())
        if cycle.result.get("commit_status") == "committed":
            self._apply_new_commands()
        self._apply_verification_commands(cycle.result)
        self.mark_llm_recovered()

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

    def _finish_epoch(
        self,
        epoch: PlanningEpoch | None,
        result: Mapping[str, Any],
        error: BaseException | None = None,
    ) -> None:
        """Close one reserved epoch and let the coordinator own retry policy."""
        if epoch is None:
            return
        epoch_result = result.get("epoch_commit_result")
        if not isinstance(epoch_result, EpochCommitResult) or epoch_result.epoch_id != epoch.epoch_id:
            message = (
                f"{type(error).__name__}: {error}"
                if error is not None
                else "carrier graph completed without a terminal result for the active epoch"
            )
            error_name = type(error).__name__.lower() if error is not None else ""
            if "semantic" in error_name or "semantic" in message.lower():
                epoch_result = EpochCommitResult(
                    epoch_id=epoch.epoch_id,
                    status="rejected",
                    validation_report_id=f"validation:{epoch.epoch_id}:rejected",
                    failure_category="semantic",
                    failure_message=message[:2000],
                )
            else:
                category: Literal["provider", "internal"] = (
                    "provider" if isinstance(error, LLMError) else "internal"
                )
                if error is None:
                    self.planning_epoch_invariant_failures = (
                        getattr(self, "planning_epoch_invariant_failures", 0) + 1
                    )
                epoch_result = EpochCommitResult(
                    epoch_id=epoch.epoch_id,
                    status="failed",
                    failure_category=category,
                    failure_message=message[:2000],
                )
        coordinator = getattr(self, "_epoch_coordinator", None)
        if coordinator is not None:
            coordinator.finish(epoch_result)
        if epoch.epoch_id == getattr(self, "_bootstrap_epoch_id", None):
            self._bootstrap_result = epoch_result

    def _apply_uuv_only_mission_plan(self, plan: Any | None = None) -> bool:
        """Apply only the latest verified executable plan in UUV-only mode."""
        engine = self._engine
        runtime = self._runtime
        if engine is None or runtime is None:
            return False
        if plan is None:
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
            "status": getattr(
                self,
                "_manifest_status",
                getattr(self, "_run_phase", "running"),
            ),
            "llm": self._config.llm.model if self._config.llm else "http",
            "llm_roles": sorted(self._clients),
            "created_at_ms": now_ms(),
            "carrier_error_count": self.carrier_error_count,
            "carrier_error_details": list(self.carrier_error_details),
            "decision_count": len(self.ledger.list_decisions(self.scenario_id)),
            "llm_call_count": len(self.ledger.list_llm_calls()),
            "active_plan_id": active.plan_id if active is not None else None,
            "active_plan_revision": active.revision if active is not None else None,
            "operational_frame_count": (
                self._publisher.frame_count
                if self._publisher is not None
                else 0
            ),
            "log_truncated": bool(
                getattr(getattr(self._publisher, "_logger", None), "log_truncated", False)
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
                self._background_local_mailbox = None
            finally:
                self._carrier_cycle_lock.release()
        else:
            self._closing = True
            self._background_mailbox = None
            self._background_local_mailbox = None
        for client in getattr(self, "_clients", {}).values():
            cancel = getattr(client, "cancel", None)
            if callable(cancel):
                cancel()
        periodic_summary_writer = getattr(self, "_periodic_summary_writer", None)
        if periodic_summary_writer is not None:
            periodic_summary_writer.stop(timeout=0.0)
        if self._memory_worker is not None:
            self._memory_worker.stop(timeout=0.0)

    def close(self, *, timeout_s: float = 10.0) -> bool:
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s
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
                result = self._close_once(deadline=deadline)
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

    def shutdown_report(self) -> ShutdownReport:
        """Return the latest bounded close result without exposing private state."""
        condition = getattr(self, "_close_condition", None)
        if condition is None:
            return self._shutdown_report.model_copy(deep=True)
        with condition:
            return self._shutdown_report.model_copy(deep=True)

    def _close_once(self, *, deadline: float | None = None) -> bool:
        if deadline is None:
            deadline = time.monotonic() + 10.0
        with self._carrier_cycle_lock:
            self._closing = True
            self._background_mailbox = None
            self._background_local_mailbox = None
        for client in self._clients.values():
            cancel = getattr(client, "cancel", None)
            if callable(cancel):
                cancel()
        remaining_resources: list[str] = []
        if self._memory_worker is not None:
            if not self._memory_worker.stop(
                timeout=max(0.0, min(5.0, deadline - time.monotonic()))
            ):
                remaining_resources.append("memory-worker")
                self._shutdown_report = ShutdownReport(
                    completed=False,
                    remaining_resources=tuple(remaining_resources),
                )
                return False
        periodic_summary_writer = getattr(self, "_periodic_summary_writer", None)
        if periodic_summary_writer is not None and not periodic_summary_writer.stop(
            timeout=max(0.0, min(5.0, deadline - time.monotonic()))
        ):
            remaining_resources.append("periodic-summary-writer")
            self._shutdown_report = ShutdownReport(
                completed=False,
                remaining_resources=tuple(remaining_resources),
            )
            return False
        background_thread = self._background_thread
        if background_thread is not None and background_thread.is_alive():
            background_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if background_thread is not None and background_thread.is_alive():
            remaining_resources.append("carrier-llm")
            self._shutdown_report = ShutdownReport(
                completed=False,
                remaining_resources=tuple(remaining_resources),
            )
            return False
        local_thread = getattr(self, "_background_local_thread", None)
        if local_thread is not None and local_thread.is_alive():
            local_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if local_thread is not None and local_thread.is_alive():
            remaining_resources.append("local-brains")
            self._shutdown_report = ShutdownReport(
                completed=False,
                remaining_resources=tuple(remaining_resources),
            )
            return False

        completed: set[int] = getattr(self, "_close_completed", set())
        self._close_completed = completed
        errors: list[BaseException] = []

        def close_resource(resource: object | None, owner_name: str) -> None:
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
                remaining_resources.append(owner_name)
            else:
                completed.add(identity)

        close_resource(self._memory_embedding_provider, "memory-embedding")
        close_resource(getattr(self, "_memory_worker_embedding_provider", None), "memory-worker-embedding")
        close_resource(getattr(self, "_memory_worker_llm", None), "memory-worker-llm")
        for role, client in self._clients.items():
            close_resource(client, f"http-client:{role}")
        close_resource(self._runtime, "carrier-runtime")
        close_resource(self._publisher, "frame-publisher")
        close_resource(getattr(self, "_memory_worker_short_term", None), "memory-worker-short-term")
        close_resource(getattr(self, "_memory_worker_long_term", None), "memory-worker-long-term")
        close_resource(getattr(self, "_memory_worker_events", None), "memory-worker-events")
        close_resource(getattr(self, "_memory_worker_ledger", None), "memory-worker-ledger")
        close_resource(getattr(self, "_memory_worker_plans", None), "memory-worker-plans")
        close_resource(self._memory_short_term, "short-term-memory")
        close_resource(self._memory_long_term, "long-term-memory")
        close_resource(self._knowledge_client, "knowledge-client")
        close_resource(getattr(self, "_epoch_repository", None), "planning-epoch-repository")
        close_resource(self.plans, "plan-repository")
        close_resource(self.events, "event-repository")
        close_resource(self.ledger, "decision-ledger")
        if errors:
            self._shutdown_report = ShutdownReport(
                completed=False,
                remaining_resources=tuple(dict.fromkeys(remaining_resources)),
            )
            raise errors[0]
        self._shutdown_report = ShutdownReport(completed=True)
        return True


if __name__ == "__main__":
    sys.exit(main())
