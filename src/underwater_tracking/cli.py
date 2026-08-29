# src/underwater_tracking/cli.py
"""Command-line entry points for the underwater tracking assistant.

``simulate`` runs the deterministic headless simulation and writes frames.
``agent-run`` runs the same scenario through the LangGraph
carrier: it loads the config, creates the SQLite repositories and
checkpointer, builds the real LongCat HTTP provider (the API key is read at
call time from the configured api_key or environment variable (env wins);
provider configuration and connectivity failures terminate the run), wires the engine's
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import json
from math import atan2, ceil, floor, isfinite
import os
import signal
import sys
import time
from threading import Condition, Event, RLock, Thread, current_thread, main_thread
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    _prior_seeded_planning_inputs,
)
from underwater_tracking.agent.graphs.adversary import build_adversary_graph
from underwater_tracking.agent.graphs.slave import build_slave_graph
from underwater_tracking.agent.llm import (
    HTTPStructuredLLM,
    LLMConfigError,
    LLMContentError,
    LLMError,
    StructuredLLM,
)
from underwater_tracking.agent.llm_factory import RoleHTTPStructuredLLM, build_role_llm
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.optimize import PlanningConfig
from underwater_tracking.agent.nodes.regions import regional_plan_to_mission_candidates
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
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
from underwater_tracking.domain.agent_models import (
    IntentHypothesis,
    PredictedTrackRef,
    TrackingPlan,
    VerificationCommand,
)
from underwater_tracking.domain.execution_models import (
    GlobalTargetTrackView,
    OperationalExecutionSnapshot,
)
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
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    UUVResourceState,
)
from underwater_tracking.domain.prediction_models import AcceptedPrediction
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch
from underwater_tracking.domain.regional_models import (
    GridSpec,
    TargetRegionPlan,
    TaskRegionProposal,
    TaskRegionProposalSet,
)
from underwater_tracking.domain.slave_models import (
    SlaveDecisionValidationError,
    SlaveSonarContext,
    SlaveSonarDecision,
)
from underwater_tracking.domain.ui_models import PlanningHealthView
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
from underwater_tracking.planning.mission_optimizer import MissionOptimizer
from underwater_tracking.planning.region_baseline import (
    FourRegionBaseline,
    build_four_region_baseline,
)
from underwater_tracking.planning.regions import build_llm_task_region_plan
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.runtime.run_controller import RunController
from underwater_tracking.runtime.models import ShutdownReport
from underwater_tracking.runtime.run_catalog import RunCatalog
from underwater_tracking.runtime.mission_controller import (
    MissionController,
    execution_snapshot_to_mission_plan,
)
from underwater_tracking.runtime.mission_epoch_commit import MissionEpochCommitPort
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.runtime.execution_snapshot_factory import (
    build_execution_snapshot,
    execution_group_status,
    execution_region_status,
)
from underwater_tracking.runtime.planning_epoch import EpochTrigger, PlanningEpochCoordinator
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation.engine import SimulationEngine

_SCENARIO_ID = "underwater-default"
_BATTERY_ROTATION_THRESHOLD = 0.3
_DEFAULT_API_PORT = 8000
_API_PORT_ENV = "UNDERWATER_TRACKING_API_PORT"


def _merge_authoritative_region_lifecycles(
    regions: Sequence[Any],
    mission_regions: Sequence[Any],
) -> tuple[Any, ...]:
    """Keep live controller lifecycle progress in semantic snapshot revisions."""
    by_identity = {
        (getattr(region, "region_id", None), getattr(region, "target_id", None)): region
        for region in mission_regions
    }
    merged: list[Any] = []
    for region in regions:
        mission_region = by_identity.get((region.region_id, region.target_id))
        if mission_region is None:
            merged.append(region)
            continue
        try:
            status = execution_region_status(mission_region.lifecycle)
        except (AttributeError, TypeError, ValueError):
            merged.append(region)
            continue
        merged.append(region.model_copy(update={"status": status}))
    return tuple(merged)


def _current_mission_lifecycles(engine: Any) -> dict[str, Any]:
    """Read lifecycle state for the runtime projection without mutating it."""
    controller = getattr(engine, "_mission_controller", None)
    snapshot_reader = getattr(controller, "snapshot", None)
    mission_snapshot = snapshot_reader() if callable(snapshot_reader) else None
    return {
        region.region_id: region.lifecycle
        for region in getattr(mission_snapshot, "regions", ())
    }


def _semantic_execution_cursor(
    regions: Sequence[Any],
    *,
    fallback_region_id: str,
) -> tuple[str, str]:
    """Recompute the semantic cursor after authoritative lifecycle merging."""
    if not regions:
        return fallback_region_id, fallback_region_id
    active_statuses = {"active", "passive", "handoff_pending"}
    current_index = next(
        (
            index
            for index, region in enumerate(regions)
            if getattr(region, "status", None) in active_statuses
        ),
        None,
    )
    if current_index is None:
        current_index = next(
            (
                index
                for index, region in enumerate(regions)
                if getattr(region, "region_id", None) == fallback_region_id
            ),
            0,
        )
    current_region_id = regions[current_index].region_id
    next_region_id = (
        regions[current_index + 1].region_id
        if current_index + 1 < len(regions)
        else current_region_id
    )
    return current_region_id, next_region_id


def _deterministic_region_proposals(
    points: tuple[tuple[float, float], ...],
    map_bounds: tuple[float, float, float, float],
) -> TaskRegionProposalSet:
    """Build four bounded handoff regions along a public prediction."""
    if not points:
        raise ValueError("deterministic region proposals require prediction points")
    x_span = max(point[0] for point in points) - min(point[0] for point in points)
    y_span = max(point[1] for point in points) - min(point[1] for point in points)
    axis = 0 if x_span >= y_span else 1
    direction = 1.0 if points[-1][axis] >= points[0][axis] else -1.0
    axis_min, axis_max = (
        (map_bounds[0], map_bounds[1])
        if axis == 0
        else (map_bounds[2], map_bounds[3])
    )
    cross_min, cross_max = (
        (map_bounds[2], map_bounds[3])
        if axis == 0
        else (map_bounds[0], map_bounds[1])
    )
    anchor = points[0]
    chain_extent = 9_000.0
    if direction > 0:
        axis_base = floor((anchor[axis] - 2_000.0) / 1_000.0) * 1_000.0
    else:
        axis_base = ceil((anchor[axis] + 1_000.0) / 1_000.0) * 1_000.0
        axis_base -= chain_extent
    axis_base = min(max(axis_base, axis_min), axis_max - chain_extent)
    cross_anchor = anchor[1 - axis]
    cross_base = min(
        max(
            floor((cross_anchor - 2_000.0) / 1_000.0) * 1_000.0,
            cross_min,
        ),
        cross_max - 4_000.0,
    )
    proposals: list[TaskRegionProposal] = []
    for index in range(4):
        start = (
            axis_base + index * 2_000.0
            if direction > 0
            else axis_base + (3 - index) * 2_000.0
        )
        if axis == 0:
            lower_left = (start, cross_base)
            upper_right = (start + 3_000.0, cross_base + 4_000.0)
        else:
            lower_left = (cross_base, start)
            upper_right = (cross_base + 4_000.0, start + 3_000.0)
        proposals.append(
            TaskRegionProposal(
                lower_left_xy=lower_left,
                upper_right_xy=upper_right,
                rationale=f"deterministic startup forecast segment {index + 1}",
            )
        )
    return TaskRegionProposalSet(regions=tuple(proposals))


class _ProviderAttestationProbeResponse(BaseModel):
    """Minimal structured response used to prove a role can answer live."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ok: bool


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
        "target_intent_change_suspected",
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
    if event.event_type in _PREDICTION_REFRESH_EVENT_TYPES:
        return True
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


def _has_current_public_execution_source(
    situation: SituationSnapshot,
    target_id: str,
) -> bool:
    """Return whether execution can still be grounded in public target data."""
    group_reports = getattr(situation, "group_reports", None)
    target_search_priors = getattr(situation, "target_search_priors", None)
    if group_reports is None and target_search_priors is None:
        return True
    for report in group_reports or ():
        belief = getattr(report, "belief", None)
        mean = getattr(belief, "mean", ())
        source_ids = getattr(belief, "source_observation_ids", ())
        if (
            getattr(report, "target_id", None) == target_id
            and len(mean) >= 2
            and bool(source_ids)
        ):
            return True
    for prior in target_search_priors or ():
        if (
            getattr(prior, "target_id", None) == target_id
            and prior.issued_at_s <= situation.sim_time_s < prior.valid_until_s
        ):
            return True
    return False


def _project_public_track_xy(
    position: tuple[float, float],
    map_bounds: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Project a finite public estimate into the declared map envelope."""
    if not all(isfinite(value) for value in (*position, *map_bounds)):
        return None
    min_x, max_x, min_y, max_y = map_bounds
    if min_x > max_x or min_y > max_y:
        return None
    return (
        min(max(position[0], min_x), max_x),
        min(max(position[1], min_y), max_y),
    )


def _preserve_execution_regions_after_partition_failure(
    accepted: AcceptedPrediction,
    *,
    current: OperationalExecutionSnapshot,
    target_id: str,
    execution_revision: int,
    origin_sim_time_s: float,
    map_bounds: tuple[float, float, float, float],
) -> FourRegionBaseline:
    """Reproject a verified current chain after a transient partition failure."""
    regions = tuple(current.regions)
    expected_ids = tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))
    if len(regions) != 4 or tuple(region.region_id for region in regions) != expected_ids:
        raise ValueError("current execution does not provide four stable regions")
    if any(region.target_id != target_id for region in regions):
        raise ValueError("current execution regions target does not match accepted target")
    if any(
        not all(
            map_bounds[0] <= point[0] <= map_bounds[1]
            and map_bounds[2] <= point[1] <= map_bounds[3]
            for point in region.geometry
        )
        for region in regions
    ):
        raise ValueError("current execution regions lie outside map bounds")
    prediction = accepted.prediction
    if prediction is None:
        raise ValueError("partition recovery requires an accepted prediction")
    if prediction.prediction_id != current.prediction_id:
        raise ValueError("execution_region_identity_unbound")

    # Reuse the validated prior-chain path without changing the public
    # AcceptedPrediction object or downgrading its health status.
    fallback_health = accepted.health.model_copy(update={"status": "unavailable"})
    preserved = build_four_region_baseline(
        AcceptedPrediction(prediction=None, health=fallback_health),
        target_id=target_id,
        execution_revision=execution_revision,
        origin_sim_time_s=origin_sim_time_s,
        map_bounds_xy=map_bounds,
        prior_regions=regions,
        prior_prediction_point_count=len(prediction.points_xy),
    )
    return FourRegionBaseline(
        regions=preserved.regions,
        mode=preserved.mode,
        reason_codes=tuple(
            dict.fromkeys(
                (
                    "current_prediction_partition_unavailable",
                    *preserved.reason_codes,
                )
            )
        ),
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
        provider = SentenceTransformerEmbeddingProvider(
            config,
            ledger=ledger,
            scenario_id=scenario_id,
        )
        provider.verify_ready()
        return provider
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
    base_execution_revision: int = 0
    sensor_controls: tuple[SensorModeControl, ...] = ()
    slave_decisions: tuple[SlaveSonarDecision, ...] = ()
    adversary_decisions: tuple[AdversaryIntentDecision | AdversaryEscapeDecision, ...] = ()
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    local_error: BaseException | None = None
    planning_done: bool = False
    planning_applied: bool = False
    done: bool = False


def _create_public_run_dir(*, output_root: Path = Path("outputs")) -> Path:
    """Create the sole public output directory for one application run."""
    return RunCatalog(output_root).create_run_dir()


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
    carrier_configs = (config.environment.carrier, *config.environment.carriers)
    initial_carrier_missions = {
        carrier.platform_id: CarrierMissionModel(
            carrier_id=carrier.platform_id,
            role=carrier.role,
            home_battle_group_id=config.scenario.home_battle_group_id,
            ready_uuv_ids=tuple(
                sorted(
                    uuv.platform_id
                    for uuv in config.environment.uuvs
                    if uuv.home_carrier_id == carrier.platform_id
                )
            ),
        )
        for carrier in carrier_configs
    }
    return MissionController(
        scenario_id=config.scenario.scenario_id,
        initial_uuv_resources=initial_resources,
        initial_carrier_missions=initial_carrier_missions,
        uuv_owner_by_id=owner_by_id,
        region_entry_probability_threshold=config.scenario.region_entry_probability_threshold,
        region_transition_confirm_cycles=config.scenario.region_transition_confirm_cycles,
        resource_warning_mileage_fraction=(
            config.scenario.resource_warning_mileage_fraction
        ),
        group_min_size=config.tracking.group_min_size,
        execution_hard_stale_s=config.tracking.prediction_health.hard_stale_s,
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
    serve.add_argument("--require-real-provider", action="store_true", default=True)
    serve.add_argument(
        "--bootstrap-planning",
        action="store_true",
        help="run the initial planning epoch before finite-step simulation",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument(
        "--web-ui-url",
        default=None,
        help="URL to open when the API root is requested",
    )
    serve.add_argument(
        "--static-ui-dir",
        type=Path,
        default=None,
        help="serve a built React app from this directory on the API port",
    )
    serve.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="root directory for the single run-* output directory",
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
    run_dir = _create_public_run_dir()
    engine = SimulationEngine(
        config,
        seed=args.seed,
        output_dir=run_dir,
        mission_controller=_mission_controller_for(config),
    )
    try:
        for _ in range(args.steps):
            engine.step()
    finally:
        engine.logger.close()
    return 0


def _agent_run(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the agent-coupled scenario and write manifest plus JSONL."""
    _require_uuv_only_live_config(config)
    run_dir = _create_public_run_dir()
    database_path = run_dir / "agent.db"
    loop = _AgentLoop(
        config,
        database_path=database_path,
        llm=None,
        run_id=run_dir.name,
        steps=args.steps,
        seed=args.seed,
        llm_execution_required=True,
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
        attestations = loop.provider_attestations(probe=True)
        missing_roles = sorted(
            str(item.get("role", "unknown"))
            for item in attestations
            if not bool(item.get("attested"))
            or not bool(item.get("probe_successful"))
        )
        if missing_roles:
            raise LLMConfigError(
                "real HTTP provider attestation failed for roles: "
                + ",".join(missing_roles)
        )
        for _ in range(args.steps):
            _step_with_llm_retries(engine, loop, config)
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
        if accepts_kwargs or "require_real_provider" in controller_parameters:
            controller_kwargs["require_real_provider"] = bool(
                getattr(args, "require_real_provider", False)
            )
        if accepts_kwargs or "bootstrap_planning" in controller_parameters:
            controller_kwargs["bootstrap_planning"] = bool(
                getattr(args, "bootstrap_planning", False)
            )
        output_root = getattr(args, "output_root", None)
        if output_root is not None and (accepts_kwargs or "output_root" in controller_parameters):
            controller_kwargs["output_root"] = output_root
        controller = RunController(config, **controller_kwargs)
        controller.start_run(config.scenario.initial_target_count, seed=args.seed)
        supervisor = getattr(controller, "process_supervisor", None)
        if supervisor is not None and callable(getattr(supervisor, "register_port", None)):
            supervisor.register_port(args.port, host=args.host, name="api")
        static_ui_dir = getattr(args, "static_ui_dir", None)
        app = create_app(
            controller=controller,
            catalog=RunCatalog(output_root or Path("outputs")),
            directive_job_limit=(
                config.agent.retention.directive_job_limit
                if config.agent is not None
                else 256
            ),
            web_ui_url=getattr(args, "web_ui_url", None),
            static_ui_dir=static_ui_dir,
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
                closed = close(
                    # An interrupt must remain prompt even if a provider call
                    # is still in flight; normal completion keeps the longer
                    # window for a complete resource drain.
                    timeout_s=1.0 if controller.aborted else 10.0
                )
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
            # ``main.py`` runs the API on a worker so it can wait for the API
            # before launching Vite. Only the main interpreter thread may
            # register OS signal handlers.
            if current_thread() is not main_thread():
                await self._serve(sockets=sockets)
                return
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
    """Build the three real role-specific HTTP clients.

    The bearer token is read at call time from the configured api_key
    (``configs/.env``, git-ignored) or the configured environment variable
    (env wins). Missing credentials are left for the real client to report as
    ``LLMConfigError`` on its first call; no unavailable or synthetic client is
    substituted. Legacy flat role settings are rejected because they cannot
    identify all three real role clients.
    """
    llm_config = config.llm
    if llm_config is None:
        raise LLMConfigError(_chat_credentials_reason(config) or "chat LLM configuration is unavailable")
    if llm_config.roles is None:
        raise LLMConfigError(
            "role-specific chat configuration is unavailable (legacy flat LLM config)"
        )
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
    configured_token = (
        os.environ[llm_config.api_key_env]
        if llm_config.api_key_env in os.environ
        else llm_config.api_key
    )
    if configured_token is None or not configured_token.strip():
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
    """Advance one physical step, failing hard when an LLM call fails."""
    del config, stop
    loop.raise_if_llm_failed()
    loop.apply_background_cycle()
    try:
        engine.step()
    except LLMError as exc:
        loop.raise_llm_failure(exc)
    loop.publish_latest()
    return True


class _AgentLoop:
    """Wires the engine's group reports into CarrierRuntime and back.

    ``on_situation`` is the engine hook called at the end of every
    observation cycle: it submits the initialization event once the belief
    history is warm, runs one carrier tick, and applies newly committed plan
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
        llm_execution_required: bool = False,
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
        self._llm_execution_required = llm_execution_required
        self.plans = PlanRepository(database_path)
        self._epoch_repository = PlanningEpochRepository(self.plans.connection)
        self._epoch_coordinator = PlanningEpochCoordinator(
            self.scenario_id,
            repository=self._epoch_repository,
        )
        self._transition_coordinator = ScenarioTransitionCoordinator(self.scenario_id)
        self._epoch_commit_port: MissionEpochCommitPort | None = None
        self._execution_coordinator: ExecutionCoordinator | None = None
        self._active_epoch: PlanningEpoch | None = None
        self._epoch_seen_event_ids: set[str] = set()
        self.events = EventRepository(database_path)
        self._periodic_summary_writer = PeriodicSituationSummaryWriter(database_path)
        self.ledger = DecisionLedger(database_path)
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
        self._fatal_llm_error: LLMError | None = None
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
        self._baseline_regional_plans: dict[str, TargetRegionPlan] = {}
        self._baseline_region_candidates: dict[str, object] = {}
        self._baseline_intent_hypotheses: dict[str, IntentHypothesis] = {}
        self._last_deterministic_region_refresh_s = 0
        self._last_strategic_review_s = 0
        self._last_battery_rotation_s: dict[str, int] = {}
        self._adversary_provider_call_ids: dict[str, str] = {}
        self._provider_probe_successes: dict[str, bool] = {}
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
            self._execution_coordinator = ExecutionCoordinator(
                scenario_id=self.scenario_id,
                plans=self.plans,
                mission_controller=mission_controller,
                evidence_resolver=lambda evidence_id: self.events.get(evidence_id),
            )
            self._epoch_commit_port = MissionEpochCommitPort(
                plans=self.plans,
                epochs=self._epoch_repository,
                mission_controller=mission_controller,
                transition_coordinator=self._transition_coordinator,
                situation_provider=self._current_commit_situation,
                execution_admission=(
                    self._uuv_execution_admission
                    if _is_uuv_only_config(self._config)
                    else None
                ),
                uuv_only=_is_uuv_only_config(self._config),
            )
            self._restore_latest_committed_epoch(mission_controller)
        self._runtime = CarrierRuntime(
            self._deps(),
            scenario_id=self.scenario_id,
            database_path=self.database_path,
            execution_coordinator=self._execution_coordinator,
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
            candidate_regions_provider=lambda: dict(self._baseline_region_candidates),
            physics_step_s=self._config.timing.physics_step_s,
            history_limit=64,
            event_history_limit=1024,
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
        """Start the initial planning epoch before any physics step is allowed."""
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

    def install_deterministic_baseline(
        self, situation: SituationSnapshot
    ) -> ExecutableMissionPlan | None:
        """Install an immediately executable UUV plan from the public forecast."""
        if not _is_uuv_only_config(self._config):
            return None
        seeded = _prior_seeded_planning_inputs(situation)
        predictions = seeded.get("predictions", {})
        intents = seeded.get("intent_hypotheses", {})
        if not predictions:
            raise RuntimeError("deterministic baseline requires a public target prediction")
        map_bounds = situation.map_bounds_xy
        if map_bounds is None:
            raise RuntimeError("deterministic baseline requires shared map bounds")
        regional_plans: dict[str, TargetRegionPlan] = {}
        candidates = []
        for target_id, prediction in sorted(predictions.items()):
            intent = intents.get(target_id)
            if intent is None:
                raise RuntimeError(
                    f"deterministic baseline requires intent for target {target_id!r}"
                )
            regional_plan = build_llm_task_region_plan(
                prediction,
                intent,
                _deterministic_region_proposals(
                    prediction.points_xy,
                    map_bounds,
                ),
                map_bounds,
                GridSpec(),
                required_quality=self._config.tracking.quality_warning,
            )
            regional_plans[target_id] = regional_plan
            region_candidates = regional_plan_to_mission_candidates(regional_plan)
            if not region_candidates:
                raise RuntimeError(
                    f"deterministic baseline produced no region for target {target_id!r}"
                )
            candidates.extend(region_candidates)
        snapshot = build_planning_snapshot(situation)
        plan = MissionOptimizer(
            home_battle_group_id=self._config.scenario.home_battle_group_id,
            goal_mode=True,
        ).optimize(
            snapshot,
            tuple(candidates),
        )
        if not plan.batches:
            raise RuntimeError("deterministic baseline produced no deployable UUV batch")
        members_by_target: dict[str, tuple[str, ...]] = {}
        roles_by_member: dict[str, str] = {}
        standby_ids: set[str] = set()
        for target_id in sorted(regional_plans):
            assignments = tuple(
                assignment
                for assignment in plan.region_assignments
                if assignment.target_id == target_id
            )
            active_ids = tuple(
                dict.fromkeys(
                    uuv_id
                    for assignment in assignments
                    for uuv_id in (
                        *assignment.active_scan_uuv_ids,
                        *assignment.passive_track_uuv_ids,
                    )
                )
            )
            members_by_target[target_id] = active_ids
            for assignment in assignments:
                roles_by_member.update(
                    {uuv_id: "active_verifier" for uuv_id in assignment.active_scan_uuv_ids}
                )
                roles_by_member.update(
                    {uuv_id: "passive_tracker" for uuv_id in assignment.passive_track_uuv_ids}
                )
                standby_ids.update(assignment.reserve_uuv_ids)
        active_ids = tuple(
            sorted({uuv_id for members in members_by_target.values() for uuv_id in members})
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for regional_plan in regional_plans.values()
                for evidence_id in regional_plan.evidence_ids
            )
        )
        for target_id, regional_plan in sorted(regional_plans.items()):
            prediction = predictions[target_id]
            for evidence_id in regional_plan.evidence_ids:
                self.events.append_if_absent(
                    event_id=evidence_id,
                    event_type="deterministic_baseline_evidence",
                    scenario_id=situation.scenario_id,
                    sim_time_s=situation.sim_time_s,
                    target_id=target_id,
                    severity="info",
                    payload={
                        "evidence_id": evidence_id,
                        "source": (
                            "prediction"
                            if evidence_id == prediction.prediction_id
                            else "belief_history"
                        ),
                        "target_id": target_id,
                        "prediction_id": prediction.prediction_id,
                        "horizon_s": prediction.horizon_s,
                        "times_s": prediction.times_s,
                        "points_xy": prediction.points_xy,
                        "corridor_radius_m": prediction.corridor_radius_m,
                        "rationale": (
                            "deterministic executable baseline derived from the "
                            "current public target prediction"
                        ),
                    },
                )
        valid_until_s = max(
            (batch.exit_s for batch in plan.batches),
            default=situation.sim_time_s + self._config.timing.prediction_horizon_s,
        )
        audit_baseline = TrackingPlan(
            plan_id=f"{situation.scenario_id}:plan:{plan.revision}",
            scenario_id=situation.scenario_id,
            revision=plan.revision,
            base_snapshot_revision=situation.snapshot_revision,
            status="active",
            valid_from_s=situation.sim_time_s,
            valid_until_s=valid_until_s,
            concept="hold_current",
            target_priorities={target_id: 1.0 for target_id in regional_plans},
            required_quality={
                target_id: self._config.tracking.quality_warning
                for target_id in regional_plans
            },
            member_ids_by_target=members_by_target,
            roles_by_member=roles_by_member,
            prediction_refs={
                target_id: regional_plan.prediction_id
                for target_id, regional_plan in regional_plans.items()
            },
            active_uuv_ids=active_ids,
            standby_uuv_ids=tuple(sorted(standby_ids)),
            predicted_quality={
                target_id: self._config.tracking.quality_warning
                for target_id in regional_plans
            },
            predicted_active_count=len(active_ids),
            evidence_ids=evidence_ids,
            regional_plans=regional_plans,
        )
        active_audit = self.plans.get_active(situation.scenario_id)
        if active_audit is None:
            self.plans.set_snapshot_revision(
                situation.scenario_id,
                situation.snapshot_revision,
            )
            self.plans.commit(audit_baseline)
        elif active_audit.revision != plan.revision:
            raise RuntimeError(
                "deterministic baseline revision conflicts with the active audit plan"
            )
        self._baseline_regional_plans = regional_plans
        self._baseline_region_candidates = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        self._baseline_intent_hypotheses = dict(intents)
        self._last_deterministic_region_refresh_s = situation.sim_time_s
        execution = self._ensure_uuv_only_execution_snapshot(
            situation,
            prediction_state=seeded,
            audit_projection=audit_baseline,
        )
        if execution is None:
            raise RuntimeError("deterministic execution snapshot could not be committed")
        authoritative = execution_snapshot_to_mission_plan(execution)
        self.situation = situation
        self.publish_latest()
        return authoritative

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

    def _uuv_execution_admission(
        self,
        situation: SituationSnapshot,
        plan: ExecutableMissionPlan,
    ) -> str | None:
        """Keep an LLM commit from getting ahead of the executable snapshot."""
        del plan
        coordinator = getattr(self, "_execution_coordinator", None)
        if coordinator is None:
            return None
        active_audit = self.plans.get_active(situation.scenario_id)
        current_reader = getattr(coordinator, "active_mission_plan", None)
        current = current_reader() if callable(current_reader) else None
        if not isinstance(current, OperationalExecutionSnapshot):
            if active_audit is None:
                return None
            return "execution_snapshot_missing"
        hard_stale_s = float(self._config.tracking.prediction_health.hard_stale_s)
        health = coordinator.execution_health(
            sim_time_s=float(situation.sim_time_s),
            hard_stale_s=hard_stale_s,
        )
        if not health.executable:
            return "execution_snapshot_not_executable"
        if active_audit is not None and active_audit.revision != current.execution_revision:
            return "audit_execution_revision_mismatch"
        if not _has_current_public_execution_source(situation, current.target_id):
            return "execution_track_source_missing"
        return None

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

    def _mark_llm_failure(self, error: LLMError) -> None:
        """Record an unrecoverable provider failure for all runtime views."""
        if getattr(self, "_fatal_llm_error", None) is None:
            self._fatal_llm_error = error
        self.paused = True
        self._llm_failure_count = getattr(self, "_llm_failure_count", 0) + 1
        self._next_llm_retry_at = float("inf")
        self.reconnectable = False
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
            runtime._llm_reconnectable = False

    def raise_llm_failure(self, error: LLMError) -> NoReturn:
        """Stop execution and propagate the provider failure to the caller."""
        self._mark_llm_failure(error)
        raise error

    def raise_if_llm_failed(self) -> None:
        """Reject every subsequent execution attempt after a provider failure."""
        error = getattr(self, "_fatal_llm_error", None)
        if error is not None:
            raise error

    def mark_llm_paused(self, error: LLMError) -> NoReturn:
        """Compatibility name for the fatal LLM failure transition."""
        self.raise_llm_failure(error)

    def mark_llm_recovered(self) -> None:
        """Clear the operator-visible pause after a successful cycle."""
        if getattr(self, "_fatal_llm_error", None) is not None:
            return
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
            getattr(planning_config, "initial_plan_timeout_s", 900.0)
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
                max_speed_mps=config.tracking.submarine_sprint_speed_mps,
                max_turn_rate_rad_s=config.tracking.submarine_turn_rate_rad_s,
                health_config=config.tracking.prediction_health,
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
            uuv_only=_is_uuv_only_config(config),
            execution_hard_stale_s=config.tracking.prediction_health.hard_stale_s,
            retention=(agent.retention if agent is not None else RuntimeRetentionConfig()),
            current_snapshot_revision=self._current_snapshot_revision,
            memory_service=self._memory_service,
            short_term_repository=self._memory_short_term,
            memory_port=self._memory_port,
            planning_epoch_provider=lambda: self._active_epoch,
            epoch_commit_port=self._epoch_commit_port,
            world_model_config=(
                config.world_model
                if config.world_model is not None and config.world_model.enabled
                else None
            ),
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
            if self._llm_execution_required:
                if isinstance(exc, LLMError):
                    raise
                raise LLMConfigError(
                    "strict live memory provider initialization failed"
                ) from exc
            return MemoryService(
                self._memory_short_term,
                self._memory_long_term,
                DegradedMemoryRetriever(self._memory_degraded_reason),
                degraded_reason=self._memory_degraded_reason,
            )

    def _current_snapshot_revision(self) -> int:
        situation = self.situation
        return situation.snapshot_revision if situation is not None else 0

    def _role_model(self, role: str) -> str:
        llm_config = self._config.llm
        if llm_config is None:
            return "http"
        if llm_config.roles is not None:
            return llm_config.for_role(role).model
        return llm_config.model

    def provider_attestations(
        self, *, probe: bool = False
    ) -> tuple[dict[str, object], ...]:
        """Describe role clients and optionally perform a live structured probe.

        Static client identity proves only wiring.  The startup gate requests a
        tiny role-specific response so a configured-but-unreachable provider
        cannot be reported as usable.
        """
        llm_config = self._config.llm
        attestations: list[dict[str, object]] = []
        for role in ("master", "slave", "adversary"):
            client = self._clients.get(role)
            configured = (
                llm_config.for_role(role)
                if llm_config is not None and llm_config.roles is not None
                else None
            )
            exact_http_client = type(client) is RoleHTTPStructuredLLM
            matches_config = bool(
                configured is not None
                and exact_http_client
                and client.role == role
                and client._model == configured.model
                and client._base_url == configured.base_url
                and client.prompt_version == configured.prompt_version
            )
            endpoint = ""
            if configured is not None:
                parsed = urlsplit(configured.base_url)
                endpoint = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            probe_successful = self._provider_probe_successes.get(role, False)
            if probe and matches_config and client is not None and configured is not None:
                try:
                    client.invoke_structured(
                        f"provider_attestation_probe:{role}",
                        {
                            "role": role,
                            "probe_version": "provider-attestation-v1",
                            "scenario_id": self.scenario_id,
                        },
                        _ProviderAttestationProbeResponse,
                        prompt_version=configured.prompt_version,
                    )
                except LLMError:
                    # Preserve the typed provider failure so callers can stop
                    # the run and surface the exact connectivity/configuration
                    # contract instead of seeing a generic health flag.
                    self._provider_probe_successes[role] = False
                    raise
                except Exception as exc:  # noqa: BLE001 - keep the provider boundary typed
                    self._provider_probe_successes[role] = False
                    raise LLMError(
                        f"provider attestation failed for role {role!r}"
                    ) from exc
                else:
                    probe_successful = True
                self._provider_probe_successes[role] = probe_successful
            attestations.append(
                {
                    "role": role,
                    "model": configured.model if configured is not None else "",
                    "client_type": (
                        f"{type(client).__module__}.{type(client).__qualname__}"
                        if client is not None
                        else "missing"
                    ),
                    "transport": "httpx" if exact_http_client else "injected",
                    "configured_endpoint": endpoint,
                    "prompt_version": (
                        configured.prompt_version if configured is not None else ""
                    ),
                    "attested": bool(
                        not self._llm_injected and exact_http_client and matches_config
                    ),
                    "probe_successful": probe_successful,
                }
            )
        return tuple(attestations)

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

        The engine gives each graph a typed, truth-safe packet. Provider
        failures remain fatal in strict production execution, while a local
        boundary rejection is safely omitted for this cycle because applying
        it would be less safe than keeping the current sensor state.
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
        """Invoke both local brains over one captured physics boundary."""
        self._set_llm_sim_time(situation.sim_time_s)
        adversary_decisions: list[AdversaryIntentDecision | AdversaryEscapeDecision] = []
        slave_decisions: list[SlaveSonarDecision] = []
        # Both local roles are required for one complete algorithm cycle. A
        # provider failure is propagated instead of allowing a partial cycle.
        adversary_graph = getattr(self, "_adversary_graph", None)
        if adversary_graph is not None:
            for adversary_context in adversary_contexts:
                try:
                    previous_calls = self.ledger.list_llm_calls(
                        scenario_id=self.scenario_id,
                        operation="adversary_mission_decision",
                        limit=1,
                    )
                    previous_call_id = previous_calls[0].id if previous_calls else 0
                    result = adversary_graph.invoke({"context": adversary_context})
                    adversary_decision = result.get("decision")
                    if not isinstance(
                        adversary_decision,
                        (AdversaryIntentDecision, AdversaryEscapeDecision),
                    ):
                        raise TypeError("adversary graph returned no typed decision")
                    provider_calls = self.ledger.list_llm_calls(
                        scenario_id=self.scenario_id,
                        operation="adversary_mission_decision",
                        limit=8,
                    )
                    provider_call = next(
                        (
                            call
                            for call in provider_calls
                            if call.id > previous_call_id
                            and call.sim_time_s == situation.sim_time_s
                            and call.error_category == ""
                            and bool(call.request_hash)
                            and bool(call.response_hash)
                        ),
                        None,
                    )
                    decision_id = getattr(adversary_decision, "decision_id", None)
                    if provider_call is not None and isinstance(decision_id, str):
                        self._adversary_provider_call_ids[
                            decision_id
                        ] = f"LLM-{provider_call.id}"
                    adversary_decisions.append(adversary_decision)
                except LLMError as exc:
                    self.raise_llm_failure(exc)
                except Exception as exc:  # LLM semantic output is a content failure
                    if self._llm_execution_required:
                        raise LLMContentError(
                            "adversary LLM decision could not be validated: "
                            f"{exc}"
                        ) from exc
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
                except LLMError as exc:
                    self.raise_llm_failure(exc)
                except Exception as exc:  # LLM semantic output is a content failure
                    if isinstance(exc, SlaveDecisionValidationError):
                        self._record_carrier_error("slave_boundary_validation", exc)
                        continue
                    if self._llm_execution_required:
                        raise LLMContentError(
                            "slave LLM decision could not be validated: "
                            f"{exc}"
                        ) from exc
                    del exc
                    continue
        return tuple(slave_decisions), tuple(adversary_decisions)

    def _apply_adversary_decision(
        self,
        engine: Any,
        decision: AdversaryIntentDecision | AdversaryEscapeDecision,
    ) -> None:
        decision_id = getattr(decision, "decision_id", None)
        provider_call_id = (
            self._adversary_provider_call_ids.pop(decision_id, None)
            if isinstance(decision_id, str)
            else None
        )
        if provider_call_id is None:
            engine.apply_adversary_decision(decision)
            return
        engine.apply_adversary_decision(
            decision,
            provider_call_id=provider_call_id,
        )

    def _set_llm_sim_time(self, sim_time_s: int) -> None:
        """Advance observability metadata without changing decision inputs."""
        for client in getattr(self, "_clients", {}).values():
            setter = getattr(client, "set_simulation_time", None)
            if callable(setter):
                setter(sim_time_s)

    def _refresh_deterministic_mission(
        self,
        situation: SituationSnapshot,
        prediction_state: Mapping[str, Any],
    ) -> None:
        """Roll the executable region chain independently of provider latency."""
        if not _is_uuv_only_config(getattr(self, "_config", None)):
            return
        engine = self._engine
        runtime = self._runtime
        if engine is None or runtime is None:
            return
        mission_snapshot = engine.mission_snapshot()
        if mission_snapshot.plan_revision < 1:
            return
        execution_coordinator = getattr(self, "_execution_coordinator", None)
        if execution_coordinator is not None:
            if not execution_coordinator.rolling_check_due(situation.sim_time_s):
                return
            installed = self._ensure_uuv_only_execution_snapshot(
                situation,
                prediction_state=prediction_state,
            )
            if installed is not None:
                execution_coordinator.mark_rolling_check(situation.sim_time_s)
            return
        else:
            refresh_interval_s = max(
                self._config.timing.observation_step_s,
                self._config.timing.prediction_horizon_s // 4,
            )
            last_refresh_s = getattr(self, "_last_deterministic_region_refresh_s", 0)
            if situation.sim_time_s - last_refresh_s < refresh_interval_s:
                return
        predictions = {
            target_id: prediction
            for target_id, prediction in (
                prediction_state.get("predictions") or {}
            ).items()
            if isinstance(prediction, PredictedTrackRef)
        }
        if not predictions or situation.map_bounds_xy is None:
            return

        seeded = _prior_seeded_planning_inputs(situation)
        seeded_intents = seeded.get("intent_hypotheses", {})
        known_intents = dict(
            getattr(self, "_baseline_intent_hypotheses", {})
        )
        regional_plans: dict[str, TargetRegionPlan] = {}
        candidates: list[Any] = []
        for target_id, prediction in sorted(predictions.items()):
            intent = known_intents.get(target_id) or seeded_intents.get(target_id)
            if not isinstance(intent, IntentHypothesis):
                continue
            regional_plan = build_llm_task_region_plan(
                prediction,
                intent,
                _deterministic_region_proposals(
                    prediction.points_xy,
                    situation.map_bounds_xy,
                ),
                situation.map_bounds_xy,
                GridSpec(),
                required_quality=self._config.tracking.quality_warning,
            )
            regional_plans[target_id] = regional_plan
            candidates.extend(regional_plan_to_mission_candidates(regional_plan))
        if not candidates:
            return

        locked_uuv_ids = {
            region.region_id: (
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            )
            for region in mission_snapshot.regions
        }
        candidate_plan = MissionOptimizer(
            home_battle_group_id=self._config.scenario.home_battle_group_id,
            goal_mode=True,
        ).optimize(
            build_planning_snapshot(situation),
            tuple(candidates),
            locked_uuv_ids_by_candidate=locked_uuv_ids,
        )
        if not candidate_plan.batches:
            raise RuntimeError("rolling deterministic plan has no executable batch")
        next_revision = mission_snapshot.plan_revision + 1
        candidate_plan = candidate_plan.model_copy(
            update={
                "revision": next_revision,
                "region_assignments": tuple(
                    assignment.model_copy(
                        update={"plan_revision": next_revision}
                    )
                    for assignment in candidate_plan.region_assignments
                ),
            }
        )
        execution = self._ensure_uuv_only_execution_snapshot(
            situation,
            prediction_state=prediction_state,
            plan=candidate_plan,
        )
        if execution is None:
            return
        self._baseline_regional_plans = regional_plans
        self._baseline_region_candidates = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        self._last_deterministic_region_refresh_s = situation.sim_time_s
        if execution_coordinator is not None:
            execution_coordinator.mark_rolling_check(situation.sim_time_s)
        self.events.append_if_absent(
            event_id=(
                f"{situation.scenario_id}:deterministic-region-refresh:"
                f"{next_revision}:{situation.sim_time_s}"
            ),
            event_type="deterministic_region_plan_refreshed",
            scenario_id=situation.scenario_id,
            sim_time_s=situation.sim_time_s,
            target_id=next(iter(sorted(regional_plans))),
            severity="info",
            payload={
                "plan_revision": next_revision,
                "prediction_ids": {
                    target_id: plan.prediction_id
                    for target_id, plan in sorted(regional_plans.items())
                },
                "region_ids": tuple(
                    candidate.candidate_id for candidate in candidates
                ),
                "reason": "rolling_prediction_horizon",
                "active_plan_preserved_on_failure": True,
            },
        )

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
        self.raise_if_llm_failed()
        runtime = getattr(self, "_runtime", None)
        refresh_predictions = getattr(runtime, "refresh_predictions", None)
        prediction_state: Mapping[str, Any] = {}
        if callable(refresh_predictions):
            try:
                prediction_state = refresh_predictions(situation)
            except LLMError as exc:
                self.raise_llm_failure(exc)
            except Exception as exc:  # noqa: BLE001 - keep physics moving; fail audit
                self._record_carrier_error("prediction_refresh", exc)
        refresh_mission = getattr(self, "_refresh_deterministic_mission", None)
        if callable(refresh_mission):
            try:
                refresh_mission(situation, prediction_state)
            except LLMError as exc:
                self.raise_llm_failure(exc)
            except Exception as exc:  # noqa: BLE001 - preserve the installed mission
                self._record_carrier_error("deterministic_region_refresh", exc)
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
        present_event_ids = {event.event_id for event in events}
        rehydrated: list[RuntimeEvent] = []
        event_reader = getattr(self.events, "get", None)
        if callable(event_reader):
            for event_id in capture.epoch.critical_event_ids:
                if event_id in present_event_ids:
                    continue
                stored = event_reader(event_id)
                if stored is None:
                    continue
                try:
                    level = EventLevel(stored.severity)
                except (TypeError, ValueError):
                    level = EventLevel.INFORMATIONAL
                rehydrated.append(
                    RuntimeEvent(
                        event_id=stored.event_id,
                        scenario_id=stored.scenario_id,
                        sim_time_s=stored.sim_time_s,
                        event_type=stored.event_type,
                        entity_id=stored.target_id,
                        level=level,
                        audiences=stored.audiences,
                        payload=dict(stored.payload),
                    )
                )
        if rehydrated:
            runtime_requeue = getattr(runtime, "requeue_events", None)
            if callable(runtime_requeue):
                runtime_requeue(tuple(rehydrated))
            events.extend(rehydrated)
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
        execution_coordinator = getattr(self, "_execution_coordinator", None)
        base_execution_revision = (
            execution_coordinator.execution_revision
            if execution_coordinator is not None
            else 0
        )
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
            self._sync_reservation_projections(engine, runtime)
            if epoch is not None or getattr(self, "_epoch_coordinator", None) is None:
                runtime.submit_events(trigger_events)
                self._set_llm_sim_time(situation.sim_time_s)
                result = runtime.tick(epoch=epoch) if epoch is not None else runtime.tick()
            else:
                result = {"commit_status": None}
            self._finish_epoch(epoch, result)
            if result.get("commit_status") == "committed":
                committed_plan = _committed_epoch_plan(result)
                if _is_uuv_only_config(self._config):
                    if committed_plan is not None:
                        if execution_coordinator is None:
                            self._apply_uuv_only_mission_plan(committed_plan)
                        else:
                            self._ensure_uuv_only_execution_snapshot(
                                situation,
                                plan=committed_plan,
                                base_execution_revision=base_execution_revision,
                            )
                    elif execution_coordinator is None:
                        self._apply_uuv_only_mission_plan()
                else:
                    self._apply_new_commands()
            self._apply_verification_commands(result)
            for slave_decision in local_slave_decisions:
                engine.apply_slave_sonar_decision(slave_decision)
            for adversary_decision in adversary_decisions:
                self._apply_adversary_decision(engine, adversary_decision)
            self.mark_llm_recovered()
            if (
                _is_uuv_only_config(self._config)
                and execution_coordinator is not None
                and execution_coordinator.current is None
            ):
                self._ensure_uuv_only_execution_snapshot(situation)
        except LLMError as exc:
            self._finish_epoch(epoch, {}, exc)
            requeue_sensor_controls = getattr(runtime, "requeue_sensor_controls", None)
            if callable(requeue_sensor_controls):
                requeue_sensor_controls(sensor_controls)
            self.raise_llm_failure(exc)
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
                base_execution_revision=(
                    self._execution_coordinator.execution_revision
                    if self._execution_coordinator is not None
                    else 0
                ),
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
            self._mark_llm_failure(exc)
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

    def drain_background_cycle(self, *, timeout_s: float = 10.0) -> bool:
        """Apply all carrier work that was queued before finite-run shutdown."""
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if not getattr(self, "_background_carrier", False):
            return True
        deadline = time.monotonic() + timeout_s
        while True:
            self.apply_background_cycle()
            with self._carrier_cycle_lock:
                active_cycle = getattr(self, "_background_cycle", None)
                mailbox = getattr(self, "_background_mailbox", None)
                background_thread = getattr(self, "_background_thread", None)
                local_thread = getattr(self, "_background_local_thread", None)
                local_mailbox = getattr(self, "_background_local_mailbox", None)
                local_results = getattr(self, "_background_local_results", None)
                pending_local_results = bool(local_results)

            if active_cycle is None and mailbox is not None:
                self._start_background_cycle(mailbox)
                continue

            background_alive = bool(
                background_thread is not None
                and background_thread.is_alive()
            )
            local_alive = bool(local_thread is not None and local_thread.is_alive())
            if (
                active_cycle is None
                and mailbox is None
                and not background_alive
                and not local_alive
                and local_mailbox is None
                and not pending_local_results
            ):
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            thread = (
                background_thread
                if background_alive
                else local_thread
                if local_alive
                else None
            )
            if thread is not None:
                thread.join(timeout=min(0.05, remaining))
            else:
                time.sleep(min(0.01, remaining))

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
                if isinstance(cycle.local_error, LLMError):
                    self.raise_llm_failure(cycle.local_error)
                self._record_carrier_error(
                    "background_local_brain_cycle", cycle.local_error
                )
                continue
            for slave_decision in cycle.slave_decisions:
                engine.apply_slave_sonar_decision(slave_decision)
            for adversary_decision in cycle.adversary_decisions:
                self._apply_adversary_decision(engine, adversary_decision)

    def _apply_background_planning_phase(
        self, cycle: _BackgroundCarrierCycle
    ) -> None:
        """Commit one finished master phase without waiting for local LLMs."""
        if cycle.error is not None:
            self._finish_epoch(cycle.epoch, {}, cycle.error)
            if isinstance(cycle.error, LLMError):
                self.raise_llm_failure(cycle.error)
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
                if getattr(self, "_execution_coordinator", None) is None:
                    self._apply_uuv_only_mission_plan(committed_plan)
                else:
                    self._ensure_uuv_only_execution_snapshot(
                        cycle.situation,
                        plan=committed_plan,
                        base_execution_revision=cycle.base_execution_revision,
                    )
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
        self._sync_reservation_projections(engine, runtime)
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

    @staticmethod
    def _sync_reservation_projections(engine: Any, runtime: Any) -> None:
        """Synchronize ordinary and dedicated operator projections independently."""
        reservations = runtime.reservations()
        controller = getattr(engine, "_mission_controller", None)
        snapshot = controller.snapshot() if controller is not None else None
        processed = set(getattr(engine, "_dedicated_release_event_ids", ()))
        visible_event_ids: set[str] = set()
        for event in getattr(snapshot, "events", ()):
            if getattr(event, "event_type", None) != "dedicated_mode_released":
                continue
            event_id = getattr(event, "event_id", None)
            if event_id is None:
                event_id = (
                    f"{getattr(event, 'event_type', '')}:"
                    f"{getattr(event, 'entity_id', '')}:"
                    f"{getattr(event, 'sim_time_s', '')}"
                )
            visible_event_ids.add(event_id)
            if event_id in processed:
                continue
            processed.add(event_id)
            target_id = (getattr(event, "payload", {}) or {}).get("target_id")
            if not target_id or not reservations.dedicated_for(target_id):
                continue
            reservations.release_dedicated(target_id)
            runtime.submit_regional_replan(
                reason="endurance",
                entity_id=target_id,
                sim_time_s=getattr(event, "sim_time_s", 0),
                payload={"source": "dedicated_mode_released"},
            )
        engine._dedicated_release_event_ids = processed & visible_event_ids
        engine.set_reservations(reservations)
        dedicated_items = getattr(reservations, "dedicated_items", None)
        set_dedicated_groups = getattr(engine, "set_dedicated_tracking_groups", None)
        if callable(dedicated_items) and callable(set_dedicated_groups):
            set_dedicated_groups(dict(dedicated_items()))

    def _ensure_uuv_only_execution_snapshot(
        self,
        situation: SituationSnapshot,
        *,
        prediction_state: Mapping[str, Any] | None = None,
        plan: ExecutableMissionPlan | None = None,
        audit_projection: TrackingPlan | None = None,
        base_execution_revision: int | None = None,
    ) -> OperationalExecutionSnapshot | None:
        """Install the one authoritative execution snapshot for a UUV cycle."""
        if not _is_uuv_only_config(self._config):
            return None
        engine = self._engine
        runtime = self._runtime
        coordinator = getattr(self, "_execution_coordinator", None)
        if engine is None or runtime is None or coordinator is None:
            return None

        current_reader = getattr(coordinator, "active_mission_plan", None)
        current = current_reader() if callable(current_reader) else None
        if not isinstance(current, OperationalExecutionSnapshot):
            current = None

        hard_stale_s = float(self._config.tracking.prediction_health.hard_stale_s)

        def retain_current_after_source_gap() -> OperationalExecutionSnapshot | None:
            """Keep the last source window; never renew it from the current clock."""
            if current is None:
                return None
            health = coordinator.execution_health(
                sim_time_s=float(situation.sim_time_s),
                hard_stale_s=hard_stale_s,
            )
            if health.status == "expired":
                if "execution_target_track_hard_stale" not in health.reason_codes:
                    coordinator.mark_expired("execution_target_track_hard_stale")
                    self.publish_latest()
                return None
            if health.status == "failed":
                return None
            return current

        if plan is not None and current is not None:
            if not _has_current_public_execution_source(situation, current.target_id):
                return retain_current_after_source_gap()
            return self._commit_semantic_execution_snapshot(
                current,
                plan=plan,
                base_execution_revision=(
                    current.execution_revision
                    if base_execution_revision is None
                    else base_execution_revision
                ),
            )

        state = runtime.get_state()
        live_state = prediction_state or {}
        raw_accepted = (
            live_state.get("accepted_predictions")
            or state.get("accepted_predictions")
            or {}
        )
        accepted_predictions = {
            target_id: value
            for target_id, value in raw_accepted.items()
            if isinstance(value, AcceptedPrediction)
        }
        if not accepted_predictions:
            coordinator.mark_failed("accepted_prediction_missing")
            self.publish_latest()
            return None
        target_id = min(accepted_predictions)
        accepted = accepted_predictions[target_id]
        if accepted.prediction is None or accepted.health.status == "unavailable":
            coordinator.mark_failed("accepted_prediction_unavailable")
            self.publish_latest()
            return None
        report = next(
            (
                candidate
                for candidate in situation.group_reports
                if candidate.target_id == target_id
                and len(candidate.belief.mean) >= 2
                and candidate.belief.source_observation_ids
            ),
            None,
        )
        prior = next(
            (
                candidate
                for candidate in situation.target_search_priors
                if candidate.target_id == target_id
                and candidate.issued_at_s <= situation.sim_time_s < candidate.valid_until_s
            ),
            None,
        )
        if report is None and prior is None:
            if current is not None and current.target_id == target_id:
                return retain_current_after_source_gap()
            coordinator.mark_failed("execution_track_source_missing")
            self.publish_latest()
            return None
        raw_intents = state.get("deterministic_intents") or {}
        intent = raw_intents.get(target_id)
        if intent is None:
            intent = (state.get("intent_hypotheses") or {}).get(target_id)
        if intent is None:
            intent = self._baseline_intent_hypotheses.get(target_id)
        controller = getattr(engine, "_mission_controller", None)
        mission = controller.snapshot() if controller is not None else None
        if mission is None:
            coordinator.mark_failed("mission_snapshot_missing")
            self.publish_latest()
            return None
        map_bounds = getattr(situation, "map_bounds_xy", None)
        if map_bounds is None:
            coordinator.mark_failed("execution_map_bounds_missing")
            self.publish_latest()
            return None
        freshness_status = "fresh"
        if report is not None:
            raw_position = (
                float(report.belief.mean[0]),
                float(report.belief.mean[1]),
            )
            position = _project_public_track_xy(raw_position, map_bounds)
            if position is None:
                coordinator.mark_failed("execution_track_source_invalid")
                self.publish_latest()
                return None
            projected_history: list[tuple[int, float, float]] = []
            for sample in engine.belief_history(target_id):
                if len(sample) < 3:
                    continue
                sample_time = float(sample[0])
                sample_position = _project_public_track_xy(
                    (float(sample[1]), float(sample[2])),
                    map_bounds,
                )
                if sample_position is None or not isfinite(sample_time):
                    continue
                projected_history.append(
                    (int(sample_time), sample_position[0], sample_position[1])
                )
            history = tuple(projected_history)
            if not history:
                history = ((int(report.sim_time_s), *position),)
            latest_time = max(float(report.sim_time_s), float(history[-1][0]))
            velocity = (
                (float(report.belief.mean[2]), float(report.belief.mean[3]))
                if len(report.belief.mean) >= 4
                else (0.0, 0.0)
            )
            source_event_ids = tuple(report.belief.source_observation_ids)
        elif prior is not None:
            position = _project_public_track_xy(
                (float(prior.center_xy[0]), float(prior.center_xy[1])),
                map_bounds,
            )
            if position is None:
                coordinator.mark_failed("execution_track_source_invalid")
                self.publish_latest()
                return None
            history = (
                (
                    int(situation.sim_time_s),
                    *position,
                ),
            )
            latest_time = float(situation.sim_time_s)
            velocity = (0.0, 0.0)
            source_event_ids = (prior.prior_id,)
        target_track = GlobalTargetTrackView(
            target_id=target_id,
            track_revision=max(1, int(situation.snapshot_revision)),
            sim_time_s=latest_time,
            position_xy=position,
            velocity_xy=velocity,
            heading_rad=atan2(velocity[1], velocity[0]) if velocity != (0.0, 0.0) else 0.0,
            acceleration_xy=(0.0, 0.0),
            turn_rate_rad_s=0.0,
            bounded_history=history,
            source_event_ids=source_event_ids,
            freshness_status=freshness_status,
        )
        prediction_revision = live_state.get(
            "prediction_snapshot_revision",
            state.get("prediction_snapshot_revision", situation.snapshot_revision),
        )
        if not isinstance(prediction_revision, int):
            prediction_revision = situation.snapshot_revision
        execution_revision = (
            current.execution_revision + 1 if current is not None else 1
        )
        try:
            try:
                baseline = build_four_region_baseline(
                    accepted,
                    target_id=target_id,
                    execution_revision=execution_revision,
                    origin_sim_time_s=float(situation.sim_time_s),
                    map_bounds_xy=map_bounds,
                    prior_regions=(current.regions if current is not None else ()),
                )
            except ValueError as exc:
                if (
                    current is None
                    or str(exc) != "map bounds cannot retain a legal four-region partition"
                ):
                    raise
                if accepted.prediction is None:
                    raise ValueError("execution_region_identity_unbound")
                if accepted.prediction.prediction_id != current.prediction_id:
                    coordinator.mark_failed("execution_region_identity_unbound")
                    self.publish_latest()
                    return None
                baseline = _preserve_execution_regions_after_partition_failure(
                    accepted,
                    current=current,
                    target_id=target_id,
                    execution_revision=execution_revision,
                    origin_sim_time_s=float(situation.sim_time_s),
                    map_bounds=map_bounds,
                )
            snapshot = build_execution_snapshot(
                situation=situation,
                target_track=target_track,
                accepted_prediction=accepted,
                baseline=baseline,
                intent=intent,
                uuv_resources=mission.uuv_resources,
                execution_revision=execution_revision,
                prediction_revision=prediction_revision,
                previous=current,
                mission_regions=mission.regions,
                plan_source="deterministic",
            )
        except (TypeError, ValueError) as exc:
            coordinator.mark_failed(
                f"execution_snapshot_build:{type(exc).__name__}"
            )
            self._record_carrier_error("execution_snapshot_build", exc)
            self.publish_latest()
            return None

        for evidence_id in snapshot.evidence_ids:
            self.events.append_if_absent(
                event_id=evidence_id,
                event_type="execution_snapshot_evidence",
                scenario_id=situation.scenario_id,
                sim_time_s=situation.sim_time_s,
                target_id=target_id,
                severity="info",
                payload={
                    "execution_revision": snapshot.execution_revision,
                    "prediction_id": snapshot.prediction_id,
                    "source": "uuv_only_execution_snapshot",
                },
            )

        audit = audit_projection
        if audit is None:
            active_audit = self.plans.get_active(situation.scenario_id)
            audit = active_audit
        if audit is not None:
            existing_audit = self.plans.get_plan(audit.plan_id)
            audit = audit.model_copy(
                update={
                    "plan_id": (
                        f"{situation.scenario_id}:execution-audit:"
                        f"{snapshot.execution_revision}:{situation.sim_time_s}"
                        if existing_audit is not None
                        else audit.plan_id
                    ),
                    "revision": snapshot.execution_revision,
                    "base_snapshot_revision": situation.snapshot_revision,
                    "valid_from_s": int(snapshot.valid_from_s),
                    "valid_until_s": int(snapshot.valid_until_s),
                    "prediction_refs": {target_id: snapshot.prediction_id},
                    "intent_refs": {
                        target_id: f"intent:{snapshot.intent_revision}"
                    },
                    "active_uuv_ids": tuple(
                        sorted(
                            {
                                member
                                for group in snapshot.task_groups
                                for member in group.member_uuv_ids
                            }
                        )
                    ),
                    "standby_uuv_ids": tuple(
                        sorted(reserve.uuv_id for reserve in snapshot.reserve_uuvs)
                    ),
                    "evidence_ids": snapshot.evidence_ids,
                    "predicted_active_count": len(
                        {
                            member
                            for group in snapshot.task_groups
                            for member in group.member_uuv_ids
                        }
                    ),
                }
            )

        result = coordinator.commit_baseline_then_optimize(
            snapshot,
            apply=self._apply_execution_snapshot_or_raise,
            audit_projection=audit,
            publish=lambda _staged: self.publish_latest(),
        )
        if not result.committed or result.snapshot is None:
            coordinator.mark_failed(
                result.reason or "execution_snapshot_commit_failed"
            )
            self._record_carrier_error(
                "execution_snapshot_commit",
                RuntimeError(result.reason or "execution snapshot was preserved"),
            )
            self.publish_latest()
            return None
        installed = execution_snapshot_to_mission_plan(
            result.snapshot,
            current_region_lifecycles=_current_mission_lifecycles(self._engine),
        )
        runtime.install_executable_baseline(installed)
        self._last_mission_revision = max(
            self._last_mission_revision,
            installed.revision,
        )
        return result.snapshot

    def _commit_semantic_execution_snapshot(
        self,
        baseline: OperationalExecutionSnapshot,
        *,
        plan: ExecutableMissionPlan,
        base_execution_revision: int,
    ) -> OperationalExecutionSnapshot:
        """Apply only semantic LLM policy through the baseline revision CAS."""

        coordinator = self._execution_coordinator
        engine = self._engine
        runtime = self._runtime
        if coordinator is None or engine is None or runtime is None:
            return baseline
        revision = baseline.execution_revision + 1
        semantic_evidence = baseline.evidence_ids
        controller = getattr(engine, "_mission_controller", None)
        snapshot_reader = getattr(controller, "snapshot", None)
        mission_snapshot = snapshot_reader() if callable(snapshot_reader) else None
        mission_regions = getattr(mission_snapshot, "regions", ())
        merged_regions = _merge_authoritative_region_lifecycles(
            baseline.regions,
            mission_regions,
        )
        regions_by_id = {region.region_id: region for region in merged_regions}
        merged_groups = tuple(
            group.model_copy(
                update={
                    "status": execution_group_status(
                        regions_by_id[group.region_id].status
                    )
                }
            )
            for group in baseline.task_groups
        )
        current_region_id, next_region_id = _semantic_execution_cursor(
            merged_regions,
            fallback_region_id=baseline.current_region_id,
        )
        candidate = baseline.model_copy(
            deep=True,
            update={
                "execution_revision": revision,
                "base_execution_revision": base_execution_revision,
                "plan_source": "llm_optimized",
                "regions": tuple(
                    region.model_copy(
                        update={
                            "execution_revision": revision,
                            "evidence_ids": region.evidence_ids,
                        }
                    )
                    for region in merged_regions
                ),
                "current_region_id": current_region_id,
                "next_region_id": next_region_id,
                "task_groups": tuple(
                    group.model_copy(
                        update={
                            "execution_revision": revision,
                            "evidence_ids": group.evidence_ids,
                        }
                    )
                    for group in merged_groups
                ),
                "evidence_ids": semantic_evidence,
            },
        )
        result = coordinator.commit_semantic_optimization(
            candidate,
            base_execution_revision=base_execution_revision,
            apply=self._apply_execution_snapshot_or_raise,
            publish=lambda _staged: self.publish_latest(),
        )
        if not result.committed or result.snapshot is None:
            return baseline
        installed = execution_snapshot_to_mission_plan(
            result.snapshot,
            current_region_lifecycles=_current_mission_lifecycles(engine),
        )
        runtime.install_executable_baseline(installed)
        self._last_mission_revision = max(
            self._last_mission_revision,
            installed.revision,
        )
        return result.snapshot

    def _apply_execution_snapshot_or_raise(
        self, snapshot: OperationalExecutionSnapshot
    ) -> bool:
        """Preserve the engine's concrete rejection reason at the coordinator boundary."""
        engine = self._engine
        if engine is None:
            raise RuntimeError("engine_missing")
        snapshot_applier = getattr(engine, "apply_verified_execution_snapshot", None)
        if callable(snapshot_applier):
            applied = snapshot_applier(snapshot)
        else:
            applied = engine.apply_verified_mission_plan(
                execution_snapshot_to_mission_plan(
                    snapshot,
                    current_region_lifecycles=_current_mission_lifecycles(engine),
                )
            )
        if applied is False:
            reason = getattr(engine, "_last_mission_plan_failure_reason", None)
            detail = "engine_apply_rejected"
            if reason:
                detail += f":{reason}"
            raise RuntimeError(detail)
        return True

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
        if not applied:
            failure_reason = getattr(
                engine, "_last_mission_plan_failure_reason", None
            )
            retryable = getattr(
                engine, "mission_plan_application_is_retryable", None
            )
            if callable(retryable) and retryable(plan):
                defer = getattr(engine, "defer_mission_plan_application", None)
                if callable(defer):
                    defer(plan, failure_reason or "transient_resource_state_change")
                return False
            # A committed executable plan that cannot be installed physically
            # is a terminal execution error. Continuing would expose the
            # controller's logical modes while all vehicles remain onboard.
            self._record_carrier_error(
                f"apply_uuv_only_mission_plan_failed:revision={plan.revision}"
                + (f":{failure_reason}" if failure_reason else "")
            )
            raise RuntimeError(
                "verified UUV mission plan could not be applied physically "
                f"(revision={plan.revision})"
            )
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
        worker_llm = getattr(self, "_memory_worker_llm", None)
        cancel = getattr(worker_llm, "cancel", None)
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
        worker_llm = getattr(self, "_memory_worker_llm", None)
        cancel = getattr(worker_llm, "cancel", None)
        if callable(cancel):
            cancel()
        remaining_resources: list[str] = []
        if self._memory_worker is not None:
            if not self._memory_worker.stop(
                timeout=max(0.0, deadline - time.monotonic())
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
        close_resource(
            getattr(getattr(self, "_engine", None), "logger", None),
            "engine-frame-log",
        )
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
