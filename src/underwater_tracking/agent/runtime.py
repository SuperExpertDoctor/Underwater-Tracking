# src/underwater_tracking/agent/runtime.py
"""Persistent scenario runtime owning the carrier graph (spec 8.4, plan Task 8).

``CarrierRuntime`` owns one scenario thread: the SQLite checkpointer, the
payload store, and the scenario ``thread_id``. Events enter through
``submit_event``, each ``tick`` advances the clock and runs one graph cycle
over the pending events, ``resume`` runs one cycle without advancing the
clock (continue after a reopen), and ``get_state`` returns the latest
checkpointed state. Expert directives (spec 10.1, plan Task 10) enter
through the non-blocking pair ``preview_directive``/``apply_directive``:
parsing and confirmation never invoke the graph, and an applied directive
only queues a strategic event for the next cycle, leaving the current plan
active until that cycle commits. Expert questions (spec 10.2, plan Task 11)
enter through the read-only ``ask``: the answer is served from the
immutable planning snapshot with bounded evidence, an optional
counterfactual is solved as an isolated ``dry-run:`` plan over a cloned
snapshot (never touching the online plan), and the run is persisted under a
deterministic run id with a ``question`` event queued for the next cycle.
Graph internals are never exposed; the injected repositories stay
caller-owned and are closed by the caller.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    REGIONAL_REPLAN_EVENT_TYPES,
    TrajectoryPredictionNode,
    assess_regional_replan_events,
    build_carrier_graph,
    live_situation_ref,
)
from underwater_tracking.agent.llm import LLMError
from underwater_tracking.agent.nodes.directives import (
    DIRECTIVE_APPLIED_EVENT_TYPE,
    DIRECTIVE_OPERATION,
    DirectiveNotApplicableError,
    assign_target_uuvs,
    build_directive_payload,
    freeze_dedicated_tracking_members,
    validate_directive,
)
from underwater_tracking.agent.nodes.conversation import (
    ConversationContext,
    process_conversation_message,
)
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.questions import (
    QUESTION_EVENT_TYPE,
    QuestionAnswer,
    answer_question,
    question_run_id,
)
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.prompts import DIRECTIVE_PROMPT_VERSION, canonical_digest
from underwater_tracking.agent.state import RegionalReplanReason
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.conversation_models import (
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.models import (
    EventLevel,
    IntelligenceReport,
    OperationalScheme,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.execution_models import (
    ExecutionContextRef,
    OperationalExecutionSnapshot,
)
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import PlanningEpoch
from underwater_tracking.persistence.checkpoints import create_checkpointer
from underwater_tracking.persistence.payloads import RuntimePayloadStore
from underwater_tracking.planning.reservations import ReservationRegistry
from underwater_tracking.world_model.adapter import build_world_model_forecasts
from underwater_tracking.runtime.execution_evidence import (
    ExecutionEvidenceResolver,
    answer_execution_question,
)


@dataclass(frozen=True, slots=True)
class SensorModeControl:
    """One operator sensor-mode write applied at the next engine boundary."""

    uuv_id: str
    mode: Literal["passive", "active"]
    target_id: str | None
    requested_at_s: int


_LIVE_PREDICTION_KEYS = (
    "predictions",
    "accepted_predictions",
    "prediction_diffs",
    "prediction_diff_gates",
    "world_model_forecasts",
    "prediction_snapshot_revision",
    "prediction_intent_verification_target_ids",
    "prediction_intent_confirmed",
    "intent_hypotheses",
    "llm_provenance",
    "confirmed_intent_labels",
)


class CarrierRuntime:
    """One scenario thread over the persistent carrier central graph."""

    def __init__(
        self,
        dependencies: CarrierDependencies,
        *,
        scenario_id: str,
        database_path: str | Path,
        thread_id: str | None = None,
        execution_coordinator: object | None = None,
    ) -> None:
        reservations = (
            dependencies.reservations
            if dependencies.reservations is not None
            else ReservationRegistry()
        )
        dependencies = replace(dependencies, reservations=reservations)
        if dependencies.monitor is None:
            dependencies = replace(
                dependencies,
                monitor=EventMonitor(
                    scenario_id=scenario_id,
                    critical_hold_s=dependencies.critical_hold_s,
                    target_lost_gap_s=dependencies.target_lost_gap_s,
                    covariance_cap_m2=dependencies.covariance_cap_m2,
                ),
            )
        if dependencies.prediction_intent_monitor is None:
            dependencies = replace(
                dependencies,
                prediction_intent_monitor=EventMonitor(
                    scenario_id=scenario_id,
                    intent_confirmation=dependencies.intent_change_confirmation,
                ),
            )
        self._dependencies = dependencies
        self._reservations = reservations
        self._scenario_id = scenario_id
        self._execution_coordinator = execution_coordinator
        retention = dependencies.retention
        self._checkpointer = create_checkpointer(
            database_path,
            max_checkpoints=retention.group_checkpoint_limit,
        )
        self._payload_store = RuntimePayloadStore(
            str(database_path),
            owner=scenario_id,
            cache_limit=retention.payload_cache_limit,
            database_limit=retention.payload_db_limit,
        )
        self._graph = build_carrier_graph(
            dependencies, self._checkpointer, self._payload_store
        )
        self._thread_id = thread_id if thread_id is not None else f"{scenario_id}:carrier"
        self._config: dict[str, Any] = {"configurable": {"thread_id": self._thread_id}}
        self._pending: list[RuntimeEvent] = []
        self._processed_event_ids: set[str] = set()
        self._processed_event_order: deque[str] = deque()
        self._pending_scheme: OperationalScheme | None = None
        self._pending_intelligence: dict[str, IntelligenceReport] = {}
        self._pending_sensor_controls: list[SensorModeControl] = []
        self._regional_replan_latches: set[tuple[str, str | None]] = set()
        self._lock = RLock()
        self._assistant_lock = RLock()
        self._simulation_time_provider: Callable[[], int] | None = None
        self._llm_paused = False
        self._llm_pause_reason: str | None = None
        self._conversation_turns: dict[tuple[str, str], ConversationTurnResult] = {}
        self._llm_reconnectable = False
        self._llm_failure: LLMError | None = None
        self._llm_degraded_event_times: set[int] = set()
        self._llm_degraded_event_order: deque[int] = deque()
        self._cycle_running = False
        self._state_cache: dict[str, Any] = {}
        self._baseline_executable_mission_plan: ExecutableMissionPlan | None = None
        self._live_prediction_lock = RLock()
        self._live_prediction_state: dict[str, Any] = {}
        self._live_prediction_events: tuple[RuntimeEvent, ...] = ()
        self._live_prediction_event_ids: set[str] = set()
        self._live_prediction_pending_events: deque[RuntimeEvent] = deque()
        self._live_prediction_snapshot_revision = -1
        self._closed = False
        self._pre_close_hooks: list[Callable[[], None]] = []

    def submit_event(
        self,
        *,
        event_type: str,
        entity_id: str | None,
        sim_time_s: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue one event for the next graph cycle (re-classified by the monitor)."""
        self.submit_events(
            (
                RuntimeEvent(
                    event_id=(
                        f"{self._scenario_id}:{event_type}:{entity_id or 'carrier'}:{sim_time_s}"
                    ),
                    scenario_id=self._scenario_id,
                    sim_time_s=sim_time_s,
                    event_type=event_type,
                    entity_id=entity_id,
                    level=EventLevel.INFORMATIONAL,
                    payload=payload or {},
                ),
            )
        )

    def submit_events(self, events: Sequence[RuntimeEvent]) -> None:
        """Queue source events once each, preserving their stable IDs and payloads."""
        with self._lock:
            pending_ids = {event.event_id for event in self._pending}
            for event in events:
                if event.event_id in pending_ids or event.event_id in self._processed_event_ids:
                    continue
                self._pending.append(event)
                pending_ids.add(event.event_id)
            limit = self._event_history_limit()
            if len(self._pending) > limit:
                del self._pending[:-limit]

    def requeue_events(self, events: Sequence[RuntimeEvent]) -> None:
        """Requeue durable planning triggers after an epoch is retried.

        A failed or invalidated epoch has already consumed its source event in
        the graph runtime.  The planning coordinator can later retry that same
        event, so the processed-event deduplication set must be released for
        this explicit replay only.
        """
        with self._lock:
            for event in events:
                self._processed_event_ids.discard(event.event_id)
            self.submit_events(events)

    def pending_events(self) -> tuple[RuntimeEvent, ...]:
        """Return source events waiting for the next graph cycle."""
        with self._lock:
            pending = tuple(self._pending)
        with self._live_prediction_lock:
            live_pending = tuple(self._live_prediction_pending_events)
        return (*pending, *live_pending)

    def refresh_predictions(self, situation: SituationSnapshot) -> dict[str, Any]:
        """Refresh deterministic prediction evidence without taking the graph lock.

        The physical loop can continue while a real provider call owns the
        graph lock.  New suspicion events are persisted immediately and kept in
        a separate mailbox; the next graph cycle consumes that mailbox and
        performs the real intent verification against the same typed diff.
        """
        if situation.scenario_id != self._scenario_id:
            raise ValueError(
                f"prediction situation belongs to {situation.scenario_id!r}, "
                f"not {self._scenario_id!r}"
            )
        with self._live_prediction_lock:
            if situation.snapshot_revision <= self._live_prediction_snapshot_revision:
                return dict(self._live_prediction_state)
            checkpoint = dict(getattr(self, "_state_cache", {}))
            previous = {
                key: self._live_prediction_state.get(key, checkpoint.get(key))
                for key in _LIVE_PREDICTION_KEYS
            }
            existing_events = self._live_prediction_events
            if not existing_events:
                existing_events = tuple(checkpoint.get("coalesced_events") or ())
            node = TrajectoryPredictionNode(
                self._dependencies.predictor,
                lambda _ref: situation,
                uuv_only=bool(getattr(self._dependencies, "uuv_only", False)),
                diff_config=getattr(
                    self._dependencies,
                    "trajectory_diff_config",
                    None,
                ),
            )
            result = node(
                {
                    "scenario_id": self._scenario_id,
                    "uuv_only": bool(getattr(self._dependencies, "uuv_only", False)),
                    "snapshot_ref": live_situation_ref(self._scenario_id),
                    "predictions": previous.get("predictions") or {},
                    "accepted_predictions": previous.get("accepted_predictions") or {},
                    "prediction_diffs": previous.get("prediction_diffs") or {},
                    "prediction_diff_gates": previous.get("prediction_diff_gates")
                    or {},
                    "prediction_snapshot_revision": previous.get(
                        "prediction_snapshot_revision"
                    ),
                    "prediction_intent_verification_target_ids": previous.get(
                        "prediction_intent_verification_target_ids"
                    )
                    or (),
                    "prediction_intent_confirmed": bool(
                        previous.get("prediction_intent_confirmed")
                    ),
                    "coalesced_events": existing_events,
                }
            )
            target_ids = {report.target_id for report in situation.group_reports}
            pending = tuple(
                result.get("prediction_intent_verification_target_ids") or ()
            )
            if not target_ids:
                pending = tuple(
                    sorted(
                        set(pending)
                        | {
                            target_id
                            for target_id, gate in (
                                result.get("prediction_diff_gates") or {}
                            ).items()
                            if gate.verification_pending
                        }
                        | set(
                            previous.get(
                                "prediction_intent_verification_target_ids"
                            )
                            or ()
                        )
                    )
                )
            result["prediction_intent_verification_target_ids"] = pending
            result["prediction_intent_confirmed"] = False
            result["prediction_snapshot_revision"] = situation.snapshot_revision
            world_model_config = getattr(self._dependencies, "world_model_config", None)
            if world_model_config is not None and world_model_config.enabled:
                result["world_model_forecasts"] = build_world_model_forecasts(
                    situation,
                    result.get("predictions") or {},
                    config=world_model_config,
                    active_plan=self._dependencies.plans.get_active(self._scenario_id),
                )
            else:
                result["world_model_forecasts"] = {}
            result_events = tuple(result.get("coalesced_events") or existing_events)
            existing_event_ids = {event.event_id for event in existing_events}
            new_events = tuple(
                event for event in result_events if event.event_id not in existing_event_ids
            )
            self._live_prediction_state = {
                key: result[key] for key in _LIVE_PREDICTION_KEYS if key in result
            }
            self._live_prediction_events = result_events
            self._live_prediction_event_ids.update(event.event_id for event in result_events)
            self._live_prediction_snapshot_revision = situation.snapshot_revision
            for event in new_events:
                append_if_absent = self._dependencies.events.append_if_absent
                append_if_absent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    scenario_id=event.scenario_id,
                    sim_time_s=event.sim_time_s,
                    payload=event.payload,
                    target_id=event.entity_id,
                    severity=event.level.value,
                    audiences=event.audiences,
                )
                self._live_prediction_pending_events.append(event)
            return dict(self._live_prediction_state)

    def _drain_live_prediction_events(self) -> None:
        """Move live prediction triggers into the next graph input mailbox."""
        lock = getattr(self, "_live_prediction_lock", None)
        if lock is None:
            return
        with lock:
            events = tuple(self._live_prediction_pending_events)
            self._live_prediction_pending_events.clear()
        pending_ids = {event.event_id for event in self._pending}
        for event in events:
            if event.event_id not in pending_ids:
                self._pending.append(event)
                pending_ids.add(event.event_id)

    def _live_prediction_fragment(self) -> dict[str, Any]:
        lock = getattr(self, "_live_prediction_lock", None)
        if lock is None:
            return {}
        with lock:
            return dict(self._live_prediction_state)

    def _merge_live_prediction_state(self, state: dict[str, Any]) -> dict[str, Any]:
        lock = getattr(self, "_live_prediction_lock", None)
        if lock is None:
            return state
        with lock:
            live_revision = self._live_prediction_snapshot_revision
            state_revision = int(state.get("prediction_snapshot_revision", -1))
            if live_revision < state_revision:
                return state
            for key in _LIVE_PREDICTION_KEYS:
                if key in self._live_prediction_state:
                    state[key] = self._live_prediction_state[key]
        return state

    def _capture_graph_prediction_state(self, state: Mapping[str, Any]) -> None:
        if "predictions" not in state:
            return
        graph_revision = int(
            state.get("prediction_snapshot_revision", state.get("snapshot_revision", -1))
        )
        lock = getattr(self, "_live_prediction_lock", None)
        if lock is None:
            return
        with lock:
            if graph_revision < self._live_prediction_snapshot_revision:
                return
            captured_state = {
                key: state[key] for key in _LIVE_PREDICTION_KEYS if key in state
            }
            if "accepted_predictions" not in state:
                accepted_predictions = self._matching_accepted_predictions(
                    self._live_prediction_state.get("accepted_predictions"),
                    state.get("predictions"),
                )
                if accepted_predictions:
                    captured_state["accepted_predictions"] = accepted_predictions
            self._live_prediction_state = captured_state
            self._live_prediction_snapshot_revision = graph_revision
            graph_events = tuple(state.get("coalesced_events") or ())
            if graph_events:
                merged_events = {
                    event.event_id: event for event in self._live_prediction_events
                }
                merged_events.update({event.event_id: event for event in graph_events})
                self._live_prediction_events = tuple(merged_events.values())
                self._live_prediction_event_ids.update(merged_events)

    @staticmethod
    def _matching_accepted_predictions(
        accepted_predictions: object,
        predictions: object,
    ) -> dict[str, Any]:
        """Keep accepted health only when it still describes public geometry."""
        if not isinstance(accepted_predictions, Mapping):
            return {}
        if not isinstance(predictions, Mapping):
            return {}
        return {
            target_id: accepted
            for target_id, accepted in accepted_predictions.items()
            if target_id in predictions
            and getattr(accepted, "prediction", None) == predictions[target_id]
        }

    def _event_history_limit(self) -> int:
        """Read the configured event bound, including for lightweight test doubles."""
        dependencies = getattr(self, "_dependencies", None)
        retention = getattr(dependencies, "retention", None)
        return int(getattr(retention, "event_history_limit", 2048))

    def submit_regional_replan(
        self,
        *,
        reason: RegionalReplanReason,
        entity_id: str | None,
        sim_time_s: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue an explicit regional-policy invalidation for the next cycle."""
        self.submit_event(
            event_type=REGIONAL_REPLAN_EVENT_TYPES[reason],
            entity_id=entity_id,
            sim_time_s=sim_time_s,
            payload=payload,
        )

    def submit_sensor_mode(
        self,
        *,
        uuv_id: str,
        mode: Literal["passive", "active"],
        target_id: str | None,
        expected_plan_version: int,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> None:
        """Queue a direct UUV sonar control without bypassing the graph audit."""
        with self._lock:
            self._validate_execution_context(execution_revision, frame_id)
            current_version = self.current_plan_version()
            if current_version != expected_plan_version:
                raise ValueError(
                    f"the operational plan changed; expected {expected_plan_version}, "
                    f"current {current_version}"
                )
            situation = self._dependencies.situation_provider(
                live_situation_ref(self._scenario_id)
            )
            uuv = next((item for item in situation.uuvs if item.uuv_id == uuv_id), None)
            if uuv is None:
                raise ValueError(f"unknown UUV {uuv_id!r}")
            if mode == "active" and not uuv.capability.active_sonar_available:
                raise ValueError(f"UUV {uuv_id!r} does not support active sonar")
            if mode == "active":
                known_targets = {
                    report.target_id for report in situation.group_reports
                } | {contact.contact_id for contact in situation.contacts}
                if target_id is None or target_id not in known_targets:
                    raise ValueError("active sonar control requires a known target id")
            control = SensorModeControl(
                uuv_id=uuv_id,
                mode=mode,
                target_id=target_id if mode == "active" else None,
                requested_at_s=self.current_sim_time_s(),
            )
            self._pending_sensor_controls = [
                item for item in self._pending_sensor_controls if item.uuv_id != uuv_id
            ]
            self._pending_sensor_controls.append(control)
            self.submit_event(
                event_type="manual_sensor_mode",
                entity_id=uuv_id,
                sim_time_s=control.requested_at_s,
                payload={
                    "uuv_id": uuv_id,
                    "mode": mode,
                    "target_id": control.target_id,
                    "passive_continuous": True,
                    "source": "operator",
                },
            )

    def drain_sensor_controls(self) -> tuple[SensorModeControl, ...]:
        """Move queued direct controls to the simulation thread."""
        with self._lock:
            controls = tuple(self._pending_sensor_controls)
            self._pending_sensor_controls.clear()
            return controls

    def requeue_sensor_controls(self, controls: Sequence[SensorModeControl]) -> None:
        """Restore controls when the surrounding engine cycle rolls back."""
        with self._lock:
            existing = {item.uuv_id for item in self._pending_sensor_controls}
            self._pending_sensor_controls.extend(
                item for item in controls if item.uuv_id not in existing
            )

    def set_operational_scheme(self, scheme: OperationalScheme) -> None:
        """Queue a replacement scheme for the next simulation snapshot."""
        with self._lock:
            current_sim_time_s = self.current_sim_time_s()
            if scheme.valid_until_s <= current_sim_time_s:
                raise ValueError(
                    "operational scheme is already expired at simulation time "
                    f"{current_sim_time_s}"
                )
            self._pending_scheme = scheme

    def submit_intelligence(self, report: IntelligenceReport) -> None:
        """Queue one intelligence report for the next simulation snapshot."""
        with self._lock:
            current_sim_time_s = self.current_sim_time_s()
            if report.valid_until_s <= current_sim_time_s:
                raise ValueError(
                    "intelligence report is already expired at simulation time "
                    f"{current_sim_time_s}"
                )
            self._pending_intelligence[report.report_id] = report

    def bind_simulation_time(self, current_sim_time_s: Callable[[], int]) -> None:
        """Use the engine clock as the authoritative input-validation clock."""
        with self._lock:
            self._simulation_time_provider = current_sim_time_s

    def current_sim_time_s(self) -> int:
        """Return the current simulation time used by adaptive input validation."""
        provider = getattr(self, "_simulation_time_provider", None)
        if provider is not None:
            return int(provider())
        return int(self._dependencies.clock.sim_time_s)

    @property
    def llm_paused(self) -> bool:
        """Whether the last graph cycle stopped on a real LLM failure."""
        if getattr(self, "_cycle_running", False):
            return self._llm_paused
        with self._lock:
            return self._llm_paused

    @property
    def llm_pause_reason(self) -> str | None:
        """Operator-facing reason for the current LLM pause, without payloads."""
        if getattr(self, "_cycle_running", False):
            return self._llm_pause_reason
        with self._lock:
            return self._llm_pause_reason

    @property
    def llm_reconnectable(self) -> bool:
        """Whether the paused LLM cycle is scheduled for a retry."""
        if getattr(self, "_cycle_running", False):
            return self._llm_reconnectable
        with self._lock:
            return self._llm_reconnectable

    def commit_operational_inputs(
        self,
        *,
        current_sim_time_s: int,
        apply_scheme: Callable[[OperationalScheme], None],
        apply_intelligence: Callable[[IntelligenceReport], None],
    ) -> None:
        """Submit all queued adaptive inputs at one engine simulation boundary.

        Pending inputs are cleared only after every engine callback succeeds.
        Boundary-time validation happens before the first callback, so an
        expired report cannot leave a scheme partially applied.
        """
        with self._lock:
            scheme = self._pending_scheme
            reports = tuple(
                report for _, report in sorted(self._pending_intelligence.items())
            )
            if scheme is not None and scheme.valid_until_s <= current_sim_time_s:
                raise ValueError(
                    "operational scheme is already expired at simulation time "
                    f"{current_sim_time_s}"
                )
            for report in reports:
                if report.valid_until_s <= current_sim_time_s:
                    raise ValueError(
                        "intelligence report is already expired at simulation time "
                        f"{current_sim_time_s}"
                    )

            if scheme is not None:
                apply_scheme(scheme)
            for report in reports:
                apply_intelligence(report)
            self._pending_scheme = None
            self._pending_intelligence.clear()

    def drain_operational_inputs(
        self,
    ) -> tuple[OperationalScheme | None, tuple[IntelligenceReport, ...]]:
        """Move queued human inputs to the simulation thread at a cycle boundary."""
        with self._lock:
            scheme = self._pending_scheme
            reports = tuple(
                report for _, report in sorted(self._pending_intelligence.items())
            )
            self._pending_scheme = None
            self._pending_intelligence.clear()
            return scheme, reports

    def tick(self, *, epoch: PlanningEpoch | None = None) -> dict[str, Any]:
        """Advance the clock and run one graph cycle over pending events.

        A real provider failure is terminal: the carrier clock returns to its
        pre-cycle value, pending events remain queued for diagnosis, and no
        later cycle is allowed to run on this runtime instance.
        """
        self._raise_if_llm_failed()
        self._cycle_running = True
        try:
            with self._lock:
                previous_time_s = self._dependencies.clock.sim_time_s
                self._dependencies.clock.tick()
                try:
                    result = self._run_cycle(epoch=epoch)
                except LLMError as exc:
                    self._dependencies.clock.sim_time_s = previous_time_s
                    self._llm_failure = exc
                    self._llm_paused = True
                    self._llm_pause_reason = str(exc)
                    self._llm_reconnectable = False
                    self._queue_llm_degraded(previous_time_s, str(exc))
                    raise
                self._llm_paused = False
                self._llm_pause_reason = None
                return result
        finally:
            self._cycle_running = False

    def resume(self, *, epoch: PlanningEpoch | None = None) -> dict[str, Any]:
        """Retry one pending cycle without advancing the carrier clock."""
        self._raise_if_llm_failed()
        self._cycle_running = True
        try:
            with self._lock:
                result = self._run_cycle(epoch=epoch)
                self._llm_paused = False
                self._llm_pause_reason = None
                return result
        except LLMError as exc:
            self._llm_failure = exc
            self._llm_paused = True
            self._llm_pause_reason = str(exc)
            self._llm_reconnectable = False
            self._queue_llm_degraded(self._dependencies.clock.sim_time_s, str(exc))
            raise
        finally:
            self._cycle_running = False

    def _raise_if_llm_failed(self) -> None:
        failure = self._llm_failure
        if failure is not None:
            raise failure

    def _queue_llm_degraded(self, sim_time_s: int, reason: str) -> None:
        """Expose the terminal provider failure without inventing a new plan."""
        if sim_time_s in self._llm_degraded_event_times:
            return
        self._llm_degraded_event_times.add(sim_time_s)
        self._llm_degraded_event_order.append(sim_time_s)
        while len(self._llm_degraded_event_order) > self._dependencies.retention.event_history_limit:
            self._llm_degraded_event_times.discard(
                self._llm_degraded_event_order.popleft()
            )
        self.submit_event(
            event_type="llm_degraded",
            entity_id=self._scenario_id,
            sim_time_s=sim_time_s,
            payload={
                "reason": reason,
                "active_plan_preserved": True,
                "execution_halted": True,
            },
        )

    def _run_cycle(self, *, epoch: PlanningEpoch | None = None) -> dict[str, Any]:
        latches = getattr(self, "_regional_replan_latches", None)
        if latches is None:
            latches = set()
            self._regional_replan_latches = latches
        previous_latches = set(latches)
        dependencies = getattr(self, "_dependencies", None)
        monitor = getattr(dependencies, "monitor", None)
        monitor_checkpoint = (
            monitor.checkpoint() if monitor is not None else None
        )
        prediction_intent_monitor = getattr(
            dependencies, "prediction_intent_monitor", None
        )
        prediction_intent_monitor_checkpoint = (
            prediction_intent_monitor.checkpoint()
            if prediction_intent_monitor is not None
            else None
        )
        try:
            self._drain_live_prediction_events()
            get_state = getattr(self._graph, "get_state", None)
            prior_state: dict[str, Any] = {}
            if get_state is not None:
                checkpoint = get_state(self._config)
                prior_state = dict(checkpoint.values or {})
                situation = self._dependencies.situation_provider(
                    live_situation_ref(self._scenario_id)
                )
                assessed_events = assess_regional_replan_events(
                    situation,
                    active_plan=self._dependencies.plans.get_active(self._scenario_id),
                    known_target_ids=tuple(prior_state.get("known_target_ids") or ()),
                    lost_target_ids=tuple(prior_state.get("lost_target_ids") or ()),
                    covariance_cap_m2=self._dependencies.covariance_cap_m2,
                )
                self.submit_events(self._latch_regional_replan_events(assessed_events))
            pending_events = tuple(self._pending)
            graph_input: dict[str, Any] = {
                    "scenario_id": self._scenario_id,
                    "uuv_only": bool(getattr(dependencies, "uuv_only", False)),
                    "snapshot_ref": live_situation_ref(self._scenario_id),
                    "pending_events": pending_events,
                    "planning_epoch": epoch,
                    # These are cycle-local terminal channels. Clear the
                    # previous epoch result before evaluating a new epoch so
                    # the finalizer cannot mistake it for a second outcome.
                    "epoch_commit_result": None,
                    "commit_status": None,
                    "selected_plan": None,
                    "node_error": None,
                    "executable_mission_plan": (
                        prior_state.get("executable_mission_plan")
                        or getattr(self, "_baseline_executable_mission_plan", None)
                    ),
            }
            graph_input.update(self._live_prediction_fragment())
            result = self._graph.invoke(
                graph_input,
                config=self._config,
            )
            processed_order = getattr(self, "_processed_event_order", None)
            if processed_order is None:
                processed_order = deque()
                self._processed_event_order = processed_order
            for event in pending_events:
                self._processed_event_ids.add(event.event_id)
                processed_order.append(event.event_id)
            dependencies = getattr(self, "_dependencies", None)
            retention = getattr(dependencies, "retention", None)
            processed_limit = int(getattr(retention, "processed_event_limit", 4096))
            while len(processed_order) > processed_limit:
                self._processed_event_ids.discard(processed_order.popleft())
            self._pending.clear()
            get_state = getattr(self._graph, "get_state", None)
            if callable(get_state):
                checkpoint = get_state(self._config)
                self._state_cache = dict(checkpoint.values or {})
            else:
                self._state_cache = dict(result)
            self._capture_graph_prediction_state(self._state_cache)
            return dict(result)
        except Exception:
            self._regional_replan_latches.clear()
            self._regional_replan_latches.update(previous_latches)
            if monitor is not None and monitor_checkpoint is not None:
                monitor.restore(monitor_checkpoint)
            if (
                prediction_intent_monitor is not None
                and prediction_intent_monitor_checkpoint is not None
            ):
                prediction_intent_monitor.restore(prediction_intent_monitor_checkpoint)
            raise

    def _latch_regional_replan_events(
        self, events: Sequence[RuntimeEvent]
    ) -> tuple[RuntimeEvent, ...]:
        """Emit one event per degraded key until the key is observed healthy."""
        active_keys = {(event.event_type, event.entity_id) for event in events}
        self._regional_replan_latches.intersection_update(active_keys)

        fresh_events: list[RuntimeEvent] = []
        seen_keys: set[tuple[str, str | None]] = set()
        for event in events:
            key = (event.event_type, event.entity_id)
            if key in self._regional_replan_latches or key in seen_keys:
                continue
            fresh_events.append(event)
            seen_keys.add(key)

        self._regional_replan_latches.update(active_keys)
        return tuple(fresh_events)

    def get_state(self) -> dict[str, Any]:
        """Latest checkpointed state of the scenario thread (empty when fresh)."""
        if getattr(self, "_cycle_running", False):
            return self._merge_live_prediction_state(
                dict(getattr(self, "_state_cache", {}))
            )
        with self._lock:
            snapshot = self._graph.get_state(self._config)
            state = dict(snapshot.values or {})
            self._state_cache = state
        return self._merge_live_prediction_state(state)

    def active_plan(self) -> TrackingPlan | None:
        """The scenario's currently broadcast plan (None before the first commit)."""
        return self._dependencies.plans.get_active(self._scenario_id)

    def current_plan_version(self) -> int:
        """Return the legacy plan version used by operator preview/apply CAS."""
        active = self._dependencies.plans.get_active(self._scenario_id)
        if active is not None:
            return active.revision
        execution = self.current_execution_snapshot()
        return execution.execution_revision if execution is not None else 0

    @property
    def memory_port(self) -> object | None:
        """The user-scoped memory adapter exposed to the HTTP boundary."""
        return self._dependencies.memory_port

    def active_mission_plan(self) -> ExecutableMissionPlan | None:
        """Return the latest verified executable plan for a UUV-only run."""
        coordinator = getattr(self, "_execution_coordinator", None)
        if coordinator is not None:
            reader = getattr(coordinator, "executable_mission_plan", None)
            if callable(reader):
                authoritative = reader(
                    sim_time_s=self.current_sim_time_s(),
                    hard_stale_s=float(
                        getattr(self._dependencies, "execution_hard_stale_s", 900.0)
                    ),
                )
                return (
                    authoritative
                    if isinstance(authoritative, ExecutableMissionPlan)
                    else None
                )
        value = self.get_state().get("executable_mission_plan")
        baseline = getattr(self, "_baseline_executable_mission_plan", None)
        plans = tuple(
            plan
            for plan in (value, baseline)
            if isinstance(plan, ExecutableMissionPlan)
        )
        return max(plans, key=lambda plan: plan.revision, default=None)

    def install_executable_baseline(self, plan: ExecutableMissionPlan) -> None:
        """Expose an already-installed deterministic plan to background planning."""
        # This is a small immutable reference exchanged between the physical
        # loop and the background graph. It must not wait for ``tick()``'s
        # graph lock while a provider call is in flight.
        self._baseline_executable_mission_plan = plan

    @property
    def execution_coordinator(self) -> object | None:
        """Expose the single authoritative execution coordinator to adapters."""

        return getattr(self, "_execution_coordinator", None)

    def current_execution_snapshot(self) -> OperationalExecutionSnapshot | None:
        """Return the current authoritative snapshot when the new path is active."""

        coordinator = self.execution_coordinator
        active_reader = getattr(coordinator, "active_mission_plan", None)
        if callable(active_reader):
            active = active_reader()
            if isinstance(active, OperationalExecutionSnapshot):
                return active
        reader = getattr(coordinator, "current", None)
        value = reader() if callable(reader) else reader
        return value if isinstance(value, OperationalExecutionSnapshot) else None

    def current_execution_context(self, *, frame_id: int | None = None) -> ExecutionContextRef | None:
        """Return transport coordinates for the current authoritative revision."""

        snapshot = self.current_execution_snapshot()
        if snapshot is None:
            return None
        return ExecutionContextRef.from_snapshot(snapshot, frame_id=frame_id)

    def _validate_execution_context(
        self, execution_revision: int | None, frame_id: int | None
    ) -> None:
        snapshot = self.current_execution_snapshot()
        if execution_revision is not None:
            current = snapshot.execution_revision if snapshot is not None else 0
            if execution_revision != current:
                raise ValueError(
                    f"the execution snapshot changed; expected {execution_revision}, "
                    f"current {current}"
                )
        if (
            frame_id is not None
            and snapshot is not None
            and snapshot.frame_id is not None
            and frame_id != snapshot.frame_id
        ):
            raise ValueError(
                f"the execution frame changed; expected {frame_id}, "
                f"current {snapshot.frame_id}"
            )

    def execution_evidence_resolver(
        self, *, frame_id: int | None = None
    ) -> ExecutionEvidenceResolver | None:
        """Build a read-only resolver over the current execution repositories."""

        snapshot = self.current_execution_snapshot()
        if snapshot is None:
            return None
        dependencies = self._dependencies
        return ExecutionEvidenceResolver(
            snapshot,
            events=dependencies.events,
            ledger=dependencies.ledger,
            plans=dependencies.plans,
            frame_id=frame_id,
        )

    def propose_execution(self, candidate: object, **kwargs: object) -> object:
        """Delegate a detached execution proposal to the scenario coordinator."""

        coordinator = self.execution_coordinator
        if coordinator is None:
            raise RuntimeError("execution coordinator is not attached")
        return coordinator.propose(candidate, **kwargs)

    def commit_execution(self, candidate: object, **kwargs: object) -> object:
        """Commit a candidate through the scenario coordinator."""

        coordinator = self.execution_coordinator
        if coordinator is None:
            raise RuntimeError("execution coordinator is not attached")
        return coordinator.commit(candidate, **kwargs)

    def preserve_execution(self, reason: str) -> object:
        """Preserve the active execution revision after a planning failure."""

        coordinator = self.execution_coordinator
        if coordinator is None:
            raise RuntimeError("execution coordinator is not attached")
        return coordinator.preserve(reason)

    def reservations(self) -> ReservationRegistry:
        """The scenario's human-assignment reservation registry (spec 17.2)."""
        return self._reservations

    def ask(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
        *,
        evidence_ids: Sequence[str] = (),
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> QuestionAnswer:
        """Answer one question while serializing access to the scenario thread."""
        with self._lock:
            self._validate_execution_context(execution_revision, frame_id)
            execution_snapshot = self.current_execution_snapshot()
            if execution_snapshot is not None:
                resolver = self.execution_evidence_resolver(frame_id=frame_id)
                payload = answer_execution_question(
                    execution_snapshot,
                    raw_text,
                    evidence_ids=evidence_ids,
                    resolver=resolver,
                    frame_id=frame_id,
                )
                answer = QuestionAnswer.model_validate(payload)
                self._persist_question_run(
                    question_run_id(
                        self._scenario_id,
                        raw_text,
                        counterfactual,
                        execution_revision=execution_snapshot.execution_revision,
                        evidence_ids=evidence_ids,
                    ),
                    raw_text,
                    answer,
                )
                return answer
            return self._ask_locked(
                raw_text,
                counterfactual,
                execution_revision=execution_revision,
                frame_id=frame_id,
                evidence_ids=evidence_ids,
            )

    def conversation_message(self, message: ConversationMessage) -> ConversationTurnResult:
        """Classify one expert turn independently of background planning."""
        with self._assistant_lock:
            self._validate_execution_context(
                getattr(message, "execution_revision", None),
                getattr(message, "frame_id", None),
            )
            execution_snapshot = self.current_execution_snapshot()
            active_plan = self._dependencies.plans.get_active(self._scenario_id)
            situation = self._dependencies.situation_provider(
                live_situation_ref(self._scenario_id)
            )
            context = ConversationContext(
                scenario_id=self._scenario_id,
                situation=situation,
                active_plan=active_plan,
                ledger=self._dependencies.ledger,
                events=self._dependencies.events,
                llm=self._dependencies.llm,
                memory_service=self._dependencies.memory_service,
                user_id=message.user_id,
                assistant_mode=message.assistant_mode,
                plans=self._dependencies.plans,
                short_term_repository=self._dependencies.short_term_repository,
                conversation_id=message.conversation_id,
                model_id=self._dependencies.model_id,
                planning_config=self._dependencies.optimizer,
                execution_snapshot=execution_snapshot,
                execution_frame_id=(
                    getattr(message, "frame_id", None)
                    if getattr(message, "frame_id", None) is not None
                    else (
                        execution_snapshot.frame_id
                        if execution_snapshot is not None
                        else None
                    )
                ),
            )
            result = process_conversation_message(message, context)
            self._conversation_turns[(result.conversation_id, result.turn_id)] = result
            self._trim_conversation_turns()
            return result

    def apply_conversation(
        self,
        conversation_id: str,
        turn_id: str,
        expected_plan_version: int,
        *,
        user_id: str = "operator",
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> ConversationTurnResult:
        """Apply one stored conversation preview after an explicit confirmation."""
        assistant_lock = getattr(self, "_assistant_lock", self._lock)
        with assistant_lock, self._lock:
            self._validate_execution_context(execution_revision, frame_id)
            result = self._conversation_turns.get((conversation_id, turn_id))
            if result is None:
                raise ValueError(f"unknown conversation turn {turn_id!r}")
            if result.user_id != user_id:
                raise ValueError(
                    "conversation turn belongs to user "
                    f"{result.user_id!r}, not {user_id!r}"
                )
            if result.applied:
                return result
            current_plan_version = self.current_plan_version()
            if expected_plan_version != current_plan_version:
                raise ValueError(
                    "conversation plan version mismatch: "
                    f"expected {expected_plan_version}, current {current_plan_version}"
                )
            if result.expected_plan_version != expected_plan_version:
                raise ValueError("conversation preview was created for another plan version")
            if result.proposal is None:
                raise ValueError("conversation turn has no plan proposal to apply")
            applied = self._apply_directive_locked(result.proposal.directive.directive_id)
            updated_proposal = result.proposal.model_copy(
                update={"directive": applied, "status": applied.status}
            )
            messages = tuple(
                item.model_copy(update={"proposal": updated_proposal})
                if item.proposal is not None
                else item
                for item in result.messages
            )
            updated = result.model_copy(
                update={
                    "messages": messages,
                    "proposal": updated_proposal,
                    "applied": True,
                }
            )
            self._conversation_turns[(conversation_id, turn_id)] = updated
            self._trim_conversation_turns()
            return updated

    def _trim_conversation_turns(self) -> None:
        limit = self._dependencies.retention.conversation_turn_limit
        while len(self._conversation_turns) > limit:
            self._conversation_turns.pop(next(iter(self._conversation_turns)))

    def _ask_locked(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
        *,
        emit_event: bool = True,
        execution_revision: int | None = None,
        frame_id: int | None = None,
        evidence_ids: Sequence[str] = (),
    ) -> QuestionAnswer:
        """Answer one expert question with evidence (read-only branch, spec 10.2).

        The answer is served from the immutable planning snapshot — the
        ledger, plan diffs, validation issues, and observations resolved by
        evidence id — and is rejected when it cites evidence absent from
        the payload. An optional ``counterfactual`` override is solved as an
        isolated dry-run (``dry-run:<uuid>`` plan id) over a cloned snapshot;
        the online plan is never touched and the graph is never invoked.
        The run is persisted under a deterministic run id (re-asking the
        same question dedupes) and a ``question`` event is queued so the
        next cycle surfaces the run on the checkpointed state.
        """
        scenario_id = self._scenario_id
        situation = self._dependencies.situation_provider(
            live_situation_ref(scenario_id)
        )
        applied = self._dependencies.ledger.list_directives(
            scenario_id, status="applied"
        )
        snapshot = build_planning_snapshot(
            situation,
            active_plan=self._dependencies.plans.get_active(scenario_id),
            applied_directives=tuple(applied),
        )
        execution_snapshot = self.current_execution_snapshot()
        if execution_snapshot is not None:
            resolver = self.execution_evidence_resolver(frame_id=frame_id)
            payload = answer_execution_question(
                execution_snapshot,
                raw_text,
                evidence_ids=evidence_ids,
                resolver=resolver,
                frame_id=frame_id,
            )
            answer = QuestionAnswer.model_validate(payload)
        else:
            answer = answer_question(
                raw_text=raw_text,
                snapshot=snapshot,
                ledger=self._dependencies.ledger,
                events=self._dependencies.events,
                llm=self._dependencies.llm,
                counterfactual=counterfactual,
                model_id=self._dependencies.model_id,
                planning_config=self._dependencies.optimizer,
            )
        if emit_event:
            self._persist_question_run(
                question_run_id(
                    scenario_id,
                    raw_text,
                    counterfactual,
                    execution_revision=answer.execution_revision,
                    evidence_ids=evidence_ids,
                ),
                raw_text,
                answer,
            )
        return answer

    def _persist_question_run(
        self, run_id: str, raw_text: str, answer: QuestionAnswer
    ) -> None:
        """Persist one completed question run exactly once (deterministic id).

        ``question_runs.run_id`` is a PRIMARY KEY and ``runtime_events``
        event ids are UNIQUE, so the ledger is checked first: re-asking the
        same question with the same overrides reuses the stored run and does
        not queue a duplicate event.
        """
        existing = {
            run.run_id
            for run in self._dependencies.ledger.list_questions(self._scenario_id)
        }
        if run_id in existing:
            return
        self._dependencies.ledger.save_question(
            run_id=run_id,
            scenario_id=self._scenario_id,
            question_text=raw_text,
            payload={"answer": answer.model_dump(mode="json")},
            status="completed",
            execution_revision=answer.execution_revision,
            frame_id=answer.frame_id,
        )
        # Persist the question event immediately so a completed run is
        # visible in runtime_events even before the next graph cycle runs
        # (spec 8.4). The event_id mirrors exactly what submit_event queues
        # below, so the next cycle's record_decision replay skips it as a
        # duplicate; the queue entry below still feeds the next cycle's
        # latest_question surfacing on the checkpointed state.
        self._dependencies.events.append(
            event_id=(
                f"{self._scenario_id}:{QUESTION_EVENT_TYPE}:{run_id}:"
                f"{self._dependencies.clock.sim_time_s}"
            ),
            event_type=QUESTION_EVENT_TYPE,
            scenario_id=self._scenario_id,
            sim_time_s=self._dependencies.clock.sim_time_s,
            payload={
                "run_id": run_id,
                "status": "completed",
                "execution_revision": answer.execution_revision,
                "frame_id": answer.frame_id,
            },
            severity="info",
        )
        self.submit_event(
            event_type=QUESTION_EVENT_TYPE,
            entity_id=run_id,
            sim_time_s=self._dependencies.clock.sim_time_s,
            payload={
                "run_id": run_id,
                "status": "completed",
                "execution_revision": answer.execution_revision,
                "frame_id": answer.frame_id,
            },
        )

    def preview_directive(
        self,
        raw_text: str,
        *,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> ExpertDirective:
        """Build one directive preview without racing the carrier tick."""
        with self._lock:
            self._validate_execution_context(execution_revision, frame_id)
            return self._preview_directive_locked(raw_text)

    def _preview_directive_locked(self, raw_text: str) -> ExpertDirective:
        """Parse one expert annotation into a persisted, validated preview.

        The directive schema is invoked over the curated payload (the raw
        text, a deterministic directive id derived from the text, and the
        scenario's known ids and applied directives), the parsed directive
        is validated (IDs, resources, conflicts) and persisted with its
        resolved status. The graph is never invoked: the running plan keeps
        executing while the expert reviews the preview.
        """
        scenario_id = self._scenario_id
        situation = self._dependencies.situation_provider(
            live_situation_ref(scenario_id)
        )
        applied = self._dependencies.ledger.list_directives(
            scenario_id, status="applied"
        )
        directive_id = f"{scenario_id}:directive:{canonical_digest(raw_text)[:12]}"
        payload = build_directive_payload(
            raw_text,
            directive_id,
            situation,
            applied,
            model_id=self._dependencies.model_id,
        )
        parsed = self._dependencies.llm.invoke_structured(
            DIRECTIVE_OPERATION,
            payload,
            ExpertDirective,
            prompt_version=DIRECTIVE_PROMPT_VERSION,
        )
        parsed = freeze_dedicated_tracking_members(parsed, situation)
        validated = validate_directive(
            parsed, situation=situation, applied_directives=applied
        )
        self._dependencies.ledger.save_directive(validated, scenario_id)
        return validated

    def preview_assignment(
        self,
        *,
        uuv_ids: Sequence[str],
        target_id: str,
        execution_revision: int | None = None,
        frame_id: int | None = None,
    ) -> ExpertDirective:
        """Build one typed assignment preview under the runtime lock."""
        with self._lock:
            self._validate_execution_context(execution_revision, frame_id)
            return self._preview_assignment_locked(uuv_ids=uuv_ids, target_id=target_id)

    def _preview_assignment_locked(
        self, *, uuv_ids: Sequence[str], target_id: str
    ) -> ExpertDirective:
        """Typed assignment preview: one directive reserving ``uuv_ids``.

        Unlike ``preview_directive`` there is no LLM parse: the assignment
        is a deterministic typed operation, so the preview is built by the
        typed shortcut, validated against the live situation, persisted,
        and returned for the expert's explicit confirmation.
        """
        scenario_id = self._scenario_id
        situation = self._dependencies.situation_provider(
            live_situation_ref(scenario_id)
        )
        applied = self._dependencies.ledger.list_directives(
            scenario_id, status="applied"
        )
        directive = assign_target_uuvs(
            directive_id=(
                f"{scenario_id}:assign:{target_id}:{','.join(sorted(uuv_ids))}"
            ),
            uuv_ids=uuv_ids,
            target_id=target_id,
            situation=situation,
            applied_directives=applied,
        )
        self._dependencies.ledger.save_directive(directive, scenario_id)
        return directive

    def apply_directive(self, directive_id: str) -> ExpertDirective:
        """Apply a reviewed directive and queue its next-cycle event safely."""
        with self._lock:
            return self._apply_directive_locked(directive_id)

    def _apply_directive_locked(self, directive_id: str) -> ExpertDirective:
        """Apply one clean preview and queue the strategic directive event.

        Rejects previews that are not cleanly applicable — low confidence,
        unresolved conflicts, a non-preview status, or a re-validation
        failure against the live situation. A clean preview is persisted as
        ``applied`` and a ``directive_applied`` event is queued for the next
        cycle, which re-plans with the directive as a hard constraint. The
        method returns immediately: the existing plan stays active until
        that commit.
        """
        scenario_id = self._scenario_id
        preview = self._find_directive(directive_id)
        if preview is None:
            raise ValueError(f"unknown directive {directive_id!r}")
        reason = self._rejection_reason(preview)
        if reason is not None:
            raise DirectiveNotApplicableError(
                f"directive {directive_id!r} cannot be applied: {reason}"
            )
        applied = ExpertDirective.model_validate(
            {**preview.model_dump(mode="json"), "status": "applied"}
        )
        self._dependencies.ledger.save_directive(applied, scenario_id)
        if applied.directive_type == "assignment":
            assigned_target = applied.assignment_target_id
            assert assigned_target is not None, "a clean assignment names its target"
            self._reservations.reserve(applied.assignment_uuv_ids, assigned_target)
        if applied.tracking_mode == "dedicated":
            target_id = applied.target_scope[0]
            self._reservations.dedicate(applied.dedicated_uuv_ids, target_id)
        elif applied.tracking_mode == "regional":
            for target_id in applied.target_scope:
                self._reservations.release_dedicated(target_id)
        self.submit_event(
            event_type=DIRECTIVE_APPLIED_EVENT_TYPE,
            entity_id=directive_id,
            sim_time_s=self._dependencies.clock.sim_time_s,
            payload={
                "directive_id": directive_id,
                "status": "applied",
                "directive_type": applied.directive_type,
                "tracking_mode": applied.tracking_mode,
                "target_scope": list(applied.target_scope),
                "region_ids": list(applied.feedback_region_ids),
            },
        )
        return applied

    def _find_directive(self, directive_id: str) -> ExpertDirective | None:
        for directive in self._dependencies.ledger.list_directives(self._scenario_id):
            if directive.directive_id == directive_id:
                return directive
        return None

    def _rejection_reason(self, preview: ExpertDirective) -> str | None:
        """Why ``preview`` cannot be applied, or None when it is clean.

        The directive is re-validated against the live situation: the world
        keeps executing between preview and confirmation, so a target or
        resource named at parse time may no longer exist.
        """
        if preview.confidence < 0.70:
            return "confidence below the 0.70 apply threshold"
        if preview.conflicts:
            return "unresolved conflicts: " + "; ".join(preview.conflicts)
        if preview.status != "preview":
            return f"status is {preview.status!r}, not 'preview'"
        situation = self._dependencies.situation_provider(
            live_situation_ref(self._scenario_id)
        )
        applied = self._dependencies.ledger.list_directives(
            self._scenario_id, status="applied"
        )
        fresh = validate_directive(
            preview, situation=situation, applied_directives=applied
        )
        if fresh.status != "preview":
            self._dependencies.ledger.save_directive(fresh, self._scenario_id)
            return "re-validation failed: " + "; ".join(fresh.conflicts)
        return None

    def close(self) -> None:
        """Close runtime-owned persistence connections (repositories stay caller-owned)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for hook in getattr(self, "_pre_close_hooks", ()):
            hook()
        self._payload_store.close()
        self._checkpointer.conn.close()

    def register_pre_close_hook(self, hook: Callable[[], None]) -> None:
        """Register an owner-controlled shutdown action before runtime resources close."""
        if self._closed:
            raise RuntimeError("cannot register a close hook after runtime shutdown")
        self._pre_close_hooks.append(hook)
