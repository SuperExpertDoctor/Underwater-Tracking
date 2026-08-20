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
    validate_directive,
)
from underwater_tracking.agent.nodes.conversation import (
    ConversationContext,
    process_conversation_message,
)
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
)
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.persistence.checkpoints import create_checkpointer
from underwater_tracking.persistence.payloads import RuntimePayloadStore
from underwater_tracking.planning.reservations import ReservationRegistry


@dataclass(frozen=True, slots=True)
class SensorModeControl:
    """One operator sensor-mode write applied at the next engine boundary."""

    uuv_id: str
    mode: Literal["passive", "active"]
    target_id: str | None
    requested_at_s: int


class CarrierRuntime:
    """One scenario thread over the persistent carrier central graph."""

    def __init__(
        self,
        dependencies: CarrierDependencies,
        *,
        scenario_id: str,
        database_path: str | Path,
        thread_id: str | None = None,
    ) -> None:
        reservations = (
            dependencies.reservations
            if dependencies.reservations is not None
            else ReservationRegistry()
        )
        dependencies = replace(dependencies, reservations=reservations)
        self._dependencies = dependencies
        self._reservations = reservations
        self._scenario_id = scenario_id
        self._checkpointer = create_checkpointer(database_path)
        retention = dependencies.retention
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
        self._simulation_time_provider: Callable[[], int] | None = None
        self._llm_paused = False
        self._llm_pause_reason: str | None = None
        self._conversation_turns: dict[tuple[str, str], ConversationTurnResult] = {}
        self._llm_reconnectable = False
        self._llm_degraded_event_times: set[int] = set()
        self._llm_degraded_event_order: deque[int] = deque()
        self._cycle_running = False
        self._state_cache: dict[str, Any] = {}

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
    ) -> None:
        """Queue a direct UUV sonar control without bypassing the graph audit."""
        with self._lock:
            active = self._dependencies.plans.get_active(self._scenario_id)
            current_version = active.revision if active is not None else 0
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

    def tick(self) -> dict[str, Any]:
        """Advance the clock and run one graph cycle over pending events.

        A real provider failure is transactional: the carrier clock returns to
        its pre-cycle value and pending events remain queued, so a reconnect
        or human-triggered retry evaluates the identical situation again.
        """
        self._cycle_running = True
        try:
            with self._lock:
                previous_time_s = self._dependencies.clock.sim_time_s
                self._dependencies.clock.tick()
                try:
                    result = self._run_cycle()
                except LLMError as exc:
                    self._dependencies.clock.sim_time_s = previous_time_s
                    self._llm_paused = True
                    self._llm_pause_reason = str(exc)
                    self._queue_llm_degraded(previous_time_s, str(exc))
                    raise
                self._llm_paused = False
                self._llm_pause_reason = None
                return result
        finally:
            self._cycle_running = False

    def resume(self) -> dict[str, Any]:
        """Retry one pending cycle without advancing the carrier clock."""
        self._cycle_running = True
        try:
            with self._lock:
                result = self._run_cycle()
                self._llm_paused = False
                self._llm_pause_reason = None
                return result
        except LLMError as exc:
            self._llm_paused = True
            self._llm_pause_reason = str(exc)
            self._queue_llm_degraded(self._dependencies.clock.sim_time_s, str(exc))
            raise
        finally:
            self._cycle_running = False

    def _queue_llm_degraded(self, sim_time_s: int, reason: str) -> None:
        """Retain the active plan and expose one strategic degradation event."""
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
            payload={"reason": reason, "active_plan_preserved": True},
        )

    def _run_cycle(self) -> dict[str, Any]:
        latches = getattr(self, "_regional_replan_latches", None)
        if latches is None:
            latches = set()
            self._regional_replan_latches = latches
        previous_latches = set(latches)
        try:
            get_state = getattr(self._graph, "get_state", None)
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
            result = self._graph.invoke(
                {
                    "scenario_id": self._scenario_id,
                    "snapshot_ref": live_situation_ref(self._scenario_id),
                    "pending_events": pending_events,
                },
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
            return dict(result)
        except Exception:
            self._regional_replan_latches.clear()
            self._regional_replan_latches.update(previous_latches)
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
            return dict(getattr(self, "_state_cache", {}))
        with self._lock:
            snapshot = self._graph.get_state(self._config)
            state = dict(snapshot.values or {})
            self._state_cache = state
            return state

    def active_plan(self) -> TrackingPlan | None:
        """The scenario's currently broadcast plan (None before the first commit)."""
        return self._dependencies.plans.get_active(self._scenario_id)

    def active_mission_plan(self) -> ExecutableMissionPlan | None:
        """Return the latest verified executable plan for a UUV-only run."""
        value = self.get_state().get("executable_mission_plan")
        return value if isinstance(value, ExecutableMissionPlan) else None

    def reservations(self) -> ReservationRegistry:
        """The scenario's human-assignment reservation registry (spec 17.2)."""
        return self._reservations

    def ask(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
    ) -> QuestionAnswer:
        """Answer one question while serializing access to the scenario thread."""
        with self._lock:
            return self._ask_locked(raw_text, counterfactual)

    def conversation_message(self, message: ConversationMessage) -> ConversationTurnResult:
        """Classify one unified expert turn without applying its proposal."""
        with self._lock:
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
                model_id=self._dependencies.model_id,
                planning_config=self._dependencies.optimizer,
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
    ) -> ConversationTurnResult:
        """Apply one stored conversation preview after an explicit confirmation."""
        with self._lock:
            result = self._conversation_turns.get((conversation_id, turn_id))
            if result is None:
                raise ValueError(f"unknown conversation turn {turn_id!r}")
            if result.applied:
                return result
            active_plan = self._dependencies.plans.get_active(self._scenario_id)
            current_plan_version = active_plan.revision if active_plan else 0
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
                question_run_id(scenario_id, raw_text, counterfactual), raw_text, answer
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
            payload={"run_id": run_id, "status": "completed"},
            severity="info",
        )
        self.submit_event(
            event_type=QUESTION_EVENT_TYPE,
            entity_id=run_id,
            sim_time_s=self._dependencies.clock.sim_time_s,
            payload={"run_id": run_id, "status": "completed"},
        )

    def preview_directive(self, raw_text: str) -> ExpertDirective:
        """Build one directive preview without racing the carrier tick."""
        with self._lock:
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
        validated = validate_directive(
            parsed, situation=situation, applied_directives=applied
        )
        self._dependencies.ledger.save_directive(validated, scenario_id)
        return validated

    def preview_assignment(
        self, *, uuv_ids: Sequence[str], target_id: str
    ) -> ExpertDirective:
        """Build one typed assignment preview under the runtime lock."""
        with self._lock:
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
        self.submit_event(
            event_type=DIRECTIVE_APPLIED_EVENT_TYPE,
            entity_id=directive_id,
            sim_time_s=self._dependencies.clock.sim_time_s,
            payload={
                "directive_id": directive_id,
                "status": "applied",
                "directive_type": applied.directive_type,
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
        self._payload_store.close()
        self._checkpointer.conn.close()
