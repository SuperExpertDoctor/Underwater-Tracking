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

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    build_carrier_graph,
    live_situation_ref,
)
from underwater_tracking.agent.nodes.directives import (
    DIRECTIVE_APPLIED_EVENT_TYPE,
    DIRECTIVE_OPERATION,
    DirectiveNotApplicableError,
    assign_target_uuvs,
    build_directive_payload,
    validate_directive,
)
from underwater_tracking.agent.nodes.questions import (
    QUESTION_EVENT_TYPE,
    QuestionAnswer,
    answer_question,
    question_run_id,
)
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.prompts import DIRECTIVE_PROMPT_VERSION, canonical_digest
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.persistence.checkpoints import create_checkpointer
from underwater_tracking.planning.reservations import ReservationRegistry


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
        self._payload_store: dict[str, Any] = {}
        self._graph = build_carrier_graph(
            dependencies, self._checkpointer, self._payload_store
        )
        self._thread_id = thread_id if thread_id is not None else f"{scenario_id}:carrier"
        self._config: dict[str, Any] = {"configurable": {"thread_id": self._thread_id}}
        self._pending: list[RuntimeEvent] = []
        self._lock = RLock()

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
                if event.event_id in pending_ids:
                    continue
                self._pending.append(event)
                pending_ids.add(event.event_id)

    def tick(self) -> dict[str, Any]:
        """Advance the clock and run one graph cycle over the pending events."""
        with self._lock:
            self._dependencies.clock.tick()
            return self._run_cycle()

    def resume(self) -> dict[str, Any]:
        """Run one cycle over the pending events without advancing the clock."""
        with self._lock:
            return self._run_cycle()

    def _run_cycle(self) -> dict[str, Any]:
        result = self._graph.invoke(
            {
                "scenario_id": self._scenario_id,
                "snapshot_ref": live_situation_ref(self._scenario_id),
                "pending_events": tuple(self._pending),
            },
            config=self._config,
        )
        self._pending.clear()
        return dict(result)

    def get_state(self) -> dict[str, Any]:
        """Latest checkpointed state of the scenario thread (empty when fresh)."""
        with self._lock:
            snapshot = self._graph.get_state(self._config)
            return dict(snapshot.values or {})

    def active_plan(self) -> TrackingPlan | None:
        """The scenario's currently broadcast plan (None before the first commit)."""
        with self._lock:
            return self._dependencies.plans.get_active(self._scenario_id)

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

    def _ask_locked(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
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
            payload={"directive_id": directive_id, "status": "applied"},
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
        """Close the checkpointer connection (repositories stay caller-owned)."""
        self._checkpointer.conn.close()
