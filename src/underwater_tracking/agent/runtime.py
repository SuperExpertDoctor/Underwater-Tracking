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
active until that cycle commits. Graph internals are never exposed; the
injected repositories stay caller-owned and are closed by the caller.
"""

from __future__ import annotations

from pathlib import Path
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
    build_directive_payload,
    validate_directive,
)
from underwater_tracking.agent.prompts import DIRECTIVE_PROMPT_VERSION, canonical_digest
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.persistence.checkpoints import create_checkpointer


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
        self._dependencies = dependencies
        self._scenario_id = scenario_id
        self._checkpointer = create_checkpointer(database_path)
        self._payload_store: dict[str, Any] = {}
        self._graph = build_carrier_graph(
            dependencies, self._checkpointer, self._payload_store
        )
        self._thread_id = thread_id if thread_id is not None else f"{scenario_id}:carrier"
        self._config: dict[str, Any] = {"configurable": {"thread_id": self._thread_id}}
        self._pending: list[RuntimeEvent] = []

    def submit_event(
        self,
        *,
        event_type: str,
        entity_id: str | None,
        sim_time_s: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue one event for the next graph cycle (re-classified by the monitor)."""
        self._pending.append(
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
            )
        )

    def tick(self) -> dict[str, Any]:
        """Advance the clock and run one graph cycle over the pending events."""
        self._dependencies.clock.tick()
        return self._run_cycle()

    def resume(self) -> dict[str, Any]:
        """Run one cycle over the pending events without advancing the clock."""
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
        snapshot = self._graph.get_state(self._config)
        return dict(snapshot.values or {})

    def active_plan(self) -> TrackingPlan | None:
        """The scenario's currently broadcast plan (None before the first commit)."""
        return self._dependencies.plans.get_active(self._scenario_id)

    def preview_directive(self, raw_text: str) -> ExpertDirective:
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

    def apply_directive(self, directive_id: str) -> ExpertDirective:
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
