# src/underwater_tracking/agent/runtime.py
"""Persistent scenario runtime owning the carrier graph (spec 8.4, plan Task 8).

``CarrierRuntime`` owns one scenario thread: the SQLite checkpointer, the
payload store, and the scenario ``thread_id``. Events enter through
``submit_event``, each ``tick`` advances the clock and runs one graph cycle
over the pending events, ``resume`` runs one cycle without advancing the
clock (continue after a reopen), and ``get_state`` returns the latest
checkpointed state. Graph internals are never exposed; the injected
repositories stay caller-owned and are closed by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    build_carrier_graph,
    live_situation_ref,
)
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

    def close(self) -> None:
        """Close the checkpointer connection (repositories stay caller-owned)."""
        self._checkpointer.conn.close()
