"""Bridge the checkpointed carrier runtime to live operational frames.

The simulation/agent loop owns LangGraph and SQLite state.  This adapter is
the narrow publication seam: it resolves the latest checkpointed semantic
outputs, combines them with the current estimator snapshot, builds a
truth-safe ``OperationalFrame``, persists it for replay, and publishes the
same object to the WebSocket hub.  It never reads the evaluation sink.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    TrackingPlan,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot
from underwater_tracking.domain.ui_models import MetricView, OperationalFrame
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger


class RuntimeFramePort(Protocol):
    def active_plan(self) -> TrackingPlan | None: ...

    def get_state(self) -> Mapping[str, object]: ...


class EventPort(Protocol):
    def list_events(
        self, *, scenario_id: str | None = None, limit: int = 1000
    ) -> Sequence[Any]: ...


class LedgerPort(Protocol):
    def list_decisions(
        self, scenario_id: str | None = None, limit: int = 100
    ) -> Sequence[Any]: ...

    def list_directives(
        self, scenario_id: str | None = None, status: str | None = None
    ) -> Sequence[ExpertDirective]: ...


class OperationalFramePublisher:
    """Publish one complete frame after every carrier observation cycle."""

    def __init__(
        self,
        *,
        runtime: RuntimeFramePort,
        ledger: LedgerPort | DecisionLedger,
        events: EventPort | EventRepository,
        hub: OperationalHub,
        logger: FrameLogger | None = None,
        history_limit: int = 300,
    ) -> None:
        self._runtime = runtime
        self._ledger = ledger
        self._events = events
        self._hub = hub
        self._logger = logger
        self._history_limit = max(1, history_limit)
        self._breadcrumbs: dict[str, list[tuple[float, float]]] = {}

    def publish(self, snapshot: SituationSnapshot) -> OperationalFrame:
        self._record_breadcrumbs(snapshot)
        state = self._runtime.get_state()
        hypotheses = _mapping_of(state.get("intent_hypotheses"), IntentHypothesis)
        predictions = _mapping_of(state.get("predictions"), PredictedTrackRef)
        stored_events = self._stored_events(snapshot)
        decisions = self._ledger.list_decisions(snapshot.scenario_id, limit=50)
        applied = self._ledger.list_directives(snapshot.scenario_id, status="applied")
        frame = build_operational_frame(
            snapshot,
            self._runtime.active_plan(),
            decisions,
            stored_events,
            _metrics(snapshot),
            intent_hypotheses=hypotheses,
            predictions=predictions,
            applied_directives=applied,
            breadcrumbs={key: tuple(value) for key, value in self._breadcrumbs.items()},
        )
        if self._logger is not None:
            self._logger.append(frame)
        self._hub.publish(frame)
        return frame

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()

    @property
    def frame_count(self) -> int:
        return self._logger.count if self._logger is not None else 0

    def _record_breadcrumbs(self, snapshot: SituationSnapshot) -> None:
        for uuv in snapshot.uuvs:
            trail = self._breadcrumbs.setdefault(uuv.uuv_id, [])
            point = (float(uuv.position_xy[0]), float(uuv.position_xy[1]))
            if not trail or trail[-1] != point:
                trail.append(point)
            del trail[:-self._history_limit]

    def _stored_events(self, snapshot: SituationSnapshot) -> tuple[RuntimeEvent, ...]:
        events: dict[str, RuntimeEvent] = {}
        for row in self._events.list_events(
            scenario_id=snapshot.scenario_id, limit=self._history_limit
        ):
            event_id = str(row.event_id)
            events[event_id] = RuntimeEvent(
                event_id=event_id,
                scenario_id=str(row.scenario_id),
                sim_time_s=int(row.sim_time_s),
                event_type=str(row.event_type),
                entity_id=row.target_id,
                level=_event_level(str(row.severity), str(row.event_type)),
                payload=dict(row.payload),
            )
        for event in snapshot.pending_events:
            events.setdefault(event.event_id, event)
        return tuple(sorted(events.values(), key=lambda event: (event.sim_time_s, event.event_id)))


def _mapping_of(value: object, expected_type: type[Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, expected_type)
    }


def _event_level(severity: str, event_type: str) -> EventLevel:
    normalized = severity.lower()
    if normalized in {level.value for level in EventLevel}:
        return cast(EventLevel, normalized)
    if any(token in event_type for token in ("plan", "directive", "target_added", "target_lost")):
        return EventLevel.STRATEGIC
    if any(token in event_type for token in ("quality", "ping", "route", "group")):
        return EventLevel.TACTICAL
    return EventLevel.INFORMATIONAL


def _metrics(snapshot: SituationSnapshot) -> tuple[MetricView, ...]:
    metrics: list[MetricView] = []
    for report in snapshot.group_reports:
        metrics.append(
            MetricView(
                metric_id=f"quality:{report.target_id}",
                label=f"{report.target_id} 编组质量",
                value=report.quality.window_mean,
                unit="score",
                threshold=None,
                window_s=300,
                series=(report.quality.instant, report.quality.window_mean, report.quality.ewma),
            )
        )
        metrics.append(
            MetricView(
                metric_id=f"fim:{report.target_id}",
                label=f"{report.target_id} FIM 最小特征值",
                value=report.belief.fim_min_eigenvalue,
                unit="m⁻²",
                threshold=0.0,
                window_s=300,
                series=(report.belief.fim_min_eigenvalue,),
            )
        )
    if snapshot.uuvs:
        metrics.append(
            MetricView(
                metric_id="fleet:energy",
                label="编队平均剩余能量",
                value=sum(uuv.energy_fraction for uuv in snapshot.uuvs) / len(snapshot.uuvs),
                unit="fraction",
                threshold=0.1,
                window_s=0,
                series=tuple(uuv.energy_fraction for uuv in snapshot.uuvs),
            )
        )
    return tuple(metrics)
