"""Truth-safe periodic situation summaries and their isolated event writer."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from underwater_tracking.domain.models import (
    EventAudience,
    EventLevel,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.runtime.mission_controller import MissionSnapshot


_MAX_SOURCE_EVENT_IDS = 64
_MAX_SUMMARY_TEXT = 4000
_LOW_ENERGY_FRACTION = 0.10
_HIGH_MILEAGE_M = 50_000.0
_SUMMARY_METRIC_EPSILON = 0.10
_PublicValue = str | int | float | bool | None
_TARGET_EVENT_TYPES = frozenset(
    {
        "intent_change_confirmed",
        "imm_confidence_shifted",
        "imm_motion_mode_changed",
        "prediction_revision",
        "target_added",
        "target_detection_acquired",
        "target_detection_lost",
        "target_depth_regime_changed",
        "target_estimate_updated",
        "target_exit_predicted",
        "target_intent_change_suspected",
        "target_intent_changed",
        "target_lost",
        "target_maneuver",
        "target_maneuver_observed",
        "target_reacquired",
        "target_removed",
        "target_speed_regime_changed",
    }
)


class _StrictSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionSummary(_StrictSummaryModel):
    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    lifecycle: str = Field(min_length=1)
    coverage: float = Field(ge=0.0, le=1.0)
    tracking_quality: float = Field(ge=0.0, le=1.0)
    active_scan_uuv_count: int = Field(ge=0)
    passive_track_uuv_count: int = Field(ge=0)
    reserve_uuv_count: int = Field(ge=0)
    active_scan_uuv_ids: tuple[str, ...] = ()
    passive_track_uuv_ids: tuple[str, ...] = ()
    reserve_uuv_ids: tuple[str, ...] = ()
    handoff_from: str | None = None
    handoff_to: str | None = None
    plan_revision: int = Field(ge=0)


class CarrierSummary(_StrictSummaryModel):
    carrier_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    route_status: str = Field(min_length=1)
    mission_type: str = Field(min_length=1)
    onboard_uuv_count: int = Field(ge=0)
    ready_uuv_count: int = Field(ge=0)
    reserved_uuv_count: int = Field(ge=0)
    recoverable_uuv_count: int = Field(ge=0)
    onboard_uuv_ids: tuple[str, ...] = ()
    ready_uuv_ids: tuple[str, ...] = ()
    reserved_uuv_ids: tuple[str, ...] = ()
    recoverable_uuv_ids: tuple[str, ...] = ()


class UUVCountSummary(_StrictSummaryModel):
    total: int = Field(ge=0)
    onboard: int = Field(ge=0)
    deployed: int = Field(ge=0)
    returning: int = Field(ge=0)
    failed: int = Field(ge=0)
    healthy: int = Field(ge=0)
    unhealthy: int = Field(ge=0)
    energy_below_reserve_count: int = Field(ge=0)
    mileage_high_count: int = Field(ge=0)
    mileage_total_m: float = Field(ge=0.0)
    mileage_max_m: float = Field(ge=0.0)
    mode_counts: Mapping[str, int] = Field(default_factory=dict)
    deployment_counts: Mapping[str, int] = Field(default_factory=dict)


class PublicTargetSummary(_StrictSummaryModel):
    target_id: str = Field(min_length=1)
    quality_score: float = Field(ge=0.0, le=1.0)
    intent: str = Field(min_length=1)
    intent_confidence: float = Field(ge=0.0, le=1.0)
    prediction_revision: int = Field(ge=0)
    prediction_state: str = Field(min_length=1)
    assigned_uuv_ids: tuple[str, ...] = ()


class SituationChange(_StrictSummaryModel):
    change_type: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    previous: _PublicValue
    current: _PublicValue


class PeriodicSituationSummary(_StrictSummaryModel):
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    plan_version: int = Field(ge=0)
    region_states: tuple[RegionSummary, ...] = Field(max_length=256)
    carrier_states: tuple[CarrierSummary, ...] = Field(max_length=32)
    uuv_counts: UUVCountSummary
    target_estimates: tuple[PublicTargetSummary, ...] = Field(max_length=64)
    changes_since_previous: tuple[SituationChange, ...] = Field(max_length=256)
    source_event_ids: tuple[str, ...] = Field(max_length=_MAX_SOURCE_EVENT_IDS)


def build_periodic_situation_summary(
    situation: SituationSnapshot,
    mission: MissionSnapshot,
    source_events: Sequence[RuntimeEvent],
    previous: PeriodicSituationSummary | None,
) -> tuple[PeriodicSituationSummary, RuntimeEvent]:
    """Build one deterministic public summary and its durable source event."""

    if situation.scenario_id != mission.scenario_id:
        raise ValueError("situation and mission scenario IDs must match")

    regions = tuple(
        RegionSummary(
            region_id=region.region_id,
            target_id=region.target_id,
            lifecycle=_value(region.lifecycle),
            coverage=region.coverage,
            tracking_quality=region.tracking_quality,
            active_scan_uuv_count=len(region.active_scan_uuv_ids),
            passive_track_uuv_count=len(region.passive_track_uuv_ids),
            reserve_uuv_count=len(region.reserve_uuv_ids),
            active_scan_uuv_ids=tuple(sorted(region.active_scan_uuv_ids)),
            passive_track_uuv_ids=tuple(sorted(region.passive_track_uuv_ids)),
            reserve_uuv_ids=tuple(sorted(region.reserve_uuv_ids)),
            handoff_from=region.handoff_from,
            handoff_to=region.handoff_to,
            plan_revision=region.plan_revision,
        )
        for region in sorted(mission.regions, key=lambda item: item.region_id)
    )
    carriers = _carrier_summaries(situation, mission)
    uuv_counts = _uuv_counts(situation, mission)
    targets = _target_summaries(situation, source_events, previous)
    source_event_ids = tuple(
        sorted({event.event_id for event in source_events})[:_MAX_SOURCE_EVENT_IDS]
    )
    changes = _changes_since_previous(
        previous,
        plan_version=mission.plan_revision,
        regions=regions,
        carriers=carriers,
        uuv_counts=uuv_counts,
        targets=targets,
    )
    summary = PeriodicSituationSummary(
        scenario_id=situation.scenario_id,
        sim_time_s=situation.sim_time_s,
        plan_version=mission.plan_revision,
        region_states=regions,
        carrier_states=carriers,
        uuv_counts=uuv_counts,
        target_estimates=targets,
        changes_since_previous=changes,
        source_event_ids=source_event_ids,
    )
    summary_text = _human_summary(summary)
    payload = summary.model_dump(mode="json")
    payload["summary"] = summary_text
    payload["memory_eligible"] = previous is None or bool(changes)
    event = RuntimeEvent(
        event_id=f"periodic_situation_summary:{situation.scenario_id}:{situation.sim_time_s}",
        scenario_id=situation.scenario_id,
        sim_time_s=situation.sim_time_s,
        event_type="periodic_situation_summary",
        entity_id=situation.scenario_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload,
    )
    return summary, event


def _carrier_summaries(
    situation: SituationSnapshot, mission: MissionSnapshot
) -> tuple[CarrierSummary, ...]:
    if mission.carrier_missions:
        return tuple(
            CarrierSummary(
                carrier_id=carrier.carrier_id,
                role=carrier.role,
                route_status=_value(carrier.route_status),
                mission_type=carrier.mission_type,
                onboard_uuv_count=len(carrier.onboard_uuv_ids),
                ready_uuv_count=len(carrier.ready_uuv_ids),
                reserved_uuv_count=len(carrier.reserved_uuv_ids),
                recoverable_uuv_count=len(carrier.recoverable_uuv_ids),
                onboard_uuv_ids=tuple(sorted(carrier.onboard_uuv_ids)),
                ready_uuv_ids=tuple(sorted(carrier.ready_uuv_ids)),
                reserved_uuv_ids=tuple(sorted(carrier.reserved_uuv_ids)),
                recoverable_uuv_ids=tuple(sorted(carrier.recoverable_uuv_ids)),
            )
            for carrier in (
                mission.carrier_missions[carrier_id]
                for carrier_id in sorted(mission.carrier_missions)
            )
        )
    return tuple(
        CarrierSummary(
            carrier_id=carrier.carrier_id,
            role=carrier.role,
            route_status=_value(carrier.status),
            mission_type="unknown",
            onboard_uuv_count=len(carrier.onboard_uuv_ids),
            ready_uuv_count=0,
            reserved_uuv_count=0,
            recoverable_uuv_count=len(carrier.returning_uuv_ids),
            onboard_uuv_ids=tuple(sorted(carrier.onboard_uuv_ids)),
            recoverable_uuv_ids=tuple(sorted(carrier.returning_uuv_ids)),
        )
        for carrier in sorted(
            situation.carriers or ((situation.carrier,) if situation.carrier else ()),
            key=lambda item: item.carrier_id,
        )
    )


def _uuv_counts(situation: SituationSnapshot, mission: MissionSnapshot) -> UUVCountSummary:
    situation_uuvs = {uuv.uuv_id: uuv for uuv in situation.uuvs}
    resources = dict(mission.uuv_resources)
    modes = dict(mission.uuv_modes)
    uuv_ids = sorted(set(situation_uuvs) | set(resources) | set(modes))
    deployment_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    healthy = unhealthy = low_energy = high_mileage = 0
    mileage_total = 0.0
    mileage_max = 0.0
    for uuv_id in uuv_ids:
        resource = resources.get(uuv_id)
        state = situation_uuvs.get(uuv_id)
        deployment = (
            resource.deployment_state
            if resource is not None
            else _value(state.deployment_state)
            if state is not None
            else "unknown"
        )
        deployment_counts[deployment] = deployment_counts.get(deployment, 0) + 1
        mode = _value(modes.get(uuv_id, "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if resource is not None:
            energy = resource.energy_fraction
            mileage = resource.mileage_m
            is_healthy = resource.healthy and resource.capability_active
        else:
            energy = state.energy_fraction if state is not None else 0.0
            mileage = 0.0
            is_healthy = state is not None and state.status.value != "failed"
        healthy += int(is_healthy)
        unhealthy += int(not is_healthy)
        low_energy += int(energy < _LOW_ENERGY_FRACTION)
        high_mileage += int(mileage >= _HIGH_MILEAGE_M)
        mileage_total += mileage
        mileage_max = max(mileage_max, mileage)
    return UUVCountSummary(
        total=len(uuv_ids),
        onboard=deployment_counts.get("onboard", 0),
        deployed=deployment_counts.get("deployed", 0),
        returning=deployment_counts.get("returning", 0),
        failed=deployment_counts.get("failed", 0),
        healthy=healthy,
        unhealthy=unhealthy,
        energy_below_reserve_count=low_energy,
        mileage_high_count=high_mileage,
        mileage_total_m=mileage_total,
        mileage_max_m=mileage_max,
        mode_counts=dict(sorted(mode_counts.items())),
        deployment_counts=dict(sorted(deployment_counts.items())),
    )


def _target_summaries(
    situation: SituationSnapshot,
    source_events: Sequence[RuntimeEvent],
    previous: PeriodicSituationSummary | None = None,
) -> tuple[PublicTargetSummary, ...]:
    reports: dict[str, GroupReport] = {}
    for report in situation.group_reports:
        current_report = reports.get(report.target_id)
        if current_report is None or (report.sim_time_s, report.group_id) > (
            current_report.sim_time_s,
            current_report.group_id,
        ):
            reports[report.target_id] = report
    previous_targets = (
        {item.target_id: item for item in previous.target_estimates} if previous is not None else {}
    )
    public_event_target_ids = {
        target_id
        for event in source_events
        if EventAudience.BLUE_PLANNING in event.audiences
        for target_id in (
            event.payload.get("target_id"),
            event.entity_id if event.event_type in _TARGET_EVENT_TYPES else None,
        )
        if isinstance(target_id, str) and target_id
    }
    target_ids = sorted(set(reports) | public_event_target_ids | set(previous_targets))
    results: list[PublicTargetSummary] = []
    for target_id in target_ids:
        target_report = reports.get(target_id)
        previous_target = previous_targets.get(target_id)
        if target_report is None:
            quality = previous_target.quality_score if previous_target is not None else 0.0
            inferred_intent = previous_target.intent if previous_target is not None else "unknown"
            intent_confidence = (
                previous_target.intent_confidence if previous_target is not None else 0.0
            )
            assigned_uuv_ids = previous_target.assigned_uuv_ids if previous_target is not None else ()
        else:
            quality = target_report.quality.ewma
            inferred_intent, intent_confidence = max(
                target_report.belief.model_probabilities.items(),
                key=lambda item: (item[1], item[0]),
                default=("unknown", 0.0),
            )
            assigned_uuv_ids = tuple(sorted(target_report.member_ids))
        prediction_revision = max(
            (
                int(event.payload["prediction_revision"])
                for event in source_events
                if _event_targets(event, target_id)
                and isinstance(event.payload.get("prediction_revision"), int)
            ),
            default=previous_target.prediction_revision if previous_target is not None else 0,
        )
        results.append(
            PublicTargetSummary(
                target_id=target_id,
                quality_score=quality,
                intent=str(inferred_intent),
                intent_confidence=max(0.0, min(1.0, float(intent_confidence))),
                prediction_revision=prediction_revision,
                prediction_state="revised" if prediction_revision else "stable",
                assigned_uuv_ids=assigned_uuv_ids,
            )
        )
    return tuple(results)


def _event_targets(event: RuntimeEvent, target_id: str) -> bool:
    return event.entity_id == target_id or event.payload.get("target_id") == target_id


def _changes_since_previous(
    previous: PeriodicSituationSummary | None,
    *,
    plan_version: int,
    regions: Sequence[RegionSummary],
    carriers: Sequence[CarrierSummary],
    uuv_counts: UUVCountSummary,
    targets: Sequence[PublicTargetSummary],
) -> tuple[SituationChange, ...]:
    if previous is None:
        return ()
    changes: list[SituationChange] = []
    if previous.plan_version != plan_version:
        changes.append(_change("plan_revision", "mission", previous.plan_version, plan_version))
    previous_regions = {item.region_id: item for item in previous.region_states}
    for region_current in regions:
        previous_region = previous_regions.get(region_current.region_id)
        if previous_region is not None and previous_region.lifecycle != region_current.lifecycle:
            changes.append(
                _change(
                    "region_lifecycle",
                    region_current.region_id,
                    previous_region.lifecycle,
                    region_current.lifecycle,
                )
            )
        if (
            previous_region is not None
            and abs(previous_region.coverage - region_current.coverage)
            >= _SUMMARY_METRIC_EPSILON
        ):
            changes.append(
                _change(
                    "region_coverage",
                    region_current.region_id,
                    previous_region.coverage,
                    region_current.coverage,
                )
            )
        if (
            previous_region is not None
            and abs(previous_region.tracking_quality - region_current.tracking_quality)
            >= _SUMMARY_METRIC_EPSILON
        ):
            changes.append(
                _change(
                    "region_tracking_quality",
                    region_current.region_id,
                    previous_region.tracking_quality,
                    region_current.tracking_quality,
                )
            )
        if (
            previous_region is not None
            and previous_region.plan_revision != region_current.plan_revision
        ):
            changes.append(
                _change(
                    "region_plan_revision",
                    region_current.region_id,
                    previous_region.plan_revision,
                    region_current.plan_revision,
                )
            )
        if previous_region is not None and _region_assignments(
            previous_region
        ) != _region_assignments(region_current):
            changes.append(
                _change(
                    "region_uuv_assignment",
                    region_current.region_id,
                    _region_assignments(previous_region),
                    _region_assignments(region_current),
                )
            )
    previous_carriers = {item.carrier_id: item for item in previous.carrier_states}
    for carrier_current in carriers:
        previous_carrier = previous_carriers.get(carrier_current.carrier_id)
        if (
            previous_carrier is not None
            and previous_carrier.route_status != carrier_current.route_status
        ):
            changes.append(
                _change(
                    "carrier_route_status",
                    carrier_current.carrier_id,
                    previous_carrier.route_status,
                    carrier_current.route_status,
                )
            )
        if previous_carrier is not None and _carrier_assignments(
            previous_carrier
        ) != _carrier_assignments(carrier_current):
            changes.append(
                _change(
                    "carrier_uuv_assignment",
                    carrier_current.carrier_id,
                    _carrier_assignments(previous_carrier),
                    _carrier_assignments(carrier_current),
                )
            )
    if previous.uuv_counts.mode_counts != uuv_counts.mode_counts:
        changes.append(
            _change(
                "uuv_mode",
                "fleet",
                _json_value(previous.uuv_counts.mode_counts),
                _json_value(uuv_counts.mode_counts),
            )
        )
    if previous.uuv_counts.deployment_counts != uuv_counts.deployment_counts:
        changes.append(
            _change(
                "uuv_deployment",
                "fleet",
                _json_value(previous.uuv_counts.deployment_counts),
                _json_value(uuv_counts.deployment_counts),
            )
        )
    if (previous.uuv_counts.healthy, previous.uuv_counts.unhealthy) != (
        uuv_counts.healthy,
        uuv_counts.unhealthy,
    ):
        changes.append(
            _change("uuv_health", "fleet", previous.uuv_counts.healthy, uuv_counts.healthy)
        )
    if previous.uuv_counts.energy_below_reserve_count != uuv_counts.energy_below_reserve_count:
        changes.append(
            _change(
                "uuv_energy",
                "fleet",
                previous.uuv_counts.energy_below_reserve_count,
                uuv_counts.energy_below_reserve_count,
            )
        )
    if previous.uuv_counts.mileage_high_count != uuv_counts.mileage_high_count:
        changes.append(
            _change(
                "uuv_mileage_risk",
                "fleet",
                previous.uuv_counts.mileage_high_count,
                uuv_counts.mileage_high_count,
            )
        )
    previous_targets = {item.target_id: item for item in previous.target_estimates}
    for target_current in targets:
        previous_target = previous_targets.get(target_current.target_id)
        if previous_target is None:
            continue
        if abs(previous_target.quality_score - target_current.quality_score) >= 0.1:
            changes.append(
                _change(
                    "target_quality",
                    target_current.target_id,
                    previous_target.quality_score,
                    target_current.quality_score,
                )
            )
        if previous_target.assigned_uuv_ids != target_current.assigned_uuv_ids:
            changes.append(
                _change(
                    "target_uuv_assignment",
                    target_current.target_id,
                    _json_value(previous_target.assigned_uuv_ids),
                    _json_value(target_current.assigned_uuv_ids),
                )
            )
        if previous_target.intent != target_current.intent:
            changes.append(
                _change(
                    "target_intent",
                    target_current.target_id,
                    previous_target.intent,
                    target_current.intent,
                )
            )
        if abs(previous_target.intent_confidence - target_current.intent_confidence) >= 0.15:
            changes.append(
                _change(
                    "target_intent_confidence",
                    target_current.target_id,
                    previous_target.intent_confidence,
                    target_current.intent_confidence,
                )
            )
        if previous_target.prediction_revision != target_current.prediction_revision:
            changes.append(
                _change(
                    "target_prediction_revision",
                    target_current.target_id,
                    previous_target.prediction_revision,
                    target_current.prediction_revision,
                )
            )
    return tuple(changes)


def _change(
    change_type: str, entity_id: str, previous: _PublicValue, current: _PublicValue
) -> SituationChange:
    return SituationChange(
        change_type=change_type,
        entity_id=entity_id,
        previous=previous,
        current=current,
    )


def _region_assignments(region: RegionSummary) -> str:
    return json.dumps(
        {
            "active_scan": region.active_scan_uuv_ids,
            "passive_track": region.passive_track_uuv_ids,
            "reserve": region.reserve_uuv_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _carrier_assignments(carrier: CarrierSummary) -> str:
    return json.dumps(
        {
            "onboard": carrier.onboard_uuv_ids,
            "ready": carrier.ready_uuv_ids,
            "reserved": carrier.reserved_uuv_ids,
            "recoverable": carrier.recoverable_uuv_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: Sequence[str] | Mapping[str, int]) -> str:
    if isinstance(value, Mapping):
        value = dict(sorted(value.items()))
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _human_summary(summary: PeriodicSituationSummary) -> str:
    regions = (
        ",".join(
            f"{region.region_id}:{region.lifecycle}:{region.coverage:.2f}"
            f":scan={','.join(region.active_scan_uuv_ids)}"
            f":track={','.join(region.passive_track_uuv_ids)}"
            f":reserve={','.join(region.reserve_uuv_ids)}"
            for region in summary.region_states
        )
        or "none"
    )
    carriers = (
        ",".join(
            f"{carrier.carrier_id}:{carrier.route_status}:onboard={carrier.onboard_uuv_count}"
            f":ready={carrier.ready_uuv_count}:reserved={carrier.reserved_uuv_count}"
            f":recoverable={carrier.recoverable_uuv_count}"
            for carrier in summary.carrier_states
        )
        or "none"
    )
    targets = (
        ",".join(
            f"{target.target_id}:quality={target.quality_score:.2f}:intent={target.intent}"
            f":confidence={target.intent_confidence:.2f}:assigned={','.join(target.assigned_uuv_ids)}"
            f":prediction={target.prediction_revision}"
            for target in summary.target_estimates
        )
        or "none"
    )
    change_details = (
        "|".join(
            f"{change.change_type}:{change.entity_id}:"
            f"{_public_value_text(change.previous)}->{_public_value_text(change.current)}"
            for change in summary.changes_since_previous
        )
        or "none"
    )
    text = (
        f"time={summary.sim_time_s}; plan={summary.plan_version}; regions={regions}; "
        f"carriers={carriers}; uuv_total={summary.uuv_counts.total}; "
        f"uuv_deployed={summary.uuv_counts.deployed}; targets={targets}; "
        f"changes={len(summary.changes_since_previous)}; change_details={change_details}; "
        f"sources={len(summary.source_event_ids)}"
    )
    return text if len(text) <= _MAX_SUMMARY_TEXT else text[: _MAX_SUMMARY_TEXT - 3] + "..."


def _public_value_text(value: _PublicValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PeriodicSituationSummaryWriterMetrics:
    queue_backlog: int
    accepted_count: int
    persisted_count: int
    duplicate_count: int
    failed_count: int
    overflow_count: int
    degraded_reason: str | None


class PeriodicSituationSummaryWriter:
    """Persist summary events on a private daemon-thread SQLite connection."""

    def __init__(
        self,
        database_path: Any,
        *,
        queue_limit: int = 64,
        repository_factory: Callable[[Any], EventRepository] | None = None,
    ) -> None:
        if queue_limit < 1:
            raise ValueError("queue_limit must be positive")
        self._database_path = database_path
        self._queue_limit = queue_limit
        self._repository_factory = repository_factory or EventRepository
        self._condition = Condition(Lock())
        self._queue: deque[RuntimeEvent] = deque()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._accepted_count = 0
        self._persisted_count = 0
        self._duplicate_count = 0
        self._failed_count = 0
        self._overflow_count = 0
        self._degraded_reason: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def metrics(self) -> PeriodicSituationSummaryWriterMetrics:
        with self._condition:
            return PeriodicSituationSummaryWriterMetrics(
                queue_backlog=len(self._queue),
                accepted_count=self._accepted_count,
                persisted_count=self._persisted_count,
                duplicate_count=self._duplicate_count,
                failed_count=self._failed_count,
                overflow_count=self._overflow_count,
                degraded_reason=self._degraded_reason,
            )

    def start(self) -> None:
        with self._condition:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = Thread(
                target=self._run,
                name="underwater-periodic-summary-writer",
                daemon=True,
            )
            self._thread.start()

    def submit(self, event: RuntimeEvent) -> bool:
        if event.event_type != "periodic_situation_summary":
            raise ValueError("writer accepts only periodic_situation_summary events")
        with self._condition:
            if len(self._queue) >= self._queue_limit:
                self._overflow_count += 1
                self._degraded_reason = "queue_full"
                return False
            self._queue.append(event)
            self._accepted_count += 1
            self._condition.notify()
            return True

    def stop(self, *, timeout: float = 5.0) -> bool:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, timeout))
        return not thread.is_alive()

    def _run(self) -> None:
        repository: EventRepository | None = None
        try:
            repository = self._repository_factory(self._database_path)
            while True:
                with self._condition:
                    while not self._queue and not self._stop_event.is_set():
                        self._condition.wait(timeout=0.1)
                    if not self._queue and self._stop_event.is_set():
                        return
                    event = self._queue.popleft()
                try:
                    row_id = repository.append_if_absent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        scenario_id=event.scenario_id,
                        sim_time_s=event.sim_time_s,
                        target_id=event.entity_id,
                        severity=event.level.value,
                        payload=event.payload,
                    )
                except Exception as error:  # noqa: BLE001 - retryable persistence boundary
                    with self._condition:
                        self._queue.appendleft(event)
                        self._failed_count += 1
                        self._degraded_reason = type(error).__name__
                    self._stop_event.wait(0.05)
                    continue
                with self._condition:
                    if row_id is None:
                        self._duplicate_count += 1
                    else:
                        self._persisted_count += 1
                    self._degraded_reason = None
        finally:
            if repository is not None:
                repository.close()


def _value(value: object) -> str:
    return str(getattr(value, "value", value))
