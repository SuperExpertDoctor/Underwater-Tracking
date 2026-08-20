from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ConfigDict, Field

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    RegionLifecycle,
    RegionMissionState,
    UUVResourceState,
    UUVMissionMode,
    validate_region_transition,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, StrictModel


class MissionSnapshot(StrictModel):
    """Immutable executable mission state exposed at an observation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    plan_revision: int = Field(ge=0)
    regions: tuple[RegionMissionState, ...] = ()
    uuv_modes: Mapping[str, UUVMissionMode] = {}
    uuv_resources: Mapping[str, UUVResourceState] = {}
    resource_episode_by_uuv: Mapping[str, int] = {}
    carrier_missions: Mapping[str, CarrierMissionModel] = {}
    events: tuple[RuntimeEvent, ...] = ()


Observation = Mapping[str, object]


class MissionController:
    """Own UUV-only lifecycle transitions and atomic executable plans.

    The controller consumes estimated observations and explicit operator/plan
    signals.  It does not move a UUV or carrier and never reads hidden target
    state; the simulation layer remains responsible for physical kinematics.
    """

    def __init__(
        self,
        *,
        scenario_id: str,
        region_entry_probability_threshold: float = 0.70,
        region_transition_confirm_cycles: int = 2,
        max_uuv_mileage_m: float = 50_000.0,
        min_energy_fraction: float = 0.10,
    ) -> None:
        if not 0.0 <= region_entry_probability_threshold <= 1.0:
            raise ValueError("region_entry_probability_threshold must be in [0, 1]")
        if region_transition_confirm_cycles < 1:
            raise ValueError("region_transition_confirm_cycles must be positive")
        if max_uuv_mileage_m <= 0.0:
            raise ValueError("max_uuv_mileage_m must be positive")
        if not 0.0 <= min_energy_fraction <= 1.0:
            raise ValueError("min_energy_fraction must be in [0, 1]")
        self._scenario_id = scenario_id
        self._entry_threshold = region_entry_probability_threshold
        self._confirm_cycles = region_transition_confirm_cycles
        self._max_mileage_m = max_uuv_mileage_m
        self._min_energy_fraction = min_energy_fraction
        self._sim_time_s = 0
        self._plan_revision = 0
        self._regions: dict[str, RegionMissionState] = {}
        self._uuv_modes: dict[str, UUVMissionMode] = {}
        self._uuv_resources: dict[str, UUVResourceState] = {}
        self._resource_episode_by_uuv: dict[str, int] = {}
        self._uuv_carrier_ids: dict[str, str] = {}
        self._carrier_missions: dict[str, CarrierMissionModel] = {}
        self._events: list[RuntimeEvent] = []
        self._emitted: set[tuple[str, str | None, int]] = set()

    def snapshot(self) -> MissionSnapshot:
        """Return a sorted immutable view of the current controller state."""
        return MissionSnapshot(
            scenario_id=self._scenario_id,
            sim_time_s=self._sim_time_s,
            plan_revision=self._plan_revision,
            regions=tuple(
                self._regions[region_id] for region_id in sorted(self._regions)
            ),
            uuv_modes=dict(sorted(self._uuv_modes.items())),
            uuv_resources=dict(sorted(self._uuv_resources.items())),
            resource_episode_by_uuv=dict(sorted(self._resource_episode_by_uuv.items())),
            carrier_missions=dict(sorted(self._carrier_missions.items())),
            events=tuple(self._events),
        )

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return self.snapshot().events

    def apply_verified_plan(self, plan: ExecutableMissionPlan) -> bool:
        """Atomically apply only a strictly newer executable plan."""
        if plan.revision <= self._plan_revision:
            return False
        new_regions = {
            region.region_id: region.model_copy(deep=True)
            for region in plan.region_assignments
        }
        new_modes: dict[str, UUVMissionMode] = {}
        new_uuv_carrier_ids: dict[str, str] = {}
        for batch in plan.batches:
            for uuv_id in batch.uuv_ids:
                new_modes[uuv_id] = UUVMissionMode.TRANSIT_TO_REGION
                new_uuv_carrier_ids[uuv_id] = batch.carrier_id
        for uuv_id in plan.reserved_uuv_ids:
            new_modes[uuv_id] = UUVMissionMode.ONBOARD
        for region in new_regions.values():
            if region.lifecycle is RegionLifecycle.PASSIVE_TRACK:
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                ):
                    new_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            elif region.lifecycle is RegionLifecycle.ACTIVE_SCAN:
                for uuv_id in region.active_scan_uuv_ids:
                    new_modes[uuv_id] = UUVMissionMode.ACTIVE_SCAN
        self._plan_revision = plan.revision
        self._regions = new_regions
        self._uuv_modes = new_modes
        self._uuv_carrier_ids = new_uuv_carrier_ids
        self._resource_episode_by_uuv = {
            uuv_id: self._resource_episode_by_uuv.get(uuv_id, 0)
            for uuv_id in new_modes
        }
        self._uuv_resources = {
            uuv_id: resource
            for uuv_id, resource in self._uuv_resources.items()
            if uuv_id in new_modes
        }
        self._carrier_missions = {
            carrier_id: carrier.model_copy(deep=True)
            for carrier_id, carrier in plan.carrier_missions.items()
        }
        return True

    def advance(
        self,
        sim_time_s: int,
        observations: Observation | Sequence[Observation],
    ) -> MissionSnapshot:
        """Advance lifecycle state using one estimated observation boundary."""
        if sim_time_s < self._sim_time_s:
            raise ValueError("mission time cannot move backwards")
        self._sim_time_s = sim_time_s
        observed = _normalize_observations(observations)
        self._record_resource_observations(observed)
        self._apply_failure_observations(observed)
        self._apply_deployment_observations(observed)
        self._apply_entry_observations(observed)
        self._apply_handoff_observations(observed)
        self._apply_recovery_observations(observed)
        self._apply_resource_observations(observed)
        self._apply_external_events(observed)
        return self.snapshot()

    def _apply_deployment_observations(self, observations: Observation) -> None:
        deployed_by_region = _mapping(observations.get("deployed_uuv_ids"))
        for region_id, region in tuple(self._regions.items()):
            deployed = set(_strings(deployed_by_region.get(region_id)))
            required = {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }
            if not required or not required.issubset(deployed):
                continue
            if region.lifecycle is RegionLifecycle.PLANNED:
                self._transition(region_id, RegionLifecycle.CARRIER_DEPLOYING)
            if self._regions[region_id].lifecycle is RegionLifecycle.CARRIER_DEPLOYING:
                self._transition(region_id, RegionLifecycle.ACTIVE_SCAN)
                for uuv_id in self._regions[region_id].active_scan_uuv_ids:
                    self._uuv_modes[uuv_id] = UUVMissionMode.ACTIVE_SCAN
            for uuv_id in required:
                self._remove_uuv_from_carrier_inventory(uuv_id)

    def _apply_entry_observations(self, observations: Observation) -> None:
        probabilities = _mapping(observations.get("entry_probability"))
        for region_id, region in tuple(self._regions.items()):
            if region.lifecycle is not RegionLifecycle.ACTIVE_SCAN:
                continue
            probability = _float(probabilities.get(region_id), 0.0)
            confirmations = (
                region.entry_confirmations + 1
                if probability >= self._entry_threshold
                else 0
            )
            updated = region.model_copy(update={"entry_confirmations": confirmations})
            self._regions[region_id] = updated
            if confirmations < self._confirm_cycles:
                continue
            self._transition(region_id, RegionLifecycle.PASSIVE_TRACK)
            for uuv_id in (
                *updated.active_scan_uuv_ids,
                *updated.passive_track_uuv_ids,
            ):
                if self._uuv_modes.get(uuv_id) is not UUVMissionMode.FAILED:
                    self._uuv_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            self._emit("target_entered_region", region_id)

    def _apply_handoff_observations(self, observations: Observation) -> None:
        handoffs = _mapping(observations.get("handoff_ready"))
        ready = _mapping(observations.get("successor_passive_ready"))
        for predecessor_id, successor_value in sorted(handoffs.items()):
            successor_id = str(successor_value)
            predecessor = self._regions.get(predecessor_id)
            successor = self._regions.get(successor_id)
            if predecessor is None or successor is None:
                continue
            if predecessor.lifecycle is not RegionLifecycle.PASSIVE_TRACK:
                continue
            if not bool(ready.get(successor_id)):
                continue
            if successor.lifecycle is RegionLifecycle.PLANNED:
                self._transition(successor_id, RegionLifecycle.CARRIER_DEPLOYING)
                self._transition(successor_id, RegionLifecycle.ACTIVE_SCAN)
            if successor.lifecycle is RegionLifecycle.ACTIVE_SCAN:
                self._transition(successor_id, RegionLifecycle.PASSIVE_TRACK)
            for uuv_id in (
                *successor.active_scan_uuv_ids,
                *successor.passive_track_uuv_ids,
            ):
                self._uuv_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            self._transition(predecessor_id, RegionLifecycle.HANDOFF_PENDING)
            self._transition(predecessor_id, RegionLifecycle.TRACKING_COMPLETED)
            for uuv_id in (
                *predecessor.active_scan_uuv_ids,
                *predecessor.passive_track_uuv_ids,
            ):
                self._mark_uuv_for_recovery(uuv_id)
            self._emit("handoff_completed", predecessor_id, {"successor_region_id": successor_id})

    def _apply_recovery_observations(self, observations: Observation) -> None:
        for uuv_id in _strings(observations.get("recovered_uuv_ids")):
            mode = self._uuv_modes.get(uuv_id)
            if mode not in {UUVMissionMode.RETURN_REQUIRED, UUVMissionMode.RECOVERING}:
                continue
            carrier_id = self._uuv_carrier_ids.get(uuv_id)
            if carrier_id is None or carrier_id not in self._carrier_missions:
                continue
            self._uuv_modes[uuv_id] = UUVMissionMode.ONBOARD
            carrier = self._carrier_missions[carrier_id]
            self._carrier_missions[carrier_id] = carrier.model_copy(
                update={
                    "recoverable_uuv_ids": tuple(
                        item for item in carrier.recoverable_uuv_ids if item != uuv_id
                    ),
                    "ready_uuv_ids": tuple(sorted({*carrier.ready_uuv_ids, uuv_id})),
                }
            )
            self._resource_episode_by_uuv[uuv_id] = (
                self._resource_episode_by_uuv.get(uuv_id, 0) + 1
            )
            self._emit("carrier_recovery_completed", uuv_id)
            for region_id, region in tuple(self._regions.items()):
                assigned = {
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                }
                if uuv_id not in assigned:
                    continue
                if region.lifecycle is RegionLifecycle.TRACKING_COMPLETED:
                    self._transition(region_id, RegionLifecycle.CARRIER_RECOVERY)
                if region.lifecycle is RegionLifecycle.CARRIER_RECOVERY:
                    remaining = assigned - {
                        item for item in _strings(observations.get("recovered_uuv_ids"))
                    }
                    if not remaining:
                        self._transition(region_id, RegionLifecycle.RECOVERED)

    def _record_resource_observations(self, observations: Observation) -> None:
        mileage = _mapping(observations.get("mileage_m"))
        energy = _mapping(observations.get("energy_fraction"))
        health = _mapping(observations.get("uuv_health"))
        capability = _mapping(observations.get("uuv_capability_active"))
        deployment = _mapping(observations.get("deployment_state"))
        for uuv_id in sorted(self._uuv_modes):
            self._uuv_resources[uuv_id] = UUVResourceState(
                uuv_id=uuv_id,
                carrier_id=self._uuv_carrier_ids.get(uuv_id),
                mileage_m=_float(mileage.get(uuv_id), 0.0),
                energy_fraction=_float(energy.get(uuv_id), 1.0),
                healthy=bool(health.get(uuv_id, True)),
                capability_active=bool(capability.get(uuv_id, True)),
                deployment_state=str(
                    deployment.get(uuv_id, self._uuv_modes[uuv_id].value)
                ),
                resource_episode=self._resource_episode_by_uuv.get(uuv_id, 0),
            )

    def _apply_failure_observations(self, observations: Observation) -> None:
        for uuv_id in _strings(observations.get("failed_uuv_ids")):
            if uuv_id not in self._uuv_modes:
                continue
            if self._uuv_modes[uuv_id] is UUVMissionMode.FAILED:
                continue
            self._uuv_modes[uuv_id] = UUVMissionMode.FAILED
            self._degrade_regions_for_uuv(uuv_id, "uuv_failed")
            self._emit("uuv_failed", uuv_id)

    def _apply_resource_observations(self, observations: Observation) -> None:
        mileage = _mapping(observations.get("mileage_m"))
        energy = _mapping(observations.get("energy_fraction"))
        for uuv_id in sorted(self._uuv_modes):
            if self._uuv_modes[uuv_id] in {
                UUVMissionMode.FAILED,
                UUVMissionMode.RETURN_REQUIRED,
                UUVMissionMode.RECOVERING,
            }:
                continue
            mileage_value = _float(mileage.get(uuv_id), 0.0)
            energy_value = _float(energy.get(uuv_id), 1.0)
            if mileage_value >= self._max_mileage_m:
                self._return_uuv(uuv_id, "uuv_range_exhausted")
            elif energy_value <= self._min_energy_fraction:
                self._return_uuv(uuv_id, "uuv_energy_depleted")

    def _return_uuv(self, uuv_id: str, event_type: str) -> None:
        self._uuv_modes[uuv_id] = UUVMissionMode.RETURN_REQUIRED
        self._degrade_regions_for_uuv(uuv_id, event_type)
        carrier_id = self._uuv_carrier_ids.get(uuv_id)
        if carrier_id is not None and carrier_id in self._carrier_missions:
            carrier = self._carrier_missions[carrier_id]
            updated = carrier.model_copy(
                update={
                    "ready_uuv_ids": tuple(
                        item for item in carrier.ready_uuv_ids if item != uuv_id
                    ),
                    "onboard_uuv_ids": tuple(
                        item for item in carrier.onboard_uuv_ids if item != uuv_id
                    ),
                    "reserved_uuv_ids": tuple(
                        item for item in carrier.reserved_uuv_ids if item != uuv_id
                    ),
                    "recoverable_uuv_ids": tuple(
                        sorted({*carrier.recoverable_uuv_ids, uuv_id})
                    ),
                }
            )
            self._carrier_missions[carrier_id] = updated
        self._emit(event_type, uuv_id)

    def _mark_uuv_for_recovery(self, uuv_id: str) -> None:
        if uuv_id not in self._uuv_modes:
            return
        if self._uuv_modes[uuv_id] in {
            UUVMissionMode.RETURN_REQUIRED,
            UUVMissionMode.RECOVERING,
            UUVMissionMode.FAILED,
        }:
            return
        self._uuv_modes[uuv_id] = UUVMissionMode.RETURN_REQUIRED
        carrier_id = self._uuv_carrier_ids.get(uuv_id)
        if carrier_id is None or carrier_id not in self._carrier_missions:
            return
        carrier = self._carrier_missions[carrier_id]
        self._carrier_missions[carrier_id] = carrier.model_copy(
            update={
                "ready_uuv_ids": tuple(item for item in carrier.ready_uuv_ids if item != uuv_id),
                "onboard_uuv_ids": tuple(item for item in carrier.onboard_uuv_ids if item != uuv_id),
                "reserved_uuv_ids": tuple(item for item in carrier.reserved_uuv_ids if item != uuv_id),
                "recoverable_uuv_ids": tuple(sorted({*carrier.recoverable_uuv_ids, uuv_id})),
            }
        )

    def _remove_uuv_from_carrier_inventory(self, uuv_id: str) -> None:
        carrier_id = self._uuv_carrier_ids.get(uuv_id)
        if carrier_id is None or carrier_id not in self._carrier_missions:
            return
        carrier = self._carrier_missions[carrier_id]
        self._carrier_missions[carrier_id] = carrier.model_copy(
            update={
                "ready_uuv_ids": tuple(item for item in carrier.ready_uuv_ids if item != uuv_id),
                "onboard_uuv_ids": tuple(item for item in carrier.onboard_uuv_ids if item != uuv_id),
                "reserved_uuv_ids": tuple(item for item in carrier.reserved_uuv_ids if item != uuv_id),
            }
        )

    def _degrade_regions_for_uuv(self, uuv_id: str, reason: str) -> None:
        for region_id, region in tuple(self._regions.items()):
            if uuv_id not in {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
                *region.reserve_uuv_ids,
            }:
                continue
            active = tuple(item for item in region.active_scan_uuv_ids if item != uuv_id)
            passive = tuple(item for item in region.passive_track_uuv_ids if item != uuv_id)
            reserve = tuple(item for item in region.reserve_uuv_ids if item != uuv_id)
            self._regions[region_id] = region.model_copy(
                update={
                    "active_scan_uuv_ids": active,
                    "passive_track_uuv_ids": passive,
                    "reserve_uuv_ids": reserve,
                    "degraded_reasons": tuple(sorted({*region.degraded_reasons, reason})),
                }
            )
            if region.lifecycle not in {
                RegionLifecycle.DEGRADED,
                RegionLifecycle.UNCOVERED,
                RegionLifecycle.RECOVERED,
                RegionLifecycle.TRACKING_COMPLETED,
            }:
                self._transition(region_id, RegionLifecycle.DEGRADED)
            self._emit("region_coverage_degraded", region_id, {"reason": reason})

    def _apply_external_events(self, observations: Observation) -> None:
        for event_type in (
            "target_intent_changed",
            "imm_confidence_shifted",
            "target_exit_predicted",
            "carrier_dispatch_completed",
            "carrier_recovery_completed",
        ):
            value = observations.get(event_type)
            if not value:
                continue
            entity_id = None if value is True else str(value)
            self._emit(event_type, entity_id)

    def _transition(self, region_id: str, next_state: RegionLifecycle) -> None:
        current = self._regions[region_id].lifecycle
        if current is next_state:
            return
        if not validate_region_transition(current, next_state):
            raise ValueError(
                f"invalid region transition {current.value}->{next_state.value}"
            )
        self._regions[region_id] = self._regions[region_id].model_copy(
            update={"lifecycle": next_state}
        )

    def _emit(
        self,
        event_type: str,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        episode = self._resource_episode_by_uuv.get(entity_id or "", 0)
        key = (event_type, entity_id, episode)
        if key in self._emitted:
            return
        self._emitted.add(key)
        self._events.append(
            RuntimeEvent(
                event_id=(
                    f"{self._scenario_id}:{event_type}:{entity_id or 'mission'}"
                    f":r{self._plan_revision}:e{episode}:{self._sim_time_s}"
                ),
                scenario_id=self._scenario_id,
                sim_time_s=self._sim_time_s,
                event_type=event_type,
                entity_id=entity_id,
                level=EventLevel.STRATEGIC,
                payload=payload or {},
            )
        )


def _normalize_observations(
    observations: Observation | Sequence[Observation],
) -> dict[str, object]:
    if isinstance(observations, Mapping):
        return dict(observations)
    merged: dict[str, object] = {}
    for observation in observations:
        merged.update(observation)
    return merged


def _mapping(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _float(value: object, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default
