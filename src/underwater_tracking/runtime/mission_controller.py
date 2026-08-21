from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from pydantic import ConfigDict, Field

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
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
    dedicated_target_by_uuv: Mapping[str, str] = {}
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
        uuv_owner_by_id: Mapping[str, str] | None = None,
        region_entry_probability_threshold: float = 0.70,
        region_transition_confirm_cycles: int = 2,
        max_uuv_mileage_m: float = 50_000.0,
        min_energy_fraction: float = 0.10,
        event_history_limit: int = 2048,
    ) -> None:
        if not 0.0 <= region_entry_probability_threshold <= 1.0:
            raise ValueError("region_entry_probability_threshold must be in [0, 1]")
        if region_transition_confirm_cycles < 1:
            raise ValueError("region_transition_confirm_cycles must be positive")
        if max_uuv_mileage_m <= 0.0:
            raise ValueError("max_uuv_mileage_m must be positive")
        if not 0.0 <= min_energy_fraction <= 1.0:
            raise ValueError("min_energy_fraction must be in [0, 1]")
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be positive")
        self._scenario_id = scenario_id
        self._configured_uuv_owner_by_id = dict(uuv_owner_by_id or {})
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
        self._dedicated_target_by_uuv: dict[str, str] = {}
        self._carrier_missions: dict[str, CarrierMissionModel] = {}
        self._recovered_uuv_ids_by_region: dict[str, set[str]] = {}
        self._events: list[RuntimeEvent] = []
        self._emitted: set[tuple[str, str | None, int, str | None]] = set()
        self._event_history_limit = event_history_limit
        self._emitted_order: deque[tuple[str, str | None, int, str | None]] = deque()

    @property
    def max_uuv_mileage_m(self) -> float:
        """Configured maximum sortie mileage used by execution preflight."""
        return self._max_mileage_m

    @property
    def min_energy_fraction(self) -> float:
        """Configured energy reserve used by execution preflight."""
        return self._min_energy_fraction

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
            dedicated_target_by_uuv=dict(sorted(self._dedicated_target_by_uuv.items())),
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

        # Carrier inventory is the ownership source for UUVs that are waiting
        # onboard or reserved for the next rolling task.  Build this mapping
        # before mutating controller state so a cross-carrier conflict rejects
        # the complete plan atomically.
        for carrier_id, carrier in sorted(plan.carrier_missions.items()):
            inventory = (
                *carrier.onboard_uuv_ids,
                *carrier.ready_uuv_ids,
                *carrier.reserved_uuv_ids,
                *carrier.recoverable_uuv_ids,
            )
            for uuv_id in inventory:
                previous = new_uuv_carrier_ids.get(uuv_id)
                if previous is not None and previous != carrier_id:
                    return False
                new_uuv_carrier_ids[uuv_id] = carrier_id
                if uuv_id in carrier.recoverable_uuv_ids:
                    new_modes[uuv_id] = UUVMissionMode.RETURN_REQUIRED
                else:
                    new_modes[uuv_id] = UUVMissionMode.ONBOARD

        for batch in plan.batches:
            for uuv_id in batch.uuv_ids:
                previous = new_uuv_carrier_ids.get(uuv_id)
                if previous is not None and previous != batch.carrier_id:
                    return False
                new_modes[uuv_id] = UUVMissionMode.TRANSIT_TO_REGION
                new_uuv_carrier_ids[uuv_id] = batch.carrier_id
        for uuv_id in plan.reserved_uuv_ids:
            if plan.carrier_missions and uuv_id not in new_uuv_carrier_ids:
                # A reserved UUV without a carrier cannot be recovered or
                # rotated into a later sortie safely.
                return False
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

        # A rolling plan is a complete fleet state, but it must not make an
        # already deployed UUV disappear between revisions.  Any old resource
        # omitted by the new plan remains observable; active members enter the
        # carrier-recovery queue so the physical engine can rotate them out.
        rotated_uuv_ids: set[str] = set()
        for uuv_id, previous_mode in self._uuv_modes.items():
            if uuv_id in new_modes:
                continue
            previous_carrier_id = self._uuv_carrier_ids.get(uuv_id)
            if previous_carrier_id is not None:
                new_uuv_carrier_ids[uuv_id] = previous_carrier_id
            new_modes[uuv_id] = (
                UUVMissionMode.RETURN_REQUIRED
                if previous_mode
                in {
                    UUVMissionMode.TRANSIT_TO_REGION,
                    UUVMissionMode.ACTIVE_SCAN,
                    UUVMissionMode.PASSIVE_TRACK,
                }
                else previous_mode
            )
            if new_modes[uuv_id] is UUVMissionMode.RETURN_REQUIRED:
                rotated_uuv_ids.add(uuv_id)

        new_carrier_missions = {
            carrier_id: carrier.model_copy(deep=True)
            for carrier_id, carrier in plan.carrier_missions.items()
        }
        for uuv_id in sorted(rotated_uuv_ids):
            carrier_id = new_uuv_carrier_ids.get(uuv_id)
            if carrier_id is None:
                continue
            carrier = new_carrier_missions.get(carrier_id)
            if carrier is None:
                carrier = self._carrier_missions.get(carrier_id)
            if carrier is None:
                continue
            new_carrier_missions[carrier_id] = carrier.model_copy(
                update={
                    "onboard_uuv_ids": tuple(
                        item for item in carrier.onboard_uuv_ids if item != uuv_id
                    ),
                    "ready_uuv_ids": tuple(
                        item for item in carrier.ready_uuv_ids if item != uuv_id
                    ),
                    "reserved_uuv_ids": tuple(
                        item for item in carrier.reserved_uuv_ids if item != uuv_id
                    ),
                    "recoverable_uuv_ids": tuple(
                        sorted({*carrier.recoverable_uuv_ids, uuv_id})
                    ),
                }
            )
        for uuv_id, carrier_id in new_uuv_carrier_ids.items():
            configured_carrier_id = self._configured_uuv_owner_by_id.get(uuv_id)
            if configured_carrier_id is not None and carrier_id != configured_carrier_id:
                return False
        self._plan_revision = plan.revision
        self._regions = new_regions
        self._uuv_modes = new_modes
        self._uuv_carrier_ids = new_uuv_carrier_ids
        self._recovered_uuv_ids_by_region = {
            region_id: set() for region_id in new_regions
        }
        self._resource_episode_by_uuv = {
            uuv_id: self._resource_episode_by_uuv.get(uuv_id, 0)
            for uuv_id in new_modes
        }
        self._uuv_resources = {
            uuv_id: resource
            for uuv_id, resource in self._uuv_resources.items()
            if uuv_id in new_modes
        }
        self._carrier_missions = new_carrier_missions
        for uuv_id, target_id in tuple(self._dedicated_target_by_uuv.items()):
            if uuv_id not in new_modes:
                self._dedicated_target_by_uuv.pop(uuv_id, None)
                continue
            if new_modes[uuv_id] not in {
                UUVMissionMode.ONBOARD,
                UUVMissionMode.FAILED,
            }:
                new_modes[uuv_id] = UUVMissionMode.DEDICATED_TRACK
        return True

    def set_dedicated_group(self, target_id: str, uuv_ids: Sequence[str]) -> bool:
        """Lock a human-selected UUV group to one target until released."""
        selected = tuple(sorted(dict.fromkeys(str(uuv_id) for uuv_id in uuv_ids)))
        if not selected or target_id not in {region.target_id for region in self._regions.values()}:
            return False
        if any(uuv_id not in self._uuv_modes for uuv_id in selected):
            return False
        for uuv_id, assigned_target in tuple(self._dedicated_target_by_uuv.items()):
            if assigned_target == target_id and uuv_id not in selected:
                self._dedicated_target_by_uuv.pop(uuv_id, None)
                self._restore_normal_mode(uuv_id)
        for uuv_id in selected:
            if self._uuv_modes[uuv_id] is UUVMissionMode.FAILED:
                continue
            self._dedicated_target_by_uuv[uuv_id] = target_id
            if self._uuv_modes[uuv_id] not in {
                UUVMissionMode.ONBOARD,
                UUVMissionMode.RETURN_TO_REGION,
            }:
                self._uuv_modes[uuv_id] = UUVMissionMode.DEDICATED_TRACK
        self._emit(
            "dedicated_group_assigned",
            target_id,
            {"uuv_ids": selected},
        )
        return True

    def clear_dedicated_group(
        self,
        target_id: str | None = None,
        uuv_ids: Sequence[str] = (),
    ) -> None:
        """Release dedicated assignments and restore each UUV's region mode."""
        selected_ids = set(str(uuv_id) for uuv_id in uuv_ids)
        released: list[str] = []
        for uuv_id, assigned_target in tuple(self._dedicated_target_by_uuv.items()):
            if target_id is not None and assigned_target != target_id:
                continue
            if selected_ids and uuv_id not in selected_ids:
                continue
            self._dedicated_target_by_uuv.pop(uuv_id, None)
            self._restore_normal_mode(uuv_id)
            released.append(uuv_id)
        if released:
            self._emit(
                "dedicated_group_released",
                target_id,
                {"uuv_ids": tuple(sorted(released))},
            )

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
        self._apply_resource_health_observations()
        self._apply_deployment_observations(observed)
        self._apply_entry_observations(observed)
        self._apply_handoff_observations(observed)
        self._apply_carrier_route_observations(observed)
        recovered_uuv_ids = self._apply_recovery_observations(observed)
        self._apply_resource_observations(observed, skip_uuv_ids=recovered_uuv_ids)
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
                for uuv_id in required:
                    self._uuv_modes[uuv_id] = (
                        UUVMissionMode.DEDICATED_TRACK
                        if uuv_id in self._dedicated_target_by_uuv
                        else UUVMissionMode.ACTIVE_SCAN
                    )
            for uuv_id in required:
                self._remove_uuv_from_carrier_inventory(uuv_id)

    def _apply_entry_observations(self, observations: Observation) -> None:
        probabilities = _mapping(observations.get("entry_probability"))
        for region_id, region in tuple(self._regions.items()):
            if region.lifecycle is not RegionLifecycle.ACTIVE_SCAN:
                continue
            if region_id not in probabilities:
                self._regions[region_id] = region.model_copy(
                    update={"entry_confirmations": 0}
                )
                continue
            probability = _float(probabilities.get(region_id), float("nan"))
            if not isfinite(probability) or not 0.0 <= probability <= 1.0:
                self._regions[region_id] = region.model_copy(
                    update={"entry_confirmations": 0}
                )
                continue
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
                if (
                    self._uuv_modes.get(uuv_id) is not UUVMissionMode.FAILED
                    and uuv_id not in self._dedicated_target_by_uuv
                ):
                    self._uuv_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            self._emit("target_entered_region", region_id)

    def _apply_handoff_observations(self, observations: Observation) -> None:
        handoffs = _mapping(observations.get("handoff_ready"))
        ready = _mapping(observations.get("successor_passive_ready"))
        for predecessor_key, successor_value in sorted(handoffs.items()):
            predecessor_id = str(predecessor_key)
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
                if uuv_id not in self._dedicated_target_by_uuv:
                    self._uuv_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            self._transition(predecessor_id, RegionLifecycle.HANDOFF_PENDING)
            self._transition(predecessor_id, RegionLifecycle.TRACKING_COMPLETED)
            for uuv_id in (
                *predecessor.active_scan_uuv_ids,
                *predecessor.passive_track_uuv_ids,
            ):
                self._mark_uuv_for_recovery(uuv_id)
            self._emit("handoff_completed", predecessor_id, {"successor_region_id": successor_id})

    def _apply_recovery_observations(self, observations: Observation) -> set[str]:
        recovered_uuv_ids: set[str] = set()
        for uuv_id in _strings(observations.get("returned_to_region_uuv_ids")):
            if self._uuv_modes.get(uuv_id) is not UUVMissionMode.RETURN_TO_REGION:
                continue
            target_id = self._dedicated_target_by_uuv.pop(uuv_id, None)
            self._restore_normal_mode(uuv_id)
            resource = self._uuv_resources.get(uuv_id)
            if resource is not None:
                self._uuv_resources[uuv_id] = resource.model_copy(
                    update={
                        "mileage_m": 0.0,
                        "deployment_state": self._uuv_modes[uuv_id].value,
                    }
                )
            self._emit(
                "dedicated_mode_released",
                uuv_id,
                {"target_id": target_id},
            )
            recovered_uuv_ids.add(uuv_id)
        for uuv_id in _strings(observations.get("recovery_requested_uuv_ids")):
            if self._uuv_modes.get(uuv_id) in {
                UUVMissionMode.ONBOARD,
                UUVMissionMode.FAILED,
                UUVMissionMode.RETURN_REQUIRED,
                UUVMissionMode.RECOVERING,
            }:
                continue
            self._mark_uuv_for_recovery(uuv_id)

        for uuv_id in _strings(observations.get("recovering_uuv_ids")):
            if self._uuv_modes.get(uuv_id) is not UUVMissionMode.RETURN_REQUIRED:
                continue
            self._uuv_modes[uuv_id] = UUVMissionMode.RECOVERING
            for region_id, region in tuple(self._regions.items()):
                assigned = {
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                }
                if uuv_id not in assigned:
                    continue
                if region.lifecycle is RegionLifecycle.TRACKING_COMPLETED:
                    self._recovered_uuv_ids_by_region.setdefault(region_id, set())
                    self._transition(region_id, RegionLifecycle.CARRIER_RECOVERY)

        for uuv_id in _strings(observations.get("recovered_uuv_ids")):
            mode = self._uuv_modes.get(uuv_id)
            if mode not in {UUVMissionMode.RETURN_REQUIRED, UUVMissionMode.RECOVERING}:
                continue
            carrier_id = self._uuv_carrier_ids.get(uuv_id)
            if carrier_id is None or carrier_id not in self._carrier_missions:
                continue
            if not _health_check_passed(observations, uuv_id):
                self._emit("carrier_recovery_health_check_pending", uuv_id)
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
            previous_resource = self._uuv_resources.get(uuv_id)
            self._uuv_resources[uuv_id] = UUVResourceState(
                uuv_id=uuv_id,
                carrier_id=carrier_id,
                mileage_m=0.0,
                energy_fraction=1.0,
                healthy=True,
                capability_active=(
                    previous_resource.capability_active
                    if previous_resource is not None
                    else True
                ),
                deployment_state=UUVMissionMode.ONBOARD.value,
                resource_episode=self._resource_episode_by_uuv[uuv_id],
            )
            recovered_uuv_ids.add(uuv_id)
            self._emit("carrier_recovery_completed", uuv_id)
            for region_id, region in tuple(self._regions.items()):
                assigned = {
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                }
                if uuv_id not in assigned:
                    continue
                current_region = self._regions[region_id]
                if current_region.lifecycle is RegionLifecycle.TRACKING_COMPLETED:
                    self._recovered_uuv_ids_by_region.setdefault(region_id, set())
                    self._transition(region_id, RegionLifecycle.CARRIER_RECOVERY)
                    current_region = self._regions[region_id]
                if current_region.lifecycle is RegionLifecycle.CARRIER_RECOVERY:
                    recovered_for_region = self._recovered_uuv_ids_by_region.setdefault(
                        region_id, set()
                    )
                    recovered_for_region.add(uuv_id)
                    remaining = assigned - recovered_for_region
                    if not remaining:
                        self._transition(region_id, RegionLifecycle.RECOVERED)
        return recovered_uuv_ids

    def _record_resource_observations(self, observations: Observation) -> None:
        mileage = _mapping(observations.get("mileage_m"))
        energy = _mapping(observations.get("energy_fraction"))
        health = _mapping(observations.get("uuv_health"))
        capability = _mapping(observations.get("uuv_capability_active"))
        deployment = _mapping(observations.get("deployment_state"))
        for uuv_id in sorted(self._uuv_modes):
            previous = self._uuv_resources.get(uuv_id)
            mileage_value = _observed_float(
                mileage,
                uuv_id,
                previous.mileage_m if previous is not None else 0.0,
            )
            energy_value = _observed_float(
                energy,
                uuv_id,
                previous.energy_fraction if previous is not None else 1.0,
            )
            healthy_value = _observed_bool(
                health,
                uuv_id,
                previous.healthy if previous is not None else True,
            )
            capability_value = _observed_bool(
                capability,
                uuv_id,
                previous.capability_active if previous is not None else True,
            )
            deployment_value = _observed_string(
                deployment,
                uuv_id,
                previous.deployment_state
                if previous is not None
                else self._uuv_modes[uuv_id].value,
            )
            self._uuv_resources[uuv_id] = UUVResourceState(
                uuv_id=uuv_id,
                carrier_id=self._uuv_carrier_ids.get(uuv_id),
                mileage_m=mileage_value,
                energy_fraction=energy_value,
                healthy=healthy_value,
                capability_active=capability_value,
                deployment_state=deployment_value,
                resource_episode=self._resource_episode_by_uuv.get(uuv_id, 0),
            )

    def _apply_failure_observations(self, observations: Observation) -> None:
        for uuv_id in _strings(observations.get("failed_uuv_ids")):
            if uuv_id not in self._uuv_modes:
                continue
            self._fail_uuv(uuv_id, "uuv_failed", "uuv_failed")

    def _apply_resource_health_observations(self) -> None:
        for uuv_id, resource in tuple(self._uuv_resources.items()):
            if uuv_id not in self._uuv_modes:
                continue
            if not resource.healthy:
                self._fail_uuv(uuv_id, "uuv_failed", "uuv_health_failed")
            elif not resource.capability_active:
                self._fail_uuv(uuv_id, "uuv_capability_lost", "uuv_capability_lost")

    def _apply_resource_observations(
        self,
        observations: Observation,
        *,
        skip_uuv_ids: set[str] | None = None,
    ) -> None:
        del observations
        skipped = skip_uuv_ids or set()
        for uuv_id in sorted(self._uuv_modes):
            if uuv_id in skipped:
                continue
            if self._uuv_modes[uuv_id] in {
                UUVMissionMode.FAILED,
                UUVMissionMode.RECOVERING,
                UUVMissionMode.RETURN_TO_REGION,
            }:
                continue
            resource = self._uuv_resources.get(uuv_id)
            if resource is None:
                continue
            mileage_value = resource.mileage_m
            energy_value = resource.energy_fraction
            if mileage_value >= self._max_mileage_m:
                self._return_uuv(uuv_id, "uuv_range_exhausted")
            elif energy_value <= self._min_energy_fraction:
                self._return_uuv(uuv_id, "uuv_energy_depleted")

    def _fail_uuv(self, uuv_id: str, event_type: str, reason: str) -> None:
        if self._uuv_modes.get(uuv_id) is UUVMissionMode.FAILED:
            return
        self._uuv_modes[uuv_id] = UUVMissionMode.FAILED
        self._degrade_regions_for_uuv(uuv_id, reason)
        self._emit(event_type, uuv_id, {"reason": reason})

    def _return_uuv(self, uuv_id: str, event_type: str) -> None:
        if uuv_id in self._dedicated_target_by_uuv:
            self._uuv_modes[uuv_id] = UUVMissionMode.RETURN_TO_REGION
            self._emit(
                "uuv_dedicated_return_to_region",
                uuv_id,
                {
                    "target_id": self._dedicated_target_by_uuv[uuv_id],
                    "reason": event_type,
                },
            )
            return
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
            UUVMissionMode.ONBOARD,
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

    def _restore_normal_mode(self, uuv_id: str) -> None:
        if uuv_id not in self._uuv_modes or self._uuv_modes[uuv_id] is UUVMissionMode.FAILED:
            return
        for region in self._regions.values():
            if uuv_id not in {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }:
                continue
            self._uuv_modes[uuv_id] = (
                UUVMissionMode.PASSIVE_TRACK
                if region.lifecycle is RegionLifecycle.PASSIVE_TRACK
                else UUVMissionMode.ACTIVE_SCAN
            )
            return
        self._uuv_modes[uuv_id] = UUVMissionMode.ONBOARD

    def _apply_external_events(self, observations: Observation) -> None:
        exit_prediction = observations.get("target_exit_predicted")
        if exit_prediction:
            region_id = str(exit_prediction)
            region = self._regions.get(region_id)
            if region is not None:
                if region.lifecycle is RegionLifecycle.PASSIVE_TRACK:
                    self._transition(region_id, RegionLifecycle.HANDOFF_PENDING)
                    self._transition(region_id, RegionLifecycle.TRACKING_COMPLETED)
                    for uuv_id in (
                        *region.active_scan_uuv_ids,
                        *region.passive_track_uuv_ids,
                    ):
                        self._mark_uuv_for_recovery(uuv_id)
                elif region.lifecycle is RegionLifecycle.ACTIVE_SCAN:
                    self._transition(region_id, RegionLifecycle.UNCOVERED)
                    for uuv_id in (
                        *region.active_scan_uuv_ids,
                        *region.passive_track_uuv_ids,
                    ):
                        self._mark_uuv_for_recovery(uuv_id)
            self._emit("target_exit_predicted", region_id)
        for event_type in (
            "target_intent_changed",
            "imm_confidence_shifted",
            "carrier_dispatch_completed",
            "carrier_recovery_completed",
        ):
            value = observations.get(event_type)
            if not value:
                continue
            if isinstance(value, Mapping):
                entity_id = value.get("entity_id")
                event_id = value.get("event_id")
                payload = {
                    str(key): child
                    for key, child in value.items()
                    if key not in {"entity_id", "event_id"}
                }
                self._emit(
                    event_type,
                    None if entity_id is None else str(entity_id),
                    payload,
                    dedupe_id=(None if event_id is None else str(event_id)),
                )
                continue
            entity_id = None if value is True else str(value)
            self._emit(event_type, entity_id)

    def _apply_carrier_route_observations(self, observations: Observation) -> None:
        statuses = _mapping(observations.get("carrier_route_status"))
        epochs = _mapping(observations.get("carrier_route_epoch"))
        for raw_carrier_id, raw_status in sorted(
            statuses.items(), key=lambda item: str(item[0])
        ):
            carrier_id = str(raw_carrier_id)
            mission = self._carrier_missions.get(carrier_id)
            if mission is None:
                continue
            if isinstance(raw_status, Mapping):
                status_value = raw_status.get("status")
                epoch_value = raw_status.get("route_epoch")
            else:
                status_value = raw_status
                epoch_value = epochs.get(raw_carrier_id, epochs.get(carrier_id))
            try:
                status = CarrierRouteStatus(str(status_value))
            except ValueError:
                continue
            if mission.route_status is status:
                continue
            self._carrier_missions[carrier_id] = mission.model_copy(
                update={"route_status": status}
            )
            if status is CarrierRouteStatus.COMPLETE:
                self._emit(
                    "carrier_returned_to_fleet",
                    carrier_id,
                    {"role": mission.role},
                    dedupe_id=(
                        None
                        if epoch_value is None
                        else f"route:{epoch_value}"
                    ),
                )

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
        *,
        dedupe_id: str | None = None,
    ) -> None:
        episode = self._resource_episode_by_uuv.get(entity_id or "", 0)
        key = (event_type, entity_id, episode, dedupe_id)
        if key in self._emitted:
            return
        self._emitted.add(key)
        self._emitted_order.append(key)
        while len(self._emitted_order) > self._event_history_limit:
            self._emitted.discard(self._emitted_order.popleft())
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
        if len(self._events) > self._event_history_limit:
            del self._events[: -self._event_history_limit]


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


def _observed_float(mapping: Mapping[object, object], key: str, default: float) -> float:
    value = mapping.get(key)
    return _float(value, default) if value is not None else default


def _observed_bool(mapping: Mapping[object, object], key: str, default: bool) -> bool:
    value = mapping.get(key)
    return _bool(value, default) if value is not None else default


def _observed_string(mapping: Mapping[object, object], key: str, default: str) -> str:
    value = mapping.get(key)
    return default if value is None else str(value)


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "healthy", "active"}:
            return True
        if normalized in {"false", "0", "no", "off", "failed", "inactive"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _health_check_passed(observations: Observation, uuv_id: str) -> bool:
    value = observations.get("health_check_passed")
    if isinstance(value, Mapping):
        return _bool(value.get(uuv_id), False)
    if value is True:
        return True
    return uuv_id in _strings(value)


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
