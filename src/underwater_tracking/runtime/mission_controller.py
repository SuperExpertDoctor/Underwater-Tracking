from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from pydantic import ConfigDict, Field

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
    ExecutableMissionPlan,
    HandoffEvidence,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionMode,
    UUVResourceState,
    validate_region_transition,
)
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, StrictModel
from underwater_tracking.planning.coverage import (
    serpentine_coverage_waypoints,
    serpentine_coverage_waypoints_by_uuv,
)
from underwater_tracking.runtime.execution_health import classify_execution_health


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


_REGION_PROGRESS_ORDER = {
    RegionLifecycle.PLANNED: 0,
    RegionLifecycle.CARRIER_DEPLOYING: 1,
    RegionLifecycle.ACTIVE_SCAN: 2,
    RegionLifecycle.PASSIVE_TRACK: 3,
    RegionLifecycle.HANDOFF_PENDING: 4,
    RegionLifecycle.TRACKING_COMPLETED: 5,
    RegionLifecycle.CARRIER_RECOVERY: 6,
    RegionLifecycle.RECOVERED: 7,
}
_REGION_PROGRESS_STATES = frozenset(
    {
        RegionLifecycle.CARRIER_DEPLOYING,
        RegionLifecycle.ACTIVE_SCAN,
        RegionLifecycle.PASSIVE_TRACK,
        RegionLifecycle.HANDOFF_PENDING,
        RegionLifecycle.TRACKING_COMPLETED,
        RegionLifecycle.CARRIER_RECOVERY,
        RegionLifecycle.RECOVERED,
    }
)
_HANDOFF_TOPOLOGY_STATES = frozenset(
    {
        RegionLifecycle.CARRIER_DEPLOYING,
        RegionLifecycle.ACTIVE_SCAN,
        RegionLifecycle.PASSIVE_TRACK,
        RegionLifecycle.HANDOFF_PENDING,
        RegionLifecycle.CARRIER_RECOVERY,
    }
)


def _region_plan_assignments_match(
    current: RegionMissionState,
    candidate: RegionMissionState,
) -> bool:
    """Return whether a candidate is a compatible rolling refresh."""
    assignments_match = (
        current.region_id == candidate.region_id
        and current.target_id == candidate.target_id
        and frozenset(current.active_scan_uuv_ids)
        == frozenset(candidate.active_scan_uuv_ids)
        and frozenset(current.passive_track_uuv_ids)
        == frozenset(candidate.passive_track_uuv_ids)
        and frozenset(current.reserve_uuv_ids)
        == frozenset(candidate.reserve_uuv_ids)
    )
    if not assignments_match:
        return False
    if current.lifecycle in _HANDOFF_TOPOLOGY_STATES:
        return (
            current.handoff_from == candidate.handoff_from
            and current.handoff_to == candidate.handoff_to
        )
    return True


def _preserve_region_progress(
    current: RegionMissionState,
    candidate: RegionMissionState,
) -> RegionMissionState:
    """Keep live lifecycle progress while adopting a compatible plan refresh."""
    if not _region_plan_assignments_match(current, candidate):
        return candidate
    if current.lifecycle not in _REGION_PROGRESS_STATES:
        return candidate
    candidate_rank = _REGION_PROGRESS_ORDER.get(candidate.lifecycle)
    current_rank = _REGION_PROGRESS_ORDER[current.lifecycle]
    if candidate_rank is None or current_rank < candidate_rank:
        return candidate
    return candidate.model_copy(
        update={
            "lifecycle": current.lifecycle,
            "coverage": max(current.coverage, candidate.coverage),
            "tracking_quality": max(
                current.tracking_quality,
                candidate.tracking_quality,
            ),
            "entry_confirmations": max(
                current.entry_confirmations,
                candidate.entry_confirmations,
            ),
            "degraded_reasons": tuple(
                dict.fromkeys(
                    (*candidate.degraded_reasons, *current.degraded_reasons)
                )
            ),
        }
    )


def execution_snapshot_to_mission_plan(
    snapshot: OperationalExecutionSnapshot,
    *,
    current_region_lifecycles: Mapping[str, RegionLifecycle] | None = None,
) -> ExecutableMissionPlan:
    """Project one authoritative execution snapshot into controller state.

    A snapshot has no separate public status for ``RECOVERED``. Keep a
    controller's terminal recovery state when a semantic refresh carries the
    corresponding terminal ``monitoring_complete`` status instead of
    regressing the physical lifecycle to active tracking.
    """

    groups_by_region = {group.region_id: group for group in snapshot.task_groups}
    lifecycle_by_status = {
        "planned": RegionLifecycle.PLANNED,
        "prepositioning": RegionLifecycle.PLANNED,
        "active": RegionLifecycle.ACTIVE_SCAN,
        "passive": RegionLifecycle.PASSIVE_TRACK,
        "handoff_pending": RegionLifecycle.HANDOFF_PENDING,
        "handoff_completed": RegionLifecycle.CARRIER_RECOVERY,
        "monitoring_complete": RegionLifecycle.TRACKING_COMPLETED,
        "degraded": RegionLifecycle.DEGRADED,
        "uncovered": RegionLifecycle.UNCOVERED,
    }
    assignments: list[RegionMissionState] = []
    resource_episodes: dict[str, int] = {}
    for region in snapshot.regions:
        group = groups_by_region[region.region_id]
        active_ids = (group.active_verifier_uuv_id,)
        passive_ids = (group.passive_tracker_uuv_id,)
        lifecycle = lifecycle_by_status[region.status]
        current_lifecycle = (current_region_lifecycles or {}).get(region.region_id)
        if current_lifecycle in {
            RegionLifecycle.CARRIER_RECOVERY,
            RegionLifecycle.RECOVERED,
        } and lifecycle in {
            RegionLifecycle.PASSIVE_TRACK,
            RegionLifecycle.TRACKING_COMPLETED,
            RegionLifecycle.CARRIER_RECOVERY,
        }:
            lifecycle = current_lifecycle
        for uuv_id in group.member_uuv_ids:
            resource_episodes[uuv_id] = 0
        coverage_ids = (*active_ids, *passive_ids)
        coverage_degraded_reasons: tuple[str, ...] = ()
        try:
            scan_waypoints = serpentine_coverage_waypoints(
                region.geometry,
                lane_count=max(1, len(coverage_ids)),
            )
            scan_waypoints_by_uuv = serpentine_coverage_waypoints_by_uuv(
                region.geometry,
                coverage_ids,
                start_point=region.geometry[0],
            )
        except ValueError:
            # Preserve the authoritative task rather than silently dropping it.
            # The fallback is explicit and degraded; valid polygons take the
            # existing deterministic multi-UUV lane splitter above.
            scan_waypoints = region.geometry
            scan_waypoints_by_uuv = {
                uuv_id: region.geometry for uuv_id in coverage_ids
            }
            coverage_degraded_reasons = ("coverage_path_unavailable",)
        assignments.append(
            RegionMissionState(
                region_id=region.region_id,
                target_id=region.target_id,
                task_group_id=group.task_group_id,
                lifecycle=lifecycle,
                active_scan_uuv_ids=active_ids,
                passive_track_uuv_ids=passive_ids,
                coverage=1.0 if region.status == "active" else 0.0,
                tracking_quality=1.0 if region.status in {"active", "passive"} else 0.0,
                handoff_from=region.predecessor_region_id,
                handoff_to=region.successor_region_id,
                plan_revision=snapshot.execution_revision,
                degraded_reasons=(
                    *(
                        ("execution_snapshot_degraded",)
                        if snapshot.degradation.degraded
                        else ()
                    ),
                    *coverage_degraded_reasons,
                ),
                region_polygon=region.geometry,
                scan_waypoints=scan_waypoints,
                scan_waypoints_by_uuv=scan_waypoints_by_uuv,
            )
        )
    for reserve in snapshot.reserve_uuvs:
        resource_episodes[reserve.uuv_id] = reserve.resource_episode
    return ExecutableMissionPlan(
        revision=snapshot.execution_revision,
        region_assignments=tuple(assignments),
        task_groups=snapshot.task_groups,
        reserve_uuvs=snapshot.reserve_uuvs,
        resource_episode_by_uuv=resource_episodes,
        degraded_reasons=snapshot.degradation.reasons,
    )


@dataclass(frozen=True)
class MissionControllerCheckpoint:
    sim_time_s: int
    plan_revision: int
    regions: dict[str, RegionMissionState]
    uuv_modes: dict[str, UUVMissionMode]
    uuv_resources: dict[str, UUVResourceState]
    resource_episode_by_uuv: dict[str, int]
    uuv_carrier_ids: dict[str, str]
    dedicated_target_by_uuv: dict[str, str]
    carrier_missions: dict[str, CarrierMissionModel]
    recovered_uuv_ids_by_region: dict[str, set[str]]
    unavailable_until_by_uuv: dict[str, int]
    pending_boundary_entries: dict[str, tuple[str, str, str | None]]
    events: list[RuntimeEvent]
    emitted: set[tuple[str, str | None, int, str | None]]
    emitted_order: deque[tuple[str, str | None, int, str | None]]


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
        initial_uuv_resources: Mapping[str, UUVResourceState] | None = None,
        initial_carrier_missions: Mapping[str, CarrierMissionModel] | None = None,
        uuv_owner_by_id: Mapping[str, str] | None = None,
        region_entry_probability_threshold: float = 0.70,
        region_transition_confirm_cycles: int = 2,
        resource_warning_mileage_fraction: float = 0.02,
        group_min_size: int = 2,
        max_uuv_mileage_m: float = 50_000.0,
        min_energy_fraction: float = 0.10,
        refuel_cooldown_s: int = 120,
        event_history_limit: int = 2048,
        execution_hard_stale_s: float = 900.0,
    ) -> None:
        if not 0.0 <= region_entry_probability_threshold <= 1.0:
            raise ValueError("region_entry_probability_threshold must be in [0, 1]")
        if region_transition_confirm_cycles < 1:
            raise ValueError("region_transition_confirm_cycles must be positive")
        if not 0.0 < resource_warning_mileage_fraction <= 1.0:
            raise ValueError("resource_warning_mileage_fraction must be in (0, 1]")
        if group_min_size < 1:
            raise ValueError("group_min_size must be positive")
        if max_uuv_mileage_m <= 0.0:
            raise ValueError("max_uuv_mileage_m must be positive")
        if not 0.0 <= min_energy_fraction <= 1.0:
            raise ValueError("min_energy_fraction must be in [0, 1]")
        if refuel_cooldown_s < 1:
            raise ValueError("refuel_cooldown_s must be positive")
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be positive")
        if not isfinite(execution_hard_stale_s) or execution_hard_stale_s <= 0.0:
            raise ValueError("execution_hard_stale_s must be finite and positive")
        self._scenario_id = scenario_id
        configured_owners = dict(uuv_owner_by_id or {})
        for uuv_id, resource in (initial_uuv_resources or {}).items():
            if resource.carrier_id is None:
                raise ValueError(f"initial UUV resource {uuv_id!r} requires a carrier_id")
            configured = configured_owners.get(uuv_id)
            if configured is not None and configured != resource.carrier_id:
                raise ValueError(f"initial UUV resource owner mismatch for {uuv_id!r}")
            configured_owners[uuv_id] = resource.carrier_id
        self._configured_uuv_owner_by_id = configured_owners
        self._entry_threshold = region_entry_probability_threshold
        self._confirm_cycles = region_transition_confirm_cycles
        self._resource_warning_mileage_fraction = resource_warning_mileage_fraction
        self._group_min_size = group_min_size
        self._max_mileage_m = max_uuv_mileage_m
        self._min_energy_fraction = min_energy_fraction
        self._refuel_cooldown_s = refuel_cooldown_s
        self._sim_time_s = 0
        self._plan_revision = 0
        self._regions: dict[str, RegionMissionState] = {}
        self._uuv_modes: dict[str, UUVMissionMode] = {}
        self._uuv_resources: dict[str, UUVResourceState] = {}
        self._resource_episode_by_uuv: dict[str, int] = {}
        self._uuv_carrier_ids: dict[str, str] = {}
        self._dedicated_target_by_uuv: dict[str, str] = {}
        self._carrier_missions = {
            carrier_id: mission.model_copy(deep=True)
            for carrier_id, mission in (initial_carrier_missions or {}).items()
        }
        self._recovered_uuv_ids_by_region: dict[str, set[str]] = {}
        self._unavailable_until_by_uuv: dict[str, int] = {}
        self._pending_boundary_entries: dict[str, tuple[str, str, str | None]] = {}
        self._events: list[RuntimeEvent] = []
        self._emitted: set[tuple[str, str | None, int, str | None]] = set()
        self._event_history_limit = event_history_limit
        self._execution_hard_stale_s = float(execution_hard_stale_s)
        self._emitted_order: deque[tuple[str, str | None, int, str | None]] = deque()
        for uuv_id, resource in sorted((initial_uuv_resources or {}).items()):
            deployment_state = resource.deployment_state.lower()
            mode = {
                "onboard": UUVMissionMode.ONBOARD,
                "deployed": UUVMissionMode.ACTIVE_SCAN,
                "returning": UUVMissionMode.RETURN_REQUIRED,
                "failed": UUVMissionMode.FAILED,
            }.get(deployment_state)
            if mode is None:
                raise ValueError(
                    f"unsupported initial deployment_state {resource.deployment_state!r}"
                )
            self._uuv_modes[uuv_id] = mode
            self._uuv_carrier_ids[uuv_id] = resource.carrier_id  # type: ignore[assignment]
            self._resource_episode_by_uuv[uuv_id] = resource.resource_episode
            self._uuv_resources[uuv_id] = resource

    @property
    def max_uuv_mileage_m(self) -> float:
        """Configured maximum sortie mileage used by execution preflight."""
        return self._max_mileage_m

    @property
    def group_min_size(self) -> int:
        """Minimum number of distinct UUV observers required for handoff."""
        return self._group_min_size

    @property
    def scenario_id(self) -> str:
        """Scenario identity owned by this controller."""
        return self._scenario_id

    @property
    def min_energy_fraction(self) -> float:
        """Configured energy reserve used by execution preflight."""
        return self._min_energy_fraction

    @property
    def resource_warning_mileage_m(self) -> float:
        """Mileage at which the current sortie enters its rotation reserve."""
        return self._max_mileage_m * self._resource_warning_mileage_fraction

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

    def checkpoint(self) -> MissionControllerCheckpoint:
        """Copy all mutable mission state for a transition rollback."""
        return MissionControllerCheckpoint(
            sim_time_s=self._sim_time_s,
            plan_revision=self._plan_revision,
            regions=deepcopy(self._regions),
            uuv_modes=deepcopy(self._uuv_modes),
            uuv_resources=deepcopy(self._uuv_resources),
            resource_episode_by_uuv=deepcopy(self._resource_episode_by_uuv),
            uuv_carrier_ids=deepcopy(self._uuv_carrier_ids),
            dedicated_target_by_uuv=deepcopy(self._dedicated_target_by_uuv),
            carrier_missions=deepcopy(self._carrier_missions),
            recovered_uuv_ids_by_region=deepcopy(self._recovered_uuv_ids_by_region),
            unavailable_until_by_uuv=deepcopy(self._unavailable_until_by_uuv),
            pending_boundary_entries=deepcopy(self._pending_boundary_entries),
            events=deepcopy(self._events),
            emitted=deepcopy(self._emitted),
            emitted_order=deepcopy(self._emitted_order),
        )

    def restore(self, checkpoint: MissionControllerCheckpoint) -> None:
        """Restore a checkpoint without retaining references to its caller."""
        self._sim_time_s = checkpoint.sim_time_s
        self._plan_revision = checkpoint.plan_revision
        self._regions = deepcopy(checkpoint.regions)
        self._uuv_modes = deepcopy(checkpoint.uuv_modes)
        self._uuv_resources = deepcopy(checkpoint.uuv_resources)
        self._resource_episode_by_uuv = deepcopy(checkpoint.resource_episode_by_uuv)
        self._uuv_carrier_ids = deepcopy(checkpoint.uuv_carrier_ids)
        self._dedicated_target_by_uuv = deepcopy(checkpoint.dedicated_target_by_uuv)
        self._carrier_missions = deepcopy(checkpoint.carrier_missions)
        self._recovered_uuv_ids_by_region = deepcopy(checkpoint.recovered_uuv_ids_by_region)
        self._unavailable_until_by_uuv = deepcopy(checkpoint.unavailable_until_by_uuv)
        self._pending_boundary_entries = deepcopy(checkpoint.pending_boundary_entries)
        self._events = deepcopy(checkpoint.events)
        self._emitted = deepcopy(checkpoint.emitted)
        self._emitted_order = deepcopy(checkpoint.emitted_order)

    def apply_revalidated_plan(
        self,
        plan: ExecutableMissionPlan,
        *,
        expected_current_revision: int,
        preserve_region_progress: bool = True,
    ) -> bool:
        """Apply a semantically revalidated plan under the transition lock."""
        if self._plan_revision != expected_current_revision:
            return False
        return self.apply_verified_plan(
            plan,
            preserve_region_progress=preserve_region_progress,
        )

    def apply_committed_plan(
        self,
        plan: ExecutableMissionPlan,
        *,
        expected_current_revision: int,
        preserve_region_progress: bool = True,
    ) -> bool:
        """Refresh the controller projection for an already committed revision."""
        if self._plan_revision != expected_current_revision:
            return False
        return self._apply_plan(
            plan,
            allow_same_revision=True,
            preserve_region_progress=preserve_region_progress,
        )

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return self.snapshot().events

    def apply_verified_plan(
        self,
        plan: ExecutableMissionPlan,
        *,
        preserve_region_progress: bool = True,
    ) -> bool:
        """Atomically apply only a strictly newer executable plan."""
        return self._apply_plan(
            plan,
            allow_same_revision=False,
            preserve_region_progress=preserve_region_progress,
        )

    @property
    def execution_revision(self) -> int:
        """Return the execution revision currently installed in the controller."""

        return self._plan_revision

    def apply_execution_snapshot(
        self,
        snapshot: OperationalExecutionSnapshot | Mapping[str, Any],
        *,
        expected_current_revision: int | None = None,
    ) -> bool:
        """Apply an authoritative snapshot without exposing carrier execution."""

        health = classify_execution_health(
            snapshot,
            sim_time_s=float(self._sim_time_s),
            hard_stale_s=self._execution_hard_stale_s,
        )
        if not health.executable:
            return False
        if not isinstance(snapshot, OperationalExecutionSnapshot):
            try:
                snapshot = OperationalExecutionSnapshot.model_validate(snapshot)
            except (TypeError, ValueError):
                return False
        if snapshot.scenario_id != self._scenario_id:
            return False
        expected = (
            self._plan_revision
            if expected_current_revision is None
            else expected_current_revision
        )
        if self._plan_revision != expected:
            return False
        if snapshot.base_execution_revision not in (None, expected):
            return False
        checkpoint = self.checkpoint()
        try:
            current_region_lifecycles = {
                region.region_id: region.lifecycle for region in self._regions.values()
            }
            applied = self._apply_plan(
                execution_snapshot_to_mission_plan(
                    snapshot,
                    current_region_lifecycles=current_region_lifecycles,
                ),
                allow_same_revision=False,
                preserve_region_progress=False,
            )
        except Exception:  # noqa: BLE001 - restore the complete controller boundary
            self.restore(checkpoint)
            return False
        if not applied:
            self.restore(checkpoint)
            return False
        return True

    def _apply_plan(
        self,
        plan: ExecutableMissionPlan,
        *,
        allow_same_revision: bool,
        preserve_region_progress: bool = True,
    ) -> bool:
        """Build and install a complete plan projection without partial writes."""
        if plan.revision < self._plan_revision or (
            plan.revision == self._plan_revision and not allow_same_revision
        ):
            return False
        new_regions = {
            region.region_id: region.model_copy(deep=True)
            for region in plan.region_assignments
        }
        previous_regions = self._regions
        previous_recovered_uuv_ids_by_region = self._recovered_uuv_ids_by_region
        previous_uuv_carrier_ids = self._uuv_carrier_ids
        previous_carrier_missions = self._carrier_missions
        if preserve_region_progress:
            new_regions = {
                region_id: _preserve_region_progress(
                    self._regions[region_id],
                    region,
                )
                if region_id in self._regions
                else region
                for region_id, region in new_regions.items()
            }
        preserved_recovered_uuv_ids_by_region: dict[str, set[str]] = {}
        for region_id, region in new_regions.items():
            previous = previous_regions.get(region_id)
            assigned = {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }
            if (
                previous is not None
                and previous.lifecycle
                in {RegionLifecycle.CARRIER_RECOVERY, RegionLifecycle.RECOVERED}
                and region.lifecycle
                in {RegionLifecycle.CARRIER_RECOVERY, RegionLifecycle.RECOVERED}
                and _region_plan_assignments_match(previous, region)
            ):
                preserved_recovered_uuv_ids_by_region[region_id] = (
                    previous_recovered_uuv_ids_by_region.get(region_id, set())
                    & assigned
                )
            else:
                preserved_recovered_uuv_ids_by_region[region_id] = set()
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
        for reserve in plan.reserve_uuvs:
            if reserve.status in {"unavailable", "exiting"}:
                new_modes[reserve.uuv_id] = UUVMissionMode.RECOVERING
            else:
                new_modes[reserve.uuv_id] = UUVMissionMode.ONBOARD
        active_execution_uuv_ids: set[str] = set()
        for group in plan.task_groups:
            if len(group.member_uuv_ids) != 2:
                return False
            new_modes[group.active_verifier_uuv_id] = UUVMissionMode.ACTIVE_SCAN
            new_modes[group.passive_tracker_uuv_id] = UUVMissionMode.PASSIVE_TRACK
            if group.status != "complete":
                active_execution_uuv_ids.add(group.active_verifier_uuv_id)
        for region in new_regions.values():
            if region.lifecycle is RegionLifecycle.PASSIVE_TRACK:
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                ):
                    if uuv_id not in active_execution_uuv_ids:
                        new_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            elif region.lifecycle is RegionLifecycle.ACTIVE_SCAN:
                for uuv_id in region.active_scan_uuv_ids:
                    new_modes[uuv_id] = UUVMissionMode.ACTIVE_SCAN
            elif region.lifecycle in {
                RegionLifecycle.TRACKING_COMPLETED,
                RegionLifecycle.CARRIER_RECOVERY,
            }:
                recovered_uuv_ids = preserved_recovered_uuv_ids_by_region.get(
                    region.region_id, set()
                )
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                ):
                    if uuv_id in recovered_uuv_ids:
                        new_modes[uuv_id] = UUVMissionMode.ONBOARD
                    elif uuv_id not in self._dedicated_target_by_uuv:
                        new_modes[uuv_id] = UUVMissionMode.RETURN_REQUIRED
            elif region.lifecycle is RegionLifecycle.RECOVERED:
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                ):
                    if uuv_id not in self._dedicated_target_by_uuv:
                        new_modes[uuv_id] = UUVMissionMode.ONBOARD

        # A rolling plan can still reference a UUV while its physical return
        # is in progress.  Rebuilding task-group modes must not cancel that
        # lifecycle transition before the boundary-exit observation arrives.
        for uuv_id, previous_mode in self._uuv_modes.items():
            if (
                previous_mode is UUVMissionMode.RETURN_REQUIRED
                and uuv_id in new_modes
            ):
                new_modes[uuv_id] = previous_mode

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

        # Task-group execution snapshots intentionally carry no carrier
        # inventory. Keep the controller's authoritative ownership metadata
        # so a rolling refresh cannot erase recovery bookkeeping.
        if not plan.carrier_missions and not plan.batches:
            for uuv_id in new_modes:
                carrier_id = previous_uuv_carrier_ids.get(uuv_id)
                if carrier_id is not None:
                    new_uuv_carrier_ids.setdefault(uuv_id, carrier_id)

        new_carrier_missions = {
            carrier_id: carrier.model_copy(deep=True)
            for carrier_id, carrier in plan.carrier_missions.items()
        }
        if not plan.carrier_missions and not plan.batches:
            new_carrier_missions = {
                carrier_id: carrier.model_copy(deep=True)
                for carrier_id, carrier in previous_carrier_missions.items()
            }
        for uuv_id in sorted(rotated_uuv_ids):
            recovery_carrier_id = new_uuv_carrier_ids.get(uuv_id)
            if recovery_carrier_id is None:
                continue
            recovery_carrier = new_carrier_missions.get(recovery_carrier_id)
            if recovery_carrier is None:
                recovery_carrier = self._carrier_missions.get(recovery_carrier_id)
            if recovery_carrier is None:
                continue
            new_carrier_missions[recovery_carrier_id] = recovery_carrier.model_copy(
                update={
                    "onboard_uuv_ids": tuple(
                        item
                        for item in recovery_carrier.onboard_uuv_ids
                        if item != uuv_id
                    ),
                    "ready_uuv_ids": tuple(
                        item
                        for item in recovery_carrier.ready_uuv_ids
                        if item != uuv_id
                    ),
                    "reserved_uuv_ids": tuple(
                        item
                        for item in recovery_carrier.reserved_uuv_ids
                        if item != uuv_id
                    ),
                    "recoverable_uuv_ids": tuple(
                        sorted({*recovery_carrier.recoverable_uuv_ids, uuv_id})
                    ),
                }
            )
        for uuv_id, carrier_id in new_uuv_carrier_ids.items():
            configured_carrier_id = self._configured_uuv_owner_by_id.get(uuv_id)
            if configured_carrier_id is not None and carrier_id != configured_carrier_id:
                return False
        for uuv_id, available_at_s in self._unavailable_until_by_uuv.items():
            if uuv_id in new_modes and self._sim_time_s < available_at_s:
                new_modes[uuv_id] = UUVMissionMode.RECOVERING
        self._plan_revision = plan.revision
        self._regions = new_regions
        self._uuv_modes = new_modes
        self._uuv_carrier_ids = new_uuv_carrier_ids
        self._recovered_uuv_ids_by_region = preserved_recovered_uuv_ids_by_region
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
        exit_uuv_ids = tuple(
            sorted(
                uuv_id
                for region in self._regions.values()
                if region.target_id == target_id
                for uuv_id in (
                    *region.active_scan_uuv_ids,
                    *region.passive_track_uuv_ids,
                )
                if uuv_id not in selected
                and self._uuv_modes.get(uuv_id)
                in {
                    UUVMissionMode.TRANSIT_TO_REGION,
                    UUVMissionMode.ACTIVE_SCAN,
                    UUVMissionMode.PASSIVE_TRACK,
                }
            )
        )
        for uuv_id in exit_uuv_ids:
            self._mark_uuv_for_boundary_exit(uuv_id)
        if exit_uuv_ids:
            self._emit(
                "dedicated_group_regional_exit_requested",
                target_id,
                {"uuv_ids": exit_uuv_ids},
            )
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
        selected_ids = {str(uuv_id) for uuv_id in uuv_ids}
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

    def begin_boundary_exit(
        self,
        uuv_id: str,
        region: RegionMissionState | str,
        *,
        reason: str = "boundary_rotation",
    ) -> bool:
        """Mark a regional UUV for physical exit through its own boundary."""
        selected = self._resolve_boundary_region(region)
        if selected is None or uuv_id not in {
            *selected.active_scan_uuv_ids,
            *selected.passive_track_uuv_ids,
        }:
            return False
        if self._uuv_modes.get(uuv_id) not in {
            UUVMissionMode.TRANSIT_TO_REGION,
            UUVMissionMode.ACTIVE_SCAN,
            UUVMissionMode.PASSIVE_TRACK,
            UUVMissionMode.RETURN_TO_REGION,
        }:
            return False
        self._mark_uuv_for_boundary_exit(uuv_id)
        self._emit(
            "uuv_rotation",
            uuv_id,
            {"region_id": selected.region_id, "reason": reason, "boundary": True},
            dedupe_id=f"boundary-exit:{selected.region_id}",
        )
        return True

    def complete_boundary_exit(self, uuv_id: str) -> bool:
        """Accept a boundary-exit observation and make the UUV unavailable."""
        if self._uuv_modes.get(uuv_id) is not UUVMissionMode.RETURN_REQUIRED:
            return False
        return bool(
            self._apply_boundary_exit_observations(
                {"boundary_exited_uuv_ids": (uuv_id,)}
            )
        )

    def begin_boundary_entry(
        self,
        uuv_id: str,
        region: RegionMissionState | str,
        *,
        role: str,
        outgoing_uuv_id: str | None = None,
    ) -> bool:
        """Reserve a task slot for a UUV entering from the same region edge."""
        selected = self._resolve_boundary_region(region)
        normalized_role = _normalize_boundary_role(role)
        if selected is None or normalized_role is None:
            return False
        if uuv_id not in selected.reserve_uuv_ids:
            return False
        if self._uuv_modes.get(uuv_id) is not UUVMissionMode.ONBOARD:
            return False
        self._pending_boundary_entries[uuv_id] = (
            selected.region_id,
            normalized_role,
            outgoing_uuv_id,
        )
        return True

    def complete_boundary_replacement(
        self,
        incoming_uuv_id: str,
        *,
        outgoing_uuv_id: str,
        observation_ids: Sequence[str] = (),
        valid_observation: bool = False,
        reason: str = "boundary_replacement",
    ) -> bool:
        """Commit an entering UUV only after current-cycle evidence is valid."""
        pending = self._pending_boundary_entries.get(incoming_uuv_id)
        if pending is None or (not valid_observation and not tuple(observation_ids)):
            return False
        region_id, role, expected_outgoing = pending
        if expected_outgoing is not None and expected_outgoing != outgoing_uuv_id:
            return False
        region = self._regions.get(region_id)
        if region is None or outgoing_uuv_id not in {
            *region.active_scan_uuv_ids,
            *region.passive_track_uuv_ids,
        }:
            return False
        if self._uuv_modes.get(incoming_uuv_id) not in {
            UUVMissionMode.ONBOARD,
            UUVMissionMode.ACTIVE_SCAN,
            UUVMissionMode.PASSIVE_TRACK,
        }:
            return False
        active_ids = tuple(
            incoming_uuv_id if item == outgoing_uuv_id else item
            for item in region.active_scan_uuv_ids
        )
        passive_ids = tuple(
            incoming_uuv_id if item == outgoing_uuv_id else item
            for item in region.passive_track_uuv_ids
        )
        routes = dict(region.scan_waypoints_by_uuv)
        outgoing_route = routes.pop(outgoing_uuv_id, None)
        if outgoing_route is not None:
            routes[incoming_uuv_id] = outgoing_route
        self._regions[region_id] = region.model_copy(
            update={
                "active_scan_uuv_ids": active_ids,
                "passive_track_uuv_ids": passive_ids,
                "reserve_uuv_ids": tuple(
                    item for item in region.reserve_uuv_ids if item != incoming_uuv_id
                ),
                "scan_waypoints_by_uuv": routes,
            }
        )
        self._uuv_modes[incoming_uuv_id] = (
            UUVMissionMode.ACTIVE_SCAN
            if role == "active_scan"
            else UUVMissionMode.PASSIVE_TRACK
        )
        self._resource_episode_by_uuv[incoming_uuv_id] = (
            self._resource_episode_by_uuv.get(incoming_uuv_id, 0) + 1
        )
        resource = self._uuv_resources.get(incoming_uuv_id)
        if resource is not None:
            self._uuv_resources[incoming_uuv_id] = resource.model_copy(
                update={
                    "deployment_state": self._uuv_modes[incoming_uuv_id].value,
                    "resource_episode": self._resource_episode_by_uuv[incoming_uuv_id],
                }
            )
        self._pending_boundary_entries.pop(incoming_uuv_id, None)
        self._emit(
            "uuv_boundary_replacement",
            outgoing_uuv_id,
            {
                "outgoing_uuv_id": outgoing_uuv_id,
                "replacement_uuv_id": incoming_uuv_id,
                "region_id": region_id,
                "role": role,
                "reason": reason,
                "observation_ids": tuple(str(item) for item in observation_ids),
            },
            dedupe_id=f"boundary-replacement:{region_id}:{incoming_uuv_id}",
        )
        return True

    def _resolve_boundary_region(
        self,
        region: RegionMissionState | str,
    ) -> RegionMissionState | None:
        if isinstance(region, RegionMissionState):
            return self._regions.get(region.region_id, region)
        return self._regions.get(str(region))

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
        boundary_exited_uuv_ids = self._apply_boundary_exit_observations(observed)
        self._release_refueled_uuvs()
        self._apply_resource_observations(
            observed,
            skip_uuv_ids=recovered_uuv_ids | boundary_exited_uuv_ids,
        )
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
                    if uuv_id in self._dedicated_target_by_uuv:
                        mode = UUVMissionMode.DEDICATED_TRACK
                    elif uuv_id in region.passive_track_uuv_ids:
                        mode = UUVMissionMode.PASSIVE_TRACK
                    else:
                        mode = UUVMissionMode.ACTIVE_SCAN
                    self._uuv_modes[uuv_id] = mode
            for uuv_id in required:
                self._remove_uuv_from_carrier_inventory(uuv_id)

    def _apply_entry_observations(self, observations: Observation) -> None:
        probabilities = _mapping(observations.get("entry_probability"))
        predicted_exit_region_id = str(
            observations.get("target_exit_predicted", "")
        )
        for region_id, region in tuple(self._regions.items()):
            if region.lifecycle is not RegionLifecycle.ACTIVE_SCAN:
                continue
            if region_id == predicted_exit_region_id:
                # An exit prediction in the same cycle takes precedence over
                # entry confirmation for the active predecessor.
                self._regions[region_id] = region.model_copy(
                    update={"entry_confirmations": 0}
                )
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
        raw_evidence = observations.get("handoff_evidence")
        if isinstance(raw_evidence, HandoffEvidence):
            evidence_items: tuple[object, ...] = (raw_evidence,)
        elif isinstance(raw_evidence, Mapping) and "predecessor_region_id" in raw_evidence:
            evidence_items = (raw_evidence,)
        elif isinstance(raw_evidence, Mapping):
            evidence_items = tuple(
                value
                for _, value in sorted(
                    raw_evidence.items(), key=lambda item: str(item[0])
                )
            )
        elif isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence, (str, bytes, bytearray)
        ):
            evidence_items = tuple(raw_evidence)
        else:
            evidence_items = ()
        for raw_item in evidence_items:
            try:
                evidence = HandoffEvidence.model_validate(raw_item)
            except (TypeError, ValueError):
                continue
            predecessor_id = evidence.predecessor_region_id
            successor_id = evidence.successor_region_id
            predecessor = self._regions.get(predecessor_id)
            successor = self._regions.get(successor_id)
            if predecessor is None or successor is None:
                continue
            if evidence.plan_revision != self._plan_revision:
                continue
            if evidence.observation_cycle_s != self._sim_time_s:
                continue
            required = {
                *successor.active_scan_uuv_ids,
                *successor.passive_track_uuv_ids,
            }
            if set(evidence.required_uuv_ids) != required:
                continue
            if evidence.blocked_reason is not None:
                self._block_handoff(predecessor_id, successor_id, evidence)
                continue
            active_exit = (
                predecessor.lifecycle is RegionLifecycle.ACTIVE_SCAN
                and str(observations.get("target_exit_predicted", ""))
                == predecessor_id
            )
            if predecessor.lifecycle not in {
                RegionLifecycle.PASSIVE_TRACK,
                RegionLifecycle.HANDOFF_PENDING,
            } and not active_exit:
                continue
            if successor.lifecycle not in {
                RegionLifecycle.PLANNED,
                RegionLifecycle.CARRIER_DEPLOYING,
                RegionLifecycle.ACTIVE_SCAN,
                RegionLifecycle.PASSIVE_TRACK,
                # The successor can already be preparing its own next
                # handoff while still providing the overlap evidence needed
                # to complete this predecessor handoff.
                RegionLifecycle.HANDOFF_PENDING,
            }:
                continue
            if not evidence.is_complete(group_min_size=self._group_min_size):
                continue
            if active_exit:
                self._transition(predecessor_id, RegionLifecycle.PASSIVE_TRACK)
                predecessor = self._regions[predecessor_id]
            if successor.lifecycle is RegionLifecycle.PLANNED:
                self._transition(successor_id, RegionLifecycle.CARRIER_DEPLOYING)
            if successor.lifecycle is RegionLifecycle.CARRIER_DEPLOYING:
                self._transition(successor_id, RegionLifecycle.ACTIVE_SCAN)
            if successor.lifecycle is RegionLifecycle.ACTIVE_SCAN:
                self._transition(successor_id, RegionLifecycle.PASSIVE_TRACK)
            for uuv_id in (
                *successor.active_scan_uuv_ids,
                *successor.passive_track_uuv_ids,
            ):
                if uuv_id not in self._dedicated_target_by_uuv:
                    self._uuv_modes[uuv_id] = UUVMissionMode.PASSIVE_TRACK
            if predecessor.lifecycle is RegionLifecycle.PASSIVE_TRACK:
                self._transition(predecessor_id, RegionLifecycle.HANDOFF_PENDING)
            self._transition(predecessor_id, RegionLifecycle.TRACKING_COMPLETED)
            for uuv_id in (
                *predecessor.active_scan_uuv_ids,
                *predecessor.passive_track_uuv_ids,
            ):
                if uuv_id not in self._dedicated_target_by_uuv:
                    self._mark_uuv_for_recovery(uuv_id)
            self._emit(
                "handoff_completed",
                predecessor_id,
                {
                    "target_id": predecessor.target_id,
                    "predecessor_region_id": predecessor_id,
                    "successor_region_id": successor_id,
                    "predecessor_uuv_ids": tuple(
                        sorted(
                            {
                                *predecessor.active_scan_uuv_ids,
                                *predecessor.passive_track_uuv_ids,
                            }
                        )
                    ),
                    "successor_uuv_ids": tuple(sorted(required)),
                    "plan_revision": evidence.plan_revision,
                    "source_observation_ids": tuple(
                        observation.observation_id
                        for observation in evidence.accepted_observations
                    ),
                },
                dedupe_id=f"handoff:r{evidence.plan_revision}:{successor_id}",
            )

    def _block_handoff(
        self,
        predecessor_id: str,
        successor_id: str,
        evidence: HandoffEvidence,
    ) -> None:
        predecessor = self._regions[predecessor_id]
        if predecessor.lifecycle is RegionLifecycle.PASSIVE_TRACK:
            self._transition(predecessor_id, RegionLifecycle.HANDOFF_PENDING)
            predecessor = self._regions[predecessor_id]
        if predecessor.lifecycle not in {
            RegionLifecycle.HANDOFF_PENDING,
            RegionLifecycle.DEGRADED,
        }:
            return
        reason = evidence.blocked_reason or "handoff_blocked"
        if predecessor.lifecycle is RegionLifecycle.HANDOFF_PENDING:
            self._transition(predecessor_id, RegionLifecycle.DEGRADED)
        current = self._regions[predecessor_id]
        if reason not in current.degraded_reasons:
            self._regions[predecessor_id] = current.model_copy(
                update={
                    "degraded_reasons": tuple(
                        sorted({*current.degraded_reasons, reason})
                    )
                }
            )
        # A typed blocker with accepted successor observations has enough
        # evidence to rotate the predecessor.  A legacy/no-observation
        # blocker only degrades the region; withdrawing that waterborne group
        # would start a carrier lifecycle without proof of a handoff.
        if evidence.accepted_observations:
            for uuv_id in (
                *current.active_scan_uuv_ids,
                *current.passive_track_uuv_ids,
            ):
                if uuv_id not in self._dedicated_target_by_uuv:
                    self._mark_uuv_for_recovery(uuv_id)
        self._emit(
            "handoff_blocked",
            predecessor_id,
            {
                "successor_region_id": successor_id,
                "plan_revision": evidence.plan_revision,
                "reason": reason,
                "source_observation_ids": tuple(
                    observation.observation_id
                    for observation in evidence.accepted_observations
                ),
            },
            dedupe_id=f"handoff-blocked:r{evidence.plan_revision}:{successor_id}:{reason}",
        )

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
        for uuv_id in sorted(set(self._uuv_modes) | set(self._uuv_resources)):
            self._uuv_modes.setdefault(uuv_id, UUVMissionMode.ONBOARD)
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

    def _apply_boundary_exit_observations(self, observations: Observation) -> set[str]:
        exited_uuv_ids: set[str] = set()
        for uuv_id in _strings(observations.get("boundary_exited_uuv_ids")):
            if self._uuv_modes.get(uuv_id) is not UUVMissionMode.RETURN_REQUIRED:
                continue
            self._uuv_modes[uuv_id] = UUVMissionMode.RECOVERING
            previous = self._uuv_resources.get(uuv_id)
            self._uuv_resources[uuv_id] = UUVResourceState(
                uuv_id=uuv_id,
                carrier_id=self._uuv_carrier_ids.get(uuv_id),
                mileage_m=previous.mileage_m if previous is not None else 0.0,
                energy_fraction=previous.energy_fraction if previous is not None else 0.0,
                healthy=previous.healthy if previous is not None else True,
                capability_active=(
                    previous.capability_active if previous is not None else True
                ),
                deployment_state="unavailable",
                resource_episode=self._resource_episode_by_uuv.get(uuv_id, 0),
            )
            self._unavailable_until_by_uuv[uuv_id] = (
                self._sim_time_s + self._refuel_cooldown_s
            )
            exited_uuv_ids.add(uuv_id)
            self._emit("uuv_boundary_exit_completed", uuv_id)
        return exited_uuv_ids

    def _release_refueled_uuvs(self) -> None:
        for uuv_id, available_at_s in tuple(sorted(self._unavailable_until_by_uuv.items())):
            if self._sim_time_s < available_at_s:
                continue
            self._unavailable_until_by_uuv.pop(uuv_id, None)
            self._uuv_modes[uuv_id] = UUVMissionMode.ONBOARD
            self._restore_normal_mode(uuv_id)
            self._resource_episode_by_uuv[uuv_id] = (
                self._resource_episode_by_uuv.get(uuv_id, 0) + 1
            )
            previous = self._uuv_resources.get(uuv_id)
            self._uuv_resources[uuv_id] = UUVResourceState(
                uuv_id=uuv_id,
                carrier_id=self._uuv_carrier_ids.get(uuv_id),
                mileage_m=0.0,
                energy_fraction=1.0,
                healthy=True,
                capability_active=(
                    previous.capability_active if previous is not None else True
                ),
                deployment_state=self._uuv_modes[uuv_id].value,
                resource_episode=self._resource_episode_by_uuv[uuv_id],
            )
            self._emit("uuv_refueled_active", uuv_id)

    def _apply_resource_observations(
        self,
        observations: Observation,
        *,
        skip_uuv_ids: set[str] | None = None,
    ) -> None:
        del observations
        skipped = skip_uuv_ids or set()
        for uuv_id in sorted(set(self._uuv_modes) | set(self._uuv_resources)):
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
            if (
                mileage_value >= self.resource_warning_mileage_m
                and mileage_value < self._max_mileage_m
            ):
                self._emit(
                    "endurance_threshold_crossed",
                    uuv_id,
                    {
                        "mileage_m": mileage_value,
                        "warning_mileage_m": self.resource_warning_mileage_m,
                        "max_mileage_m": self._max_mileage_m,
                        "energy_fraction": energy_value,
                        "resource_episode": resource.resource_episode,
                    },
                    dedupe_id=f"warning:r{resource.resource_episode}",
                )
                if self._uuv_requires_post_handoff_rotation(uuv_id):
                    self._return_uuv(uuv_id, "battery_rotation")
            if self._uuv_modes[uuv_id] is UUVMissionMode.RETURN_REQUIRED:
                if mileage_value >= self._max_mileage_m:
                    self._emit(
                        "uuv_range_exhausted",
                        uuv_id,
                        {"uuv_id": uuv_id, "reason": "uuv_range_exhausted"},
                    )
                continue
            if (
                uuv_id in self._dedicated_target_by_uuv
                and self._max_mileage_m - mileage_value
                <= self.resource_warning_mileage_m
            ):
                # A human-directed group stays with its target across normal
                # region handoffs, but must keep a configured reserve to
                # rejoin the autonomous regional workflow safely.
                self._return_uuv(uuv_id, "dedicated_range_reserve")
                continue
            if mileage_value >= self._max_mileage_m:
                self._return_uuv(uuv_id, "uuv_range_exhausted")
            elif (
                self._uuv_is_regional_worker(uuv_id)
                and self._max_mileage_m - mileage_value
                <= self.resource_warning_mileage_m
            ):
                self._return_uuv(uuv_id, "uuv_range_reserve")
            elif energy_value <= self._min_energy_fraction:
                self._return_uuv(uuv_id, "uuv_energy_depleted")

    def _uuv_requires_post_handoff_rotation(self, uuv_id: str) -> bool:
        """Rotate a resource only after its completed region was handed off."""
        if uuv_id in self._dedicated_target_by_uuv:
            return False
        return any(
            region.lifecycle is RegionLifecycle.TRACKING_COMPLETED
            and uuv_id
            in {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }
            for region in self._regions.values()
        )

    def _uuv_is_regional_worker(self, uuv_id: str) -> bool:
        return any(
            uuv_id in {
                *region.active_scan_uuv_ids,
                *region.passive_track_uuv_ids,
            }
            for region in self._regions.values()
        )

    def _replace_regional_uuv(self, outgoing_uuv_id: str, reason: str) -> bool:
        region = next(
            (
                candidate
                for candidate in self._regions.values()
                if outgoing_uuv_id
                in {
                    *candidate.active_scan_uuv_ids,
                    *candidate.passive_track_uuv_ids,
                }
            ),
            None,
        )
        if region is None or region.lifecycle is RegionLifecycle.TRACKING_COMPLETED:
            return False
        role = (
            "active_scan"
            if outgoing_uuv_id in region.active_scan_uuv_ids
            else "passive_track"
        )
        assigned_elsewhere = {
            uuv_id
            for candidate in self._regions.values()
            if candidate.region_id != region.region_id
            for uuv_id in (
                *candidate.active_scan_uuv_ids,
                *candidate.passive_track_uuv_ids,
                *candidate.reserve_uuv_ids,
            )
        }
        candidates = tuple(region.reserve_uuv_ids) + tuple(
            uuv_id
            for uuv_id in sorted(self._uuv_modes)
            if uuv_id not in region.reserve_uuv_ids
        )
        replacement_uuv_id = next(
            (
                uuv_id
                for uuv_id in candidates
                if uuv_id != outgoing_uuv_id
                and uuv_id not in assigned_elsewhere
                and self._uuv_modes.get(uuv_id) is UUVMissionMode.ONBOARD
                and uuv_id not in self._dedicated_target_by_uuv
                and (
                    (resource := self._uuv_resources.get(uuv_id)) is None
                    or (resource.healthy and resource.capability_active)
                )
            ),
            None,
        )
        if replacement_uuv_id is None:
            return False

        active_ids = tuple(
            replacement_uuv_id if item == outgoing_uuv_id else item
            for item in region.active_scan_uuv_ids
        )
        passive_ids = tuple(
            replacement_uuv_id if item == outgoing_uuv_id else item
            for item in region.passive_track_uuv_ids
        )
        routes = dict(region.scan_waypoints_by_uuv)
        outgoing_route = routes.pop(outgoing_uuv_id, None)
        if outgoing_route is not None:
            routes[replacement_uuv_id] = outgoing_route
        self._regions[region.region_id] = region.model_copy(
            update={
                "active_scan_uuv_ids": active_ids,
                "passive_track_uuv_ids": passive_ids,
                "reserve_uuv_ids": tuple(
                    item
                    for item in region.reserve_uuv_ids
                    if item != replacement_uuv_id
                ),
                "scan_waypoints_by_uuv": routes,
            }
        )
        replacement_mode = (
            UUVMissionMode.ACTIVE_SCAN
            if role == "active_scan"
            else UUVMissionMode.PASSIVE_TRACK
        )
        self._uuv_modes[replacement_uuv_id] = replacement_mode
        replacement_resource = self._uuv_resources.get(replacement_uuv_id)
        if replacement_resource is not None:
            self._uuv_resources[replacement_uuv_id] = replacement_resource.model_copy(
                update={"deployment_state": replacement_mode.value}
            )
        self._emit(
            "uuv_boundary_replacement",
            outgoing_uuv_id,
            {
                "outgoing_uuv_id": outgoing_uuv_id,
                "replacement_uuv_id": replacement_uuv_id,
                "region_id": region.region_id,
                "role": role,
                "reason": reason,
            },
            dedupe_id=f"{region.region_id}:{replacement_uuv_id}",
        )
        return True

    def _fail_uuv(self, uuv_id: str, event_type: str, reason: str) -> None:
        if self._uuv_modes.get(uuv_id) is UUVMissionMode.FAILED:
            return
        dedicated_target = self._dedicated_target_by_uuv.pop(uuv_id, None)
        self._uuv_modes[uuv_id] = UUVMissionMode.FAILED
        self._degrade_regions_for_uuv(uuv_id, reason)
        self._emit(event_type, uuv_id, {"reason": reason})
        if dedicated_target is None:
            return
        for member_id, target_id in tuple(self._dedicated_target_by_uuv.items()):
            if target_id != dedicated_target:
                continue
            self._dedicated_target_by_uuv.pop(member_id, None)
            self._restore_normal_mode(member_id)
        self._emit(
            "dedicated_mode_released",
            uuv_id,
            {"target_id": dedicated_target, "reason": "member_failure"},
        )

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
        replaced = self._replace_regional_uuv(uuv_id, event_type)
        self._uuv_modes[uuv_id] = UUVMissionMode.RETURN_REQUIRED
        carrier_id = self._uuv_carrier_ids.get(uuv_id)
        if not replaced and carrier_id is not None and carrier_id in self._carrier_missions:
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
        self._emit(
            event_type,
            uuv_id,
            {
                "uuv_id": uuv_id,
                "reason": event_type,
            },
        )

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

    def _mark_uuv_for_boundary_exit(self, uuv_id: str) -> None:
        """Remove a UUV from regional execution without scheduling recovery."""
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
        self._remove_uuv_from_carrier_inventory(uuv_id)

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
                UUVMissionMode.ACTIVE_SCAN
                if uuv_id in region.active_scan_uuv_ids
                else UUVMissionMode.PASSIVE_TRACK
            )
            return
        self._uuv_modes[uuv_id] = UUVMissionMode.ONBOARD

    def _apply_external_events(self, observations: Observation) -> None:
        exit_prediction = observations.get("target_exit_predicted")
        if exit_prediction:
            region_id = str(exit_prediction)
            region = self._regions.get(region_id)
            if (
                region is not None
                and region.lifecycle is RegionLifecycle.PASSIVE_TRACK
                and region.handoff_to is not None
            ):
                self._transition(region_id, RegionLifecycle.HANDOFF_PENDING)
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


def _normalize_boundary_role(role: str) -> str | None:
    normalized = str(role).casefold()
    if normalized in {"active_scan", "active_verifier", "active"}:
        return "active_scan"
    if normalized in {"passive_track", "passive_tracker", "passive"}:
        return "passive_track"
    return None


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
        if value is None:
            return default
        if not isinstance(value, (int, float, str)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
