from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    MissionCandidate,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.domain.regional_models import RegionalMissionCandidate


@dataclass(frozen=True, slots=True)
class _PlatformPool:
    carrier_id: str
    home_battle_group_id: str
    uuv_ids: tuple[str, ...]
    active_capable_uuv_ids: tuple[str, ...] = ()
    carrier_ids: tuple[str, ...] = ()
    uuv_ids_by_carrier: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class MissionOptimizer:
    """Select one rolling UUV deployment batch while protecting future demand.

    The optimizer consumes estimated candidate metadata only.  It does not
    inspect target truth or mutate the platform snapshot.  The current time
    slice is the earliest candidate window; later high-probability candidates
    become reserve assignments for the next rolling cycle.
    """

    def __init__(self, *, home_battle_group_id: str = "home_battle_group") -> None:
        self._home_battle_group_id = home_battle_group_id

    def optimize(
        self,
        snapshot: Any,
        candidates: Sequence[MissionCandidate | RegionalMissionCandidate],
        *,
        locked_uuv_ids_by_candidate: Mapping[str, Sequence[str]] | None = None,
    ) -> ExecutableMissionPlan:
        normalized = _normalize_candidates(candidates)
        pool = _platform_pool(snapshot, self._home_battle_group_id)
        if not normalized:
            return _empty_plan(snapshot, pool)

        current_entry = min(candidate.entry_s for candidate in normalized)
        current = tuple(
            candidate for candidate in normalized if candidate.entry_s == current_entry
        )
        future = tuple(
            candidate for candidate in normalized if candidate.entry_s > current_entry
        )
        future_reserve_count = _future_reserve_count(future)
        current_candidate = current[0] if current else None

        all_ids = list(pool.uuv_ids)
        locks = {
            candidate_id: tuple(sorted(set(uuv_ids)))
            for candidate_id, uuv_ids in (locked_uuv_ids_by_candidate or {}).items()
        }
        unknown_locked = set(
            uuv_id for uuv_ids in locks.values() for uuv_id in uuv_ids
        ) - set(all_ids)
        if unknown_locked:
            raise ValueError(f"locked UUV is unavailable: {sorted(unknown_locked)}")
        current_locked = (
            locks.get(current_candidate.candidate_id, ())
            if current_candidate is not None
            else ()
        )
        future_locked = tuple(
            sorted(
                uuv_id
                for candidate in future
                for uuv_id in locks.get(candidate.candidate_id, ())
            )
        )
        reserve_candidates = [
            uuv_id for uuv_id in all_ids if uuv_id not in set(current_locked)
        ]
        future_locked_set = set(future_locked)
        future_reserve_ids = tuple(
            sorted(
                {
                    *future_locked_set,
                    *reserve_candidates[
                        max(0, len(reserve_candidates) - future_reserve_count) :
                    ],
                }
            )
        )
        remaining_ids = [uuv_id for uuv_id in all_ids if uuv_id not in future_reserve_ids]

        current_reserve_ids: tuple[str, ...] = ()
        selected_ids: tuple[str, ...] = ()
        if current_candidate is not None:
            current_reserve_count = current_candidate.reserve_uuv_count
            if current_reserve_count:
                reserve_start = max(0, len(remaining_ids) - current_reserve_count)
                current_reserve_ids = tuple(remaining_ids[reserve_start:])
                remaining_ids = remaining_ids[:reserve_start]
            minimum = _minimum_uuvs(current_candidate)
            maximum = max(
                minimum + current_candidate.optional_uuv_count,
                len(current_locked),
            )
            selected_count = min(maximum, len(remaining_ids))
            ordered_selection = [
                *current_locked,
                *(uuv_id for uuv_id in remaining_ids if uuv_id not in current_locked),
            ]
            if current_candidate.active_scan_uuv_count:
                active_capable = set(pool.active_capable_uuv_ids)
                ordered_selection = [
                    *(
                        uuv_id
                        for uuv_id in ordered_selection
                        if uuv_id in active_capable
                    ),
                    *(
                        uuv_id
                        for uuv_id in ordered_selection
                        if uuv_id not in active_capable
                    ),
                ]
            selected_ids = tuple(ordered_selection[:selected_count])

        reserved_ids = tuple(sorted((*current_reserve_ids, *future_reserve_ids)))
        assignments: list[RegionMissionState] = []
        batches: dict[str, tuple[UUVMissionBatch, ...]] = {}
        if current_candidate is not None:
            current_assignment, _ = _current_assignment(
                current_candidate,
                pool.carrier_id,
                selected_ids,
                current_reserve_ids,
                active_capable_uuv_ids=pool.active_capable_uuv_ids,
            )
            current_assignment = current_assignment.model_copy(
                update={"plan_revision": _snapshot_revision(snapshot)}
            )
            assignments.append(current_assignment)
            carrier_batches: dict[str, list[UUVMissionBatch]] = {}
            for carrier_id in sorted(pool.carrier_ids or (pool.carrier_id,)):
                carrier_uuv_ids = set(
                    pool.uuv_ids_by_carrier.get(carrier_id, ())
                )
                selected_for_carrier = tuple(
                    uuv_id for uuv_id in selected_ids if uuv_id in carrier_uuv_ids
                )
                if not selected_for_carrier:
                    continue
                active_ids = tuple(
                    uuv_id
                    for uuv_id in current_assignment.active_scan_uuv_ids
                    if uuv_id in selected_for_carrier
                )
                passive_ids = tuple(
                    uuv_id
                    for uuv_id in current_assignment.passive_track_uuv_ids
                    if uuv_id in selected_for_carrier
                )
                carrier_batches.setdefault(carrier_id, []).append(
                    UUVMissionBatch(
                        carrier_id=carrier_id,
                        candidate_id=current_candidate.candidate_id,
                        uuv_ids=selected_for_carrier,
                        active_scan_uuv_ids=active_ids,
                        passive_track_uuv_ids=passive_ids,
                        deployment_point=current_candidate.perimeter_points[0],
                        recovery_point=current_candidate.perimeter_points[-1],
                        entry_s=current_candidate.entry_s,
                        exit_s=current_candidate.exit_s,
                    )
                )
            batches = {
                carrier_id: tuple(carrier_batches[carrier_id])
                for carrier_id in sorted(carrier_batches)
            }
            for candidate in current[1:]:
                assignments.append(
                    _uncovered_assignment(
                        candidate,
                        "current_batch_priority",
                        plan_revision=_snapshot_revision(snapshot),
                    )
                )

        reserved_for_future = tuple(
            future_reserve_ids
        )
        for candidate in sorted(
            future,
            key=lambda item: (-item.probability, item.entry_s, item.candidate_id),
        ):
            if not reserved_for_future:
                assignments.append(
                    _uncovered_assignment(
                        candidate,
                        "future_reserve_unavailable",
                        plan_revision=_snapshot_revision(snapshot),
                    )
                )
                continue
            need = _minimum_uuvs(candidate) + candidate.reserve_uuv_count
            if need <= len(reserved_for_future):
                assignment_ids = reserved_for_future[:need]
                # Future members stay onboard as reserve until the next
                # rolling revision, including the members that will later
                # become active/passive task members.
                reserve_ids = assignment_ids
                assignments.append(
                    RegionMissionState(
                        region_id=candidate.candidate_id,
                        target_id=candidate.target_id,
                        lifecycle=RegionLifecycle.PLANNED,
                        reserve_uuv_ids=tuple(reserve_ids),
                        plan_revision=_snapshot_revision(snapshot),
                    )
                )
                # Keep the reservation attached to the highest-priority future
                # candidate; later candidates are reconsidered next cycle.
                reserved_for_future = ()
            else:
                assignments.append(
                    _uncovered_assignment(
                        candidate,
                        "future_reserve_infeasible",
                        plan_revision=_snapshot_revision(snapshot),
                    )
                )

        assignments.sort(key=lambda assignment: assignment.region_id)
        carrier_missions = {
            carrier_id: CarrierMissionModel(
                carrier_id=carrier_id,
                home_battle_group_id=pool.home_battle_group_id,
                route_xy=(),
                stop_ids=(),
                onboard_uuv_ids=(),
                ready_uuv_ids=tuple(
                    uuv_id
                    for uuv_id in pool.uuv_ids_by_carrier.get(carrier_id, ())
                    if uuv_id not in reserved_ids
                ),
                reserved_uuv_ids=tuple(
                    uuv_id
                    for uuv_id in reserved_ids
                    if uuv_id in pool.uuv_ids_by_carrier.get(carrier_id, ())
                ),
                recoverable_uuv_ids=(),
            )
            for carrier_id in pool.carrier_ids or (pool.carrier_id,)
        }
        degraded = tuple(
            f"{assignment.region_id}:{reason}"
            for assignment in assignments
            for reason in assignment.degraded_reasons
        )
        return ExecutableMissionPlan(
            revision=_snapshot_revision(snapshot),
            uuv_batches_by_carrier=batches,
            reserved_uuv_ids=reserved_ids,
            region_assignments=tuple(assignments),
            carrier_missions=carrier_missions,
            degraded_reasons=tuple(sorted(degraded)),
        )


def required_active_uuvs(region: MissionCandidate, snapshot: Any) -> int:
    """Return the deterministic active-scan minimum for a candidate."""
    del snapshot
    return max(0, int(region.active_scan_uuv_count))


def required_passive_uuvs(region: MissionCandidate, snapshot: Any) -> int:
    """Return the deterministic passive-tracking minimum for a candidate."""
    del snapshot
    return max(0, int(region.passive_track_uuv_count))


def _minimum_uuvs(candidate: MissionCandidate) -> int:
    return candidate.active_scan_uuv_count + candidate.passive_track_uuv_count


def _future_reserve_count(candidates: Iterable[MissionCandidate]) -> int:
    """Reserve enough members for the most demanding future candidate."""
    return max(
        (
            _minimum_uuvs(candidate) + candidate.reserve_uuv_count
            for candidate in candidates
        ),
        default=0,
    )


def _current_assignment(
    candidate: MissionCandidate,
    carrier_id: str,
    selected_ids: tuple[str, ...],
    reserve_ids: tuple[str, ...],
    *,
    active_capable_uuv_ids: Iterable[str] = (),
) -> tuple[RegionMissionState, UUVMissionBatch | None]:
    active_capable = set(active_capable_uuv_ids)
    active_count = min(candidate.active_scan_uuv_count, len(selected_ids))
    active_ids = tuple(
        uuv_id for uuv_id in selected_ids if uuv_id in active_capable
    )[:active_count]
    remaining = tuple(uuv_id for uuv_id in selected_ids if uuv_id not in active_ids)
    passive_count = min(candidate.passive_track_uuv_count, len(remaining))
    passive_ids = remaining[:passive_count]
    extra_ids = remaining[passive_count:]
    # Extra optional capacity improves pre-entry discovery, so it joins the
    # active scan set when it can actually perform active sonar.  A passive
    # extra member stays passive instead of being assigned an impossible role.
    active_ids = (
        *active_ids,
        *(uuv_id for uuv_id in extra_ids if uuv_id in active_capable),
    )
    passive_ids = (
        *passive_ids,
        *(uuv_id for uuv_id in extra_ids if uuv_id not in active_capable),
    )
    minimum = _minimum_uuvs(candidate)
    degraded_reasons: list[str] = []
    if len(selected_ids) == 0:
        assignment = _uncovered_assignment(candidate, "no_uuv_available")
    else:
        if len(selected_ids) < minimum:
            degraded_reasons.append("insufficient_uuv")
        if len(active_ids) < candidate.active_scan_uuv_count:
            degraded_reasons.append("active_capability_unavailable")
        lifecycle = (
            RegionLifecycle.DEGRADED if degraded_reasons else RegionLifecycle.PLANNED
        )
        assignment = RegionMissionState(
            region_id=candidate.candidate_id,
            target_id=candidate.target_id,
            lifecycle=lifecycle,
            active_scan_uuv_ids=tuple(active_ids),
            passive_track_uuv_ids=tuple(passive_ids),
            reserve_uuv_ids=reserve_ids,
            plan_revision=1,
            degraded_reasons=tuple(degraded_reasons),
        )
    batch = None
    if selected_ids:
        batch = UUVMissionBatch(
            carrier_id=carrier_id,
            candidate_id=candidate.candidate_id,
            uuv_ids=selected_ids,
            active_scan_uuv_ids=tuple(active_ids),
            passive_track_uuv_ids=tuple(passive_ids),
            deployment_point=candidate.perimeter_points[0],
            recovery_point=candidate.perimeter_points[-1],
            entry_s=candidate.entry_s,
            exit_s=candidate.exit_s,
        )
    return assignment, batch


def _uncovered_assignment(
    candidate: MissionCandidate,
    reason: str,
    *,
    plan_revision: int = 1,
) -> RegionMissionState:
    return RegionMissionState(
        region_id=candidate.candidate_id,
        target_id=candidate.target_id,
        lifecycle=RegionLifecycle.UNCOVERED,
        plan_revision=plan_revision,
        degraded_reasons=(reason,),
    )


def _normalize_candidates(
    candidates: Sequence[MissionCandidate | RegionalMissionCandidate],
) -> tuple[MissionCandidate, ...]:
    normalized = tuple(
        candidate
        if isinstance(candidate, MissionCandidate)
        else MissionCandidate(
            candidate_id=candidate.candidate_id,
            target_id=_target_id_from_candidate_id(candidate.candidate_id),
            entry_s=candidate.time_window.start_s,
            exit_s=candidate.time_window.end_s,
            probability=0.5,
            perimeter_points=candidate.perimeter_points,
            predecessor_candidate_ids=candidate.predecessor_candidate_ids,
            successor_candidate_ids=candidate.successor_candidate_ids,
        )
        for candidate in candidates
    )
    ids = [candidate.candidate_id for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("mission candidate IDs must be unique")
    return tuple(
        sorted(
            normalized,
            key=lambda candidate: (
                candidate.entry_s,
                -candidate.probability,
                candidate.candidate_id,
            ),
        )
    )


def _target_id_from_candidate_id(candidate_id: str) -> str:
    if ":r" in candidate_id:
        return candidate_id.split(":r", 1)[0]
    return candidate_id.split(":", 1)[0]


def _snapshot_revision(snapshot: Any) -> int:
    return max(1, int(getattr(snapshot, "snapshot_revision", 1)))


def _platform_pool(snapshot: Any, home_battle_group_id: str) -> _PlatformPool:
    situation = getattr(snapshot, "situation", snapshot)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    if platform_snapshot is not None:
        roster = platform_snapshot.roster
        eligible_uuvs = tuple(
            sorted(
                (
                    platform
                    for platform in roster.uuvs
                    if platform.deployment_state in {"onboard", "deployed"}
                    and platform.energy_fraction > 0.10
                ),
                key=lambda platform: platform.platform_id,
            )
        )
        uuv_ids = tuple(platform.platform_id for platform in eligible_uuvs)
        active_capable_uuv_ids = tuple(
            platform.platform_id
            for platform in eligible_uuvs
            if platform.capability.sonar.active_capable
        )
        carriers = tuple(getattr(platform_snapshot, "carriers", ()) or ())
        if not carriers:
            carriers = (platform_snapshot.carrier,)
        primary = platform_snapshot.carrier
        carrier_uuv_ids: dict[str, tuple[str, ...]] = {}
        assigned: set[str] = set()
        for carrier in sorted(carriers, key=lambda item: item.carrier_id):
            listed = tuple(
                platform_id
                for platform_id in (
                    *carrier.onboard_platform_ids,
                    *carrier.deployed_platform_ids,
                )
                if platform_id in uuv_ids
            )
            carrier_uuv_ids[carrier.carrier_id] = tuple(sorted(set(listed)))
            assigned.update(listed)
        # Older platform snapshots do not identify a carrier per UUV. Treat
        # unlisted eligible UUVs as belonging to the primary carrier.
        carrier_uuv_ids[primary.carrier_id] = tuple(
            sorted({*carrier_uuv_ids.get(primary.carrier_id, ()), *(set(uuv_ids) - assigned)})
        )
        return _PlatformPool(
            carrier_id=primary.carrier_id,
            home_battle_group_id=home_battle_group_id,
            uuv_ids=uuv_ids,
            active_capable_uuv_ids=active_capable_uuv_ids,
            carrier_ids=tuple(sorted(carrier.carrier_id for carrier in carriers)),
            uuv_ids_by_carrier=carrier_uuv_ids,
        )
    legacy_uuvs = getattr(situation, "uuvs", ())
    eligible_legacy_uuvs = tuple(
        sorted(
            (
                uuv
                for uuv in legacy_uuvs
                if getattr(
                    getattr(uuv, "status", None),
                    "value",
                    getattr(uuv, "status", ""),
                )
                not in {"failed", "returning"}
            ),
            key=lambda uuv: getattr(uuv, "uuv_id", ""),
        )
    )
    uuv_ids = tuple(
        getattr(uuv, "uuv_id", "") for uuv in eligible_legacy_uuvs if getattr(uuv, "uuv_id", "")
    )
    active_capable_uuv_ids = tuple(
        uuv_id
        for uuv_id, uuv in zip(uuv_ids, eligible_legacy_uuvs, strict=False)
        if getattr(getattr(uuv, "capability", None), "active_sonar_available", True)
    )
    return _PlatformPool(
        carrier_id="carrier-01",
        home_battle_group_id=home_battle_group_id,
        uuv_ids=tuple(uuv_id for uuv_id in uuv_ids if uuv_id),
        active_capable_uuv_ids=tuple(
            uuv_id for uuv_id in active_capable_uuv_ids if uuv_id
        ),
        carrier_ids=("carrier-01",),
        uuv_ids_by_carrier={"carrier-01": tuple(uuv_id for uuv_id in uuv_ids if uuv_id)},
    )


def _empty_plan(snapshot: Any, pool: _PlatformPool) -> ExecutableMissionPlan:
    return ExecutableMissionPlan(
        revision=_snapshot_revision(snapshot),
        carrier_missions={
            carrier_id: CarrierMissionModel(
                carrier_id=carrier_id,
                home_battle_group_id=pool.home_battle_group_id,
                ready_uuv_ids=pool.uuv_ids_by_carrier.get(carrier_id, ()),
            )
            for carrier_id in pool.carrier_ids or (pool.carrier_id,)
        },
    )
