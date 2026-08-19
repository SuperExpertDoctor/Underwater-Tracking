from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
            selected_ids = tuple(ordered_selection[:selected_count])

        reserved_ids = tuple(sorted((*current_reserve_ids, *future_reserve_ids)))
        assignments: list[RegionMissionState] = []
        batches: dict[str, tuple[UUVMissionBatch, ...]] = {}
        if current_candidate is not None:
            current_assignment, batch = _current_assignment(
                current_candidate,
                pool.carrier_id,
                selected_ids,
                current_reserve_ids,
            )
            current_assignment = current_assignment.model_copy(
                update={"plan_revision": _snapshot_revision(snapshot)}
            )
            assignments.append(current_assignment)
            if batch is not None:
                batches[pool.carrier_id] = (batch,)
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
        carrier = CarrierMissionModel(
            carrier_id=pool.carrier_id,
            home_battle_group_id=pool.home_battle_group_id,
            route_xy=(),
            stop_ids=(),
            onboard_uuv_ids=(),
            ready_uuv_ids=tuple(
                uuv_id for uuv_id in pool.uuv_ids if uuv_id not in reserved_ids
            ),
            reserved_uuv_ids=reserved_ids,
            recoverable_uuv_ids=(),
        )
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
            carrier_missions={pool.carrier_id: carrier},
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
) -> tuple[RegionMissionState, UUVMissionBatch | None]:
    active_count = min(candidate.active_scan_uuv_count, len(selected_ids))
    active_ids = selected_ids[:active_count]
    remaining = selected_ids[active_count:]
    passive_count = min(candidate.passive_track_uuv_count, len(remaining))
    passive_ids = remaining[:passive_count]
    extra_ids = remaining[passive_count:]
    # Extra optional capacity improves pre-entry discovery, so it joins the
    # active scan set rather than being misrepresented as passive tracking.
    active_ids = (*active_ids, *extra_ids)
    minimum = _minimum_uuvs(candidate)
    if len(selected_ids) == 0:
        assignment = _uncovered_assignment(candidate, "no_uuv_available")
    elif len(selected_ids) < minimum:
        assignment = RegionMissionState(
            region_id=candidate.candidate_id,
            target_id=candidate.target_id,
            lifecycle=RegionLifecycle.DEGRADED,
            active_scan_uuv_ids=tuple(active_ids),
            passive_track_uuv_ids=tuple(passive_ids),
            reserve_uuv_ids=reserve_ids,
            plan_revision=1,
            degraded_reasons=("insufficient_uuv",),
        )
    else:
        assignment = RegionMissionState(
            region_id=candidate.candidate_id,
            target_id=candidate.target_id,
            lifecycle=RegionLifecycle.PLANNED,
            active_scan_uuv_ids=tuple(active_ids),
            passive_track_uuv_ids=tuple(passive_ids),
            reserve_uuv_ids=reserve_ids,
            plan_revision=1,
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
        uuv_ids = tuple(
            sorted(
                platform.platform_id
                for platform in roster.uuvs
                if platform.deployment_state in {"onboard", "deployed"}
            )
        )
        carrier = platform_snapshot.carrier
        return _PlatformPool(
            carrier_id=carrier.carrier_id,
            home_battle_group_id=home_battle_group_id,
            uuv_ids=uuv_ids,
        )
    legacy_uuvs = getattr(situation, "uuvs", ())
    uuv_ids = tuple(
        sorted(
            getattr(uuv, "uuv_id", "")
            for uuv in legacy_uuvs
            if getattr(getattr(uuv, "status", None), "value", getattr(uuv, "status", ""))
            not in {"failed", "returning"}
        )
    )
    return _PlatformPool(
        carrier_id="carrier-01",
        home_battle_group_id=home_battle_group_id,
        uuv_ids=tuple(uuv_id for uuv_id in uuv_ids if uuv_id),
    )


def _empty_plan(snapshot: Any, pool: _PlatformPool) -> ExecutableMissionPlan:
    carrier = CarrierMissionModel(
        carrier_id=pool.carrier_id,
        home_battle_group_id=pool.home_battle_group_id,
        ready_uuv_ids=pool.uuv_ids,
    )
    return ExecutableMissionPlan(
        revision=_snapshot_revision(snapshot),
        carrier_missions={pool.carrier_id: carrier},
    )
