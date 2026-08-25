from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Literal, cast

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    MissionCandidate,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.domain.models import ContactClassification
from underwater_tracking.domain.regional_models import RegionalMissionCandidate
from underwater_tracking.planning.coverage import (
    serpentine_coverage_waypoints,
    serpentine_coverage_waypoints_by_uuv,
)
from underwater_tracking.planning.cooperative_auction import (
    AuctionTask,
    AuctionUUV,
    market_allocate,
    rank_positive_bidders,
)
from underwater_tracking.planning.region_cap import (
    MAX_EXECUTABLE_REGIONS_PER_TARGET,
    cap_candidate_regions,
)


@dataclass(frozen=True, slots=True)
class _PlatformPool:
    carrier_id: str
    home_battle_group_id: str
    uuv_ids: tuple[str, ...]
    active_capable_uuv_ids: tuple[str, ...] = ()
    carrier_ids: tuple[str, ...] = ()
    uuv_ids_by_carrier: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    carrier_roles: Mapping[str, str] = field(default_factory=dict)
    uuv_positions: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    uuv_speeds_mps: Mapping[str, float] = field(default_factory=dict)
    uuv_energy_fraction: Mapping[str, float] = field(default_factory=dict)


class MissionOptimizer:
    """Select one rolling UUV deployment batch while protecting future demand.

    The optimizer consumes estimated candidate metadata only.  It does not
    inspect target truth or mutate the platform snapshot.  The current time
    slice is the earliest candidate window; later high-probability candidates
    become reserve assignments for the next rolling cycle.
    """

    def __init__(
        self,
        *,
        home_battle_group_id: str = "home_battle_group",
        max_regions_per_target: int = MAX_EXECUTABLE_REGIONS_PER_TARGET,
        goal_mode: bool = False,
    ) -> None:
        self._home_battle_group_id = home_battle_group_id
        self._max_regions_per_target = max_regions_per_target
        self._goal_mode = goal_mode

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
        archived_candidates = tuple(
            candidate
            for candidate in normalized
            if candidate.exit_s < _snapshot_sim_time(snapshot)
        )
        normalized = _archive_expired_candidates(normalized, _snapshot_sim_time(snapshot))
        if not normalized:
            empty = _empty_plan(snapshot, pool)
            return empty.model_copy(
                update={
                    "region_assignments": tuple(
                        _uncovered_assignment(candidate, "candidate_window_expired")
                        for candidate in archived_candidates
                    )
                }
            )

        all_ids = list(pool.uuv_ids)
        normalized_ids = {candidate.candidate_id for candidate in normalized}
        locks = {
            candidate_id: tuple(sorted(set(uuv_ids)))
            for candidate_id, uuv_ids in (locked_uuv_ids_by_candidate or {}).items()
            if candidate_id in normalized_ids
        }
        # Provider assignments are observations from the previous rolling
        # epoch, not a license to dispatch a member that is now returning,
        # failed, depleted, or otherwise absent from the live pool. Preserve
        # the stale IDs for the audit trail, then let the deterministic auction
        # replace them with currently eligible resources.
        unavailable_locked_by_candidate = {
            candidate_id: tuple(
                uuv_id for uuv_id in uuv_ids if uuv_id not in set(all_ids)
            )
            for candidate_id, uuv_ids in locks.items()
        }
        unavailable_locked_by_candidate = {
            candidate_id: uuv_ids
            for candidate_id, uuv_ids in unavailable_locked_by_candidate.items()
            if uuv_ids
        }
        if unavailable_locked_by_candidate:
            unavailable_ids = {
                uuv_id
                for uuv_ids in unavailable_locked_by_candidate.values()
                for uuv_id in uuv_ids
            }
            locks = {
                candidate_id: tuple(
                    uuv_id for uuv_id in uuv_ids if uuv_id not in unavailable_ids
                )
                for candidate_id, uuv_ids in locks.items()
            }
        temporal_paths: dict[str, tuple[str, ...]] = {}
        protected_for_cap = set(locks)
        if self._goal_mode:
            # Goal mode must produce a physically executable current batch. The
            # bounded temporal chain is still protected from region capping so
            # later windows can be carried as explicit handoff intent.
            normalized, temporal_paths = _ensure_goal_search_chain(
                normalized,
                snapshot,
                pool,
                max_regions=self._max_regions_per_target,
            )
            protected_for_cap.update(
                candidate_id
                for path in temporal_paths.values()
                for candidate_id in path
            )
        elif (
            normalized
            and any(
                candidate.predecessor_candidate_ids or candidate.successor_candidate_ids
                for candidate in normalized
            )
        ):
            temporal_paths = _preferred_temporal_paths(
                normalized,
                max_regions=self._max_regions_per_target,
                locked_ids=locks,
                snapshot=snapshot,
            )
            protected_for_cap.update(
                candidate_id
                for path in temporal_paths.values()
                for candidate_id in path
            )
        normalized, excluded = cap_candidate_regions(
            normalized,
            max_regions=self._max_regions_per_target,
            protected_ids=protected_for_cap,
        )
        if not normalized:
            return _empty_plan(snapshot, pool)
        normalized = _relink_temporal_paths(normalized, temporal_paths)

        if self._goal_mode:
            # Provider policies remain authoritative inputs. Goal mode adds
            # only the minimum executable contract needed to realize the
            # provider's public candidate sequence: an active/passive seed and
            # a bounded successor path for physical handoff.
            normalized = _ensure_goal_search_pair(normalized, snapshot, pool)

        current_entry = min(candidate.entry_s for candidate in normalized)
        current = tuple(
            candidate for candidate in normalized if candidate.entry_s == current_entry
        )
        future = tuple(
            candidate for candidate in normalized if candidate.entry_s > current_entry
        )
        current_candidate = (
            _current_candidate_for_snapshot(snapshot, current) if current else None
        )
        execution_members_by_region = _execution_members_by_region(snapshot)
        current_continuity_ids = (
            tuple(execution_members_by_region.get(current_candidate.candidate_id, ()))
            if current_candidate is not None
            else ()
        )
        if current_candidate is not None and current_continuity_ids:
            # A rolling epoch may finish while the current physical group is
            # still underwater. Keep that exact group attached to the same
            # candidate so a provider refresh cannot orphan its sortie. The
            # physical group replaces stale provider locks for this candidate;
            # appending both sets can exceed the candidate's role slots and
            # makes the auction reject an otherwise valid active/passive pair.
            locks[current_candidate.candidate_id] = tuple(
                dict.fromkeys(current_continuity_ids)
            )
        provider_locked_ids = {
            uuv_id
            for uuv_ids in locks.values()
            for uuv_id in uuv_ids
        }
        busy_deployed_ids = (
            _deployed_uuv_ids(snapshot)
            - set(current_continuity_ids)
            - provider_locked_ids
        )
        if busy_deployed_ids:
            pool = _restrict_platform_pool(
                pool,
                set(pool.uuv_ids) - busy_deployed_ids,
            )
            all_ids = list(pool.uuv_ids)
            all_id_set = set(all_ids)
            late_unavailable_by_candidate = {
                candidate_id: tuple(
                    uuv_id for uuv_id in uuv_ids if uuv_id not in all_id_set
                )
                for candidate_id, uuv_ids in locks.items()
            }
            late_unavailable_by_candidate = {
                candidate_id: uuv_ids
                for candidate_id, uuv_ids in late_unavailable_by_candidate.items()
                if uuv_ids
            }
            for candidate_id, unavailable_ids in late_unavailable_by_candidate.items():
                unavailable_locked_by_candidate[candidate_id] = tuple(
                    dict.fromkeys(
                        (
                            *unavailable_locked_by_candidate.get(candidate_id, ()),
                            *unavailable_ids,
                        )
                    )
                )
            unavailable_ids = {
                uuv_id
                for uuv_ids in unavailable_locked_by_candidate.values()
                for uuv_id in uuv_ids
            }
            locks = {
                candidate_id: tuple(
                    uuv_id for uuv_id in uuv_ids if uuv_id not in unavailable_ids
                )
                for candidate_id, uuv_ids in locks.items()
            }
        provider_lock_conflicts: dict[str, tuple[str, ...]] = {}
        locks, provider_lock_conflicts = _deduplicate_provider_locks(
            locks,
            normalized,
            preferred_candidate_ids=(
                (current_candidate.candidate_id,) if current_candidate is not None else ()
            ),
        )
        candidates_by_id = {candidate.candidate_id: candidate for candidate in normalized}
        topology_chain = _topology_chain(current_candidate, candidates_by_id)
        auction_order_by_candidate = _auction_selection_orders(
            snapshot,
            topology_chain,
            pool,
            locks,
        )
        topology_ids = {candidate.candidate_id for candidate in topology_chain}
        future_reserve_candidates = tuple(
            candidate for candidate in future if candidate.candidate_id not in topology_ids
        )
        future_reserve_count = _future_reserve_count(future_reserve_candidates)

        current_locked = (
            locks.get(current_candidate.candidate_id, ())
            if current_candidate is not None
            else ()
        )
        topology_locked = tuple(
            sorted(
                uuv_id
                for candidate in topology_chain[1:]
                for uuv_id in locks.get(candidate.candidate_id, ())
            )
        )
        future_locked = tuple(
            sorted(
                uuv_id
                for candidate in future_reserve_candidates
                for uuv_id in locks.get(candidate.candidate_id, ())
            )
        )
        lock_owners: dict[str, str] = {}
        for candidate_id, uuv_ids in locks.items():
            for uuv_id in uuv_ids:
                previous = lock_owners.setdefault(uuv_id, candidate_id)
                if previous != candidate_id:
                    raise ValueError(
                        f"locked UUV is assigned to multiple candidates: {uuv_id}"
                    )
        current_locked_set = set(current_locked)
        topology_locked_set = set(topology_locked)
        future_locked_set = set(future_locked)
        if current_locked_set & (topology_locked_set | future_locked_set):
            raise ValueError("current candidate locks overlap future candidate locks")
        if topology_locked_set & future_locked_set:
            raise ValueError("topology locks overlap future reserve locks")
        reserve_candidates = [
            uuv_id
            for uuv_id in all_ids
            if uuv_id not in current_locked_set
            and uuv_id not in topology_locked_set
            and uuv_id not in future_locked_set
        ]
        future_tail_count = max(0, future_reserve_count - len(future_locked_set))
        future_reserve_ids = tuple(
            sorted(
                {
                    *future_locked_set,
                    *reserve_candidates[max(0, len(reserve_candidates) - future_tail_count) :],
                }
            )
        )
        remaining_ids = [uuv_id for uuv_id in all_ids if uuv_id not in future_reserve_ids]

        assignments: list[RegionMissionState] = []
        carrier_batches: dict[str, list[UUVMissionBatch]] = {}
        reserved_ids_set: set[str] = set(future_reserve_ids)
        available_ids = [
            uuv_id
            for uuv_id in remaining_ids
            if uuv_id not in current_locked_set
            and uuv_id not in topology_locked_set
        ]
        assignment_by_candidate: dict[str, RegionMissionState] = {}
        batch_by_candidate: dict[str, bool] = {}

        if current_candidate is not None:
            current_reserve_count = current_candidate.reserve_uuv_count
            current_minimum = _minimum_uuvs(current_candidate)
            current_maximum = max(
                current_minimum + current_candidate.optional_uuv_count,
                len(current_locked),
            )
            # A rolling plan must keep the earliest executable window alive.
            # Protect only the immediate successor here; later topology nodes
            # are reconsidered after the next observation/replan boundary.
            # Reserving the entire future chain can consume every member and
            # silently turn the current search window into an uncovered one.
            successor_capacity = (
                _minimum_uuvs(topology_chain[1])
                + topology_chain[1].reserve_uuv_count
                if len(topology_chain) > 1
                else 0
            )
            successor_capacity = min(
                successor_capacity,
                max(0, len(remaining_ids) - current_reserve_count - current_minimum),
            )
            current_capacity = max(
                0,
                len(remaining_ids) - current_reserve_count - successor_capacity,
            )
            selected_count = min(current_maximum, current_capacity)
            selected_count = max(selected_count, len(current_locked))
            current_selection_pool = [
                uuv_id
                for uuv_id in auction_order_by_candidate.get(
                    current_candidate.candidate_id, ()
                )
                if uuv_id in remaining_ids
                if uuv_id not in topology_locked_set
            ]
            selected_ids = _ordered_selection(
                current_locked,
                current_selection_pool,
                active_capable_uuv_ids=pool.active_capable_uuv_ids,
                prioritize_active=bool(current_candidate.active_scan_uuv_count),
                preferred_ids=auction_order_by_candidate.get(
                    current_candidate.candidate_id, ()
                ),
            )[:selected_count]
            selected_set = set(selected_ids)
            available_ids = [
                uuv_id
                for uuv_id in remaining_ids
                if uuv_id not in selected_set
                and uuv_id not in topology_locked_set
            ]
            current_reserve_pool = tuple(
                uuv_id for uuv_id in available_ids if uuv_id not in topology_locked_set
            )
            current_reserve_ids = tuple(
                current_reserve_pool[:current_reserve_count]
            )
            available_ids = [
                uuv_id for uuv_id in available_ids if uuv_id not in current_reserve_ids
            ]
            reserved_ids_set.update(current_reserve_ids)
            current_assignment, _ = _current_assignment(
                current_candidate,
                pool.carrier_id,
                selected_ids,
                current_reserve_ids,
                active_capable_uuv_ids=pool.active_capable_uuv_ids,
                # The current window may still execute a physically valid
                # partial role set when the live pool is short. Keep the
                # assignment degraded and auditable; only future handoff
                # windows must be rejected when their role contract is
                # incomplete.
                reject_partial=False,
            )
            current_assignment = current_assignment.model_copy(
                update={"plan_revision": _snapshot_revision(snapshot)}
            )
            assignment_by_candidate[current_candidate.candidate_id] = current_assignment
            batch_by_candidate[current_candidate.candidate_id] = (
                bool(selected_ids)
                and current_assignment.lifecycle is not RegionLifecycle.UNCOVERED
            )
            if batch_by_candidate[current_candidate.candidate_id]:
                _append_carrier_batches(
                    carrier_batches,
                    pool,
                    current_candidate,
                    current_assignment,
                )
            for candidate in current[1:]:
                if candidate.candidate_id == current_candidate.candidate_id:
                    # The public estimate may select a later item in the
                    # same-window tuple as the current region.  Keep its
                    # materialized assignment aligned with the batch created
                    # above; only the other same-window branches are skipped.
                    continue
                assignment_by_candidate[candidate.candidate_id] = _uncovered_assignment(
                    candidate,
                    "current_batch_priority",
                    plan_revision=_snapshot_revision(snapshot),
                )

            for candidate in topology_chain[1:]:
                candidate_locked = locks.get(candidate.candidate_id, ())
                candidate_maximum = _minimum_uuvs(candidate) + candidate.optional_uuv_count
                candidate_capacity = max(
                    0,
                    len(available_ids) - candidate.reserve_uuv_count,
                )
                candidate_count = min(candidate_maximum, candidate_capacity)
                candidate_count = max(candidate_count, len(candidate_locked))
                selected_for_candidate = _ordered_selection(
                    candidate_locked,
                    tuple(
                        uuv_id
                        for uuv_id in auction_order_by_candidate.get(
                            candidate.candidate_id, ()
                        )
                        if uuv_id in available_ids
                    ),
                    active_capable_uuv_ids=pool.active_capable_uuv_ids,
                    prioritize_active=bool(candidate.active_scan_uuv_count),
                    preferred_ids=auction_order_by_candidate.get(
                        candidate.candidate_id, ()
                    ),
                )[:candidate_count]
                selected_set = set(selected_for_candidate)
                available_ids = [
                    uuv_id
                    for uuv_id in available_ids
                    if uuv_id not in selected_set
                    and uuv_id not in topology_locked_set
                ]
                reserve_for_candidate = tuple(
                    available_ids[: candidate.reserve_uuv_count]
                )
                available_ids = [
                    uuv_id
                    for uuv_id in available_ids
                    if uuv_id not in reserve_for_candidate
                ]
                reserved_ids_set.update(reserve_for_candidate)
                assignment, _ = _current_assignment(
                    candidate,
                    pool.carrier_id,
                    selected_for_candidate,
                    reserve_for_candidate,
                    active_capable_uuv_ids=pool.active_capable_uuv_ids,
                    reject_partial=self._goal_mode,
                )
                assignment = assignment.model_copy(
                    update={"plan_revision": _snapshot_revision(snapshot)}
                )
                if self._goal_mode and selected_for_candidate:
                    assignment = assignment.model_copy(
                        update={
                            "degraded_reasons": tuple(
                                dict.fromkeys(
                                    (
                                        *assignment.degraded_reasons,
                                        "future_window_reservation",
                                    )
                                )
                            )
                        }
                    )
                assignment_by_candidate[candidate.candidate_id] = assignment
                batch_by_candidate[candidate.candidate_id] = (
                    bool(selected_for_candidate)
                    and assignment.lifecycle is not RegionLifecycle.UNCOVERED
                )
                materialize_batch = not self._goal_mode or (
                    len(topology_chain) > 1
                    and candidate.candidate_id == topology_chain[1].candidate_id
                )
                if batch_by_candidate[candidate.candidate_id] and materialize_batch:
                    # Materialize the bounded topology successor now. Its
                    # carrier route can deploy during the overlap window,
                    # which is required for physical handoff evidence.
                    _append_carrier_batches(
                        carrier_batches,
                        pool,
                        candidate,
                        assignment,
                    )

        reserved_for_future = tuple(sorted(future_reserve_ids))
        for candidate in sorted(
            future_reserve_candidates,
            key=lambda item: (-item.probability, item.entry_s, item.candidate_id),
        ):
            if not reserved_for_future:
                assignment_by_candidate[candidate.candidate_id] = _uncovered_assignment(
                    candidate,
                    "future_reserve_unavailable",
                    plan_revision=_snapshot_revision(snapshot),
                )
                continue
            need = _minimum_uuvs(candidate) + candidate.reserve_uuv_count
            if need <= len(reserved_for_future):
                assignment_ids = reserved_for_future[:need]
                # Future members stay onboard as reserve until the next
                # rolling revision, including the members that will later
                # become active/passive task members.
                reserve_ids = assignment_ids
                assignment_by_candidate[candidate.candidate_id] = RegionMissionState(
                    region_id=candidate.candidate_id,
                    target_id=candidate.target_id,
                    lifecycle=RegionLifecycle.PLANNED,
                    reserve_uuv_ids=tuple(reserve_ids),
                    plan_revision=_snapshot_revision(snapshot),
                )
                # Keep the reservation attached to the highest-priority future
                # candidate; later candidates are reconsidered next cycle.
                reserved_for_future = ()
            else:
                assignment_by_candidate[candidate.candidate_id] = _uncovered_assignment(
                    candidate,
                    "future_reserve_infeasible",
                    plan_revision=_snapshot_revision(snapshot),
                )

        excluded_assignments = [
            _uncovered_assignment(
                candidate,
                "region_cap_not_selected",
                plan_revision=_snapshot_revision(snapshot),
            )
            for candidate in excluded
        ]
        expired_assignments = [
            _uncovered_assignment(
                candidate,
                "candidate_window_expired",
                plan_revision=_snapshot_revision(snapshot),
            )
            for candidate in archived_candidates
        ]
        executable_chain = tuple(
            candidate
            for candidate in topology_chain
            if batch_by_candidate.get(candidate.candidate_id, False)
            and (
                assignment := assignment_by_candidate.get(candidate.candidate_id)
            ) is not None
            and (
                assignment.active_scan_uuv_ids
                or assignment.passive_track_uuv_ids
            )
        )
        for index, candidate in enumerate(executable_chain):
            assignment = assignment_by_candidate[candidate.candidate_id]
            assignment_by_candidate[candidate.candidate_id] = assignment.model_copy(
                update={
                    "handoff_from": (
                        executable_chain[index - 1].candidate_id
                        if index > 0
                        else None
                    ),
                    "handoff_to": (
                        executable_chain[index + 1].candidate_id
                        if index + 1 < len(executable_chain)
                        else None
                    ),
                }
            )
        for candidate_id, unavailable_ids in unavailable_locked_by_candidate.items():
            assignment = assignment_by_candidate.get(candidate_id)
            if assignment is None:
                continue
            provider_reason = (
                "provider_assignment_unavailable:"
                + ",".join(unavailable_ids)
            )
            assignment_by_candidate[candidate_id] = assignment.model_copy(
                update={
                    "degraded_reasons": tuple(
                        dict.fromkeys(
                            (*assignment.degraded_reasons, provider_reason)
                        )
                    )
                }
            )
        for candidate_id, conflict_ids in provider_lock_conflicts.items():
            assignment = assignment_by_candidate.get(candidate_id)
            if assignment is None:
                continue
            assignment_by_candidate[candidate_id] = assignment.model_copy(
                update={
                    "degraded_reasons": tuple(
                        dict.fromkeys(
                            (
                                *assignment.degraded_reasons,
                                *(
                                    f"provider_assignment_conflict:{uuv_id}"
                                    for uuv_id in conflict_ids
                                ),
                            )
                        )
                    )
                }
            )
        assignments = [
            *assignment_by_candidate.values(),
            *excluded_assignments,
            *expired_assignments,
        ]
        assignments.sort(key=lambda assignment: assignment.region_id)
        reserved_ids = tuple(sorted(reserved_ids_set))
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
            uuv_batches_by_carrier={
                carrier_id: tuple(carrier_batches[carrier_id])
                for carrier_id in sorted(carrier_batches)
            },
            reserved_uuv_ids=reserved_ids,
            region_assignments=tuple(assignments),
            carrier_missions=carrier_missions,
            degraded_reasons=tuple(sorted(degraded)),
            resource_episode_by_uuv=_resource_episodes(snapshot),
        )


def required_active_uuvs(region: MissionCandidate, snapshot: Any) -> int:
    """Return the deterministic active-scan minimum for a candidate."""
    del snapshot
    return max(0, int(region.active_scan_uuv_count))


def required_passive_uuvs(region: MissionCandidate, snapshot: Any) -> int:
    """Return the deterministic passive-tracking minimum for a candidate."""
    del snapshot
    return max(0, int(region.passive_track_uuv_count))


def _goal_current_locks(
    candidates: Sequence[MissionCandidate],
    locks: Mapping[str, Sequence[str]],
    snapshot: Any,
) -> dict[str, tuple[str, ...]]:
    """Keep only provider locks that belong to the current public window."""
    if not candidates:
        return {}
    earliest = min(candidate.entry_s for candidate in candidates)
    current = tuple(candidate for candidate in candidates if candidate.entry_s == earliest)
    selected = _current_candidate_for_snapshot(snapshot, current)
    selected_ids = tuple(sorted(set(locks.get(selected.candidate_id, ()))))
    return {selected.candidate_id: selected_ids} if selected_ids else {}


def _deduplicate_provider_locks(
    locks: Mapping[str, Sequence[str]],
    candidates: Sequence[MissionCandidate],
    *,
    preferred_candidate_ids: Sequence[str] = (),
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Keep preferred/current locks first and audit duplicate provider claims."""
    preferred = set(preferred_candidate_ids)
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.candidate_id not in preferred,
            candidate.entry_s,
            -candidate.probability,
            candidate.candidate_id,
        ),
    )
    owners: dict[str, str] = {}
    deduplicated: dict[str, tuple[str, ...]] = {}
    conflicts: dict[str, tuple[str, ...]] = {}
    for candidate in ordered:
        kept: list[str] = []
        rejected: list[str] = []
        for uuv_id in tuple(
            dict.fromkeys(locks.get(candidate.candidate_id, ()))
        ):
            previous = owners.get(uuv_id)
            if previous is not None and previous != candidate.candidate_id:
                rejected.append(uuv_id)
                continue
            owners[uuv_id] = candidate.candidate_id
            kept.append(uuv_id)
        if kept:
            deduplicated[candidate.candidate_id] = tuple(kept)
        if rejected:
            conflicts[candidate.candidate_id] = tuple(rejected)
    return deduplicated, conflicts


def _ensure_goal_search_chain(
    candidates: Sequence[MissionCandidate],
    snapshot: Any,
    pool: _PlatformPool,
    *,
    max_regions: int,
) -> tuple[tuple[MissionCandidate, ...], dict[str, tuple[str, ...]]]:
    """Materialize a bounded active/passive temporal handoff chain.

    A regional provider may label future cells as reserve-only because it is
    describing the next auction round.  In goal mode the executable contract
    is stronger: a bounded sequence of cells must already contain enough
    active and passive roles for the next physical handoff.  The chain is
    still selected from public candidate topology and never from target truth.
    """
    if (
        not candidates
        or len(pool.uuv_ids) < 2
        or not pool.active_capable_uuv_ids
        or max_regions < 1
    ):
        return tuple(candidates), {}

    by_target: dict[str, list[MissionCandidate]] = {}
    for candidate in candidates:
        by_target.setdefault(candidate.target_id, []).append(candidate)

    updates: dict[str, MissionCandidate] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for target_id, target_candidates in sorted(by_target.items()):
        by_id = {candidate.candidate_id: candidate for candidate in target_candidates}
        earliest = min(candidate.entry_s for candidate in target_candidates)
        current_candidates = tuple(
            candidate
            for candidate in target_candidates
            if candidate.entry_s == earliest
        )
        current = _current_candidate_for_snapshot(snapshot, current_candidates)
        chain = list(_topology_chain(current, by_id))
        visited = {candidate.candidate_id for candidate in chain}
        for candidate in sorted(
            target_candidates,
            key=lambda item: (item.entry_s, -item.probability, item.candidate_id),
        ):
            if len(chain) >= max_regions:
                break
            if candidate.candidate_id in visited or candidate.entry_s <= chain[-1].entry_s:
                continue
            chain.append(candidate)
            visited.add(candidate.candidate_id)
        chain = chain[:max_regions]
        paths[target_id] = tuple(candidate.candidate_id for candidate in chain)
        for index, candidate in enumerate(chain):
            predecessor_ids = list(candidate.predecessor_candidate_ids)
            successor_ids = list(candidate.successor_candidate_ids)
            if index > 0:
                predecessor_ids.append(chain[index - 1].candidate_id)
            if index + 1 < len(chain):
                successor_ids.append(chain[index + 1].candidate_id)
            updates[candidate.candidate_id] = candidate.model_copy(
                update={
                    "active_scan_uuv_count": max(1, candidate.active_scan_uuv_count),
                    "passive_track_uuv_count": max(1, candidate.passive_track_uuv_count),
                    "predecessor_candidate_ids": tuple(dict.fromkeys(predecessor_ids)),
                    "successor_candidate_ids": tuple(dict.fromkeys(successor_ids)),
                }
            )

    return (
        tuple(updates.get(candidate.candidate_id, candidate) for candidate in candidates),
        paths,
    )


def _ensure_goal_search_pair(
    candidates: Sequence[MissionCandidate],
    snapshot: Any,
    pool: _PlatformPool,
) -> tuple[MissionCandidate, ...]:
    """Require a feasible active/passive seed for the first goal window.

    The provider decides the topology and can leave assignments empty, but a
    goal-mode bootstrap must still create a useful cooperative search group
    whenever the live roster has both capabilities.  This is a deterministic
    execution guard, not a replacement result for a missing provider.
    """
    if not candidates or not pool.active_capable_uuv_ids:
        return tuple(candidates)
    earliest = min(candidate.entry_s for candidate in candidates)
    current = tuple(candidate for candidate in candidates if candidate.entry_s == earliest)
    selected = _current_candidate_for_snapshot(snapshot, current)
    if len(pool.uuv_ids) < 2:
        return tuple(candidates)
    active_count = max(1, selected.active_scan_uuv_count)
    passive_count = max(1, selected.passive_track_uuv_count)
    if active_count == selected.active_scan_uuv_count and passive_count == selected.passive_track_uuv_count:
        return tuple(candidates)
    return tuple(
        candidate.model_copy(
            update={
                "active_scan_uuv_count": (
                    active_count if candidate.candidate_id == selected.candidate_id
                    else candidate.active_scan_uuv_count
                ),
                "passive_track_uuv_count": (
                    passive_count if candidate.candidate_id == selected.candidate_id
                    else candidate.passive_track_uuv_count
                ),
            }
        )
        for candidate in candidates
    )


def _topology_chain(
    current: MissionCandidate | None,
    candidates_by_id: Mapping[str, MissionCandidate],
) -> tuple[MissionCandidate, ...]:
    """Follow one deterministic successor chain from the current candidate."""
    if current is None:
        return ()
    chain = [current]
    visited = {current.candidate_id}
    while True:
        previous = chain[-1]
        successor = _first_later_successor(
            previous,
            candidates_by_id,
            visited,
        )
        if successor is None:
            successor = next(
                (
                    candidate
                    for candidate in sorted(
                        candidates_by_id.values(),
                        key=lambda item: (
                            item.entry_s,
                            -item.probability,
                            item.candidate_id,
                        ),
                    )
                    if previous.candidate_id in candidate.predecessor_candidate_ids
                    and candidate.candidate_id not in visited
                    and candidate.entry_s > previous.entry_s
                ),
                None,
            )
        if successor is None:
            break
        chain.append(successor)
        visited.add(successor.candidate_id)
    return tuple(chain)


def _current_candidate(candidates: Sequence[MissionCandidate]) -> MissionCandidate:
    """Prefer a topology root over a same-window downstream branch."""
    current_ids = {candidate.candidate_id for candidate in candidates}
    return min(
        candidates,
        key=lambda candidate: (
            bool(current_ids.intersection(candidate.predecessor_candidate_ids)),
            -len(candidate.successor_candidate_ids),
            -candidate.probability,
            candidate.candidate_id,
        ),
    )


def _current_candidate_for_snapshot(
    snapshot: Any,
    candidates: Sequence[MissionCandidate],
) -> MissionCandidate:
    """Select the candidate closest to the latest public target estimate.

    A same-window topology root is only a deterministic tie-breaker.  The
    public belief geometry must select the current search surface before graph
    ancestry does, otherwise a valid downstream branch can move the physical
    batch away from the only available evidence.
    """
    if not candidates:
        raise ValueError("at least one current candidate is required")
    public_point_by_target = {
        candidate.target_id: point
        for candidate in candidates
        if (point := _public_target_point(snapshot, candidate.target_id)) is not None
    }
    if not public_point_by_target:
        return _current_candidate(candidates)
    current_ids = {candidate.candidate_id for candidate in candidates}
    return min(
        candidates,
        key=lambda candidate: (
            _candidate_public_distance(
                candidate,
                public_point_by_target.get(candidate.target_id),
            ),
            bool(current_ids.intersection(candidate.predecessor_candidate_ids)),
            -candidate.priority,
            -candidate.probability,
            -len(candidate.successor_candidate_ids),
            candidate.candidate_id,
        ),
    )


def _public_target_point(snapshot: Any, target_id: str) -> tuple[float, float] | None:
    """Read only public contact/prior/belief geometry for planning and bidding."""
    situation = getattr(snapshot, "situation", snapshot)
    contacts = tuple(getattr(situation, "contacts", ()) or ())
    known_contacts = tuple(
        contact
        for contact in contacts
        if getattr(contact, "contact_id", None) == target_id
        and getattr(contact, "classification", None)
        is ContactClassification.SUBMARINE
        and getattr(contact, "estimated_position_xy", None) is not None
    )
    if known_contacts:
        contact = max(known_contacts, key=lambda item: int(getattr(item, "sim_time_s", 0)))
        point = contact.estimated_position_xy
        return float(point[0]), float(point[1])
    sim_time_s = int(getattr(situation, "sim_time_s", getattr(snapshot, "sim_time_s", 0)))
    priors = tuple(getattr(situation, "target_search_priors", ()) or ())
    active_priors = tuple(
        prior
        for prior in priors
        if getattr(prior, "target_id", None) == target_id
        and int(getattr(prior, "issued_at_s", 0))
        <= sim_time_s
        < int(getattr(prior, "valid_until_s", 0))
    )
    if active_priors:
        prior = max(
            active_priors,
            key=lambda item: (
                float(getattr(item, "confidence", 0.0)),
                int(getattr(item, "issued_at_s", 0)),
                str(getattr(item, "prior_id", "")),
            ),
        )
        point = getattr(prior, "center_xy", None)
        if point is not None:
            return float(point[0]), float(point[1])
    reports = tuple(getattr(situation, "group_reports", ()) or ())
    reports = tuple(report for report in reports if getattr(report, "target_id", None) == target_id)
    if reports:
        report = max(reports, key=lambda item: int(getattr(item, "sim_time_s", 0)))
        mean = getattr(getattr(report, "belief", None), "mean", None)
        if mean is not None and len(mean) >= 2:
            return float(mean[0]), float(mean[1])
    return None


def _candidate_public_distance(
    candidate: MissionCandidate,
    point: tuple[float, float] | None,
) -> float:
    """Distance from a public point to a candidate polygon."""
    if point is None:
        return 0.0
    if _point_in_polygon(point, candidate.perimeter_points):
        return 0.0
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(
            candidate.perimeter_points,
            (*candidate.perimeter_points[1:], candidate.perimeter_points[0]),
        )
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return whether a public point is inside a candidate polygon."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        ratio = 0.0
    else:
        ratio = max(
            0.0,
            min(
                1.0,
                ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
                / denominator,
            ),
        )
    closest = (start[0] + ratio * dx, start[1] + ratio * dy)
    return hypot(point[0] - closest[0], point[1] - closest[1])


def _polygon_center(
    polygon: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    if not polygon:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def _auction_selection_orders(
    snapshot: Any,
    candidates: Sequence[MissionCandidate],
    pool: _PlatformPool,
    locks: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Globally award the current topology chain's heterogeneous role slots."""
    if not candidates or not pool.uuv_ids:
        return {}
    tasks = tuple(
        AuctionTask(
            task_id=candidate.candidate_id,
            center_xy=_polygon_center(candidate.perimeter_points),
            entry_s=candidate.entry_s,
            exit_s=candidate.exit_s,
            probability=candidate.probability,
            priority=candidate.priority,
            active_slots=candidate.active_scan_uuv_count,
            passive_slots=candidate.passive_track_uuv_count + candidate.optional_uuv_count,
            reserve_slots=candidate.reserve_uuv_count,
            continuity_uuv_ids=tuple(locks.get(candidate.candidate_id, ())),
        )
        for candidate in candidates
    )
    carrier_by_uuv = {
        uuv_id: carrier_id
        for carrier_id, uuv_ids in pool.uuv_ids_by_carrier.items()
        for uuv_id in uuv_ids
    }
    uuvs = tuple(
        AuctionUUV(
            uuv_id=uuv_id,
            position_xy=pool.uuv_positions.get(uuv_id, (0.0, 0.0)),
            speed_mps=max(pool.uuv_speeds_mps.get(uuv_id, 1.0), 1e-6),
            energy_fraction=pool.uuv_energy_fraction.get(uuv_id, 1.0),
            active_capable=uuv_id in pool.active_capable_uuv_ids,
            carrier_id=carrier_by_uuv.get(uuv_id, pool.carrier_id),
        )
        for uuv_id in pool.uuv_ids
    )
    allocation = market_allocate(
        tasks,
        uuvs,
        locked_uuv_ids_by_task={
            task.task_id: tuple(locks.get(task.task_id, ())) for task in tasks
        },
    )
    award_by_task = {award.task_id: award for award in allocation.awards}
    order_by_candidate: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        award = award_by_task.get(task.task_id)
        awarded = (
            (
                *award.active_uuv_ids,
                *award.passive_uuv_ids,
                *award.reserve_uuv_ids,
            )
            if award is not None
            else ()
        )
        order_by_candidate[task.task_id] = tuple(
            dict.fromkeys(
                (*sorted(awarded), *rank_positive_bidders(task, uuvs))
            )
        )
    return order_by_candidate


def _first_later_successor(
    previous: MissionCandidate,
    candidates_by_id: Mapping[str, MissionCandidate],
    visited: set[str],
) -> MissionCandidate | None:
    """Walk same-window bridge cells until a later executable window appears."""
    queue = list(previous.successor_candidate_ids)
    queued = set(queue)
    while queue:
        candidate_id = queue.pop(0)
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None or candidate_id in visited:
            continue
        if candidate.entry_s > previous.entry_s:
            return candidate
        for successor_id in candidate.successor_candidate_ids:
            if successor_id not in queued and successor_id not in visited:
                queue.append(successor_id)
                queued.add(successor_id)
    return None


def _preferred_temporal_paths(
    candidates: Sequence[MissionCandidate],
    *,
    max_regions: int,
    locked_ids: Mapping[str, Sequence[str]],
    snapshot: Any | None = None,
) -> dict[str, tuple[str, ...]]:
    """Choose a bounded topology path for unprioritized generated regions."""
    result: dict[str, tuple[str, ...]] = {}
    by_target: dict[str, list[MissionCandidate]] = {}
    for candidate in candidates:
        by_target.setdefault(candidate.target_id, []).append(candidate)
    for target_id, target_candidates in sorted(by_target.items()):
        target_locked = {
            candidate_id
            for candidate_id in locked_ids
            if any(item.candidate_id == candidate_id for item in target_candidates)
        }
        budget = max(0, max_regions - len(target_locked))
        by_id = {candidate.candidate_id: candidate for candidate in target_candidates}
        earliest_entry = min(candidate.entry_s for candidate in target_candidates)
        earliest_candidates = tuple(
            candidate
            for candidate in target_candidates
            if candidate.entry_s == earliest_entry
        )
        current = (
            _current_candidate_for_snapshot(snapshot, earliest_candidates)
            if snapshot is not None
            else _current_candidate(earliest_candidates)
        )
        path: list[str] = []
        for candidate in _topology_chain(current, by_id):
            if candidate.candidate_id in target_locked:
                continue
            if len(path) >= budget:
                break
            path.append(candidate.candidate_id)
        if len(path) < budget:
            for candidate in sorted(
                target_candidates,
                key=lambda item: (item.entry_s, -item.probability, item.candidate_id),
            ):
                if candidate.candidate_id in target_locked or candidate.candidate_id in path:
                    continue
                path.append(candidate.candidate_id)
                if len(path) >= budget:
                    break
        result[target_id] = tuple(path)
    return result


def _relink_temporal_paths(
    candidates: Sequence[MissionCandidate],
    paths: Mapping[str, Sequence[str]],
) -> tuple[MissionCandidate, ...]:
    """Make selected path edges explicit after the executable cap is applied."""
    if not paths:
        return tuple(candidates)
    selected_ids = {candidate.candidate_id for candidate in candidates}
    path_positions = {
        candidate_id: (index, tuple(path))
        for path in paths.values()
        for index, candidate_id in enumerate(path)
    }
    relinked: list[MissionCandidate] = []
    for candidate in candidates:
        predecessor_ids = tuple(
            relation
            for relation in candidate.predecessor_candidate_ids
            if relation in selected_ids
        )
        successor_ids = tuple(
            relation
            for relation in candidate.successor_candidate_ids
            if relation in selected_ids
        )
        path_data = path_positions.get(candidate.candidate_id)
        if path_data is not None:
            index, path = path_data
            if index > 0:
                predecessor_ids = tuple(dict.fromkeys((*predecessor_ids, path[index - 1])))
            if index + 1 < len(path):
                successor_ids = tuple(dict.fromkeys((*successor_ids, path[index + 1])))
        relinked.append(
            candidate.model_copy(
                update={
                    "predecessor_candidate_ids": predecessor_ids,
                    "successor_candidate_ids": successor_ids,
                }
            )
        )
    return tuple(relinked)


def _ordered_selection(
    locked_ids: Sequence[str],
    available_ids: Sequence[str],
    *,
    active_capable_uuv_ids: Iterable[str],
    prioritize_active: bool,
    preferred_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Order locked and available resources without inventing fleet members."""
    locked = tuple(dict.fromkeys(locked_ids))
    locked_set = set(locked)
    available = [uuv_id for uuv_id in available_ids if uuv_id not in locked_set]
    preferred_set = set(preferred_ids)
    preferred = [
        uuv_id for uuv_id in preferred_ids if uuv_id in available and uuv_id not in locked_set
    ]
    remaining = [uuv_id for uuv_id in available if uuv_id not in preferred_set]
    if not prioritize_active:
        return (*locked, *preferred, *remaining)
    active_capable = set(active_capable_uuv_ids)
    return tuple(
        [*locked]
        + [uuv_id for uuv_id in preferred if uuv_id in active_capable]
        + [uuv_id for uuv_id in preferred if uuv_id not in active_capable]
        + [uuv_id for uuv_id in remaining if uuv_id in active_capable]
        + [uuv_id for uuv_id in remaining if uuv_id not in active_capable]
    )


def _append_carrier_batches(
    carrier_batches: dict[str, list[UUVMissionBatch]],
    pool: _PlatformPool,
    candidate: MissionCandidate,
    assignment: RegionMissionState,
) -> None:
    """Materialize one logical assignment into physical carrier batches."""
    deployment_point, recovery_point = _carrier_service_points(candidate.perimeter_points)
    selected_ids = (
        *assignment.active_scan_uuv_ids,
        *assignment.passive_track_uuv_ids,
    )
    for carrier_id in sorted(pool.carrier_ids or (pool.carrier_id,)):
        carrier_uuv_ids = set(pool.uuv_ids_by_carrier.get(carrier_id, ()))
        selected_for_carrier = tuple(
            uuv_id for uuv_id in selected_ids if uuv_id in carrier_uuv_ids
        )
        if not selected_for_carrier:
            continue
        selected_set = set(selected_for_carrier)
        carrier_batches.setdefault(carrier_id, []).append(
            UUVMissionBatch(
                carrier_id=carrier_id,
                candidate_id=candidate.candidate_id,
                uuv_ids=selected_for_carrier,
                active_scan_uuv_ids=tuple(
                    uuv_id
                    for uuv_id in assignment.active_scan_uuv_ids
                    if uuv_id in selected_set
                ),
                passive_track_uuv_ids=tuple(
                    uuv_id
                    for uuv_id in assignment.passive_track_uuv_ids
                    if uuv_id in selected_set
                ),
                deployment_point=deployment_point,
                recovery_point=recovery_point,
                entry_s=candidate.entry_s,
                exit_s=candidate.exit_s,
            )
        )


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
    reject_partial: bool = False,
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
    # Optional capacity must not escalate a provider-authorized passive policy
    # into active sonar. Extra members remain passive observers.
    passive_ids = (*passive_ids, *extra_ids)
    minimum = _minimum_uuvs(candidate)
    if not selected_ids:
        return _uncovered_assignment(candidate, "no_uuv_available"), None
    if reject_partial and len(selected_ids) < minimum:
        return _uncovered_assignment(candidate, "insufficient_uuv"), None
    if reject_partial and len(active_ids) < candidate.active_scan_uuv_count:
        return _uncovered_assignment(candidate, "active_capability_unavailable"), None
    degraded_reasons: list[str] = []
    if len(selected_ids) < minimum:
        degraded_reasons.append("insufficient_uuv")
    if len(active_ids) < candidate.active_scan_uuv_count:
        degraded_reasons.append("active_capability_unavailable")
    coverage_ids = (*active_ids, *passive_ids)
    scan_waypoints: tuple[tuple[float, float], ...] = ()
    scan_waypoints_by_uuv: dict[str, tuple[tuple[float, float], ...]] = {}
    try:
        scan_waypoints = serpentine_coverage_waypoints(
            candidate.perimeter_points,
            lane_count=max(1, len(coverage_ids)),
        )
        scan_waypoints_by_uuv = serpentine_coverage_waypoints_by_uuv(
            candidate.perimeter_points,
            coverage_ids,
            start_point=candidate.perimeter_points[0],
        )
    except ValueError:
        degraded_reasons.append("coverage_path_unavailable")
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
        region_polygon=candidate.perimeter_points,
        scan_waypoints=scan_waypoints,
        scan_waypoints_by_uuv=scan_waypoints_by_uuv,
    )
    batch = None
    if selected_ids:
        deployment_point, recovery_point = _carrier_service_points(candidate.perimeter_points)
        batch = UUVMissionBatch(
            carrier_id=carrier_id,
            candidate_id=candidate.candidate_id,
            uuv_ids=selected_ids,
            active_scan_uuv_ids=tuple(active_ids),
            passive_track_uuv_ids=tuple(passive_ids),
            deployment_point=deployment_point,
            recovery_point=recovery_point,
            entry_s=candidate.entry_s,
            exit_s=candidate.exit_s,
        )
    return assignment, batch


def _carrier_service_points(
    perimeter_points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use one stable task-boundary rendezvous for deployment and recovery."""
    if len(perimeter_points) < 3:
        raise ValueError("task region requires at least three perimeter points")
    rendezvous = tuple(perimeter_points[0])
    return rendezvous, rendezvous


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
        region_polygon=candidate.perimeter_points,
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


def _archive_expired_candidates(
    candidates: Sequence[MissionCandidate],
    sim_time_s: int,
) -> tuple[MissionCandidate, ...]:
    """Drop expired prefixes when at least one candidate window remains executable."""
    valid_ids = {
        candidate.candidate_id
        for candidate in candidates
        if candidate.exit_s >= sim_time_s
    }
    if not valid_ids:
        return ()
    return tuple(
        candidate.model_copy(
            update={
                "predecessor_candidate_ids": tuple(
                    item
                    for item in candidate.predecessor_candidate_ids
                    if item in valid_ids
                ),
                "successor_candidate_ids": tuple(
                    item
                    for item in candidate.successor_candidate_ids
                    if item in valid_ids
                ),
            }
        )
        for candidate in candidates
        if candidate.candidate_id in valid_ids
    )


def _target_id_from_candidate_id(candidate_id: str) -> str:
    if ":r" in candidate_id:
        return candidate_id.split(":r", 1)[0]
    return candidate_id.split(":", 1)[0]


def _snapshot_revision(snapshot: Any) -> int:
    return max(1, int(getattr(snapshot, "snapshot_revision", 1)))


def _snapshot_sim_time(snapshot: Any) -> int:
    situation = getattr(snapshot, "situation", snapshot)
    return max(0, int(getattr(situation, "sim_time_s", getattr(snapshot, "sim_time_s", 0))))


def _resource_episodes(snapshot: Any) -> dict[str, int]:
    situation = getattr(snapshot, "situation", snapshot)
    return dict(getattr(situation, "uuv_resource_episodes", {}) or {})


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
        support_carriers = tuple(
            carrier
            for carrier in carriers
            if getattr(carrier, "role", "carrier") == "mother_ship"
        ) or carriers
        support_ids = {carrier.carrier_id for carrier in support_carriers}
        carrier_uuv_ids: dict[str, tuple[str, ...]] = {}
        uuv_positions = {
            platform.platform_id: tuple(platform.position_xy)
            for platform in eligible_uuvs
        }
        uuv_speeds_mps = {
            platform.platform_id: max(
                float(platform.speed_mps),
                float(platform.capability.motion.max_speed_mps),
            )
            for platform in eligible_uuvs
        }
        uuv_energy_fraction = {
            platform.platform_id: float(platform.energy_fraction)
            for platform in eligible_uuvs
        }
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
        # unlisted eligible UUVs as belonging to the first support ship when
        # roles are available, otherwise preserve the legacy primary carrier.
        fallback_carrier_id = next(iter(sorted(support_ids)), primary.carrier_id)
        carrier_uuv_ids[fallback_carrier_id] = tuple(
            sorted(
                {
                    *carrier_uuv_ids.get(fallback_carrier_id, ()),
                    *(set(uuv_ids) - assigned),
                }
            )
        )
        return _PlatformPool(
            carrier_id=primary.carrier_id,
            home_battle_group_id=home_battle_group_id,
            uuv_ids=uuv_ids,
            active_capable_uuv_ids=active_capable_uuv_ids,
            carrier_ids=tuple(sorted(carrier.carrier_id for carrier in carriers)),
            uuv_ids_by_carrier=carrier_uuv_ids,
            carrier_roles={
                carrier.carrier_id: getattr(carrier, "role", "carrier")
                for carrier in carriers
            },
            uuv_positions=uuv_positions,
            uuv_speeds_mps=uuv_speeds_mps,
            uuv_energy_fraction=uuv_energy_fraction,
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
        carrier_roles={"carrier-01": "carrier"},
        uuv_positions={
            uuv_id: tuple(getattr(uuv, "position_xy", (0.0, 0.0)))
            for uuv_id, uuv in zip(uuv_ids, eligible_legacy_uuvs, strict=False)
            if uuv_id
        },
        uuv_speeds_mps={
            uuv_id: max(float(getattr(uuv, "speed_mps", 0.0)), 1.0)
            for uuv_id, uuv in zip(uuv_ids, eligible_legacy_uuvs, strict=False)
            if uuv_id
        },
        uuv_energy_fraction={
            uuv_id: float(getattr(uuv, "energy_fraction", 1.0))
            for uuv_id, uuv in zip(uuv_ids, eligible_legacy_uuvs, strict=False)
            if uuv_id
        },
    )


def _execution_members_by_region(snapshot: Any) -> dict[str, tuple[str, ...]]:
    """Return currently waterborne members keyed by their public region ID."""
    situation = getattr(snapshot, "situation", snapshot)
    result: dict[str, tuple[str, ...]] = {}
    mode_order = {"active_scan": 0, "passive_track": 1, "returning": 2}
    groups = sorted(
        getattr(situation, "execution_groups", ()) or (),
        key=lambda group: (
            str(getattr(group, "region_id", "")),
            mode_order.get(str(getattr(getattr(group, "mode", None), "value", getattr(group, "mode", ""))), 99),
            str(getattr(group, "group_id", "")),
        ),
    )
    for group in groups:
        mode = getattr(group, "mode", None)
        mode_value = getattr(mode, "value", mode)
        if mode_value not in {"active_scan", "passive_track"}:
            continue
        region_id = getattr(group, "region_id", None)
        if not isinstance(region_id, str) or not region_id:
            continue
        members = tuple(
            dict.fromkeys(
                member_id
                for member_id in getattr(group, "member_ids", ())
                if isinstance(member_id, str) and member_id
            )
        )
        result[region_id] = tuple(
            dict.fromkeys((*result.get(region_id, ()), *members))
        )
    return result


def _deployed_uuv_ids(snapshot: Any) -> set[str]:
    """Return UUVs already waterborne at the planning snapshot boundary."""
    situation = getattr(snapshot, "situation", snapshot)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    roster = getattr(platform_snapshot, "roster", None)
    uuvs = tuple(getattr(roster, "uuvs", ()) or ())
    if not uuvs:
        uuvs = tuple(getattr(situation, "uuvs", ()) or ())
    return {
        str(getattr(uuv, "platform_id", getattr(uuv, "uuv_id", "")))
        for uuv in uuvs
        if getattr(uuv, "deployment_state", None) == "deployed"
        and getattr(uuv, "platform_id", getattr(uuv, "uuv_id", None))
    }


def _restrict_platform_pool(
    pool: _PlatformPool,
    allowed_ids: set[str],
) -> _PlatformPool:
    """Project a pool onto resources physically eligible for the new sortie."""
    uuv_ids = tuple(uuv_id for uuv_id in pool.uuv_ids if uuv_id in allowed_ids)
    return _PlatformPool(
        carrier_id=pool.carrier_id,
        home_battle_group_id=pool.home_battle_group_id,
        uuv_ids=uuv_ids,
        active_capable_uuv_ids=tuple(
            uuv_id for uuv_id in pool.active_capable_uuv_ids if uuv_id in allowed_ids
        ),
        carrier_ids=pool.carrier_ids,
        uuv_ids_by_carrier={
            carrier_id: tuple(
                uuv_id for uuv_id in uuv_ids_by_carrier if uuv_id in allowed_ids
            )
            for carrier_id, uuv_ids_by_carrier in pool.uuv_ids_by_carrier.items()
        },
        carrier_roles=pool.carrier_roles,
        uuv_positions={
            uuv_id: position
            for uuv_id, position in pool.uuv_positions.items()
            if uuv_id in allowed_ids
        },
        uuv_speeds_mps={
            uuv_id: speed
            for uuv_id, speed in pool.uuv_speeds_mps.items()
            if uuv_id in allowed_ids
        },
        uuv_energy_fraction={
            uuv_id: energy
            for uuv_id, energy in pool.uuv_energy_fraction.items()
            if uuv_id in allowed_ids
        },
    )


def _empty_plan(snapshot: Any, pool: _PlatformPool) -> ExecutableMissionPlan:
    return ExecutableMissionPlan(
        revision=_snapshot_revision(snapshot),
        carrier_missions={
            carrier_id: CarrierMissionModel(
                carrier_id=carrier_id,
                role=cast(
                    Literal["carrier", "mother_ship"],
                    pool.carrier_roles.get(carrier_id, "carrier"),
                ),
                home_battle_group_id=pool.home_battle_group_id,
                ready_uuv_ids=pool.uuv_ids_by_carrier.get(carrier_id, ()),
            )
            for carrier_id in pool.carrier_ids or (pool.carrier_id,)
        },
        resource_episode_by_uuv=_resource_episodes(snapshot),
    )
