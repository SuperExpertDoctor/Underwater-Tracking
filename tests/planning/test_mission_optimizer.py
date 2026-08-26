from types import SimpleNamespace

import pytest

from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    MissionCandidate,
    RegionLifecycle,
)
from underwater_tracking.domain.models import (
    ExecutionGroupState,
    IntelligenceSource,
    TargetSearchPrior,
)
from underwater_tracking.domain.regional_models import RegionalMissionCandidate, TimeWindow
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    UUVPlatformState,
)
from underwater_tracking.planning.mission_optimizer import (
    MissionOptimizer,
    required_active_uuvs,
    required_passive_uuvs,
)
from underwater_tracking.planning.mission_validation import validate_executable_mission_plan


def _capability() -> PlatformCapability:
    return PlatformCapability(
        kind=PlatformKind.UUV,
        motion=MotionLimits(
            max_speed_mps=8.0,
            max_acceleration_mps2=1.0,
            max_turn_rate_rad_s=1.0,
        ),
        sonar=SonarCapability(
            passive_range_m=2_000.0,
            passive_bearing_variance_rad2=0.1,
            active_source_range_m=1_000.0,
            active_receive_range_m=1_000.0,
            active_range_sigma_m=5.0,
            active_bearing_sigma_rad=0.1,
            active_capable=True,
            ping_cooldown_s=10,
            ping_energy_cost_fraction=0.1,
            clutter_sensitivity=0.1,
            exposure_cost=0.1,
        ),
        communications=CommunicationCapability(
            surface_range_m=2_000.0,
            acoustic_range_m=1_000.0,
        ),
    )


def _snapshot(uuv_count: int = 4) -> SimpleNamespace:
    uuvs = tuple(
        UUVPlatformState(
            platform_id=f"U{index:02d}",
            platform_index=index,
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=4.0,
            energy_fraction=0.9,
            deployment_state="onboard",
            capability=_capability(),
            master_connected=True,
        )
        for index in range(1, uuv_count + 1)
    )
    platform_snapshot = PlatformSnapshot(
        scenario_id="S1",
        sim_time_s=100,
        carrier=CarrierPlatformState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=5.0,
            support_radius_m=2_000.0,
            onboard_platform_ids=tuple(uuv.platform_id for uuv in uuvs),
            deployed_platform_ids=(),
            returning_platform_ids=(),
        ),
        roster=PlatformRoster(usvs=(), uuvs=uuvs),
        communication_links=(),
    )
    situation = SimpleNamespace(
        sim_time_s=100,
        snapshot_revision=7,
        platform_snapshot=platform_snapshot,
    )
    return SimpleNamespace(
        sim_time_s=100,
        snapshot_revision=7,
        situation=situation,
    )


def _candidate(
    candidate_id: str,
    *,
    entry_s: int,
    exit_s: int,
    probability: float,
    active: int = 1,
    passive: int = 1,
    reserve: int = 0,
    optional: int = 0,
    predecessors: tuple[str, ...] = (),
    successors: tuple[str, ...] = (),
) -> MissionCandidate:
    return MissionCandidate(
        candidate_id=candidate_id,
        target_id="T1",
        entry_s=entry_s,
        exit_s=exit_s,
        probability=probability,
        perimeter_points=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        active_scan_uuv_count=active,
        passive_track_uuv_count=passive,
        reserve_uuv_count=reserve,
        optional_uuv_count=optional,
        predecessor_candidate_ids=predecessors,
        successor_candidate_ids=successors,
    )


def test_required_scanning_and_tracking_counts_are_explicit_and_deterministic() -> None:
    region = _candidate("T1:r1", entry_s=100, exit_s=200, probability=0.8, active=2, passive=3)

    assert required_active_uuvs(region, _snapshot()) == 2
    assert required_passive_uuvs(region, _snapshot()) == 3


def test_optimizer_reserves_future_high_probability_region() -> None:
    current = _candidate("T1:r1", entry_s=0, exit_s=150, probability=0.9)
    future = _candidate(
        "T1:r2",
        entry_s=200,
        exit_s=300,
        probability=0.95,
        active=0,
        passive=2,
    )

    result = MissionOptimizer().optimize(_snapshot(), (current, future))

    assert isinstance(result, ExecutableMissionPlan)
    batch = result.uuv_batches_by_carrier["carrier-01"][0]
    assert batch.uuv_ids == ("U01", "U02")
    assert result.reserved_uuv_ids == ("U03", "U04")
    assert result.assignments_by_candidate["T1:r2"].reserve_uuv_ids == ("U03", "U04")


def test_optimizer_uses_one_task_region_boundary_point_for_deployment_and_recovery() -> None:
    result = MissionOptimizer().optimize(
        _snapshot(),
        (_candidate("T1:r1", entry_s=0, exit_s=150, probability=0.9),),
    )

    batch = result.uuv_batches_by_carrier["carrier-01"][0]
    assert batch.deployment_point is not None
    assert batch.recovery_point is not None
    assert batch.deployment_point == (0.0, 0.0)
    assert batch.recovery_point == batch.deployment_point


def test_larger_current_batch_is_rejected_when_it_breaks_future_reserve() -> None:
    current = _candidate(
        "T1:r1",
        entry_s=0,
        exit_s=150,
        probability=0.9,
        optional=2,
    )
    future = _candidate(
        "T1:r2",
        entry_s=200,
        exit_s=300,
        probability=0.95,
        active=0,
        passive=2,
    )

    result = MissionOptimizer().optimize(_snapshot(5), (current, future))

    batch = result.uuv_batches_by_carrier["carrier-01"][0]
    assert len(batch.uuv_ids) == 3
    assert len(result.reserved_uuv_ids) == 2


def test_optimizer_prefers_higher_probability_marginal_benefit() -> None:
    lower = _candidate("T1:lower", entry_s=0, exit_s=150, probability=0.55)
    higher = _candidate("T1:higher", entry_s=0, exit_s=150, probability=0.95)

    result = MissionOptimizer().optimize(_snapshot(2), (lower, higher))

    assert result.uuv_batches_by_carrier["carrier-01"][0].candidate_id == "T1:higher"
    assert result.assignments_by_candidate["T1:lower"].lifecycle is RegionLifecycle.UNCOVERED


def test_optimizer_keeps_region_cap_assignments_for_audit() -> None:
    candidates = tuple(
        _candidate(
            f"T1:r{index}",
            entry_s=index * 100,
            exit_s=index * 100 + 80,
            probability=0.5,
        )
        for index in range(6)
    )

    result = MissionOptimizer().optimize(_snapshot(4), candidates)

    assert len(result.region_assignments) == 6
    excluded = [
        assignment
        for assignment in result.region_assignments
        if "region_cap_not_selected" in assignment.degraded_reasons
    ]
    expired = [
        assignment
        for assignment in result.region_assignments
        if "candidate_window_expired" in assignment.degraded_reasons
    ]
    assert len(excluded) == 1
    assert len(expired) == 1
    assert all(assignment.lifecycle is RegionLifecycle.UNCOVERED for assignment in excluded)
    assert all(assignment.lifecycle is RegionLifecycle.UNCOVERED for assignment in expired)


def test_initial_zero_snapshot_revision_accepts_first_executable_plan() -> None:
    snapshot = _snapshot(4)
    snapshot.snapshot_revision = 0
    snapshot.situation.snapshot_revision = 0
    snapshot.situation.uuvs = tuple(
        SimpleNamespace(
            uuv_id=platform.platform_id,
            deployment_state="onboard",
            energy_fraction=platform.energy_fraction,
        )
        for platform in snapshot.situation.platform_snapshot.roster.uuvs
    )
    candidate = _candidate("T1:r1", entry_s=0, exit_s=100, probability=0.8)

    result = MissionOptimizer().optimize(snapshot, (candidate,))
    issues = validate_executable_mission_plan(snapshot, result)

    assert result.revision == 1
    assert "mission_revision_mismatch" not in issues


def test_resource_shortage_degrades_without_fabricating_uuv_ids() -> None:
    candidate = _candidate(
        "T1:r1",
        entry_s=0,
        exit_s=150,
        probability=0.9,
        active=2,
        passive=1,
    )

    result = MissionOptimizer().optimize(_snapshot(1), (candidate,))

    assignment = result.region_assignments[0]
    assert assignment.lifecycle is RegionLifecycle.DEGRADED
    assert assignment.active_scan_uuv_ids == ("U01",)
    assert assignment.passive_track_uuv_ids == ()
    assert all("U99" not in ids for ids in result.all_uuv_ids)


def test_optimizer_rejects_invalid_candidate_window() -> None:
    with pytest.raises(ValueError):
        _candidate("T1:bad", entry_s=10, exit_s=10, probability=0.5)


def test_optimizer_accepts_strict_regional_mission_candidates() -> None:
    candidate = RegionalMissionCandidate(
        candidate_id="T1:r1:square:0:0:1",
        cell_ids=("T1:r1:cell:0:0",),
        time_window=TimeWindow(start_s=0, end_s=100),
        perimeter_points=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    )

    result = MissionOptimizer().optimize(_snapshot(2), (candidate,))

    assert result.batches[0].candidate_id == candidate.candidate_id


def test_manually_locked_uuv_is_a_hard_assignment_constraint() -> None:
    candidate = _candidate("T1:r1", entry_s=0, exit_s=100, probability=0.8)

    result = MissionOptimizer().optimize(
        _snapshot(4),
        (candidate,),
        locked_uuv_ids_by_candidate={candidate.candidate_id: ("U04",)},
    )

    assert "U04" in result.batches[0].uuv_ids


def test_unavailable_provider_lock_is_reallocated_and_audited() -> None:
    snapshot = _snapshot(4)
    uuvs = tuple(
        uuv.model_copy(update={"deployment_state": "returning"})
        if uuv.platform_id == "U04"
        else uuv
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )
    candidate = _candidate("T1:r1", entry_s=0, exit_s=100, probability=0.8)

    result = MissionOptimizer().optimize(
        snapshot,
        (candidate,),
        locked_uuv_ids_by_candidate={candidate.candidate_id: ("U04",)},
    )

    assignment = result.assignments_by_candidate[candidate.candidate_id]
    assert "U04" not in result.all_uuv_ids
    assert "U04" not in result.batches[0].uuv_ids
    assert any(
        reason == "provider_assignment_unavailable:U04"
        for reason in assignment.degraded_reasons
    )
    assert any(
        reason.endswith(":provider_assignment_unavailable:U04")
        for reason in result.degraded_reasons
    )


def test_duplicate_provider_locks_keep_earliest_window_and_reallocate_later() -> None:
    first = _candidate("T1:locked-first", entry_s=0, exit_s=100, probability=0.9)
    second = _candidate("T1:locked-second", entry_s=110, exit_s=210, probability=0.8)

    result = MissionOptimizer().optimize(
        _snapshot(6),
        (first, second),
        locked_uuv_ids_by_candidate={
            first.candidate_id: ("U01", "U02"),
            second.candidate_id: ("U01", "U03"),
        },
    )

    assert "U01" in result.batches[0].uuv_ids
    later = result.assignments_by_candidate[second.candidate_id]
    assert "U01" not in later.active_scan_uuv_ids + later.passive_track_uuv_ids
    assert "provider_assignment_conflict:U01" in later.degraded_reasons


def test_topology_lock_is_not_consumed_by_an_earlier_successor_window() -> None:
    current = _candidate(
        "T1:current",
        entry_s=0,
        exit_s=250,
        probability=0.8,
        active=0,
        passive=2,
        reserve=1,
        successors=("T1:bridge",),
    )
    bridge = _candidate(
        "T1:bridge",
        entry_s=100,
        exit_s=350,
        probability=0.7,
        active=1,
        passive=0,
    )
    locked_successor = _candidate(
        "T1:locked-successor",
        entry_s=200,
        exit_s=350,
        probability=0.6,
        active=1,
        passive=0,
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        _snapshot(4),
        (current, bridge, locked_successor),
        locked_uuv_ids_by_candidate={locked_successor.candidate_id: ("U01", "U02")},
    )

    assigned = {
        uuv_id
        for assignment in result.region_assignments
        for uuv_id in (
            *assignment.active_scan_uuv_ids,
            *assignment.passive_track_uuv_ids,
            *assignment.reserve_uuv_ids,
        )
    }
    assert len(assigned) == sum(
        len(
            {
                *assignment.active_scan_uuv_ids,
                *assignment.passive_track_uuv_ids,
                *assignment.reserve_uuv_ids,
            }
        )
        for assignment in result.region_assignments
    )
    assert result.assignments_by_candidate[locked_successor.candidate_id].active_scan_uuv_ids == (
        "U01",
    )
    assert result.assignments_by_candidate[locked_successor.candidate_id].passive_track_uuv_ids == (
        "U02",
    )


def test_deployed_uuv_is_not_reused_by_a_new_candidate_window() -> None:
    snapshot = _snapshot(4)
    uuvs = tuple(
        uuv.model_copy(update={"deployment_state": "deployed"})
        if uuv.platform_id == "U04"
        else uuv
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )
    candidate = _candidate("T1:new-window", entry_s=0, exit_s=100, probability=0.8)

    result = MissionOptimizer().optimize(snapshot, (candidate,))

    assert "U04" not in result.batches[0].uuv_ids


def test_current_execution_group_members_stay_in_current_batch() -> None:
    snapshot = _snapshot(4)
    uuvs = tuple(
        uuv.model_copy(update={"deployment_state": "deployed"})
        if uuv.platform_id == "U04"
        else uuv
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )
    snapshot.situation.execution_groups = (
        ExecutionGroupState(
            group_id="G-T1:current",
            target_id="T1",
            region_id="T1:current",
            member_ids=("U04",),
            mode="active_scan",
        ),
    )
    current = _candidate(
        "T1:current",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        active=1,
        passive=0,
        successors=("T1:future",),
    )
    same_window_branch = _candidate(
        "T1:aaa-branch",
        entry_s=0,
        exit_s=100,
        probability=0.8,
        active=1,
        passive=0,
    )
    future = _candidate(
        "T1:future",
        entry_s=110,
        exit_s=210,
        probability=0.8,
        active=1,
        passive=0,
        predecessors=("T1:current",),
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        snapshot,
        (current, same_window_branch, future),
        locked_uuv_ids_by_candidate={same_window_branch.candidate_id: ("U04",)},
    )

    assert "U04" in result.batches[0].uuv_ids
    assignment = result.assignments_by_candidate["T1:current"]
    assert "U04" in assignment.active_scan_uuv_ids + assignment.passive_track_uuv_ids
    assert "provider_assignment_conflict:U04" in result.assignments_by_candidate[
        same_window_branch.candidate_id
    ].degraded_reasons


def test_current_execution_members_replace_extra_same_candidate_provider_locks() -> None:
    snapshot = _snapshot(4)
    uuvs = tuple(
        uuv.model_copy(update={"deployment_state": "deployed"})
        if uuv.platform_id in {"U02", "U04"}
        else uuv
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )
    snapshot.situation.execution_groups = (
        ExecutionGroupState(
            group_id="G-T1:current-active",
            target_id="T1",
            region_id="T1:current",
            member_ids=("U04",),
            mode="active_scan",
        ),
        ExecutionGroupState(
            group_id="G-T1:current-passive",
            target_id="T1",
            region_id="T1:current",
            member_ids=("U02",),
            mode="passive_track",
        ),
    )
    candidate = _candidate(
        "T1:current",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        active=1,
        passive=1,
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        snapshot,
        (candidate,),
        locked_uuv_ids_by_candidate={candidate.candidate_id: ("U01", "U02", "U04")},
    )

    assert result.batches[0].uuv_ids == ("U04", "U02")
    assert result.assignments_by_candidate[candidate.candidate_id].active_scan_uuv_ids == (
        "U04",
    )
    assert result.assignments_by_candidate[candidate.candidate_id].passive_track_uuv_ids == (
        "U02",
    )


def test_optimizer_excludes_low_energy_returning_and_failed_uuvs() -> None:
    snapshot = _snapshot(4)
    uuvs = tuple(
        uuv.model_copy(
            update=(
                {"energy_fraction": 0.05}
                if uuv.platform_id == "U02"
                else {"deployment_state": "returning"}
                if uuv.platform_id == "U03"
                else {"deployment_state": "failed"}
                if uuv.platform_id == "U04"
                else {}
            )
        )
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={
            "roster": PlatformRoster(
                usvs=(),
                uuvs=uuvs,
            )
        }
    )

    result = MissionOptimizer().optimize(
        snapshot,
        (_candidate("T1:r1", entry_s=0, exit_s=100, probability=0.9),),
    )

    assert result.all_uuv_ids == ("U01",)


def test_passive_only_candidate_can_use_uuv_without_active_sonar() -> None:
    snapshot = _snapshot(2)
    passive_capability = _capability().model_copy(
        update={"sonar": _capability().sonar.model_copy(update={"active_capable": False})}
    )
    uuvs = tuple(
        uuv.model_copy(update={"capability": passive_capability})
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )

    result = MissionOptimizer().optimize(
        snapshot,
        (_candidate("T1:passive", entry_s=0, exit_s=100, probability=0.8, active=0, passive=2),),
    )

    assert result.batches[0].passive_track_uuv_ids == ("U01", "U02")
    assert result.assignments_by_candidate["T1:passive"].degraded_reasons == ()


def test_optional_uuv_does_not_upgrade_provider_passive_policy_to_active() -> None:
    result = MissionOptimizer().optimize(
        _snapshot(2),
        (
            _candidate(
                "T1:passive-optional",
                entry_s=100,
                exit_s=300,
                probability=0.8,
                active=0,
                passive=1,
                optional=1,
            ),
        ),
    )

    assignment = result.assignments_by_candidate["T1:passive-optional"]
    assert assignment.active_scan_uuv_ids == ()
    assert assignment.passive_track_uuv_ids == ("U01", "U02")


def test_global_topology_auction_preserves_active_uuv_for_active_successor() -> None:
    snapshot = _snapshot(2)
    passive_capability = _capability().model_copy(
        update={"sonar": _capability().sonar.model_copy(update={"active_capable": False})}
    )
    uuvs = tuple(
        uuv.model_copy(update={"capability": passive_capability})
        if uuv.platform_id == "U02"
        else uuv
        for uuv in snapshot.situation.platform_snapshot.roster.uuvs
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"roster": PlatformRoster(usvs=(), uuvs=uuvs)}
    )
    current = _candidate(
        "T1:current",
        entry_s=100,
        exit_s=200,
        probability=0.9,
        active=0,
        passive=1,
        successors=("T1:successor",),
    )
    successor = _candidate(
        "T1:successor",
        entry_s=200,
        exit_s=300,
        probability=0.9,
        active=1,
        passive=0,
        predecessors=("T1:current",),
    )

    result = MissionOptimizer().optimize(snapshot, (current, successor))

    current_assignment = result.assignments_by_candidate["T1:current"]
    successor_assignment = result.assignments_by_candidate["T1:successor"]
    assert current_assignment.passive_track_uuv_ids == ("U02",)
    assert successor_assignment.active_scan_uuv_ids == ("U01",)
    assert successor_assignment.lifecycle is not RegionLifecycle.UNCOVERED


def test_optimizer_archives_expired_root_when_successor_window_is_still_valid() -> None:
    expired = _candidate(
        "T1:expired",
        entry_s=0,
        exit_s=90,
        probability=0.9,
        successors=("T1:valid",),
    )
    valid = _candidate(
        "T1:valid",
        entry_s=120,
        exit_s=300,
        probability=0.8,
        predecessors=("T1:expired",),
    )

    result = MissionOptimizer().optimize(_snapshot(4), (expired, valid))

    assert result.assignments_by_candidate["T1:expired"].lifecycle is RegionLifecycle.UNCOVERED
    assert "candidate_window_expired" in result.assignments_by_candidate[
        "T1:expired"
    ].degraded_reasons
    assert all(batch.candidate_id != "T1:expired" for batch in result.batches)
    assert result.assignments_by_candidate["T1:valid"].lifecycle is not RegionLifecycle.UNCOVERED
    assert all(batch.candidate_id == "T1:valid" for batch in result.batches)


def test_optimizer_does_not_reuse_uuvs_rejected_by_the_auction() -> None:
    distant = _candidate(
        "T1:distant",
        entry_s=0,
        exit_s=100,
        probability=0.1,
        active=1,
        passive=1,
    ).model_copy(
        update={
            "perimeter_points": (
                (100_000.0, 100_000.0),
                (100_000.0, 100_100.0),
                (100_100.0, 100_000.0),
                (100_100.0, 100_100.0),
            )
        }
    )

    result = MissionOptimizer().optimize(_snapshot(2), (distant,))

    assignment = result.assignments_by_candidate[distant.candidate_id]
    assert result.batches == ()
    assert assignment.lifecycle is RegionLifecycle.UNCOVERED
    assert assignment.degraded_reasons == ("no_uuv_available",)


def test_optimizer_preserves_declared_passive_only_role_counts() -> None:
    result = MissionOptimizer().optimize(
        _snapshot(2),
        (_candidate("T1:goal", entry_s=0, exit_s=100, probability=0.8, active=0, passive=1),),
    )

    assignment = result.assignments_by_candidate["T1:goal"]
    assert assignment.active_scan_uuv_ids == ()
    assert assignment.passive_track_uuv_ids
    assert result.batches[0].active_scan_uuv_ids == ()


def test_goal_mode_bootstraps_active_passive_seed_and_temporal_successor() -> None:
    first = _candidate(
        "T1:goal-bootstrap",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        active=0,
        passive=1,
    )
    successor = _candidate(
        "T1:goal-successor",
        entry_s=110,
        exit_s=210,
        probability=0.8,
        active=0,
        passive=0,
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        _snapshot(4),
        (first, successor),
    )

    first_assignment = result.assignments_by_candidate[first.candidate_id]
    successor_assignment = result.assignments_by_candidate[successor.candidate_id]
    assert first_assignment.active_scan_uuv_ids
    assert first_assignment.passive_track_uuv_ids
    assert first_assignment.handoff_to == successor.candidate_id
    assert successor_assignment.handoff_from == first.candidate_id
    assert successor_assignment.passive_track_uuv_ids
    assert "future_window_reservation" in successor_assignment.degraded_reasons
    # Goal mode must publish a physical successor batch so the next carrier
    # deployment can overlap the predecessor and produce real handoff evidence.
    assert tuple(batch.candidate_id for batch in result.batches) == (
        first.candidate_id,
        successor.candidate_id,
    )
    assert all(
        batch.deployment_point is None and batch.recovery_point is None
        for batch in result.batches
    )


def test_goal_mode_materializes_only_the_immediate_physical_successor() -> None:
    candidates = (
        _candidate("T1:goal-current", entry_s=0, exit_s=100, probability=0.9),
        _candidate("T1:goal-next", entry_s=110, exit_s=210, probability=0.8),
        _candidate("T1:goal-future", entry_s=220, exit_s=320, probability=0.7),
    )

    result = MissionOptimizer(goal_mode=True).optimize(_snapshot(8), candidates)

    assert tuple(batch.candidate_id for batch in result.batches) == (
        "T1:goal-current",
        "T1:goal-next",
    )


def test_goal_mode_does_not_publish_a_partial_future_role_assignment() -> None:
    current = _candidate(
        "T1:goal-current",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        active=1,
        passive=1,
    )
    future = _candidate(
        "T1:goal-future",
        entry_s=110,
        exit_s=210,
        probability=0.8,
        active=1,
        passive=1,
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        _snapshot(3),
        (current, future),
        locked_uuv_ids_by_candidate={future.candidate_id: ("U03",)},
    )

    future_assignment = result.assignments_by_candidate[future.candidate_id]
    assert future_assignment.lifecycle is RegionLifecycle.UNCOVERED
    assert future_assignment.active_scan_uuv_ids == ()
    assert future_assignment.passive_track_uuv_ids == ()
    assert future_assignment.reserve_uuv_ids == ()
    assert "insufficient_uuv" in future_assignment.degraded_reasons
    assert all(batch.candidate_id != future.candidate_id for batch in result.batches)


def test_goal_mode_does_not_link_to_an_uncovered_topology_bridge() -> None:
    current = _candidate(
        "T1:goal-current",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        successors=("T1:goal-bridge",),
    )
    bridge = _candidate(
        "T1:goal-bridge",
        entry_s=0,
        exit_s=100,
        probability=0.8,
        predecessors=("T1:goal-current",),
        successors=("T1:goal-next",),
    )
    next_region = _candidate(
        "T1:goal-next",
        entry_s=110,
        exit_s=210,
        probability=0.7,
        predecessors=("T1:goal-bridge",),
    )

    result = MissionOptimizer(goal_mode=True).optimize(
        _snapshot(2),
        (current, bridge, next_region),
    )

    assignments = result.assignments_by_candidate
    executable_ids = {
        candidate_id
        for candidate_id, assignment in assignments.items()
        if assignment.active_scan_uuv_ids or assignment.passive_track_uuv_ids
    }
    for assignment in assignments.values():
        for linked_id in (assignment.handoff_from, assignment.handoff_to):
            if linked_id is not None:
                assert linked_id in executable_ids


def test_optimizer_materializes_declared_active_passive_handoff_chain() -> None:
    first = _candidate(
        "T1:goal-first",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        active=1,
        passive=1,
        successors=("T1:goal-second",),
    )
    second = _candidate(
        "T1:goal-second",
        entry_s=110,
        exit_s=210,
        probability=0.8,
        active=1,
        passive=1,
        predecessors=("T1:goal-first",),
        successors=("T1:goal-third",),
    )
    third = _candidate(
        "T1:goal-third",
        entry_s=220,
        exit_s=320,
        probability=0.7,
        active=1,
        passive=1,
        predecessors=("T1:goal-second",),
    )

    result = MissionOptimizer().optimize(
        _snapshot(8),
        (first, second, third),
        locked_uuv_ids_by_candidate={first.candidate_id: ("U01", "U02")},
    )

    batches = result.uuv_batches_by_carrier["carrier-01"]
    assert tuple(batch.candidate_id for batch in batches) == (
        "T1:goal-first",
        "T1:goal-second",
        "T1:goal-third",
    )
    assert all(batch.active_scan_uuv_ids for batch in batches)
    assert all(batch.passive_track_uuv_ids for batch in batches)
    assert result.assignments_by_candidate["T1:goal-first"].handoff_to == "T1:goal-second"
    assert result.assignments_by_candidate["T1:goal-second"].handoff_from == "T1:goal-first"
    assert result.assignments_by_candidate["T1:goal-second"].handoff_to == "T1:goal-third"


def test_optimizer_preserves_all_configured_carriers_in_the_plan() -> None:
    snapshot = _snapshot(2)
    primary = snapshot.situation.platform_snapshot.carrier
    secondary = primary.model_copy(
        update={
            "carrier_id": "carrier-02",
            "position_xy": (100.0, 0.0),
            "onboard_platform_ids": (),
        }
    )
    snapshot.situation.platform_snapshot = SimpleNamespace(
        roster=snapshot.situation.platform_snapshot.roster,
        carrier=primary,
        carriers=(primary, secondary),
    )

    result = MissionOptimizer().optimize(
        snapshot,
        (_candidate("T1:r1", entry_s=0, exit_s=100, probability=0.9),),
    )

    assert set(result.carrier_missions) == {"carrier-01", "carrier-02"}


def test_mission_validation_reports_missing_configured_carrier_mission() -> None:
    base = _snapshot(2)
    primary = base.situation.platform_snapshot.carrier
    secondary = primary.model_copy(update={"carrier_id": "carrier-02"})
    platform_snapshot = SimpleNamespace(
        roster=base.situation.platform_snapshot.roster,
        carrier=primary,
        carriers=(primary, secondary),
    )
    situation = SimpleNamespace(
        snapshot_revision=7,
        platform_snapshot=platform_snapshot,
        uuvs=(),
        uuv_resource_episodes={},
    )
    snapshot = PlanningSnapshot(situation, None, ())
    plan = ExecutableMissionPlan(
        revision=7,
        carrier_missions={
            "carrier-01": CarrierMissionModel(
                carrier_id="carrier-01",
                home_battle_group_id="battle-group-01",
            )
        },
    )

    issues = validate_executable_mission_plan(snapshot, plan)

    assert "missing_carrier_mission:carrier-02" in issues

    extra_plan = plan.model_copy(
        update={
            "carrier_missions": {
                **plan.carrier_missions,
                "carrier-99": CarrierMissionModel(
                    carrier_id="carrier-99",
                    home_battle_group_id="battle-group-01",
                ),
            }
        }
    )
    extra_issues = validate_executable_mission_plan(snapshot, extra_plan)

    assert "unknown_carrier_mission:carrier-99" in extra_issues


def test_optimizer_splits_current_batch_by_physical_carrier_ownership() -> None:
    snapshot = _snapshot(4)
    primary = snapshot.situation.platform_snapshot.carrier
    secondary = primary.model_copy(
        update={
            "carrier_id": "carrier-02",
            "position_xy": (100.0, 0.0),
            "onboard_platform_ids": ("U02", "U04"),
            "deployed_platform_ids": (),
        }
    )
    primary = primary.model_copy(
        update={
            "onboard_platform_ids": ("U01", "U03"),
            "deployed_platform_ids": (),
        }
    )
    snapshot.situation.platform_snapshot = snapshot.situation.platform_snapshot.model_copy(
        update={"carrier": primary, "carriers": (primary, secondary)}
    )

    result = MissionOptimizer().optimize(
        snapshot,
        (_candidate("T1:r1", entry_s=0, exit_s=100, probability=0.9),),
    )

    assert {
        carrier_id: batch.uuv_ids
        for carrier_id, batches in result.uuv_batches_by_carrier.items()
        for batch in batches
    } == {"carrier-01": ("U01",), "carrier-02": ("U02",)}


def test_optimizer_materializes_topology_chain_as_handoff_batches() -> None:
    first = _candidate(
        "T1:r1",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        successors=("T1:r2",),
    )
    second = _candidate(
        "T1:r2",
        entry_s=110,
        exit_s=200,
        probability=0.9,
        predecessors=("T1:r1",),
        successors=("T1:r3",),
    )
    third = _candidate(
        "T1:r3",
        entry_s=210,
        exit_s=300,
        probability=0.9,
        predecessors=("T1:r2",),
    )

    result = MissionOptimizer().optimize(_snapshot(8), (first, second, third))

    batches = result.uuv_batches_by_carrier["carrier-01"]
    assert tuple(batch.candidate_id for batch in batches) == ("T1:r1", "T1:r2", "T1:r3")
    assignments = result.assignments_by_candidate
    assert assignments["T1:r1"].handoff_to == "T1:r2"
    assert assignments["T1:r2"].handoff_from == "T1:r1"
    assert assignments["T1:r2"].handoff_to == "T1:r3"
    assert assignments["T1:r3"].handoff_from == "T1:r2"


def test_optimizer_preserves_a_temporal_path_when_unprioritized_regions_are_capped() -> None:
    bridge = _candidate(
        "T1:bridge",
        entry_s=0,
        exit_s=80,
        probability=0.5,
        predecessors=("T1:r1",),
        successors=("T1:r2",),
    )
    first = _candidate(
        "T1:r1",
        entry_s=0,
        exit_s=80,
        probability=0.5,
        successors=("T1:bridge",),
    )
    second = _candidate(
        "T1:r2",
        entry_s=100,
        exit_s=180,
        probability=0.5,
        predecessors=("T1:bridge",),
        successors=("T1:r3",),
    )
    third = _candidate(
        "T1:r3",
        entry_s=200,
        exit_s=280,
        probability=0.5,
        predecessors=("T1:r2",),
        successors=("T1:r4",),
    )
    fourth = _candidate(
        "T1:r4",
        entry_s=300,
        exit_s=380,
        probability=0.5,
        predecessors=("T1:r3",),
    )
    fifth = _candidate(
        "T1:r5",
        entry_s=400,
        exit_s=480,
        probability=0.5,
    )

    candidates = tuple(
        candidate.model_copy(update={"priority": 1.0})
        for candidate in (first, bridge, second, third, fourth, fifth)
    )

    result = MissionOptimizer().optimize(_snapshot(8), candidates)

    batches = result.uuv_batches_by_carrier["carrier-01"]
    assert tuple(batch.candidate_id for batch in batches) == (
        "T1:r2",
        "T1:r3",
        "T1:r4",
        "T1:r5",
    )


def test_optimizer_keeps_the_earliest_temporal_window_before_future_reserves() -> None:
    candidates = (
        _candidate(
            "T1:r1",
            entry_s=0,
            exit_s=100,
            probability=0.9,
            reserve=1,
            successors=("T1:r2",),
        ),
        _candidate(
            "T1:r2",
            entry_s=110,
            exit_s=200,
            probability=0.9,
            reserve=1,
            predecessors=("T1:r1",),
            successors=("T1:r3",),
        ),
        _candidate(
            "T1:r3",
            entry_s=210,
            exit_s=300,
            probability=0.9,
            reserve=1,
            predecessors=("T1:r2",),
            successors=("T1:r4",),
        ),
        _candidate(
            "T1:r4",
            entry_s=310,
            exit_s=400,
            probability=0.9,
            reserve=1,
            predecessors=("T1:r3",),
        ),
    )

    result = MissionOptimizer().optimize(_snapshot(8), candidates)

    assert result.uuv_batches_by_carrier["carrier-01"][0].candidate_id == "T1:r1"
    assignment = result.assignments_by_candidate["T1:r1"]
    assert len((*assignment.active_scan_uuv_ids, *assignment.passive_track_uuv_ids)) >= 2
    assert assignment.lifecycle is not RegionLifecycle.UNCOVERED


def test_optimizer_uses_public_search_prior_to_select_the_current_region() -> None:
    snapshot = _snapshot(2)
    snapshot.situation.target_search_priors = (
        TargetSearchPrior(
            prior_id="prior-1",
            target_id="T1",
            source=IntelligenceSource.TECHNICAL_RECONNAISSANCE,
            issued_at_s=0,
            valid_until_s=900,
            center_xy=(-4_200.0, -6_200.0),
            covariance_xy=((360_000.0, 0.0), (0.0, 360_000.0)),
            confidence=0.8,
        ),
    )
    near = _candidate("T1:near", entry_s=0, exit_s=100, probability=0.5).model_copy(
        update={
            "perimeter_points": (
                (-4_400.0, -6_400.0),
                (-4_400.0, -6_000.0),
                (-4_000.0, -6_400.0),
                (-4_000.0, -6_000.0),
            )
        }
    )
    far = _candidate("T1:far", entry_s=0, exit_s=100, probability=0.5).model_copy(
        update={
            "perimeter_points": (
                (4_000.0, 4_000.0),
                (4_000.0, 4_400.0),
                (4_400.0, 4_000.0),
                (4_400.0, 4_400.0),
            )
        }
    )

    result = MissionOptimizer().optimize(snapshot, (far, near))

    assert result.batches[0].candidate_id == "T1:near"


def test_public_prior_current_assignment_matches_its_physical_batch() -> None:
    snapshot = _snapshot(2)
    snapshot.situation.target_search_priors = (
        TargetSearchPrior(
            prior_id="prior-1",
            target_id="T1",
            source=IntelligenceSource.TECHNICAL_RECONNAISSANCE,
            issued_at_s=0,
            valid_until_s=900,
            center_xy=(-4_200.0, -6_200.0),
            covariance_xy=((360_000.0, 0.0), (0.0, 360_000.0)),
            confidence=0.8,
        ),
    )
    near = _candidate("T1:near", entry_s=0, exit_s=100, probability=0.5).model_copy(
        update={
            "perimeter_points": (
                (-4_400.0, -6_400.0),
                (-4_400.0, -6_000.0),
                (-4_000.0, -6_400.0),
                (-4_000.0, -6_000.0),
            )
        }
    )
    far = _candidate("T1:far", entry_s=0, exit_s=100, probability=0.5).model_copy(
        update={
            "perimeter_points": (
                (4_000.0, 4_000.0),
                (4_000.0, 4_400.0),
                (4_400.0, 4_000.0),
                (4_400.0, 4_400.0),
            )
        }
    )

    result = MissionOptimizer().optimize(snapshot, (far, near))

    batch = result.batches[0]
    assignment = result.assignments_by_candidate["T1:near"]
    assert batch.candidate_id == "T1:near"
    assert assignment.lifecycle is not RegionLifecycle.UNCOVERED
    assert {
        *assignment.active_scan_uuv_ids,
        *assignment.passive_track_uuv_ids,
    } == set(batch.uuv_ids)


def test_public_prior_beats_same_window_topology_root_tiebreaker() -> None:
    snapshot = _snapshot(2)
    snapshot.situation.target_search_priors = (
        TargetSearchPrior(
            prior_id="prior-1",
            target_id="T1",
            source=IntelligenceSource.TECHNICAL_RECONNAISSANCE,
            issued_at_s=0,
            valid_until_s=900,
            center_xy=(-4_200.0, -6_200.0),
            covariance_xy=((360_000.0, 0.0), (0.0, 360_000.0)),
            confidence=0.8,
        ),
    )
    topology_root = _candidate(
        "T1:root",
        entry_s=0,
        exit_s=100,
        probability=0.9,
        successors=("T1:near",),
    )
    near = _candidate(
        "T1:near",
        entry_s=0,
        exit_s=100,
        probability=0.5,
        predecessors=("T1:root",),
    ).model_copy(
        update={
            "perimeter_points": (
                (-4_400.0, -6_400.0),
                (-4_400.0, -6_000.0),
                (-4_000.0, -6_400.0),
                (-4_000.0, -6_000.0),
            )
        }
    )

    result = MissionOptimizer().optimize(snapshot, (topology_root, near))

    assert result.batches[0].candidate_id == "T1:near"
