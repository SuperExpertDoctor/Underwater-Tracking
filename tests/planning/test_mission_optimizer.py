from types import SimpleNamespace

import pytest

from underwater_tracking.domain.mission_models import (
    ExecutableMissionPlan,
    MissionCandidate,
    RegionLifecycle,
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
) -> MissionCandidate:
    return MissionCandidate(
        candidate_id=candidate_id,
        target_id="T1",
        entry_s=entry_s,
        exit_s=exit_s,
        probability=probability,
        perimeter_points=((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)),
        active_scan_uuv_count=active,
        passive_track_uuv_count=passive,
        reserve_uuv_count=reserve,
        optional_uuv_count=optional,
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
        perimeter_points=((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)),
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
