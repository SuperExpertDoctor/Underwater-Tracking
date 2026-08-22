from __future__ import annotations

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    RegionMissionState,
    UUVResourceState,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.domain.planning_epoch_models import PlanningEpoch
from underwater_tracking.planning.mission_revalidation import (
    revalidate_executable_mission_plan,
)
from underwater_tracking.runtime.mission_controller import MissionSnapshot


def epoch() -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id="epoch:S1:1:a1",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=30,
        observation_batch_id="observation:S1:1",
        critical_event_ids=("event-1",),
        public_target_prior_ids=(),
        public_target_estimate_ids=(),
        resource_manifest_hash="manifest-1",
        active_plan_version=0,
    )


def situation(revision: int = 20) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=revision,
        sim_time_s=100,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )


def mission(plan_revision: int = 0) -> MissionSnapshot:
    return MissionSnapshot(
        scenario_id="S1",
        sim_time_s=100,
        plan_revision=plan_revision,
        uuv_resources={
            "uuv-1": UUVResourceState(
                uuv_id="uuv-1",
                carrier_id="carrier-1",
                mileage_m=100.0,
                energy_fraction=0.9,
                deployment_state="onboard",
            )
        },
        carrier_missions={
            "carrier-1": CarrierMissionModel(
                carrier_id="carrier-1",
                home_battle_group_id="group-1",
                onboard_uuv_ids=("uuv-1",),
            )
        },
    )


def candidate() -> ExecutableMissionPlan:
    return ExecutableMissionPlan(
        revision=1,
        carrier_missions=mission().carrier_missions,
        region_assignments=(
            RegionMissionState(region_id="region-1", target_id="target-00"),
        ),
    )


def test_revision_drift_rebases_eta_without_changing_strategy() -> None:
    report = revalidate_executable_mission_plan(
        epoch=epoch(),
        candidate=candidate(),
        current_situation=situation(),
        current_mission=mission(),
        current_expert_request_version=None,
        recovered_event_ids=frozenset(),
    )
    assert report.valid is True
    assert report.rebased_plan is not None
    assert report.rebased_plan.region_assignments[0].region_id == "region-1"
    assert report.rebased_plan.revision == 1


def test_advanced_plan_is_one_stable_invalidation_reason() -> None:
    report = revalidate_executable_mission_plan(
        epoch=epoch(),
        candidate=candidate(),
        current_situation=situation(),
        current_mission=mission(plan_revision=1),
        current_expert_request_version=None,
        recovered_event_ids=frozenset(),
    )
    assert report.valid is False
    assert {issue.code for issue in report.issues} == {"active_plan_advanced"}
    assert report.rebased_plan is None


def test_owner_change_invalidates_candidate() -> None:
    changed = mission().model_copy(
        update={
            "uuv_resources": {
                "uuv-1": UUVResourceState(
                    uuv_id="uuv-1",
                    carrier_id="carrier-2",
                    mileage_m=100.0,
                    energy_fraction=0.9,
                    deployment_state="onboard",
                )
            }
        }
    )
    report = revalidate_executable_mission_plan(
        epoch=epoch(),
        candidate=candidate(),
        current_situation=situation(),
        current_mission=changed,
        current_expert_request_version=None,
        recovered_event_ids=frozenset(),
    )
    assert report.valid is False
    assert "owner_changed" in {issue.code for issue in report.issues}


def test_recovered_trigger_invalidates_candidate() -> None:
    report = revalidate_executable_mission_plan(
        epoch=epoch(),
        candidate=candidate(),
        current_situation=situation(),
        current_mission=mission(),
        current_expert_request_version=None,
        recovered_event_ids=frozenset({"event-1"}),
    )
    assert report.valid is False
    assert "trigger_recovered" in {issue.code for issue in report.issues}

