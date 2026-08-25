import pytest
from pydantic import ValidationError

from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    SonarPolicy,
    TaskRegionProposal,
    TaskRegionProposalSet,
    TargetRegionPlan,
    TimeWindow,
)


def cell() -> RegionCell:
    return RegionCell(
        region_id="T1:cell:0:0",
        target_id="T1",
        grid_x=0,
        grid_y=0,
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=100.0,
        center_xy=(50.0, 50.0),
        cell_size_m=100.0,
        first_entry_s=100,
        last_exit_s=180,
        visit_windows=(TimeWindow(start_s=100, end_s=180),),
        evidence_ids=("belief:T1", "intent:T1"),
    )


def task() -> RegionTask:
    return RegionTask(
        region_id="T1:cell:0:0",
        target_id="T1",
        active_window=TimeWindow(start_s=100, end_s=180),
        required_quality=0.8,
        required_uuv_count=1,
        uuv_roles=("passive_tracker",),
        sonar_policy=SonarPolicy(passive_required=True, active_allowed=False),
        communication=CommunicationRequirement(),
        evidence_ids=("belief:T1", "intent:T1"),
    )


def test_grid_spec_rejects_inverted_cell_limits() -> None:
    with pytest.raises(ValidationError, match="max_cell_size_m"):
        GridSpec(min_cell_size_m=200.0, max_cell_size_m=100.0)


def test_current_regional_contract_rejects_legacy_usv_fields() -> None:
    with pytest.raises(ValidationError):
        GridSpec(require_usv_per_region=True)
    with pytest.raises(ValidationError):
        CommunicationRequirement(usv_relay_required=True)
    with pytest.raises(ValidationError):
        RegionTask(
            region_id="T1:cell:0:0",
            target_id="T1",
            active_window=TimeWindow(start_s=100, end_s=180),
            tracking_mode="uuv_primary_usv_relay",
            assigned_usv_ids=("USV-1",),
        )


def test_region_cell_has_stable_id_and_axis_aligned_geometry() -> None:
    value = cell()
    assert value.region_id == "T1:cell:0:0"
    assert value.center_xy == (50.0, 50.0)


def test_region_task_requires_passive_sonar() -> None:
    with pytest.raises(ValidationError, match="passive"):
        RegionTask(**{**task().model_dump(), "sonar_policy": {"passive_required": False}})


def test_task_region_proposal_set_requires_exactly_four_regions() -> None:
    proposal = TaskRegionProposal(
        lower_left_xy=(0.0, 0.0),
        upper_right_xy=(1_000.0, 1_000.0),
        rationale="forecast segment",
    )

    with pytest.raises(ValidationError):
        TaskRegionProposalSet(regions=(proposal,) * 3)
    with pytest.raises(ValidationError):
        TaskRegionProposalSet(regions=(proposal,) * 5)

    accepted = TaskRegionProposalSet(regions=(proposal,) * 4)
    assert len(accepted.regions) == 4


def test_target_region_plan_round_trips_without_losing_evidence() -> None:
    plan = TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=(cell(),),
        tasks=(task(),),
        prediction_id="pred:T1",
        intent_label="patrol",
        intent_confidence=0.8,
        evidence_ids=("belief:T1", "intent:T1"),
    )
    restored = TargetRegionPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert restored.region_ids == ("T1:cell:0:0",)
