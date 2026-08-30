import itertools

import pytest

from underwater_tracking.agent.nodes.regions import regional_plan_to_mission_candidates
from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.regional_models import (
    GridSpec,
    TaskRegionProposal,
    TaskRegionProposalSet,
)
from underwater_tracking.planning.regions import (
    TASK_REGION_CELL_SIZE_M,
    build_llm_task_region_plan,
    compute_cell_size,
    generate_target_region_plan,
    rectangles_overlap,
)
from underwater_tracking.planning.regional_plan_validator import RegionalPlanError


def prediction(points, *, fallback=False) -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id="pred:T1",
        target_id="T1",
        sim_time_s=0,
        horizon_s=400.0,
        sample_step_s=100.0,
        times_s=tuple(float(index * 100) for index in range(len(points))),
        points_xy=tuple(points),
        corridor_radius_m=tuple(10.0 for _ in points),
        fallback_used=fallback,
        fallback_reason="history_short" if fallback else None,
    )


INTENT = IntentHypothesis(
    label="patrol",
    confidence=0.8,
    evidence_ids=("intent:T1",),
    model_id="fake",
    prompt_version="test",
)


def fixed_spec() -> GridSpec:
    return GridSpec(
        origin_xy=(0.0, 0.0),
        target_grid_cells=9,
        min_cell_size_m=100.0,
        max_cell_size_m=100.0,
        cell_size_rounding_m=50.0,
    )


def test_cell_size_uses_area_then_clamps_and_rounds() -> None:
    spec = GridSpec(
        target_grid_cells=16,
        min_cell_size_m=100.0,
        max_cell_size_m=400.0,
        cell_size_rounding_m=50.0,
    )
    assert compute_cell_size(10_000.0, spec) == 100.0
    assert compute_cell_size(10_000_000.0, spec) == 400.0
    assert compute_cell_size(160_000.0, spec) == 100.0
    assert compute_cell_size(360_000.0, spec) == 150.0


def test_default_grid_uses_finer_adaptive_density_without_member_quotas() -> None:
    spec = GridSpec()

    assert spec.target_grid_cells == 64
    assert spec.require_uuv_per_region is False

    plan = generate_target_region_plan(
        prediction(
            ((150.0, 450.0), (650.0, 450.0), (1_150.0, 450.0), (1_650.0, 450.0)),
        ),
        INTENT,
        (0.0, 2_000.0, 0.0, 900.0),
        GridSpec(
            target_grid_cells=64,
            min_cell_size_m=100.0,
            max_cell_size_m=500.0,
            cell_size_rounding_m=50.0,
            lateral_half_width_cells=2,
            max_uncertainty_margin_cells=1,
        ),
    )

    assert len(plan.cells) >= 32
    assert all(task.required_uuv_count == 0 for task in plan.tasks)


def test_generation_contains_the_mandatory_lateral_band() -> None:
    plan = generate_target_region_plan(
        prediction(((150.0, 150.0), (250.0, 150.0), (350.0, 150.0))),
        INTENT,
        (-500.0, 1_000.0, -500.0, 1_000.0),
        fixed_spec(),
    )
    keys = {(cell.grid_x, cell.grid_y) for cell in plan.cells}
    assert {(1, -1), (1, 0), (1, 1), (1, 2), (1, 3)} <= keys


def test_generation_deduplicates_cells_and_keeps_cells_disjoint() -> None:
    plan = generate_target_region_plan(
        prediction(((50.0, 50.0), (150.0, 50.0), (250.0, 50.0))),
        INTENT,
        (-500.0, 1_000.0, -500.0, 1_000.0),
        fixed_spec(),
    )
    keys = [(cell.grid_x, cell.grid_y) for cell in plan.cells]
    assert len(keys) == len(set(keys))
    assert all(
        not rectangles_overlap(left, right)
        for left, right in itertools.combinations(plan.cells, 2)
    )


def test_generation_records_multiple_visit_windows_for_loop_back() -> None:
    plan = generate_target_region_plan(
        prediction(((50.0, 50.0), (150.0, 50.0), (50.0, 50.0))),
        INTENT,
        (-500.0, 1_000.0, -500.0, 1_000.0),
        fixed_spec(),
    )
    cell = next(item for item in plan.cells if item.grid_x == 0 and item.grid_y == 0)
    assert len(cell.visit_windows) == 2


def test_generation_clips_cells_and_propagates_fallback_evidence() -> None:
    plan = generate_target_region_plan(
        prediction(((50.0, 50.0), (150.0, 150.0)), fallback=True),
        INTENT,
        (0.0, 500.0, 0.0, 500.0),
        fixed_spec(),
    )
    assert all(cell.min_x >= 0 and cell.max_x <= 500 for cell in plan.cells)
    assert plan.fallback_used is True
    assert "prediction:fallback" in plan.evidence_ids


def test_generation_rejects_empty_prediction() -> None:
    with pytest.raises(ValueError, match="prediction points"):
        generate_target_region_plan(prediction(()), INTENT, (0.0, 500.0, 0.0, 500.0), fixed_spec())


def test_llm_task_region_plan_requires_exactly_four_regions() -> None:
    proposal_set = TaskRegionProposalSet.model_construct(
        regions=(
            TaskRegionProposal(
                lower_left_xy=(0.0, 0.0),
                upper_right_xy=(1_000.0, 1_000.0),
                rationale="covers the forecast",
            ),
        )
    )

    with pytest.raises(ValueError, match="exactly four task regions"):
        build_llm_task_region_plan(
            prediction(((500.0, 500.0),)),
            INTENT,
            proposal_set,
            (0.0, 5_000.0, 0.0, 5_000.0),
            fixed_spec(),
        )


def test_llm_task_regions_reject_undersized_short_prediction_tiles() -> None:
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=lower_left,
                upper_right_xy=(lower_left[0] + 1_000.0, lower_left[1] + 1_000.0),
                rationale="short-track uncertainty coverage",
            )
            for lower_left in (
                (0.0, 0.0),
                (1_000.0, 0.0),
                (0.0, 1_000.0),
                (1_000.0, 1_000.0),
            )
        )
    )

    with pytest.raises(ValueError, match="at least 3000 m"):
        build_llm_task_region_plan(
            prediction(((500.0, 500.0),)),
            INTENT,
            proposals,
            (0.0, 5_000.0, 0.0, 5_000.0),
            fixed_spec(),
        )


def test_llm_task_regions_reject_undersized_stationary_corner_tiles() -> None:
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=lower_left,
                upper_right_xy=(lower_left[0] + 1_000.0, lower_left[1] + 1_000.0),
                rationale="corner uncertainty coverage",
            )
            for lower_left in (
                (0.0, 0.0),
                (1_000.0, 0.0),
                (0.0, 1_000.0),
                (1_000.0, 1_000.0),
            )
        )
    )

    with pytest.raises(ValueError, match="at least 3000 m"):
        build_llm_task_region_plan(
            prediction(((1.0, 1.0),)),
            INTENT,
            proposals,
            (0.0, 5_000.0, 0.0, 5_000.0),
            fixed_spec(),
        )


def test_llm_task_regions_are_materialized_in_prediction_time_order() -> None:
    track = prediction(
        (
            (500.0, 500.0),
            (3_500.0, 500.0),
            (6_500.0, 500.0),
            (9_500.0, 500.0),
        )
    )
    lower_lefts = (
        (9_000.0, 0.0),
        (6_000.0, 0.0),
        (3_000.0, 0.0),
        (0.0, 0.0),
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=lower_left,
                upper_right_xy=(lower_left[0] + 4_000.0, 4_000.0),
                rationale="reversed provider order",
            )
            for lower_left in lower_lefts
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 5_000.0),
        fixed_spec(),
    )

    assert [region.lower_left_xy for region in plan.task_regions] == [
        (0.0, 0.0),
        (3_000.0, 0.0),
        (6_000.0, 0.0),
        (9_000.0, 0.0),
    ]


def test_llm_task_regions_remain_square_after_grid_normalization() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                top_left_xy=(0.0, 4_000.0),
                bottom_right_xy=(4_000.0, 0.0),
                rationale="square provider region",
            )
            for _ in range(4)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 5_000.0),
        fixed_spec(),
    )

    assert all(
        region.bottom_right_xy[0] - region.top_left_xy[0]
        == region.top_left_xy[1] - region.bottom_right_xy[1]
        for region in plan.task_regions
    )


def test_llm_task_region_normalization_handles_stationary_centerline() -> None:
    track = prediction(((500.0, 500.0),) * 4)
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(0.0, 0.0),
                upper_right_xy=(4_000.0, 4_000.0),
                rationale="keep a four-slot stationary search surface",
            )
            for _ in range(4)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 5_000.0, 0.0, 5_000.0),
        fixed_spec(),
    )

    assert len(plan.task_regions) == 4
    assert all(region.active_window.end_s > region.active_window.start_s for region in plan.task_regions)


def test_llm_task_region_normalization_repairs_stationary_missing_centerline() -> None:
    track = prediction(((7_500.0, 7_500.0),) * 4)
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(0.0, 0.0),
                upper_right_xy=(4_000.0, 4_000.0),
                rationale="provider omitted the stationary centerline",
            )
            for _ in range(4)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 10_000.0, 0.0, 10_000.0),
        fixed_spec(),
    )

    assert len(plan.task_regions) == 4
    assert [region.active_window.start_s for region in plan.task_regions] == [0, 100, 200, 300]
    assert all(
        region.lower_left_xy[0] <= 7_500.0 <= region.upper_right_xy[0]
        and region.lower_left_xy[1] <= 7_500.0 <= region.upper_right_xy[1]
        for region in plan.task_regions
    )


def test_llm_task_region_normalization_handles_compact_nonstationary_centerline() -> None:
    track = prediction(
        (
            (500.0, 500.0),
            (1_500.0, 500.0),
            (2_500.0, 500.0),
            (3_500.0, 500.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(0.0, 0.0),
                upper_right_xy=(4_000.0, 4_000.0),
                rationale="keep compact forecast coverage in chronological slots",
            )
            for _ in range(4)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 5_000.0, 0.0, 5_000.0),
        fixed_spec(),
    )

    assert len(plan.task_regions) == 4
    assert [
        region.active_window.start_s for region in plan.task_regions
    ] == [0, 100, 200, 300]
    assert all(
        region.active_window.end_s > region.active_window.start_s
        for region in plan.task_regions
    )


def test_llm_task_regions_share_global_1km_grid_and_scale_uuv_demand() -> None:
    """LLM bounds become global-grid regions with area-based UUV demand."""
    track = PredictedTrackRef(
        prediction_id="pred:T1:evasive",
        target_id="T1",
        sim_time_s=0,
        horizon_s=400.0,
        sample_step_s=100.0,
        times_s=(0.0, 100.0, 200.0, 300.0),
        points_xy=(
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        ),
        corridor_radius_m=(100.0, 150.0, 400.0, 600.0),
    )
    plan = build_llm_task_region_plan(
        track,
        INTENT.model_copy(update={"label": "evade", "confidence": 0.35}),
        TaskRegionProposalSet(
            regions=(
                TaskRegionProposal(
                    lower_left_xy=(0.0, 0.0),
                    upper_right_xy=(4_000.0, 4_000.0),
                    rationale="early evasive segment",
                ),
                TaskRegionProposal(
                    lower_left_xy=(3_000.0, 0.0),
                    upper_right_xy=(7_000.0, 4_000.0),
                    rationale="early handoff segment",
                ),
                TaskRegionProposal(
                    lower_left_xy=(6_000.0, 0.0),
                    upper_right_xy=(10_000.0, 4_000.0),
                    rationale="later evasive segment",
                ),
                TaskRegionProposal(
                    lower_left_xy=(9_000.0, 0.0),
                    upper_right_xy=(15_000.0, 6_000.0),
                    rationale="later uncertain segment",
                ),
            )
        ),
        (0.0, 15_000.0, 0.0, 6_000.0),
        fixed_spec(),
        uuv_scan_range_m=2_000.0,
    )

    assert len(plan.task_regions) == 4
    assert all(cell.cell_size_m == TASK_REGION_CELL_SIZE_M for cell in plan.cells)
    assert all(
        cell.min_x % TASK_REGION_CELL_SIZE_M == 0
        and cell.min_y % TASK_REGION_CELL_SIZE_M == 0
        for cell in plan.cells
    )
    assert plan.task_regions[0].required_uuv_count == 2
    assert plan.task_regions[3].required_uuv_count == 3


def test_llm_task_regions_form_large_overlapping_windows_along_prediction() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(start_x, 0.0),
                upper_right_xy=(start_x + 4_000.0, 4_000.0),
                rationale="large tracking window with a one-cell handoff overlap",
            )
            for start_x in (0.0, 3_000.0, 6_000.0, 9_000.0)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 5_000.0),
        fixed_spec(),
    )

    assert len(plan.task_regions) == 4
    assert all(
        region.upper_right_xy[0] - region.lower_left_xy[0] >= 4_000.0
        and region.upper_right_xy[1] - region.lower_left_xy[1] >= 4_000.0
        for region in plan.task_regions
    )
    assert all(
        set(left.cell_ids) & set(right.cell_ids)
        for left, right in zip(plan.task_regions, plan.task_regions[1:])
    )
    assert not (set(plan.task_regions[0].cell_ids) & set(plan.task_regions[2].cell_ids))
    assert all(
        region.lower_left_xy[0] <= point[0] <= region.upper_right_xy[0]
        and region.lower_left_xy[1] <= point[1] <= region.upper_right_xy[1]
        for region, point in zip(plan.task_regions, track.points_xy, strict=True)
    )
    candidates = regional_plan_to_mission_candidates(plan)
    assert [candidate.predecessor_candidate_ids for candidate in candidates] == [
        (),
        ("T1:task:01",),
        ("T1:task:02",),
        ("T1:task:03",),
    ]
    assert [candidate.successor_candidate_ids for candidate in candidates] == [
        ("T1:task:02",),
        ("T1:task:03",),
        ("T1:task:04",),
        (),
    ]


def test_llm_task_regions_fill_missing_prediction_centerline() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(start_x, 5_000.0),
                upper_right_xy=(start_x + 3_000.0, 8_000.0),
                rationale="provider omitted the forecast centerline",
            )
            for start_x in (0.0, 3_000.0, 6_000.0, 9_000.0)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 8_000.0),
        fixed_spec(),
    )

    assert all(
        any(
            region.lower_left_xy[0] <= point[0] <= region.upper_right_xy[0]
            and region.lower_left_xy[1] <= point[1] <= region.upper_right_xy[1]
            for point in track.points_xy
        )
        for region in plan.task_regions
    )


def test_llm_task_regions_clip_excessive_and_non_adjacent_overlap() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(0.0, 0.0),
                upper_right_xy=(13_000.0, 6_000.0),
                rationale=f"overlapping provider region {index}",
            )
            for index in range(4)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 6_000.0),
        fixed_spec(),
    )

    regions = plan.task_regions
    assert all(
        set(left.cell_ids) & set(right.cell_ids)
        for left, right in itertools.pairwise(regions)
    )
    assert all(
        not (set(regions[left].cell_ids) & set(regions[right].cell_ids))
        for left in range(len(regions))
        for right in range(left + 2, len(regions))
    )


def test_llm_task_regions_clip_map_overflow_before_centerline_repair() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(start_x, 20_000.0),
                upper_right_xy=(start_x + 4_000.0, 24_000.0),
                rationale="provider rectangle exceeded the shared map",
            )
            for start_x in (-4_000.0, 2_000.0, 8_000.0, 14_000.0)
        )
    )

    plan = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 8_000.0),
        fixed_spec(),
    )

    assert all(
        0.0 <= region.lower_left_xy[0] < region.upper_right_xy[0] <= 13_000.0
        and 0.0 <= region.lower_left_xy[1] < region.upper_right_xy[1] <= 8_000.0
        for region in plan.task_regions
    )
    assert all(
        any(
            region.lower_left_xy[0] <= point[0] <= region.upper_right_xy[0]
            and region.lower_left_xy[1] <= point[1] <= region.upper_right_xy[1]
            for point in track.points_xy
        )
        for region in plan.task_regions
    )


def test_task_region_uuv_demand_uses_uuv_scan_range() -> None:
    track = prediction(
        (
            (500.0, 2_000.0),
            (3_500.0, 2_000.0),
            (6_500.0, 2_000.0),
            (9_500.0, 2_000.0),
        )
    )
    proposals = TaskRegionProposalSet(
        regions=tuple(
            TaskRegionProposal(
                lower_left_xy=(start_x, 0.0),
                upper_right_xy=(start_x + 4_000.0, 4_000.0),
                rationale="coverage-capacity comparison",
            )
            for start_x in (0.0, 3_000.0, 6_000.0, 9_000.0)
        )
    )

    short_range = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 5_000.0),
        fixed_spec(),
        uuv_scan_range_m=1_000.0,
    )
    long_range = build_llm_task_region_plan(
        track,
        INTENT,
        proposals,
        (0.0, 13_000.0, 0.0, 5_000.0),
        fixed_spec(),
        uuv_scan_range_m=4_000.0,
    )

    assert short_range.task_regions[0].required_uuv_count == 4
    assert long_range.task_regions[0].required_uuv_count == 2


@pytest.mark.parametrize("forbidden_field", ["coordinates", "time_window", "successor_candidate_id"])
def test_live_llm_region_policy_rejects_geometry_windows_and_topology(forbidden_field: str) -> None:
    candidates = regional_plan_to_mission_candidates(
        build_llm_task_region_plan(
            prediction(((500.0, 500.0),) * 4),
            INTENT,
            TaskRegionProposalSet(
                regions=tuple(
                    TaskRegionProposal(
                        lower_left_xy=(0.0, 0.0),
                        upper_right_xy=(4_000.0, 4_000.0),
                        rationale="legacy replay geometry",
                    )
                    for _ in range(4)
                )
            ),
            (0.0, 5_000.0, 0.0, 5_000.0),
            fixed_spec(),
        )
    )
    response = {
        "policies": [
            {
                "candidate_id": candidate.candidate_id,
                "coverage_mode": "required",
                "tracking_mode": "passive_track",
                "priority": 1.0,
                "required_quality": 0.8,
                "active_scan_uuv_count": 0,
                "passive_track_uuv_count": 1,
                "assigned_uuv_ids": [],
                "rationale": "semantic choice",
                "evidence_ids": ["belief:T1:1"],
                forbidden_field: [[0.0, 0.0]],
            }
            for candidate in candidates
        ]
    }

    with pytest.raises(RegionalPlanError, match="strict UUV regional decision schema"):
        build_llm_task_region_plan(candidates, response, ())


def test_live_llm_region_policy_rejects_unknown_candidate_id() -> None:
    candidate = regional_plan_to_mission_candidates(
        build_llm_task_region_plan(
            prediction(((500.0, 500.0),) * 4),
            INTENT,
            TaskRegionProposalSet(
                regions=tuple(
                    TaskRegionProposal(
                        lower_left_xy=(0.0, 0.0),
                        upper_right_xy=(4_000.0, 4_000.0),
                        rationale="legacy replay geometry",
                    )
                    for _ in range(4)
                )
            ),
            (0.0, 5_000.0, 0.0, 5_000.0),
            fixed_spec(),
        )
    )[0]
    response = {
        "policies": [
            {
                "candidate_id": "T1:task:unknown",
                "coverage_mode": "required",
                "tracking_mode": "passive_track",
                "priority": 1.0,
                "required_quality": 0.8,
                "active_scan_uuv_count": 0,
                "passive_track_uuv_count": 1,
                "assigned_uuv_ids": [],
                "rationale": "semantic choice",
                "evidence_ids": ["belief:T1:1"],
            }
        ]
    }

    with pytest.raises(RegionalPlanError, match="unknown regional policy"):
        build_llm_task_region_plan((candidate,), response, ())
