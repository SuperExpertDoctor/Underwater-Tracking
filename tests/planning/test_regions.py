import itertools

import pytest

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


def test_llm_task_regions_share_global_1km_grid_and_scale_uuv_demand() -> None:
    """LLM bounds become global-grid regions with area-based UUV demand."""
    track = PredictedTrackRef(
        prediction_id="pred:T1:evasive",
        target_id="T1",
        sim_time_s=0,
        horizon_s=600.0,
        sample_step_s=100.0,
        times_s=(0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
        points_xy=(
            (500.0, 500.0),
            (1_000.0, 500.0),
            (1_500.0, 750.0),
            (2_000.0, 1_250.0),
            (2_500.0, 1_250.0),
            (3_000.0, 1_750.0),
        ),
        corridor_radius_m=(100.0, 100.0, 150.0, 250.0, 400.0, 600.0),
    )
    plan = build_llm_task_region_plan(
        track,
        INTENT.model_copy(update={"label": "evade", "confidence": 0.35}),
        TaskRegionProposalSet(
            regions=(
                TaskRegionProposal(
                    lower_left_xy=(350.0, 350.0),
                    upper_right_xy=(1_650.0, 1_650.0),
                    rationale="early evasive segment",
                ),
                TaskRegionProposal(
                    lower_left_xy=(2_000.0, 350.0),
                    upper_right_xy=(3_350.0, 2_650.0),
                    rationale="later uncertain segment",
                ),
            )
        ),
        (0.0, 4_500.0, 0.0, 4_500.0),
        fixed_spec(),
    )

    assert len(plan.task_regions) == 2
    assert all(cell.cell_size_m == TASK_REGION_CELL_SIZE_M for cell in plan.cells)
    assert all(
        cell.min_x % TASK_REGION_CELL_SIZE_M == 0
        and cell.min_y % TASK_REGION_CELL_SIZE_M == 0
        for cell in plan.cells
    )
    assert plan.task_regions[1].required_uuv_count > plan.task_regions[0].required_uuv_count
