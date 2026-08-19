from __future__ import annotations

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.models import TargetBelief
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.planning.prediction_grid import build_prediction_grid


def _prediction() -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id="pred:T1:4",
        target_id="T1",
        sim_time_s=30,
        horizon_s=300.0,
        sample_step_s=60.0,
        times_s=(30.0, 90.0, 150.0, 210.0),
        points_xy=((100.0, 100.0), (700.0, 100.0), (1300.0, 100.0), (1900.0, 100.0)),
        corridor_radius_m=(100.0, 100.0, 100.0, 100.0),
        source_belief_history_ids=("belief:T1:4",),
    )


def _belief(covariance_scale: float) -> TargetBelief:
    return TargetBelief(
        target_id="T1",
        sim_time_s=30,
        mean=(100.0, 100.0, 8.0, 0.0),
        covariance=tuple(
            tuple(covariance_scale if row == column else 0.0 for column in range(4))
            for row in range(4)
        ),
        model_probabilities={"CV": 0.8, "CT": 0.2},
    )


def _intent() -> IntentHypothesis:
    return IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("intent:T1:4",),
        model_id="deterministic-test",
        prompt_version="test-v1",
    )


def test_prediction_grid_is_deterministic_and_ids_include_revision() -> None:
    spec = GridSpec(
        target_grid_cells=16,
        min_cell_size_m=100.0,
        max_cell_size_m=1000.0,
        cell_size_rounding_m=50.0,
    )
    first = build_prediction_grid(_belief(100.0), _prediction(), _intent(), 4, spec)
    second = build_prediction_grid(_belief(100.0), _prediction(), _intent(), 4, spec)

    assert first == second
    assert all(cell.region_id.startswith("T1:r4:cell:") for cell in first.cells)
    assert tuple(cell.region_id for cell in first.cells) == tuple(
        sorted(cell.region_id for cell in first.cells)
    )


def test_high_covariance_expands_or_keeps_prediction_cells_coarse() -> None:
    spec = GridSpec(
        target_grid_cells=16,
        min_cell_size_m=100.0,
        max_cell_size_m=2000.0,
        cell_size_rounding_m=50.0,
    )
    tight = build_prediction_grid(_belief(100.0), _prediction(), _intent(), 1, spec)
    diffuse = build_prediction_grid(_belief(10_000.0), _prediction(), _intent(), 1, spec)

    assert diffuse.cell_size_m >= tight.cell_size_m


def test_grid_cells_stay_inside_requested_map_bounds() -> None:
    spec = GridSpec(
        target_grid_cells=9,
        min_cell_size_m=100.0,
        max_cell_size_m=500.0,
        cell_size_rounding_m=50.0,
    )
    grid = build_prediction_grid(_belief(100.0), _prediction(), _intent(), 2, spec)

    assert all(
        0.0 <= cell.min_x < cell.max_x <= 2500.0
        and 0.0 <= cell.min_y < cell.max_y <= 1000.0
        for cell in grid.cells
    )
