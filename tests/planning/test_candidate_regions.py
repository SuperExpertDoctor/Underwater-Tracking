from __future__ import annotations

from underwater_tracking.domain.mission_models import PredictionGrid, PredictionGridCell
from underwater_tracking.planning.candidate_regions import (
    generate_candidate_regions,
)


def _grid() -> PredictionGrid:
    cells = tuple(
        PredictionGridCell(
            target_id="T1",
            revision=3,
            grid_x=x,
            grid_y=y,
            min_x=x * 100.0,
            max_x=(x + 1) * 100.0,
            min_y=y * 100.0,
            max_y=(y + 1) * 100.0,
            cell_size_m=100.0,
            probability=0.7 - 0.05 * (x + y),
            first_entry_s=30 + 10 * (x + y),
            last_exit_s=120 + 10 * (x + y),
            covariance_summary=(100.0, 100.0, 0.0),
            intent_label="transit",
            intent_confidence=0.8,
        )
        for y in range(2)
        for x in range(2)
    )
    return PredictionGrid(
        target_id="T1",
        revision=3,
        origin=(0.0, 0.0),
        cell_size_m=100.0,
        cells=cells,
        centerline_region_ids=tuple(cell.region_id for cell in cells),
    )


def test_candidates_are_contiguous_axis_aligned_squares() -> None:
    regions = generate_candidate_regions(_grid(), (0.0, 300.0, 0.0, 300.0))

    assert regions
    assert any(len(region.cell_ids) == 4 for region in regions)
    for region in regions:
        assert region.max_x > region.min_x
        assert region.max_y > region.min_y
        assert region.max_x - region.min_x == region.max_y - region.min_y
        assert 0.0 <= region.min_x < region.max_x <= 300.0
        assert 0.0 <= region.min_y < region.max_y <= 300.0


def test_candidates_have_stable_ids_and_perimeter_points() -> None:
    regions = generate_candidate_regions(_grid(), (0.0, 300.0, 0.0, 300.0))
    ids = [region.candidate_id for region in regions]

    assert ids == sorted(ids)
    assert all(len(region.perimeter_points) >= 4 for region in regions)
    assert all(region.entry_s <= region.exit_s for region in regions)
