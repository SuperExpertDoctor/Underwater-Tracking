from __future__ import annotations

from itertools import product
from typing import Annotated

from pydantic import Field

from underwater_tracking.domain.models import StrictModel
from underwater_tracking.domain.mission_models import PredictionGrid
from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    TimeWindow,
)

Finite = Annotated[float, Field(allow_inf_nan=False)]
Bounds = tuple[float, float, float, float]


class CandidateRegion(StrictModel):
    candidate_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    cell_ids: tuple[str, ...] = ()
    min_x: Finite
    max_x: Finite
    min_y: Finite
    max_y: Finite
    entry_s: int = Field(ge=0)
    exit_s: int = Field(ge=0)
    probability: float = Field(ge=0, le=1)
    perimeter_points: tuple[tuple[Finite, Finite], ...] = ()
    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()


def candidate_region_to_mission_candidate(
    region: CandidateRegion,
) -> RegionalMissionCandidate:
    """Convert deterministic geometry into the strict semantic candidate model."""
    return RegionalMissionCandidate(
        candidate_id=region.candidate_id,
        cell_ids=region.cell_ids,
        time_window=TimeWindow(start_s=region.entry_s, end_s=region.exit_s),
        perimeter_points=region.perimeter_points,
        predecessor_candidate_ids=region.predecessor_ids,
        successor_candidate_ids=region.successor_ids,
    )


def generate_candidate_regions(
    grid: PredictionGrid, map_bounds_xy: Bounds
) -> tuple[CandidateRegion, ...]:
    """Enumerate contiguous square windows from a validated prediction grid."""
    min_map_x, max_map_x, min_map_y, max_map_y = map_bounds_xy
    cells = {(cell.grid_x, cell.grid_y): cell for cell in grid.cells}
    if not cells:
        return ()
    max_side = min(
        max(x for x, _ in cells) - min(x for x, _ in cells) + 1,
        max(y for _, y in cells) - min(y for _, y in cells) + 1,
    )
    candidates: list[CandidateRegion] = []
    for side, start_x, start_y in product(
        range(1, max_side + 1),
        range(min(x for x, _ in cells), max(x for x, _ in cells) + 1),
        range(min(y for _, y in cells), max(y for _, y in cells) + 1),
    ):
        keys = tuple((start_x + dx, start_y + dy) for dy in range(side) for dx in range(side))
        selected = [cells.get(key) for key in keys]
        if any(cell is None for cell in selected):
            continue
        typed = tuple(cell for cell in selected if cell is not None)
        min_x = min(cell.min_x for cell in typed)
        max_x = max(cell.max_x for cell in typed)
        min_y = min(cell.min_y for cell in typed)
        max_y = max(cell.max_y for cell in typed)
        if min_x < min_map_x or max_x > max_map_x or min_y < min_map_y or max_y > max_map_y:
            continue
        candidate_id = f"{grid.target_id}:r{grid.revision}:square:{start_x}:{start_y}:{side}"
        candidates.append(
            CandidateRegion(
                candidate_id=candidate_id,
                target_id=grid.target_id,
                revision=grid.revision,
                cell_ids=tuple(cell.region_id for cell in typed),
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                entry_s=min(cell.first_entry_s for cell in typed),
                exit_s=max(cell.last_exit_s for cell in typed),
                probability=sum(cell.probability for cell in typed) / len(typed),
                perimeter_points=(
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                ),
            )
        )
    candidates.sort(key=lambda region: region.candidate_id)
    return tuple(candidates)
