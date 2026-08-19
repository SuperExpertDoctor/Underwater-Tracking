from __future__ import annotations

from collections.abc import Callable

from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.regional_models import (
    GridSpec,
    RegionalMissionCandidate,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.regions import generate_target_region_plan


class RegionGenerationNode:
    """Build deterministic target region plans from stored prediction references."""

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        map_bounds_provider: Callable[[PlanningSnapshot], tuple[float, float, float, float]],
        grid_spec: GridSpec,
        required_quality: float = 0.0,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._map_bounds_provider = map_bounds_provider
        self._grid_spec = grid_spec
        self._required_quality = required_quality

    def __call__(self, state: CarrierState) -> CarrierState:
        snapshot_ref = state.get("snapshot_ref")
        if not snapshot_ref:
            raise ValueError("region generation requires snapshot_ref")
        snapshot = self._snapshot_provider(snapshot_ref)
        intents = state.get("intent_hypotheses", {})
        predictions = state.get("predictions", {})
        map_bounds = self._map_bounds_provider(snapshot)
        plans: dict[str, TargetRegionPlan] = {}
        for target_id, prediction in sorted(predictions.items()):
            intent = intents.get(target_id)
            if intent is None:
                raise ValueError(f"region generation requires intent for target {target_id!r}")
            plans[target_id] = generate_target_region_plan(
                prediction,
                intent,
                map_bounds,
                self._grid_spec,
                required_quality=self._required_quality,
            )
        return {
            "regional_plans": plans,
            "regional_candidates": {
                target_id: regional_plan_to_mission_candidates(plan)
                for target_id, plan in sorted(plans.items())
            },
        }


def regional_plan_to_mission_candidates(
    plan: TargetRegionPlan,
) -> tuple[RegionalMissionCandidate, ...]:
    """Expose planner-owned region geometry as strict UUV candidates."""
    return tuple(
        RegionalMissionCandidate(
            candidate_id=cell.region_id,
            cell_ids=(cell.region_id,),
            time_window=TimeWindow(
                start_s=cell.first_entry_s,
                end_s=max(cell.first_entry_s + 1, cell.last_exit_s),
            ),
            perimeter_points=tuple(
                sorted(
                    (
                        (cell.min_x, cell.min_y),
                        (cell.min_x, cell.max_y),
                        (cell.max_x, cell.min_y),
                        (cell.max_x, cell.max_y),
                    )
                )
            ),
            predecessor_candidate_ids=tuple(cell.predecessor_region_ids),
            successor_candidate_ids=tuple(cell.successor_region_ids),
        )
        for cell in plan.cells
    )
