from pathlib import Path

from underwater_tracking.agent.nodes.commit import validate_plan
from underwater_tracking.agent.nodes.optimize import PlanningConfig, optimize_candidates
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import StrategyProposal, StrategySet
from underwater_tracking.simulation.engine import SimulationEngine


def test_explicit_platform_core_can_commit_first_uuv_dispatch(tmp_path: Path) -> None:
    """An onboard fleet can be planned as a staged dispatch from the carrier."""
    config = load_app_config("configs/scenario/segmented_single_target.yaml")
    situations = []
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        carrier=situations.append,
    )
    for _ in range(3):
        engine.step()

    situation = situations[-1]
    assert all(uuv.deployment_state.value == "onboard" for uuv in situation.uuvs)
    planning_config = PlanningConfig(bounds=config.environment.map_bounds_xy)
    evidence_ids = tuple(
        sorted(
            {
                observation_id
                for report in situation.group_reports
                for observation_id in report.belief.source_observation_ids
            }
        )
    )
    proposal = StrategyProposal(
        concept="balanced",
        target_priorities={"target_00": 1.0},
        required_quality={"target_00": 0.75},
        reinforcement_policy={"target_00": "active_passive_fusion"},
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=evidence_ids,
        rationale="dispatch toward the predicted tracking sector",
    )
    snapshot = build_planning_snapshot(situation)

    candidate = optimize_candidates(
        snapshot,
        StrategySet(proposals=(proposal,)),
        planning_config,
    )[0].plan

    assert len(candidate.member_ids_by_target["target_00"]) == 3
    assert set(candidate.member_ids_by_target["target_00"]) <= {
        uuv.uuv_id for uuv in situation.uuvs
    }
    assert validate_plan(snapshot, candidate, planning_config) == ()
