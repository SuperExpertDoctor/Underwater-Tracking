from __future__ import annotations

import pytest

import underwater_tracking.agent.graphs.central as central_graph
from underwater_tracking.agent.graphs.central import CommitPlanNode
from underwater_tracking.cli import _AgentLoop
from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch


def _epoch() -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id="epoch:S1:1:a1",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=1,
        observation_batch_id="observation:S1:1",
        resource_manifest_hash="manifest",
        active_plan_version=0,
    )


def _audit_projection() -> TrackingPlan:
    return TrackingPlan(
        plan_id="plan:S1:1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=1,
    )


class _InvalidatingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[PlanningEpoch, TrackingPlan, ExecutableMissionPlan]] = []

    def commit(
        self,
        *,
        epoch: PlanningEpoch,
        audit_projection: TrackingPlan,
        executable_plan: ExecutableMissionPlan,
    ) -> EpochCommitResult:
        self.calls.append((epoch, audit_projection, executable_plan))
        return EpochCommitResult(
            epoch_id=epoch.epoch_id,
            status="invalidated",
            validation_report_id="validation:S1:1",
            invalidated_reason="active_plan_advanced",
        )


def test_uuv_only_commit_delegates_revision_drift_to_epoch_port() -> None:
    port = _InvalidatingPort()
    node = CommitPlanNode(
        object(),  # type: ignore[arg-type]
        {"candidate": _audit_projection()},
        uuv_only=True,
        epoch_commit_port=port,
    )

    result = node(
        {
            "selected_plan_ref": "candidate",
            "planning_epoch": _epoch(),
            "executable_mission_plan": ExecutableMissionPlan(revision=1),
        }
    )

    assert len(port.calls) == 1
    assert result["commit_status"] == "invalidated"
    assert result["selected_plan"] is None
    assert result["epoch_commit_result"] is not None


def test_active_epoch_error_is_finalized_into_a_terminal_result() -> None:
    finalize = getattr(central_graph, "FinalizeEpochNode", None)
    if finalize is None:
        pytest.fail("FinalizeEpochNode is not implemented")

    result = finalize()(
        {
            "planning_epoch": _epoch(),
            "node_error": "regional semantic rejection",
        }
    )

    terminal = result["epoch_commit_result"]
    assert terminal.epoch_id == _epoch().epoch_id
    assert terminal.status in {"rejected", "failed"}


def test_active_epoch_without_terminal_result_is_an_invariant_failure() -> None:
    finalize = getattr(central_graph, "FinalizeEpochNode", None)
    if finalize is None:
        pytest.fail("FinalizeEpochNode is not implemented")

    with pytest.raises(Exception, match="terminal|invariant"):
        finalize()({"planning_epoch": _epoch()})


def test_finish_epoch_counts_a_missing_terminal_result_as_an_invariant_failure() -> None:
    finished: list[EpochCommitResult] = []
    loop = _AgentLoop.__new__(_AgentLoop)
    loop.planning_epoch_invariant_failures = 0
    loop._epoch_coordinator = type("Coordinator", (), {"finish": finished.append})()
    loop._finish_epoch(_epoch(), {}, None)

    assert loop.planning_epoch_invariant_failures == 1
    assert finished[0].status == "failed"
    assert finished[0].failure_category == "internal"
