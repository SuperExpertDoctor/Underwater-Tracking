"""Every active planning epoch error has one durable terminal outcome."""

from __future__ import annotations

import pytest

from underwater_tracking.agent.graphs.central import (
    FinalizeEpochNode,
    PlanningEpochInvariantError,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch


def _epoch() -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id="epoch:S1:1:contract",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=30,
        observation_batch_id="observation:S1:1",
        resource_manifest_hash="manifest",
        active_plan_version=0,
    )


@pytest.mark.parametrize(
    "node_name,error_message",
    (
        ("regional_strategy", "regional_strategy semantic rejection"),
        ("regional_strategy_adapter", "regional_strategy_adapter content rejection"),
        ("verify_strategy", "verify_strategy semantic verification failed"),
        ("resource_optimizer", "resource_optimizer content failure"),
        ("verify_plan", "verify_plan schema failure"),
    ),
)
def test_error_exit_produces_one_terminal_epoch_result(
    node_name: str, error_message: str
) -> None:
    event = RuntimeEvent(
        event_id=f"event:{node_name}",
        scenario_id="S1",
        sim_time_s=30,
        event_type="strategic_review",
        entity_id="S1",
        level=EventLevel.STRATEGIC,
    )
    result = FinalizeEpochNode()(
        {
            "planning_epoch": _epoch(),
            "coalesced_events": (event,),
            "node_error": error_message,
        }
    )
    terminal = result["epoch_commit_result"]
    assert isinstance(terminal, EpochCommitResult)
    assert terminal.epoch_id == _epoch().epoch_id
    assert terminal.status in {"rejected", "failed"}
    assert terminal.consumed_event_ids == (event.event_id,)


def test_existing_terminal_result_is_preserved_and_second_outcome_is_rejected() -> None:
    terminal = EpochCommitResult(
        epoch_id=_epoch().epoch_id,
        status="failed",
        failure_category="provider",
        failure_message="provider unavailable",
    )
    preserved = FinalizeEpochNode()(
        {"planning_epoch": _epoch(), "epoch_commit_result": terminal}
    )
    assert preserved["epoch_commit_result"] == terminal
    assert preserved["epoch_finalization_route"] == "record"

    with pytest.raises(PlanningEpochInvariantError, match="second"):
        FinalizeEpochNode()(
            {
                "planning_epoch": _epoch(),
                "epoch_commit_result": terminal,
                "node_error": "another failure",
            }
        )


def test_active_epoch_without_terminal_result_is_an_invariant_failure() -> None:
    with pytest.raises(PlanningEpochInvariantError, match="terminal"):
        FinalizeEpochNode()({"planning_epoch": _epoch()})
