from types import SimpleNamespace

from underwater_tracking.agent.graphs.central import (
    RecordDecisionNode,
    _route_after_prediction_intent,
)
from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.models import EventLevel


class _Events:
    def __init__(self) -> None:
        self.appended: list[object] = []

    def get(self, event_id: str) -> object | None:
        del event_id
        return None

    def append(self, **kwargs: object) -> None:
        self.appended.append(kwargs)


class _Ledger:
    def __init__(self) -> None:
        self.records: list[object] = []

    def record(self, decision: object) -> None:
        self.records.append(decision)


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        sim_time_s=90,
        snapshot_revision=3,
        digest="snapshot-digest",
        situation=SimpleNamespace(group_reports=()),
        applied_directives=(),
    )


def _candidate() -> TrackingPlan:
    return TrackingPlan(
        plan_id="plan:S1:2",
        scenario_id="S1",
        revision=2,
        base_snapshot_revision=3,
    )


def test_invalidated_epoch_is_recorded_without_a_final_plan() -> None:
    events = _Events()
    ledger = _Ledger()
    node = RecordDecisionNode(
        events, ledger, lambda ref: _snapshot(), {"candidate": _candidate()}
    )

    node(
        {
            "route": EventLevel.TACTICAL,
            "selected_plan_ref": "candidate",
            "commit_status": "invalidated",
            "snapshot_ref": "live",
            "coalesced_events": (),
        }
    )

    assert len(ledger.records) == 1
    assert ledger.records[0].final_plan_id is None


def test_unconfirmed_prediction_intent_does_not_reenter_strategic_provider_chain() -> None:
    state = {
        "prediction_intent_confirmed": False,
        "uuv_only": True,
        "predictions": {"target_00": object()},
        "regional_plans": {"target_00": object()},
        "coalesced_events": (SimpleNamespace(event_type="target_estimate_updated"),),
    }

    assert _route_after_prediction_intent(state) == "tactical"


def test_confirmed_prediction_intent_still_reenters_strategic_provider_chain() -> None:
    state = {
        "prediction_intent_confirmed": True,
        "uuv_only": True,
        "predictions": {"target_00": object()},
        "regional_plans": {"target_00": object()},
        "coalesced_events": (SimpleNamespace(event_type="target_estimate_updated"),),
    }

    assert _route_after_prediction_intent(state) == "strategic"
