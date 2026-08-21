from underwater_tracking.agent.event_policy import (
    EventDisposition,
    EventEpisodeGate,
    evaluate_plan_impact,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent


def _event(event_type: str, *, entity_id: str = "T1", payload: dict | None = None) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"{event_type}:{entity_id}",
        scenario_id="S1",
        sim_time_s=100,
        event_type=event_type,
        entity_id=entity_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload or {},
    )


def test_normal_carrier_lifecycle_event_is_audit_only_without_plan_impact() -> None:
    assessment = evaluate_plan_impact(
        _event("carrier_dispatch_completed", entity_id="carrier-1"),
        active_region_ids=("R1",),
        active_uuv_ids=("U1",),
    )

    assert assessment.disposition is EventDisposition.AUDIT_ONLY
    assert assessment.plan_impact is False


def test_periodic_situation_summary_is_memory_only() -> None:
    assessment = evaluate_plan_impact(
        _event("periodic_situation_summary", entity_id="S1"),
        active_region_ids=("R1",),
        active_uuv_ids=("U1",),
    )

    assert assessment.disposition is EventDisposition.AUDIT_ONLY
    assert assessment.plan_impact is False


def test_handoff_blocked_and_rendezvous_infeasible_are_eligible_for_impact() -> None:
    for event_type in ("handoff_blocked", "carrier_rendezvous_infeasible"):
        assessment = evaluate_plan_impact(
            _event(event_type, payload={"plan_impact": True}),
            active_region_ids=("R1",),
            active_uuv_ids=("U1",),
        )

        assert assessment.disposition is EventDisposition.KEY
        assert assessment.plan_impact is True


def test_quality_event_becomes_key_only_when_active_quality_is_below_requirement() -> None:
    assessment = evaluate_plan_impact(
        _event("region_coverage_degraded", payload={"region_id": "R1"}),
        active_region_ids=("R1",),
        quality_by_target={"T1": 0.42},
        required_quality_by_target={"T1": 0.70},
    )

    assert assessment.disposition is EventDisposition.KEY
    assert assessment.plan_impact is True
    assert assessment.affected_region_ids == ("R1",)


def test_intent_change_without_current_plan_impact_is_not_a_key_event() -> None:
    assessment = evaluate_plan_impact(
        _event("target_intent_changed", payload={"target_id": "T1"}),
        active_region_ids=("R1",),
        active_target_ids=(),
        target_corridor_changed=False,
    )

    assert assessment.disposition is EventDisposition.CANDIDATE
    assert assessment.plan_impact is False


def test_episode_gate_emits_once_until_recovery_then_allows_new_episode() -> None:
    gate = EventEpisodeGate()

    assert gate.observe("T1:quality", 0, active=True, hold_s=20, confirmations=2) is False
    assert gate.observe("T1:quality", 10, active=True, hold_s=20, confirmations=2) is False
    assert gate.observe("T1:quality", 20, active=True, hold_s=20, confirmations=2) is True
    assert gate.observe("T1:quality", 30, active=True, hold_s=20, confirmations=2) is False
    assert gate.observe("T1:quality", 40, active=False, hold_s=20, confirmations=2) is False
    assert gate.observe("T1:quality", 50, active=True, hold_s=20, confirmations=2) is False
    assert gate.observe("T1:quality", 70, active=True, hold_s=20, confirmations=2) is True
