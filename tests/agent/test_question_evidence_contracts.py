"""Offline contracts for the bounded expert-question evidence payload."""

from underwater_tracking.agent.nodes.questions import (
    QuestionEntities,
    QuestionEvidence,
    build_question_payload,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.models import SituationSnapshot


def test_question_payload_keeps_ordinary_event_evidence() -> None:
    """Ordinary event evidence remains citable in the bounded payload."""
    snapshot = PlanningSnapshot(
        situation=SituationSnapshot(
            scenario_id="scenario-question-contract",
            snapshot_revision=1,
            sim_time_s=60,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        ),
        active_plan=None,
        applied_directives=(),
    )
    evidence = QuestionEvidence(
        known_evidence_ids=("event-1",),
        observations=(),
        decisions=(),
        plan_diffs=(),
        validation_issues=(),
    )

    payload = build_question_payload(
        "Why did the plan change?",
        QuestionEntities(target_ids=(), uuv_ids=(), plan_ids=()),
        snapshot,
        evidence,
        None,
    )

    assert payload["evidence_ids"] == ["event-1"]
