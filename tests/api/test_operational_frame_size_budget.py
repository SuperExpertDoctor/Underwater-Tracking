from __future__ import annotations

from types import SimpleNamespace

from underwater_tracking.api.live import OperationalFramePublisher, compact_operational_frame
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.domain.models import SituationSnapshot


class _Runtime:
    llm_paused = False

    def active_plan(self):
        return None

    def get_state(self):
        return {"intent_hypotheses": {}, "predictions": {}}


class _Ledger:
    def list_decisions(self, scenario_id: str, limit: int = 100):
        return []

    def list_directives(self, scenario_id: str, status: str | None = None):
        return []


class _Events:
    def list_events(self, **kwargs):
        return []


def test_compact_operational_frame_uses_the_release_size_tails() -> None:
    publisher = OperationalFramePublisher(
        runtime=_Runtime(),
        ledger=_Ledger(),
        events=_Events(),
        hub=OperationalHub(),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    frame = publisher.publish(snapshot).model_copy(
        update={
            "mission_events": tuple(
                SimpleNamespace(event_id=f"mission-{index}")
                for index in range(100)
            ),
            "ledger": tuple(
                SimpleNamespace(decision_id=f"decision-{index}")
                for index in range(100)
            ),
            "operator_audit_event_ids": tuple(
                f"audit-{index}" for index in range(200)
            ),
            "plan_timeline": tuple(
                SimpleNamespace(plan_id=f"plan-{index}")
                for index in range(100)
            ),
        }
    )

    compact = compact_operational_frame(frame)

    assert len(compact.mission_events) == 16
    assert len(compact.ledger) == 32
    assert len(compact.operator_audit_event_ids) == 64
    assert len(compact.plan_timeline) == 32
    assert compact.mission_events[0].event_id == "mission-84"
    assert compact.ledger[0].decision_id == "decision-68"
    assert compact.operator_audit_event_ids[0] == "audit-136"
    assert compact.plan_timeline[0].plan_id == "plan-68"
    publisher.close()
