from __future__ import annotations

import time

import pytest

from underwater_tracking.api.hub import RuntimeDirectiveQueue
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan


def _preview(status: str = "preview") -> ExpertDirective:
    return ExpertDirective(
        directive_id="S1:assign:T1:UUV-1",
        raw_text="assignment: UUV-1 -> T1",
        target_scope=("T1",),
        directive_type="assignment",
        assignment_target_id="T1",
        assignment_uuv_ids=("UUV-1",),
        confidence=1.0,
        status=status,
    )


class Runtime:
    def __init__(self) -> None:
        self.revision = 4
        self.applied: list[str] = []

    def active_plan(self) -> TrackingPlan:
        return TrackingPlan(
            plan_id="plan-4",
            scenario_id="S1",
            revision=self.revision,
            base_snapshot_revision=0,
            valid_from_s=0,
        )

    def preview_directive(self, raw_text: str) -> ExpertDirective:
        del raw_text
        return _preview()

    def preview_assignment(self, *, uuv_ids: tuple[str, ...], target_id: str) -> ExpertDirective:
        assert uuv_ids == ("UUV-1",)
        assert target_id == "T1"
        return _preview()

    def apply_directive(self, directive_id: str) -> ExpertDirective:
        self.applied.append(directive_id)
        return _preview("applied")


def _wait_for(queue: RuntimeDirectiveQueue, request_id: str, status: str) -> dict[str, object]:
    for _ in range(100):
        result = queue.status(request_id)
        if result["status"] == status:
            return result
        time.sleep(0.01)
    pytest.fail(f"request {request_id} did not reach {status!r}")


def test_assignment_preview_requires_current_plan_at_apply() -> None:
    runtime = Runtime()
    queue = RuntimeDirectiveQueue(runtime, workers=1)
    try:
        request_id = queue.submit_assignment(
            uuv_ids=["UUV-1"], target_id="T1", expected_plan_version=4
        )
        preview = _wait_for(queue, request_id, "preview")
        directive = preview["directive"]
        assert isinstance(directive, dict)
        assert directive["directive_type"] == "assignment"

        runtime.revision = 5
        with pytest.raises(ValueError, match="plan changed"):
            queue.apply(request_id)
        assert queue.status(request_id)["status"] == "preview"

        runtime.revision = 4
        queue.apply(request_id)
        _wait_for(queue, request_id, "applied")
        assert runtime.applied == ["S1:assign:T1:UUV-1"]
    finally:
        queue.close()
