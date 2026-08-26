from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tests.domain.test_execution_models import _snapshot as execution_snapshot
from underwater_tracking.api.frame_builder import (
    build_operational_frame,
    operational_frame_payload,
)
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.models import SituationSnapshot


def _frame():
    snapshot = execution_snapshot()
    situation = SituationSnapshot(
        scenario_id=snapshot.scenario_id,
        snapshot_revision=snapshot.source_snapshot_revision,
        sim_time_s=int(snapshot.source_sim_time_s),
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    return build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
        execution_snapshot=snapshot,
    )


def test_execution_frame_projects_one_authoritative_four_region_snapshot() -> None:
    frame = _frame()

    assert frame.execution is not None
    assert frame.execution.execution_revision == 9
    assert frame.execution.source_snapshot_revision == 12
    assert len(frame.execution.regions) == 4
    assert len(frame.execution.task_groups) == 4
    assert tuple(row.region_id for row in frame.execution.regions) == (
        "target_00:task:01",
        "target_00:task:02",
        "target_00:task:03",
        "target_00:task:04",
    )
    assert frame.execution.regions[0].task_group_id == "TG-01"
    assert frame.execution.reserve_uuv_ids == ("uuv_08", "uuv_09", "uuv_10", "uuv_11")
    assert frame.execution.degraded is False


def test_execution_frame_rejects_mixed_revision_and_candidate_grid_surface() -> None:
    payload = _frame().model_dump(mode="json")
    payload["execution"]["regions"][1]["execution_revision"] = 8
    with pytest.raises(ValidationError, match="revision"):
        type(_frame()).model_validate(payload)

    frame_payload = operational_frame_payload(_frame())
    assert "prediction_grids" not in frame_payload["execution"]
    assert "candidate_regions" not in frame_payload["execution"]


def test_hub_and_replay_use_the_same_execution_frame_payload(tmp_path) -> None:
    frame = _frame()
    hub = OperationalHub()
    hub.publish(frame)
    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame)
    logger.close()

    replayed = ReplayService(path).last()
    assert replayed is not None
    assert hub.snapshot() == frame
    assert replayed == frame
    assert json.loads(frame.model_dump_json()) == operational_frame_payload(frame)


def test_legacy_frame_without_execution_remains_readable() -> None:
    frame = _frame().model_copy(
        update={"execution": None, "execution_consistency": None}
    )

    restored = type(frame).model_validate_json(frame.model_dump_json())

    assert restored.execution is None


def test_live_publisher_reads_the_runtime_execution_snapshot() -> None:
    snapshot = execution_snapshot()

    class Runtime:
        def active_plan(self):
            return None

        def get_state(self):
            return {}

        @property
        def current_execution_snapshot(self):
            return snapshot

    class Ledger:
        def list_decisions(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

        def list_directives(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    class Events:
        def list_events(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=12,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    frame = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    ).publish(situation)

    assert frame.uuv_only is True
    assert frame.execution is not None
    assert frame.execution.execution_revision == snapshot.execution_revision
