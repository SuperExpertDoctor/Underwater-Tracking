from __future__ import annotations

import json
from hashlib import sha256

import pytest

from underwater_tracking.api.frame_builder import operational_frame_json, operational_frame_payload
from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from tests.api.test_execution_frame_contract import _frame


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_runtime_frame_round_trips_through_jsonl_replay_without_adapters(tmp_path) -> None:
    frame = _frame()
    path = tmp_path / "runtime.jsonl"
    path.write_text(operational_frame_json(frame) + "\n", encoding="utf-8")

    restored = ReplayService(path).last()

    assert restored == frame
    assert restored is not None
    assert _digest(operational_frame_payload(restored)) == _digest(
        operational_frame_payload(frame)
    )
    assert len(restored.execution.task_groups) == 4
    assert all(len(group.member_uuv_ids) == 3 for group in restored.execution.task_groups)


def test_legacy_frame_defaults_refresh_projection_to_idle_unknown() -> None:
    frame = _frame()

    assert frame.execution is not None
    assert frame.execution.refresh_status == "idle"
    assert frame.execution.refresh_last_result == "unknown"
    assert frame.execution.refresh_due_at_s is None
    assert frame.execution.refresh_last_attempt_s is None
    assert frame.execution.refresh_source_snapshot_revision is None


def test_replay_drops_nested_truth_fields_before_public_frame_validation(tmp_path) -> None:
    payload = json.loads(operational_frame_json(_frame()))
    payload["region_probability_evidence"] = {
        "target_00:task:01": {
            "probability": 0.8,
            "truth_velocity": [3.0, 0.0],
            "targetTruth": {"x": 10.0},
        }
    }
    path = tmp_path / "nested-evaluation-field.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    restored = ReplayService(path).last()

    assert restored is not None
    assert restored.region_probability_evidence == {
        "target_00:task:01": {"probability": 0.8}
    }


def test_replay_rejects_legacy_usv_projection(tmp_path) -> None:
    payload = json.loads(operational_frame_json(_frame()))
    payload["usvs"] = [{"usv_id": "USV-OLD", "position": {"x": 0, "y": 0}}]
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReplayIndexError, match=r"(?s)line 1:.*Extra inputs are not permitted"):
        ReplayService(path)


@pytest.mark.parametrize("field", ("task_groups", "tracking_policy", "tracking_control"))
def test_replay_rejects_incomplete_runtime_execution(field: str, tmp_path) -> None:
    payload = json.loads(operational_frame_json(_frame()))
    payload["execution"].pop(field)
    path = tmp_path / f"missing-{field}.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReplayIndexError, match=r"line 1:"):
        ReplayService(path)
