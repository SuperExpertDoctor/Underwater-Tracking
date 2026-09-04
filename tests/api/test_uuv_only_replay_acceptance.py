from __future__ import annotations

import json

import pytest

from underwater_tracking.api.frame_builder import operational_frame_json, operational_frame_payload
from underwater_tracking.api.replay import ReplayIndexError, ReplayService
from underwater_tracking.domain.ui_models import MapBounds, OperationalFrame


def _frame(frame_id: int, sim_time_s: int) -> OperationalFrame:
    return OperationalFrame(
        schema_version="1.0",
        frame_id=frame_id,
        sim_time_s=sim_time_s,
        plan_version=frame_id,
        map_bounds=MapBounds(min_x=-100.0, min_y=-100.0, max_x=100.0, max_y=100.0),
        uuv_only=True,
    )


def test_uuv_only_jsonl_replay_preserves_canonical_frames(tmp_path) -> None:
    frames = (_frame(1, 10), _frame(2, 20))
    path = tmp_path / "uuv-only.jsonl"
    path.write_text(
        "".join(f"{operational_frame_json(frame)}\n" for frame in frames),
        encoding="utf-8",
    )

    replayed = ReplayService(path).range(10, 20)

    assert [frame.sim_time_s for frame in replayed] == [10, 20]
    assert [operational_frame_payload(frame) for frame in replayed] == [
        operational_frame_payload(frame) for frame in frames
    ]
    assert all("usvs" not in operational_frame_json(frame).lower() for frame in frames)


def test_uuv_only_replay_rejects_old_usv_fields(tmp_path) -> None:
    payload = json.loads(operational_frame_json(_frame(1, 10)))
    payload["usvs"] = [{"usv_id": "USV-OLD", "position": {"x": 1, "y": 1}}]
    path = tmp_path / "old-usv.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReplayIndexError, match=r"line 1:"):
        ReplayService(path)
