from __future__ import annotations

import json
from hashlib import sha256

from underwater_tracking.api.frame_builder import operational_frame_json, operational_frame_payload
from underwater_tracking.api.legacy_frame_adapter import read_legacy_frame
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.ui_models import MapBounds, OperationalFrame


def _frame(frame_id: int, sim_time_s: int) -> OperationalFrame:
    return OperationalFrame(
        frame_id=frame_id,
        sim_time_s=sim_time_s,
        plan_version=frame_id,
        map_bounds=MapBounds(min_x=-100.0, min_y=-100.0, max_x=100.0, max_y=100.0),
        uuv_only=True,
    )


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_uuv_only_jsonl_is_stable_and_old_usv_fields_are_read_only_compatibility(tmp_path) -> None:
    frames = (_frame(1, 10), _frame(2, 20))
    jsonl = "".join(f"{operational_frame_json(frame)}\n" for frame in frames)
    path = tmp_path / "uuv-only.jsonl"
    path.write_text(jsonl, encoding="utf-8")

    replayed = ReplayService(path).range(10, 20)
    assert [frame.sim_time_s for frame in replayed] == [10, 20]
    assert all("usvs" not in operational_frame_payload(frame) for frame in replayed)
    assert [_digest(operational_frame_payload(frame)) for frame in replayed] == [
        _digest(operational_frame_payload(frame)) for frame in frames
    ]
    assert all("usvs" not in operational_frame_json(frame).lower() for frame in frames)

    legacy_payload = json.loads(operational_frame_json(frames[0]))
    legacy_payload["usvs"] = [{"usv_id": "USV-OLD", "position": {"x": 1, "y": 1}}]
    legacy = read_legacy_frame(legacy_payload)
    assert legacy.uuv_only is True
    assert not hasattr(legacy, "usvs")
    assert "usvs" not in operational_frame_payload(legacy)
