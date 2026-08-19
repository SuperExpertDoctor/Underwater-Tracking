from __future__ import annotations

import json

from underwater_tracking.api.legacy_frame_adapter import read_legacy_frame
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.ui_models import MapBounds, OperationalFrame


def _legacy_payload() -> dict[str, object]:
    frame = OperationalFrame(
        frame_id=1,
        sim_time_s=30,
        plan_version=0,
        map_bounds=MapBounds(min_x=-100.0, min_y=-100.0, max_x=100.0, max_y=100.0),
        uuv_only=True,
    )
    payload = frame.model_dump(mode="json")
    payload["usvs"] = [{"usv_id": "USV-legacy", "position": {"x": 0, "y": 0}}]
    return payload


def test_legacy_frame_reader_discards_usv_projection() -> None:
    frame = read_legacy_frame(_legacy_payload())

    assert frame.uuvs == ()
    assert "usvs" not in frame.model_dump(mode="json")


def test_replay_accepts_old_usv_fields_but_exposes_uuv_only_view(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(_legacy_payload()) + "\n", encoding="utf-8")

    frames = ReplayService(path).range()

    assert len(frames) == 1
    assert frames[0].uuv_only is True
    assert frames[0].usvs == ()
    assert "usvs" not in frames[0].model_dump(mode="json")
