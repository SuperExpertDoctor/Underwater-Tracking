from __future__ import annotations

import json

import pytest

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
    prediction = payload["target_estimates"][0]["prediction"] if payload["target_estimates"] else None
    if prediction is not None:
        for field in (
            "prediction_id",
            "prediction_revision",
            "origin_sim_time_s",
            "health",
        ):
            prediction.pop(field, None)
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
    assert not hasattr(frames[0], "usvs")
    assert "usvs" not in frames[0].model_dump(mode="json")


def test_legacy_prediction_without_health_is_explicitly_unknown() -> None:
    from tests.api.test_frame_contracts import _full_frame

    payload = _full_frame().model_dump(mode="json")
    prediction = payload["target_estimates"][0]["prediction"]
    for field in (
        "prediction_id",
        "prediction_revision",
        "origin_sim_time_s",
        "health",
    ):
        prediction.pop(field, None)

    restored = read_legacy_frame(payload)
    restored_prediction = restored.target_estimates[0].prediction

    assert restored_prediction is not None
    assert restored_prediction.centerline_xy == _full_frame().target_estimates[0].prediction.centerline_xy
    assert restored_prediction.health.status == "legacy_unknown"
    assert restored_prediction.health.regime == "legacy_unknown"
    assert "legacy_health_missing" in restored_prediction.health.reason_codes


def test_modern_prediction_health_survives_replay_without_legacy_default() -> None:
    from tests.api.test_frame_contracts import _full_frame

    frame = _full_frame()
    restored = read_legacy_frame(frame.model_dump(mode="json"))

    assert restored.target_estimates[0].prediction.health.status == "valid"
    assert "legacy_health_missing" not in restored.target_estimates[0].prediction.health.reason_codes


@pytest.mark.parametrize(
    ("legacy_status", "health_status"),
    (("stale", "degraded"), ("unavailable", "failed")),
)
def test_legacy_execution_status_is_normalized_before_validation(
    legacy_status: str,
    health_status: str,
) -> None:
    from tests.api.test_execution_frame_contract import _frame

    payload = _frame().model_dump(mode="json")
    execution = payload["execution"]
    for field in (
        "valid_from_s",
        "valid_until_s",
        "health_status",
        "health_reasons",
        "region_generation_mode",
    ):
        execution.pop(field, None)
    execution["data_status"] = legacy_status

    restored = read_legacy_frame(payload)

    assert restored.execution.health_status == health_status
