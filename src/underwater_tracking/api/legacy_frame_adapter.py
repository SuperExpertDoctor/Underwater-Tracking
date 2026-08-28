"""Compatibility boundary for reading legacy operational frame payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from underwater_tracking.domain.mission_adapters import legacy_frame_to_uuv_view
from underwater_tracking.domain.ui_models import OperationalFrame


def read_legacy_frame(payload: Mapping[str, Any]) -> OperationalFrame:
    """Validate a legacy frame while discarding its surface-node projection."""
    had_usv_projection = bool(payload.get("usvs"))
    normalized = legacy_frame_to_uuv_view(payload)
    if "uuv_only" not in normalized:
        normalized["uuv_only"] = had_usv_projection
    _normalize_prediction_health(normalized)
    _normalize_execution_health(normalized)
    return OperationalFrame.model_validate(normalized)


def _normalize_prediction_health(payload: dict[str, Any]) -> None:
    estimates = payload.get("target_estimates")
    if not isinstance(estimates, (list, tuple)):
        return
    execution = payload.get("execution")
    frame_id = max(1, int(payload.get("frame_id", 1)))
    sim_time_s = max(0.0, float(payload.get("sim_time_s", 0.0)))
    for estimate in estimates:
        if not isinstance(estimate, dict):
            continue
        prediction = estimate.get("prediction")
        if not isinstance(prediction, dict) or "health" in prediction:
            continue
        target_id = str(estimate.get("target_id", "unknown"))
        matching_execution = (
            execution
            if isinstance(execution, dict)
            and execution.get("target_id") == target_id
            else None
        )
        revision = (
            int(matching_execution["prediction_revision"])
            if matching_execution is not None
            and matching_execution.get("prediction_revision") is not None
            else frame_id
        )
        prediction_id = (
            str(matching_execution.get("prediction_id"))
            if matching_execution is not None
            and matching_execution.get("prediction_id")
            else f"legacy:{target_id}:{revision}"
        )
        radii = prediction.get("radius_m", ())
        centerline = prediction.get("centerline_xy", ())
        prediction.setdefault("prediction_id", prediction_id)
        prediction.setdefault("prediction_revision", max(1, revision))
        prediction.setdefault("origin_sim_time_s", sim_time_s)
        if not prediction.get("point_confidence") and centerline:
            prediction["point_confidence"] = tuple(1.0 for _ in centerline)
        prediction["health"] = {
            "status": "legacy_unknown",
            "regime": "legacy_unknown",
            "reason_codes": ("legacy_health_missing",),
            "source_track_age_s": 0.0,
            "clipped_point_fraction": 0.0,
            "maximum_radius_m": max((float(radius) for radius in radii), default=0.0),
            "raw_prediction_id": None,
        }


def _normalize_execution_health(payload: dict[str, Any]) -> None:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        return
    legacy_status = execution.get("data_status")
    if "health_status" not in execution and legacy_status is not None:
        execution["health_status"] = {
            "current": "current",
            "stale": "degraded",
            "unavailable": "failed",
        }.get(str(legacy_status), "failed")
    execution.pop("data_status", None)
    sim_time_s = max(0.0, float(payload.get("sim_time_s", 0.0)))
    age_s = max(0.0, float(execution.get("data_age_s", 0.0)))
    valid_from_s = max(0.0, sim_time_s - age_s)
    execution.setdefault("valid_from_s", valid_from_s)
    execution.setdefault("valid_until_s", max(valid_from_s + 1.0, sim_time_s + 1.0))
    execution.setdefault("health_reasons", ("legacy_execution_health",))
    execution.setdefault("region_generation_mode", "reprojected_previous")
