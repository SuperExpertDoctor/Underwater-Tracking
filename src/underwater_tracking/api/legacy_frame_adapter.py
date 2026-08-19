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
    return OperationalFrame.model_validate(normalized)
