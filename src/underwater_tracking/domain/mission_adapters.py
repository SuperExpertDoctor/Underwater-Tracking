from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def legacy_frame_to_uuv_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read old operational payloads while dropping legacy USV projections."""
    normalized = dict(payload)
    normalized.pop("usvs", None)
    plans = normalized.get("regional_plans")
    if isinstance(plans, Mapping):
        normalized["regional_plans"] = {
            key: _drop_legacy_region_fields(value) for key, value in plans.items()
        }
    return normalized


def _drop_legacy_region_fields(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    result.pop("usv_assignments", None)
    result.pop("assigned_usv_ids", None)
    return result
