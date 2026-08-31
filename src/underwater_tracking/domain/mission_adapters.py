from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def legacy_frame_to_uuv_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read old operational payloads while dropping legacy USV projections."""
    return _drop_legacy_fields(payload)


_LEGACY_USV_KEYS = frozenset(
    {
        "usvs",
        "usv_assignments",
        "assigned_usv_ids",
        "usv_ids_by_target",
        "usv_ids",
        "usv_roles_by_member",
        "usv_actions",
        "usv_role",
        "relay_usv_ids",
        "require_usv_per_region",
        "usv_relay_required",
        "relay_overlap_policy",
    }
)


def _drop_legacy_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {
            key: _drop_legacy_fields(child)
            for key, child in value.items()
            if key not in _LEGACY_USV_KEYS
        }
        if result.get("tracking_mode") in {
            "uuv_primary_usv_relay",
            "heuristic_usv",
        }:
            result["tracking_mode"] = "heuristic_uuv"
        return result
    if isinstance(value, list):
        return [_drop_legacy_fields(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_drop_legacy_fields(child) for child in value)
    return value
