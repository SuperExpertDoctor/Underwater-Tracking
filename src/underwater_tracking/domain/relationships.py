"""Compatibility normalization for carrier relationship payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_RELATIONSHIP_FIELDS = {
    "onboard": "onboard_uuv_ids",
    "deployed": "deployed_uuv_ids",
    "returning": "returning_uuv_ids",
}


def normalize_legacy_carrier_relationships(value: Any) -> Any:
    """Derive omitted carrier lists from UUV deployment defaults in old payloads."""
    if not isinstance(value, Mapping):
        return value
    carrier = value.get("carrier")
    if not isinstance(carrier, Mapping):
        return value
    missing = [field for field in _RELATIONSHIP_FIELDS.values() if field not in carrier]
    if not missing:
        return value
    memberships: dict[str, list[str]] = {
        field: [] for field in _RELATIONSHIP_FIELDS.values()
    }
    for uuv in value.get("uuvs", ()):
        if not isinstance(uuv, Mapping):
            continue
        uuv_id = uuv.get("uuv_id")
        deployment_state = uuv.get("deployment_state", "deployed")
        if (
            not isinstance(uuv_id, str)
            or uuv.get("status") == "failed"
            or deployment_state == "failed"
        ):
            continue
        relationship_field = _RELATIONSHIP_FIELDS.get(str(deployment_state))
        if relationship_field is not None:
            memberships[relationship_field].append(uuv_id)
    normalized = dict(value)
    normalized_carrier = dict(carrier)
    for field in missing:
        normalized_carrier[field] = tuple(sorted(memberships[field]))
    normalized["carrier"] = normalized_carrier
    return normalized
