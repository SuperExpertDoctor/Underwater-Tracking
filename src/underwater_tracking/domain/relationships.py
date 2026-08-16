"""Compatibility normalization for carrier relationship payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_RELATIONSHIP_FIELDS = {
    "onboard": "onboard_uuv_ids",
    "deployed": "deployed_uuv_ids",
    "returning": "returning_uuv_ids",
}


def expected_carrier_status(
    speed_mps: float, onboard_uuv_ids: tuple[str, ...], deployed_uuv_ids: tuple[str, ...], returning_uuv_ids: tuple[str, ...]
) -> str:
    """Derive the only valid operational carrier status from public state."""
    if returning_uuv_ids:
        return "recovering"
    if onboard_uuv_ids and deployed_uuv_ids:
        return "deploying"
    return "transit" if speed_mps > 0.0 else "standby"


def normalize_legacy_uuv_deployment_state(value: Any) -> Any:
    """Align old returning and failed statuses with their deployment state."""
    if not isinstance(value, Mapping) or "deployment_state" in value:
        return value
    status = value.get("status")
    if status not in {"returning", "failed"}:
        return value
    normalized = dict(value)
    normalized["deployment_state"] = status
    return normalized


def _field_value(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _legacy_deployment_state(uuv: Any) -> str:
    deployment_state = _field_value(uuv, "deployment_state")
    if deployment_state is not None:
        return str(deployment_state)
    status = str(_field_value(uuv, "status", ""))
    return status if status in {"returning", "failed"} else "deployed"


def normalize_legacy_carrier_relationships(value: Any) -> Any:
    """Derive omitted carrier lists from UUV deployment defaults in old payloads."""
    if not isinstance(value, Mapping):
        return value
    carrier = value.get("carrier")
    if carrier is None:
        return value
    if isinstance(carrier, Mapping):
        missing = [field for field in _RELATIONSHIP_FIELDS.values() if field not in carrier]
        missing_status = "status" not in carrier
    else:
        fields_set: set[str] = set(getattr(carrier, "model_fields_set", ()))
        missing = [field for field in _RELATIONSHIP_FIELDS.values() if field not in fields_set]
        missing_status = "status" not in fields_set
    if not missing and not missing_status:
        return value
    memberships: dict[str, list[str]] = {
        field: [] for field in _RELATIONSHIP_FIELDS.values()
    }
    for uuv in value.get("uuvs", ()):
        uuv_id = _field_value(uuv, "uuv_id")
        deployment_state = _legacy_deployment_state(uuv)
        if (
            not isinstance(uuv_id, str)
            or str(_field_value(uuv, "status")) == "failed"
            or deployment_state == "failed"
        ):
            continue
        relationship_field = _RELATIONSHIP_FIELDS.get(str(deployment_state))
        if relationship_field is not None:
            memberships[relationship_field].append(uuv_id)
    normalized = dict(value)
    updates: dict[str, Any] = {field: tuple(sorted(memberships[field])) for field in missing}
    relationship_values = {
        field: updates.get(field, tuple(_field_value(carrier, field, ())))
        for field in _RELATIONSHIP_FIELDS.values()
    }
    if missing_status:
        updates["status"] = expected_carrier_status(
            float(_field_value(carrier, "speed_mps", 0.0)),
            relationship_values["onboard_uuv_ids"],
            relationship_values["deployed_uuv_ids"],
            relationship_values["returning_uuv_ids"],
        )
    if isinstance(carrier, Mapping):
        normalized_carrier = dict(carrier)
        normalized_carrier.update(updates)
        normalized["carrier"] = normalized_carrier
    else:
        normalized["carrier"] = type(carrier).model_validate(
            {**carrier.model_dump(), **updates}
        )
    return normalized
