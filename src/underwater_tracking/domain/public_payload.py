"""Recursive sanitization for payloads crossing the public evidence boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


FORBIDDEN_TRUTH_KEYS = frozenset(
    {
        "actual_position",
        "actual_targets",
        "actual_velocity",
        "evaluation",
        "evaluation_frame",
        "evaluation_only",
        "evaluation_only_state",
        "evaluation_result",
        "evaluation_state",
        "truth",
        "truth_position",
        "truth_velocity",
        "ground_truth",
        "target_truth",
        "global_trajectory_history",
        "scenario_truth_label",
        "simulation_truth",
        "target_position_truth",
        "true_course",
        "true_intent",
        "true_location",
        "true_position",
        "true_state",
        "true_targets",
        "true_velocity",
    }
)

_NORMALIZED_FORBIDDEN_KEYS = frozenset(
    key.replace("_", "") for key in FORBIDDEN_TRUTH_KEYS
)


def sanitize_public_payload(value: object) -> object:
    """Drop forbidden evaluation fields recursively without exposing a reason."""

    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, child in value.items():
            key_text = str(key)
            normalized = re.sub(r"[^a-z0-9]+", "_", key_text.casefold()).strip("_")
            if (
                normalized in FORBIDDEN_TRUTH_KEYS
                or normalized.replace("_", "") in _NORMALIZED_FORBIDDEN_KEYS
            ):
                continue
            cleaned[key_text] = sanitize_public_payload(child)
        return cleaned
    if isinstance(value, tuple):
        return tuple(sanitize_public_payload(child) for child in value)
    if isinstance(value, list):
        return [sanitize_public_payload(child) for child in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_public_payload(child) for child in value]
    return value


def sanitize_public_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a mapping-shaped sanitized payload for typed builder APIs."""

    cleaned = sanitize_public_payload(value)
    return cleaned if isinstance(cleaned, dict) else {}


__all__ = [
    "FORBIDDEN_TRUTH_KEYS",
    "sanitize_public_mapping",
    "sanitize_public_payload",
]
