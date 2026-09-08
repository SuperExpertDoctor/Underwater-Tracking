"""One fail-closed quality contract for public, observation-backed estimates."""

from __future__ import annotations

from math import isfinite
from typing import Literal
from pydantic import Field
from underwater_tracking.domain.models import StrictModel, TargetBelief


class EstimateHealth(StrictModel):
    status: Literal["current", "degraded", "expired", "unavailable"] = "unavailable"
    source_track_revision: int = Field(default=0, ge=0)
    state_time_s: float | None = None
    last_observed_at_s: float | None = None
    valid_until_s: float | None = None
    source_age_s: float | None = None
    source_observation_ids: tuple[str, ...] = ()
    accepted_observation_ids_this_cycle: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ("estimate_provenance_missing",)


def assess_public_estimate(belief: TargetBelief, now_s: float) -> EstimateHealth:
    """Do not infer freshness from publication time or silently repair covariance."""
    reasons: list[str] = []
    observed = belief.last_observed_at_s
    expires = belief.valid_until_s
    if (
        belief.track_revision < 1
        or observed is None
        or expires is None
        or not belief.source_observation_ids
    ):
        reasons.append("estimate_provenance_missing")
    if (
        not isfinite(now_s)
        or now_s < belief.sim_time_s
        or (observed is not None and observed > belief.sim_time_s)
    ):
        reasons.append("estimate_time_invalid")
    try:
        _x, _y = belief.mean[:2]
        a, b = belief.covariance[0][:2]
        c, d = belief.covariance[1][:2]
        if not all(isfinite(v) for v in (*belief.mean, a, b, c, d)):
            reasons.append("estimate_non_finite")
        elif a <= 0 or d <= 0 or abs(b - c) > 1e-9 * max(1.0, abs(b), abs(c)) or a * d - b * c <= 0:
            reasons.append("estimate_covariance_invalid")
    except (ValueError, IndexError, TypeError):
        reasons.append("estimate_shape_invalid")
    if expires is not None and observed is not None and expires <= observed:
        reasons.append("estimate_validity_invalid")
    status = "unavailable"
    if not reasons:
        if now_s >= expires:
            status = "expired"
            reasons.append("estimate_expired")
        elif now_s > observed:
            status = "degraded"
            reasons.append("estimate_extrapolated")
        else:
            status = "current"
    return EstimateHealth(
        status=status,
        source_track_revision=belief.track_revision,
        state_time_s=belief.sim_time_s,
        last_observed_at_s=observed,
        valid_until_s=expires,
        source_age_s=max(0.0, now_s - observed)
        if observed is not None and isfinite(now_s)
        else None,
        source_observation_ids=belief.source_observation_ids,
        accepted_observation_ids_this_cycle=belief.accepted_observation_ids_this_cycle,
        reason_codes=tuple(reasons),
    )
