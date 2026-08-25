"""Canonical deployability predicate for planning and assignment resources."""

from __future__ import annotations

from underwater_tracking.domain.models import DeploymentState, UUVState, UUVStatus


def is_deployable(uuv: UUVState) -> bool:
    """Whether a UUV may receive a planning, assignment, or ping command."""
    return (
        uuv.status is not UUVStatus.UNAVAILABLE
        and uuv.deployment_state is DeploymentState.DEPLOYED
    )


def deployability_conflict(uuv: UUVState) -> str:
    """Return the deterministic reason a non-deployable UUV is excluded."""
    if uuv.deployment_state is DeploymentState.FAILED:
        return "failed"
    if uuv.deployment_state is DeploymentState.RETURNING:
        return "returning"
    if uuv.status is UUVStatus.UNAVAILABLE:
        return "unavailable"
    return uuv.deployment_state.value
