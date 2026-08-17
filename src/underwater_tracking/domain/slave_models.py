"""Strict contracts for the first executable group-slave sonar graph.

The slave brain receives an estimated operational view only.  These models
contain platform capability, distance-derived connectivity, belief quality,
acoustic conditions and the committed rotation segments, but no target truth,
target coordinates or simulator evaluation fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _SlaveStrictModel(BaseModel):
    """Frozen, reject-unknown-fields base for the slave boundary."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteScalar = Annotated[float, Field(allow_inf_nan=False)]


class SlavePlatformCapability(_SlaveStrictModel):
    """Capabilities and current limits visible to the group slave brain."""

    platform_id: str = Field(min_length=1)
    platform_kind: Literal["usv", "uuv"]
    passive_capable: bool
    active_capable: bool
    active_receive_capable: bool
    passive_range_m: PositiveFinite
    active_range_m: PositiveFinite
    energy_fraction: UnitInterval
    ping_energy_cost_fraction: UnitInterval
    exposure_cost: UnitInterval
    ping_cooldown_s: int = Field(ge=1)
    cooldown_remaining_s: int = Field(ge=0)
    available: bool
    sensor_mode: Literal["passive", "active"]
    max_speed_mps: PositiveFinite = 1.0
    max_turn_rate_rad_s: PositiveFinite = 0.1
    endurance_s: PositiveFinite = 1.0
    deployment_state: Literal["onboard", "deployed", "returning", "failed"] = "deployed"
    group_id: str | None = Field(default=None, min_length=1)
    is_group_leader: bool = False
    master_connected: bool
    carrier_connected: bool
    passive_bearing_variance_rad2: PositiveFinite = 0.01
    active_bearing_sigma_rad: PositiveFinite = 0.1
    active_range_sigma_m: PositiveFinite = 50.0
    clutter_sensitivity: UnitInterval = 0.3
    distance_to_carrier_m: FiniteNonNegative | None = None
    carrier_support_radius_m: PositiveFinite | None = None

    @model_validator(mode="after")
    def carrier_support_is_consistent(self) -> SlavePlatformCapability:
        if self.platform_kind == "usv":
            if self.distance_to_carrier_m is None or self.carrier_support_radius_m is None:
                raise ValueError("USV requires carrier distance and support radius")
            if self.distance_to_carrier_m > self.carrier_support_radius_m:
                raise ValueError("USV is outside the carrier support radius")
        return self


class SlaveCommunicationLink(_SlaveStrictModel):
    """A distance-derived communication edge in the slave's local graph."""

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    medium: Literal["surface", "acoustic"]
    distance_m: FiniteNonNegative
    range_m: PositiveFinite

    @property
    def connected(self) -> bool:
        """Whether the link is connected under the configured distance rule."""

        return self.distance_m <= self.range_m


class SlaveBeliefSummary(_SlaveStrictModel):
    """Target information derived from the group's current belief only."""

    target_id: str = Field(min_length=1)
    quality: UnitInterval
    covariance_trace_m2: FiniteNonNegative
    covariance_max_eigenvalue_m2: FiniteNonNegative
    last_observation_age_s: FiniteNonNegative
    passive_snr_db: FiniteScalar
    background_noise_db: FiniteScalar
    active_clutter_level: UnitInterval
    target_lost: bool
    candidate_count: int = Field(ge=0)
    candidate_ids: tuple[str, ...] = ()
    association_confidence: UnitInterval

    @model_validator(mode="after")
    def candidate_summary_is_consistent(self) -> SlaveBeliefSummary:
        if len(self.candidate_ids) > self.candidate_count:
            raise ValueError("candidate_ids cannot exceed candidate_count")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        return self


class SlaveHandoffSegment(_SlaveStrictModel):
    """A predicted spatial-temporal relay segment available to the slave."""

    segment_id: str = Field(min_length=1)
    start_s: int = Field(ge=0)
    end_s: int = Field(gt=0)
    predicted_quality: UnitInterval
    predicted_covariance_trace_m2: FiniteNonNegative
    owner_group_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> SlaveHandoffSegment:
        if self.end_s <= self.start_s:
            raise ValueError("handoff segment end_s must be after start_s")
        return self


class SlaveSonarContext(_SlaveStrictModel):
    """Bounded input admitted by the group-slave graph."""

    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    group_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    master_id: str = Field(min_length=1)
    master_connected: bool
    platforms: tuple[SlavePlatformCapability, ...] = Field(min_length=1)
    communication_links: tuple[SlaveCommunicationLink, ...] = ()
    belief: SlaveBeliefSummary
    handoff_segments: tuple[SlaveHandoffSegment, ...] = Field(min_length=1)
    current_segment_id: str | None = Field(default=None, min_length=1)
    predicted_intent: Literal[
        "transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown"
    ] = "unknown"
    intent_confidence: UnitInterval = 0.0

    @model_validator(mode="after")
    def context_references_are_valid(self) -> SlaveSonarContext:
        platform_ids = {platform.platform_id for platform in self.platforms}
        if len(platform_ids) != len(self.platforms):
            raise ValueError("platform IDs must be unique")
        if self.belief.target_id != self.target_id:
            raise ValueError("belief target_id must match context target_id")

        segment_ids = {segment.segment_id for segment in self.handoff_segments}
        if len(segment_ids) != len(self.handoff_segments):
            raise ValueError("handoff segment IDs must be unique")
        if self.current_segment_id is not None and self.current_segment_id not in segment_ids:
            raise ValueError("current_segment_id must reference a handoff segment")

        known_endpoint_ids = platform_ids | {self.master_id}
        for link in self.communication_links:
            if link.source_id not in known_endpoint_ids or link.target_id not in known_endpoint_ids:
                raise ValueError("communication link references an unknown endpoint")
            if link.source_id == link.target_id:
                raise ValueError("communication links cannot connect a platform to itself")
        return self

    def platform(self, platform_id: str) -> SlavePlatformCapability | None:
        """Return one roster entry without exposing any simulator object."""

        return next(
            (platform for platform in self.platforms if platform.platform_id == platform_id),
            None,
        )

    def is_connected(self, source_id: str, target_id: str) -> bool:
        """Evaluate a pairwise link using only configured distance/range."""

        return any(
            link.connected
            and (
                (link.source_id == source_id and link.target_id == target_id)
                or (link.source_id == target_id and link.target_id == source_id)
            )
            for link in self.communication_links
        )


class SlaveSonarDecision(_SlaveStrictModel):
    """The structured sonar-mode and relay decision emitted by the slave LLM."""

    mode: Literal["passive", "active"]
    emitter: str | None = Field(default=None, min_length=1)
    receiver_ids: tuple[str, ...] = Field(min_length=1)
    target_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    handoff_segment: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)
    confidence: UnitInterval
    expected_information_gain: UnitInterval
    energy_cost_fraction: UnitInterval
    exposure_cost: UnitInterval
    cooldown_s: int = Field(ge=0)
    passive_continuous: Literal[True] = True

    @model_validator(mode="after")
    def receiver_ids_are_unique(self) -> SlaveSonarDecision:
        if len(set(self.receiver_ids)) != len(self.receiver_ids):
            raise ValueError("receiver_ids must be unique")
        return self


class SlaveDecisionValidationError(ValueError):
    """A schema-valid LLM decision that violates the current local boundary."""


def validate_slave_decision(
    decision: SlaveSonarDecision,
    context: SlaveSonarContext,
) -> SlaveSonarDecision:
    """Validate a decision against the admitted roster and estimated context.

    This function is intentionally rejecting. It never changes the LLM
    decision and never produces a replacement policy.
    """

    errors: list[str] = []
    platforms = {platform.platform_id: platform for platform in context.platforms}

    if decision.target_id != context.target_id:
        errors.append("target_id does not match the slave context")
    if decision.group_id != context.group_id:
        errors.append("group_id does not match the slave context")
    if decision.handoff_segment not in {
        segment.segment_id for segment in context.handoff_segments
    }:
        errors.append("handoff_segment is not in the available segment roster")

    receivers = [platforms.get(platform_id) for platform_id in decision.receiver_ids]
    missing_receivers = [
        platform_id
        for platform_id, platform in zip(decision.receiver_ids, receivers, strict=True)
        if platform is None
    ]
    if missing_receivers:
        errors.append(f"receiver_ids contain unknown platforms: {sorted(missing_receivers)}")
    valid_receivers = [platform for platform in receivers if platform is not None]
    if any(not platform.available for platform in valid_receivers):
        errors.append("receiver_ids contain unavailable platforms")
    if any(platform.deployment_state == "failed" for platform in valid_receivers):
        errors.append("receiver_ids contain failed platforms")

    if decision.mode == "passive":
        if decision.emitter is not None:
            errors.append("passive mode must not select an emitter")
        if any(not platform.passive_capable for platform in valid_receivers):
            errors.append("passive mode requires passive-capable receivers")
        if decision.energy_cost_fraction != 0.0:
            errors.append("passive mode must have zero energy cost")
        if decision.exposure_cost != 0.0:
            errors.append("passive mode must have zero exposure cost")
        if decision.cooldown_s != 0:
            errors.append("passive mode must remain continuously open with cooldown_s=0")
    else:
        emitter = platforms.get(decision.emitter or "")
        if emitter is None:
            errors.append("active mode requires an emitter in the platform roster")
        else:
            if not emitter.available or emitter.deployment_state == "failed":
                errors.append("active emitter is unavailable")
            if not emitter.active_capable:
                errors.append("active emitter is not active-sonar capable")
            if emitter.cooldown_remaining_s > 0:
                errors.append("active emitter is still inside its cooldown")
            if decision.energy_cost_fraction <= 0.0:
                errors.append("active mode must declare a positive energy cost")
            if decision.energy_cost_fraction > emitter.energy_fraction:
                errors.append("active energy cost exceeds emitter energy")
            if decision.energy_cost_fraction > emitter.ping_energy_cost_fraction:
                errors.append("active energy cost exceeds emitter capability")
            if decision.exposure_cost <= 0.0:
                errors.append("active mode must declare positive exposure")
            if decision.exposure_cost > emitter.exposure_cost:
                errors.append("active exposure exceeds emitter capability")
            if decision.cooldown_s < emitter.ping_cooldown_s:
                errors.append("active cooldown is shorter than emitter ping cooldown")
            for receiver in valid_receivers:
                if not receiver.active_receive_capable:
                    errors.append(f"receiver {receiver.platform_id!r} cannot receive active sonar")
                if receiver.platform_id != emitter.platform_id and not context.is_connected(
                    emitter.platform_id, receiver.platform_id
                ):
                    errors.append(
                        f"active receiver {receiver.platform_id!r} is disconnected from emitter"
                    )
            if decision.expected_information_gain <= 0.0:
                errors.append("active mode must declare positive expected information gain")

    if errors:
        raise SlaveDecisionValidationError("; ".join(errors))
    return decision


def platform_ids(platforms: Iterable[SlavePlatformCapability]) -> tuple[str, ...]:
    """Return stable IDs for prompt construction and diagnostics."""

    return tuple(sorted(platform.platform_id for platform in platforms))
