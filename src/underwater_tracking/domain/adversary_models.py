"""Structured contracts for the submarine adversary decision loop.

The adversary controller is deliberately separated from simulation truth.  Its
input is a target-owned belief, target-owned observations, summarized platform
threats, acoustic/communications exposure, and prior decisions.  Kinematic and
operating-area limits are explicit input constraints so the LLM cannot choose a
physically impossible escape maneuver.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import pi
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Heading = Annotated[float, Field(ge=-pi, le=pi, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Point2D = tuple[FiniteFloat, FiniteFloat]

Maneuver = Literal[
    "course_change",
    "speed_change",
    "depth_change",
    "silent_running",
    "decoy_evasion",
    "hold_course",
]
EscapeIntent = Literal[
    "break_contact",
    "reposition",
    "deception",
    "silent_transit",
    "evade",
    "hold_course",
]
DecoyAction = Literal["none", "deploy", "reposition", "recover"]
CommunicationsDiscipline = Literal["normal", "restricted", "burst_only", "silent"]
PlatformKind = Literal["usv", "uuv", "unknown"]
ObservationKind = Literal[
    "passive_sonar",
    "active_sonar",
    "communication_intercept",
    "visual",
    "other",
]
ThreatLevel = Literal["low", "medium", "high", "critical"]
DecisionOutcome = Literal["unknown", "inconclusive", "contact_maintained", "contact_lost"]


class AdversaryStrictModel(BaseModel):
    """Strict, immutable base for all adversary-side contracts."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AdversaryBelief(AdversaryStrictModel):
    """The target's own estimated state and uncertainty, never simulator state."""

    target_id: str = Field(min_length=1)
    as_of_s: int = Field(ge=0)
    estimated_position_xy: Point2D
    estimated_velocity_xy: Point2D
    position_uncertainty_m: NonNegativeFloat
    velocity_uncertainty_mps: NonNegativeFloat
    estimated_heading: Heading
    estimated_speed_mps: NonNegativeFloat
    intent_hypothesis: EscapeIntent | Literal["unknown"]
    intent_confidence: Probability


class AdversaryObservation(AdversaryStrictModel):
    """One observation retained by the target-side belief manager."""

    observation_id: str = Field(min_length=1)
    observed_at_s: int = Field(ge=0)
    kind: ObservationKind
    bearing_rad: Heading | None = None
    range_m: NonNegativeFloat | None = None
    confidence: Probability
    assessment: Literal["platform", "emission", "communication", "uncertain"]


class PlatformThreatSummary(AdversaryStrictModel):
    """A bounded threat summary derived from the target's observations."""

    platform_id: str = Field(min_length=1)
    platform_kind: PlatformKind
    observed_at_s: int = Field(ge=0)
    threat_level: ThreatLevel
    estimated_range_m: NonNegativeFloat
    relative_bearing_rad: Heading
    passive_detection_risk: Probability
    active_ping_risk: Probability
    relay_detection_risk: Probability
    surface_relay_available: bool


class CommunicationsAcousticExposure(AdversaryStrictModel):
    """The target's estimate of its own acoustic and communications exposure."""

    as_of_s: int = Field(ge=0)
    passive_signature_level: Probability
    active_emitter_exposure: Probability
    communication_intercept_risk: Probability
    relay_detection_risk: Probability
    acoustic_clutter_level: Probability
    last_burst_age_s: NonNegativeFloat | None = None
    own_emission_mode: Literal["passive", "active", "mixed", "unknown"]


class AdversaryDecisionRecord(AdversaryStrictModel):
    """A prior adversary decision and its observed operational outcome."""

    decision_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    maneuver: Maneuver
    intent: EscapeIntent
    segment: str = Field(min_length=1)
    speed: NonNegativeFloat
    heading: Heading
    decoy_action: DecoyAction
    decoy_count: NonNegativeInt
    outcome: DecisionOutcome


class AdversaryKinematicLimits(AdversaryStrictModel):
    """Current maneuver limits supplied by the target's own platform model."""

    max_speed_mps: NonNegativeFloat
    max_turn_rate_rad_s: NonNegativeFloat
    decision_horizon_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    max_decoy_count: NonNegativeInt
    decoy_inventory: NonNegativeInt

    @model_validator(mode="after")
    def inventory_cannot_exceed_capacity(self) -> AdversaryKinematicLimits:
        if self.decoy_inventory > self.max_decoy_count:
            raise ValueError("decoy_inventory cannot exceed max_decoy_count")
        return self


class AdversaryOperatingBoundary(AdversaryStrictModel):
    """Axis-aligned operating boundary used for waypoint validation."""

    min_x: FiniteFloat
    max_x: FiniteFloat
    min_y: FiniteFloat
    max_y: FiniteFloat

    @model_validator(mode="after")
    def bounds_have_positive_area(self) -> AdversaryOperatingBoundary:
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("operating boundary must have positive area")
        return self


_FORBIDDEN_PRIVATE_STATE_MARKERS = (
    "truth",
    "groundtruth",
    "ground_truth",
    "trueposition",
    "true_position",
    "actualposition",
    "actual_position",
    "simulationtruth",
)


def _contains_private_state_marker(value: object) -> bool:
    if isinstance(value, str):
        compact = "".join(value.casefold().split())
        return any(marker in compact for marker in _FORBIDDEN_PRIVATE_STATE_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _contains_private_state_marker(key) or _contains_private_state_marker(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_state_marker(child) for child in value)
    return False


class AdversaryEscapeInput(AdversaryStrictModel):
    """Complete evidence packet accepted by the adversary LangGraph."""

    target_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    belief: AdversaryBelief
    observations: tuple[AdversaryObservation, ...] = ()
    platform_threats: tuple[PlatformThreatSummary, ...] = ()
    communications_acoustic_exposure: CommunicationsAcousticExposure
    decision_history: tuple[AdversaryDecisionRecord, ...] = ()
    kinematic_limits: AdversaryKinematicLimits
    operating_boundary: AdversaryOperatingBoundary

    @model_validator(mode="after")
    def target_and_evidence_are_consistent(self) -> AdversaryEscapeInput:
        if self.belief.target_id != self.target_id:
            raise ValueError("belief.target_id must match target_id")
        evidence = self.model_dump(mode="json")
        if _contains_private_state_marker(evidence):
            raise ValueError("adversary input contains unavailable simulator state")
        return self


class AdversaryEscapeDecision(AdversaryStrictModel):
    """Strict structured output of the adversary controller."""

    target_id: str = Field(min_length=1)
    maneuver: Maneuver
    intent: EscapeIntent
    waypoint: Point2D
    segment: str = Field(min_length=1)
    speed: NonNegativeFloat
    heading: Heading
    decoy_action: DecoyAction
    decoy_count: NonNegativeInt
    confidence: Probability
    rationale: str = Field(min_length=1, max_length=2000)
    communications_discipline: CommunicationsDiscipline

    @field_validator("rationale")
    @classmethod
    def rationale_must_use_available_evidence(cls, value: str) -> str:
        if _contains_private_state_marker(value):
            raise ValueError("rationale cannot claim unavailable simulator state")
        return value


__all__ = [
    "AdversaryBelief",
    "AdversaryDecisionRecord",
    "AdversaryEscapeDecision",
    "AdversaryEscapeInput",
    "AdversaryKinematicLimits",
    "AdversaryObservation",
    "AdversaryOperatingBoundary",
    "CommunicationsAcousticExposure",
    "PlatformThreatSummary",
]
