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
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
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
    "continue_mission",
    "avoid_contact",
    "escape_to_region",
    "hold_position",
]
AdversaryIntent = Literal[
    "continue_mission",
    "avoid_contact",
    "break_contact",
    "escape_to_region",
    "hold_position",
]
DecoyAction = Literal["none", "deploy", "reposition", "recover"]
CommunicationsDiscipline = Literal["normal", "restricted", "burst_only", "silent"]
TargetPlatformKind = Literal["carrier", "mother_ship", "uuv"]
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


class LocalPlatformDetection(AdversaryStrictModel):
    """Noisy target-side estimate produced inside the local sensing boundary."""

    platform_id: str = Field(min_length=1)
    platform_kind: TargetPlatformKind
    observed_at_s: int = Field(ge=0)
    estimated_range_m: NonNegativeFloat
    relative_bearing_rad: Heading
    confidence: Probability
    sensor_mode: Literal["active", "passive"]
    relay_available: bool


class TargetLocalContact(AdversaryStrictModel):
    """Target-owned contact episode state; no simulator coordinates."""

    platform_id: str = Field(min_length=1)
    platform_kind: TargetPlatformKind
    first_seen_s: int = Field(ge=0)
    last_seen_s: int = Field(ge=0)
    estimated_range_m: NonNegativeFloat
    relative_bearing_rad: Heading
    threat_level: ThreatLevel
    status: Literal["active", "lost"]

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> TargetLocalContact:
        if self.last_seen_s < self.first_seen_s:
            raise ValueError("last_seen_s must not precede first_seen_s")
        return self


class PlatformThreatSummary(AdversaryStrictModel):
    """A bounded threat summary derived from the target's observations."""

    platform_id: str = Field(min_length=1)
    platform_kind: TargetPlatformKind
    observed_at_s: int = Field(ge=0)
    threat_level: ThreatLevel
    estimated_range_m: NonNegativeFloat
    relative_bearing_rad: Heading
    passive_detection_risk: Probability
    active_ping_risk: Probability
    relay_detection_risk: Probability
    surface_relay_available: bool


class AdversaryTrigger(AdversaryStrictModel):
    """A target-visible event that may cause a new escape decision."""

    trigger_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    severity: Literal["strategic", "tactical", "informational"]
    summary: str = Field(min_length=1, max_length=240)


class AdversaryMissionState(AdversaryStrictModel):
    """Target-private mission orders and navigation progress."""

    target_id: str = Field(min_length=1)
    task_region_id: str = Field(min_length=1)
    task_region_polygon_xy: tuple[Point2D, ...] = Field(min_length=3)
    mission_route_xy: tuple[Point2D, ...] = Field(min_length=2)
    escape_regions: Mapping[str, tuple[Point2D, ...]]
    current_intent: AdversaryIntent = "continue_mission"
    current_route_index: int = Field(ge=0)
    local_contact_ids: tuple[str, ...] = ()
    last_decision_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def route_index_is_valid(self) -> AdversaryMissionState:
        if self.current_route_index >= len(self.mission_route_xy):
            raise ValueError("current_route_index must reference the mission route")
        if not self.escape_regions:
            raise ValueError("at least one configured escape region is required")
        if any(not region_id for region_id in self.escape_regions):
            raise ValueError("escape region IDs must be non-empty")
        return self


class AdversaryOperationalSummary(AdversaryStrictModel):
    """Truth-safe target brain output for the operator frame."""

    target_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    detection_range_m: PositiveFinite
    detected_platform_ids: tuple[str, ...] = ()
    trigger_event_ids: tuple[str, ...] = ()
    decision_id: str | None = Field(default=None, min_length=1)
    maneuver: Maneuver | None = None
    intent: EscapeIntent | None = None
    segment: str | None = Field(default=None, min_length=1)
    speed: NonNegativeFloat | None = None
    heading: Heading | None = None
    decoy_count: NonNegativeInt = 0
    confidence: Probability | None = None
    rationale: str | None = Field(default=None, min_length=1, max_length=2000)
    communications_discipline: CommunicationsDiscipline | None = None
    decision_status: DecisionOutcome = "unknown"
    escape_region_id: str | None = None
    decision_source: Literal["llm", "mission_route", "boundary_avoidance", "safe_hold"] | None = None
    guidance_id: str | None = None
    guidance_waypoint_xy: Point2D | None = None
    guidance_speed_mps: NonNegativeFloat | None = None
    guidance_heading_rad: Heading | None = None
    guidance_valid_until_s: int | None = Field(default=None, ge=0)
    degraded_reason: str | None = Field(default=None, min_length=1, max_length=500)


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
    confidence: Probability | None = None
    rationale: str | None = Field(default=None, min_length=1, max_length=2000)
    communications_discipline: CommunicationsDiscipline | None = None
    trigger_event_ids: tuple[str, ...] = ()


class AdversaryKinematicLimits(AdversaryStrictModel):
    """Current maneuver limits supplied by the target's own platform model."""

    max_speed_mps: NonNegativeFloat
    max_turn_rate_rad_s: NonNegativeFloat
    max_acceleration_mps2: PositiveFinite = 0.25
    max_deceleration_mps2: PositiveFinite = 0.25
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
    mission_state: AdversaryMissionState
    belief: AdversaryBelief
    local_contacts: tuple[TargetLocalContact, ...] = ()
    observations: tuple[AdversaryObservation, ...] = ()
    platform_threats: tuple[PlatformThreatSummary, ...] = ()
    trigger_events: tuple[AdversaryTrigger, ...] = ()
    communications_acoustic_exposure: CommunicationsAcousticExposure
    decision_history: tuple[AdversaryDecisionRecord, ...] = ()
    kinematic_limits: AdversaryKinematicLimits
    operating_boundary: AdversaryOperatingBoundary

    @model_validator(mode="before")
    @classmethod
    def fill_legacy_mission_state(cls, value: object) -> object:
        """Keep pre-mission replay fixtures loadable without weakening live input."""
        if not isinstance(value, Mapping) or "mission_state" in value:
            return value
        target_id = value.get("target_id")
        belief = value.get("belief")
        boundary = value.get("operating_boundary")
        if not isinstance(target_id, str) or belief is None or boundary is None:
            return value
        if isinstance(belief, Mapping):
            position = belief.get("estimated_position_xy", (0.0, 0.0))
        else:
            position = getattr(belief, "estimated_position_xy", (0.0, 0.0))
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            position = (0.0, 0.0)
        if isinstance(boundary, Mapping):
            min_x = boundary.get("min_x", 0.0)
            max_x = boundary.get("max_x", 5000.0)
            min_y = boundary.get("min_y", 0.0)
            max_y = boundary.get("max_y", 5000.0)
        else:
            min_x = getattr(boundary, "min_x", 0.0)
            max_x = getattr(boundary, "max_x", 5000.0)
            min_y = getattr(boundary, "min_y", 0.0)
            max_y = getattr(boundary, "max_y", 5000.0)
        min_x, max_x, min_y, max_y = (
            float(min_x),
            float(max_x),
            float(min_y),
            float(max_y),
        )
        route_end = (
            min(max_x, max(min_x, float(position[0]) + 1000.0)),
            min(max_y, max(min_y, float(position[1]))),
        )
        enriched = dict(value)
        enriched["mission_state"] = AdversaryMissionState(
            target_id=target_id,
            task_region_id="legacy_task",
            task_region_polygon_xy=(
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ),
            mission_route_xy=(tuple(position), route_end),
            escape_regions={
                "legacy_escape": (
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                )
            },
            current_intent="continue_mission",
            current_route_index=0,
        )
        return enriched

    @model_validator(mode="after")
    def target_and_evidence_are_consistent(self) -> AdversaryEscapeInput:
        if self.belief.target_id != self.target_id:
            raise ValueError("belief.target_id must match target_id")
        if self.mission_state.target_id != self.target_id:
            raise ValueError("mission_state.target_id must match target_id")
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
    trigger_event_ids: tuple[str, ...] = ()

    @field_validator("rationale")
    @classmethod
    def rationale_must_use_available_evidence(cls, value: str) -> str:
        if _contains_private_state_marker(value):
            raise ValueError("rationale cannot claim unavailable simulator state")
        return value


class AdversaryIntentDecision(AdversaryStrictModel):
    """High-level target decision; physical guidance is deterministic."""

    decision_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    intent: AdversaryIntent
    escape_region_id: str | None = Field(default=None, min_length=1)
    confidence: Probability
    rationale: str = Field(min_length=1, max_length=1200)
    trigger_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def escape_region_matches_intent(self) -> AdversaryIntentDecision:
        if self.intent == "escape_to_region" and self.escape_region_id is None:
            raise ValueError("escape_to_region requires escape_region_id")
        if self.intent != "escape_to_region" and self.escape_region_id is not None:
            raise ValueError("escape_region_id is only valid for escape_to_region")
        if _contains_private_state_marker(self.rationale):
            raise ValueError("rationale cannot claim unavailable simulator state")
        return self


__all__ = [
    "AdversaryIntent",
    "AdversaryIntentDecision",
    "AdversaryMissionState",
    "AdversaryOperationalSummary",
    "AdversaryBelief",
    "AdversaryDecisionRecord",
    "AdversaryEscapeDecision",
    "AdversaryEscapeInput",
    "AdversaryKinematicLimits",
    "LocalPlatformDetection",
    "TargetLocalContact",
    "AdversaryObservation",
    "AdversaryOperatingBoundary",
    "AdversaryTrigger",
    "CommunicationsAcousticExposure",
    "PlatformThreatSummary",
    "TargetPlatformKind",
]
