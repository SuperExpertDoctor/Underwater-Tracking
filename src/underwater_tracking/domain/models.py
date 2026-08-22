# src/underwater_tracking/domain/models.py
from __future__ import annotations
from copy import deepcopy
from collections.abc import Iterator, Mapping
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value
from math import isfinite, pi
import re
from typing import Annotated, Any, Literal, cast
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from underwater_tracking.domain.relationships import (
    expected_carrier_status,
    normalize_legacy_carrier_relationships,
    normalize_legacy_uuv_deployment_state,
)
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.adversary_models import AdversaryOperationalSummary


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


from underwater_tracking.domain.platforms import PlatformKind, PlatformSnapshot  # noqa: E402


class EventLevel(StrEnum):
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    INFORMATIONAL = "informational"


class EventAudience(StrEnum):
    """Subsystems allowed to consume one runtime event."""

    BLUE_PLANNING = "blue_planning"
    ADVERSARY_PRIVATE = "adversary_private"
    OPERATOR_AUDIT = "operator_audit"
    MEMORY_SOURCE = "memory_source"


DEFAULT_EVENT_AUDIENCES = frozenset(
    {
        EventAudience.BLUE_PLANNING,
        EventAudience.OPERATOR_AUDIT,
        EventAudience.MEMORY_SOURCE,
    }
)


class UUVStatus(StrEnum):
    AVAILABLE = "available"
    TRACKING = "tracking"
    RETURNING = "returning"
    FAILED = "failed"


class CarrierStatus(StrEnum):
    STANDBY = "standby"
    TRANSIT = "transit"
    DEPLOYING = "deploying"
    RECOVERING = "recovering"


class DeploymentState(StrEnum):
    ONBOARD = "onboard"
    DEPLOYED = "deployed"
    RETURNING = "returning"
    FAILED = "failed"


class ContactClassification(StrEnum):
    UNVERIFIED = "unverified"
    SUBMARINE = "submarine"
    DECOY = "decoy"


class IntelligenceSource(StrEnum):
    TECHNICAL_RECONNAISSANCE = "technical_reconnaissance"
    SIGINT = "sigint"
    ELINT = "elint"
    HUMINT = "humint"
    SONAR = "sonar"


_FinitePositive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
_FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_UnitInterval = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
_NonEmptyIdentifier = Annotated[str, Field(min_length=1)]
_FORBIDDEN_INTELLIGENCE_KEYS = frozenset(
    {
        "truth",
        "true_position",
        "true_targets",
        "target_truth",
        "ground_truth",
        "evaluation",
        "evaluation_frame",
        "evaluation_only",
        "evaluation_only_state",
        "evaluation_state",
    }
)
_FORBIDDEN_SUMMARY_PATTERNS = (
    re.compile(r"\bground[\s_-]*truth\b", re.IGNORECASE),
    re.compile(r"\btrue[\s_-]*(?:position|targets?|state|course|intent|location)\b", re.IGNORECASE),
    re.compile(r"\bactual[\s_-]*(?:position|targets?|state|course|intent|location)\b", re.IGNORECASE),
    re.compile(r"\b(?:evaluation|eval)[\s_-]*(?:state|frame|only|result|target|metrics?|label|score)\b", re.IGNORECASE),
    re.compile(r"[\"'](?:truth|ground_truth|true_position|true_targets|evaluation_state)[\"']\s*[:=]", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])(?:truth|groundtruth|evaluation|evaluation_result)\s*[:=]", re.IGNORECASE),
    re.compile(r"(?:\u771f\u503c|\u771f\u5b9e(?:\u4f4d\u7f6e|\u76ee\u6807|\u72b6\u6001|\u822a\u8ff9|\u610f\u56fe)|\u5b9e\u9645(?:\u4f4d\u7f6e|\u76ee\u6807|\u72b6\u6001|\u822a\u8ff9|\u610f\u56fe)|\u8bc4\u4f30(?:\u7ed3\u679c|\u72b6\u6001|\u6307\u6807|\u5206\u6570|\u6807\u7b7e))"),
)


class TargetSearchPrior(StrictModel):
    """Public search intelligence; never a sensor-derived target estimate."""

    prior_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    source: IntelligenceSource
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(gt=0)
    center_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_geometry_and_validity(self) -> "TargetSearchPrior":
        if self.valid_until_s <= self.issued_at_s:
            raise ValueError("valid_until_s must be after issued_at_s")
        if not all(isfinite(value) for value in self.center_xy):
            raise ValueError("target prior center_xy must contain finite values")
        p00, p01 = self.covariance_xy[0]
        p10, p11 = self.covariance_xy[1]
        if not all(isfinite(value) for value in (p00, p01, p10, p11)):
            raise ValueError("target prior covariance_xy must contain finite values")
        if p01 != p10:
            raise ValueError("target prior covariance_xy must be symmetric")
        if p00 <= 0 or p11 <= 0 or p00 * p11 - p01 * p10 <= 0:
            raise ValueError("target prior covariance_xy must be positive definite")
        return self


class _FrozenMapping(Mapping[str, Any]):
    """An immutable JSON mapping backed only by immutable item tuples."""

    _items: tuple[tuple[str, Any], ...]
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple((key, _freeze_json(child)) for key, child in value.items()),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("mapping is immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("mapping is immutable")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return bool(_thaw_json(self) == _thaw_json(other))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenMapping:
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        object.__setattr__(
            copied,
            "_items",
            tuple((deepcopy(key, memo), deepcopy(value, memo)) for key, value in self._items),
        )
        return copied


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


class SurveillanceCapability(StrictModel):
    """The sensing and maneuver limits available to one UUV."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passive_range_m: _FinitePositive = 4000.0
    active_range_m: _FinitePositive = 3000.0
    bearing_variance_rad2: _FinitePositive = 1e-2
    passive_sonar_available: bool = True
    active_sonar_available: bool = True
    max_speed_mps: _FinitePositive = 4.0
    max_turn_rate_rad_s: _FinitePositive = pi / 60.0
    endurance_s: _FinitePositive = 28_800.0
    availability: _UnitInterval = 1.0


class OperationalScheme(StrictModel):
    """A time-bounded, traceable set of deterministic tracking constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    target_priorities: Mapping[_NonEmptyIdentifier, _FiniteNonNegative] = Field(default_factory=dict)
    minimum_quality: Mapping[_NonEmptyIdentifier, _UnitInterval] = Field(default_factory=dict)
    valid_from_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    constraints: tuple[str, ...] = ()

    @field_validator("target_priorities", "minimum_quality", mode="after")
    @classmethod
    def mappings_are_immutable(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return cast(Mapping[str, float], _freeze_json(value))

    @field_serializer("target_priorities", "minimum_quality")
    def serialize_mappings(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    @model_validator(mode="after")
    def validity_interval_is_positive(self) -> OperationalScheme:
        if self.valid_until_s <= self.valid_from_s:
            raise ValueError("valid_until_s must be after valid_from_s")
        return self


class IntelligenceReport(StrictModel):
    """A source-attributed operational assessment with a finite lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(min_length=1)
    source: IntelligenceSource
    target_id: str = Field(min_length=1)
    confidence: _UnitInterval
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    content_summary: str | None = None
    assessment: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_validator("content_summary")
    @classmethod
    def content_summary_is_operational(cls, value: str | None) -> str | None:
        if value is not None and any(pattern.search(value) for pattern in _FORBIDDEN_SUMMARY_PATTERNS):
            raise ValueError("content_summary must not contain truth or evaluation payloads")
        return value

    @field_validator("assessment")
    @classmethod
    def assessment_is_safe_operational_json(
        cls, value: Mapping[str, JsonValue]
    ) -> Mapping[str, JsonValue]:
        _validate_intelligence_assessment(value)
        return cast(Mapping[str, JsonValue], _freeze_json(value))

    @field_serializer("assessment")
    def serialize_assessment(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], _thaw_json(value))

    @model_validator(mode="after")
    def expiry_is_after_issue_time(self) -> IntelligenceReport:
        if self.valid_until_s <= self.issued_at_s:
            raise ValueError("valid_until_s must be after issued_at_s")
        return self


def _validate_intelligence_assessment(value: Any, path: str = "assessment") -> None:
    """Reject non-finite JSON numbers and evaluation/truth data at the input boundary."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_intelligence_assessment(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.casefold() in _FORBIDDEN_INTELLIGENCE_KEYS:
                raise ValueError(f"{path}.{key} is not permitted in operational intelligence")
            _validate_intelligence_assessment(child, f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain only JSON values")


class Contact(StrictModel):
    """One operational sonar contact (spec 11.1 amendment, R5).

    The classification is the operational measurement produced by active
    pings; it is not truth (decoy truth stays truth-side only). Targets
    being tracked enter classified SUBMARINE (already dispatched); decoys
    enter UNVERIFIED and are pinged by the active-verification protocol.
    """

    contact_id: str
    sim_time_s: int = Field(ge=0)
    bearing_rays: tuple[BearingObservation, ...] = ()
    classification: ContactClassification = ContactClassification.UNVERIFIED
    classification_evidence: tuple[str, ...] = ()
    estimated_position_xy: tuple[float, float] | None = None


class BearingObservation(StrictModel):
    observation_id: str
    scenario_id: str
    sim_time_s: int = Field(ge=0)
    uuv_id: str
    target_id: str
    azimuth_rad: float
    variance_rad2: float = Field(gt=0)
    detection_confidence: float = Field(ge=0, le=1)
    is_false_alarm: bool = False

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi


class UUVState(StrictModel):
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    remaining_range_m: float = Field(default=0.0, ge=0)
    status: UUVStatus
    deployment_state: DeploymentState = DeploymentState.DEPLOYED
    physically_exposed: bool = True
    group_id: str | None = None
    sensor_mode: Literal["passive", "active"] = "passive"
    capability: SurveillanceCapability = Field(default_factory=SurveillanceCapability)
    reserved: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_deployment_state(cls, value: Any) -> Any:
        return normalize_legacy_uuv_deployment_state(value)

    @model_validator(mode="after")
    def status_matches_deployment_state(self) -> UUVState:
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.ONBOARD:
            raise ValueError("tracking status cannot be onboard")
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.FAILED:
            raise ValueError("tracking status cannot be failed")
        if self.status is UUVStatus.RETURNING and self.deployment_state is not DeploymentState.RETURNING:
            raise ValueError("returning status requires returning deployment_state")
        if self.status is UUVStatus.FAILED and self.deployment_state is not DeploymentState.FAILED:
            raise ValueError("failed status requires failed deployment_state")
        if self.deployment_state is DeploymentState.RETURNING and self.status is not UUVStatus.RETURNING:
            raise ValueError("returning deployment_state requires returning status")
        if self.deployment_state is DeploymentState.FAILED and self.status is not UUVStatus.FAILED:
            raise ValueError("failed deployment_state requires failed status")
        return self


class CarrierState(StrictModel):
    carrier_id: str
    role: Literal["carrier", "mother_ship"] = "carrier"
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    status: CarrierStatus = CarrierStatus.TRANSIT
    onboard_uuv_ids: tuple[str, ...] = ()
    deployed_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierState:
        raw_lists = (
            self.onboard_uuv_ids,
            self.deployed_uuv_ids,
            self.returning_uuv_ids,
        )
        if any(len(ids) != len(set(ids)) for ids in raw_lists):
            raise ValueError("carrier relationship lists must not contain duplicate IDs")
        lists = tuple(set(ids) for ids in raw_lists)
        if any(left & right for index, left in enumerate(lists) for right in lists[index + 1 :]):
            raise ValueError("carrier relationship lists must be disjoint")
        expected = expected_carrier_status(
            self.speed_mps, self.onboard_uuv_ids, self.deployed_uuv_ids, self.returning_uuv_ids
        )
        if str(self.status) != expected:
            if self.returning_uuv_ids:
                raise ValueError("returning UUVs require recovering status")
            if self.status is CarrierStatus.RECOVERING:
                raise ValueError("recovering status requires returning UUVs")
            if self.status is CarrierStatus.DEPLOYING:
                raise ValueError("deploying status requires onboard and deployed UUVs")
            if self.status is CarrierStatus.STANDBY:
                raise ValueError("standby status requires zero speed")
            if self.status is CarrierStatus.TRANSIT:
                raise ValueError("transit status requires movement")
        return self


class TargetBelief(StrictModel):
    target_id: str
    sim_time_s: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    model_probabilities: dict[str, float]
    source_observation_ids: tuple[str, ...] = ()
    fim_min_eigenvalue: float = 0.0
    fim_condition: float = float("inf")


class GroupQuality(StrictModel):
    instant: float = Field(ge=0, le=1)
    window_mean: float = Field(ge=0, le=1)
    ewma: float = Field(ge=0, le=1)
    components: dict[str, float]
    hard_guard_reasons: tuple[str, ...] = ()


class GroupReport(StrictModel):
    group_id: str
    target_id: str
    sim_time_s: int
    member_ids: tuple[str, ...]
    belief: TargetBelief
    quality: GroupQuality
    plan_revision: int
    event_types: tuple[str, ...] = ()


class ExecutionGroupState(StrictModel):
    """Waterborne scan membership, deliberately separate from target belief."""

    group_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    member_ids: tuple[str, ...] = Field(min_length=1)
    mode: Literal["active_scan", "passive_track", "returning"]

    @model_validator(mode="after")
    def members_are_unique(self) -> "ExecutionGroupState":
        if len(self.member_ids) != len(set(self.member_ids)):
            raise ValueError("execution group member IDs must be unique")
        return self


class RuntimeEvent(StrictModel):
    event_id: str
    scenario_id: str
    sim_time_s: int
    event_type: str
    entity_id: str | None = None
    level: EventLevel
    audiences: frozenset[EventAudience] = Field(
        default_factory=lambda: DEFAULT_EVENT_AUDIENCES
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_audience_contract(self) -> RuntimeEvent:
        if not self.audiences:
            raise ValueError("runtime event audiences must not be empty")
        if self.event_type == "target_mission_decision":
            required = {
                EventAudience.ADVERSARY_PRIVATE,
                EventAudience.OPERATOR_AUDIT,
                EventAudience.MEMORY_SOURCE,
            }
            if not required <= self.audiences:
                raise ValueError(
                    "target_mission_decision requires private, audit and memory audiences"
                )
            if EventAudience.BLUE_PLANNING in self.audiences:
                raise ValueError("target_mission_decision cannot enter blue planning")
        from underwater_tracking.domain.event_registry import (
            event_audiences,
            validate_event_payload,
        )

        try:
            registered_audiences = event_audiences(self.event_type)
        except ValueError:
            registered_audiences = None
        if registered_audiences is not None and self.audiences != registered_audiences:
            raise ValueError(
                f"runtime event audiences do not match the registry for {self.event_type!r}"
            )
        validate_event_payload(self.event_type, dict(self.payload))
        return self


class SituationSnapshot(StrictModel):
    scenario_id: str
    snapshot_revision: int
    sim_time_s: int
    uuvs: tuple[UUVState, ...]
    carrier: CarrierState | None = None
    carriers: tuple[CarrierState, ...] = ()
    group_reports: tuple[GroupReport, ...]
    execution_groups: tuple[ExecutionGroupState, ...] = ()
    pending_events: tuple[RuntimeEvent, ...]
    contacts: tuple[Contact, ...] = ()
    active_plan_id: str | None = None
    active_plan_revision: int | None = None
    operational_scheme: OperationalScheme | None = None
    intelligence_reports: tuple[IntelligenceReport, ...] = ()
    platform_snapshot: PlatformSnapshot | None = None
    platform_observations: tuple[PassiveSonarObservation, ...] = ()
    adversary_summaries: tuple[AdversaryOperationalSummary, ...] = ()
    target_search_priors: tuple[TargetSearchPrior, ...] = ()
    map_bounds_xy: tuple[float, float, float, float] | None = None
    uuv_resource_episodes: dict[str, int] = {}

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_carrier_relationships(cls, value: Any) -> Any:
        normalized = normalize_legacy_carrier_relationships(value)
        if not isinstance(normalized, Mapping):
            return normalized
        platform_snapshot = normalized.get("platform_snapshot")
        normalized_value: dict[str, Any] = dict(normalized)
        if isinstance(platform_snapshot, Mapping):
            normalized_value["platform_snapshot"] = _tupleize_platform_payload(
                platform_snapshot
            )
        adversary_summaries = normalized.get("adversary_summaries")
        if isinstance(adversary_summaries, (list, tuple)):
            normalized_value["adversary_summaries"] = _tupleize_platform_payload(
                adversary_summaries
            )
        if not isinstance(platform_snapshot, Mapping) and not isinstance(
            adversary_summaries, (list, tuple)
        ):
            return normalized
        return {
            **normalized,
            **normalized_value,
        }

    @model_validator(mode="after")
    def carrier_relationships_match_uuvs(self) -> SituationSnapshot:
        carriers = self.carriers or ((self.carrier,) if self.carrier is not None else ())
        if not carriers:
            return self
        uuvs_by_id = {uuv.uuv_id: uuv for uuv in self.uuvs}
        listed_ids: set[str] = set()
        for carrier in carriers:
            relationships = {
                DeploymentState.ONBOARD: carrier.onboard_uuv_ids,
                DeploymentState.DEPLOYED: carrier.deployed_uuv_ids,
                DeploymentState.RETURNING: carrier.returning_uuv_ids,
            }
            relationship_sets = tuple(set(ids) for ids in relationships.values())
            if any(len(ids) != len(set(ids)) for ids in relationships.values()):
                raise ValueError("carrier relationship lists must not contain duplicate IDs")
            if any(
                left & right
                for index, left in enumerate(relationship_sets)
                for right in relationship_sets[index + 1 :]
            ):
                raise ValueError("carrier relationship lists must be disjoint")
            carrier_listed_ids = {
                uuv_id for ids in relationships.values() for uuv_id in ids
            }
            if listed_ids & carrier_listed_ids:
                raise ValueError("carrier relationship lists must be disjoint across carriers")
            listed_ids.update(carrier_listed_ids)
            for expected_state, ids in relationships.items():
                for uuv_id in ids:
                    uuv = uuvs_by_id.get(uuv_id)
                    if uuv is None:
                        raise ValueError(f"carrier lists unknown UUV {uuv_id!r}")
                    if (
                        uuv.status is UUVStatus.RETURNING
                        and uuv.deployment_state is not DeploymentState.RETURNING
                    ) or (
                        uuv.status is UUVStatus.FAILED
                        and uuv.deployment_state is not DeploymentState.FAILED
                    ):
                        raise ValueError(f"uuv {uuv_id!r} status contradicts deployment_state")
                    if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                        raise ValueError(f"carrier lists must omit failed UUV {uuv_id!r}")
                    if uuv.deployment_state is not expected_state:
                        raise ValueError(
                            f"carrier list {expected_state.value!r} contains {uuv_id!r} "
                            f"with deployment_state {uuv.deployment_state.value!r}"
                        )
        for uuv in self.uuvs:
            if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                if uuv.uuv_id in listed_ids:
                    raise ValueError(f"carrier lists must omit failed UUV {uuv.uuv_id!r}")
                continue
            if uuv.uuv_id not in listed_ids:
                raise ValueError(f"carrier lists omit non-failed UUV {uuv.uuv_id!r}")
        return self

    @model_validator(mode="after")
    def map_bounds_are_ordered(self) -> SituationSnapshot:
        if self.map_bounds_xy is None:
            return self
        min_x, max_x, min_y, max_y = self.map_bounds_xy
        if not all(isfinite(value) for value in self.map_bounds_xy):
            raise ValueError("map_bounds_xy must contain finite values")
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("map_bounds_xy must have positive area")
        return self


def _tupleize_platform_payload(value: Any) -> Any:
    """Adapt JSON arrays to the strict tuple contract of PlatformSnapshot."""
    if isinstance(value, Mapping):
        return {
            key: (
                PlatformKind(child)
                if key == "kind" and isinstance(child, str)
                else _tupleize_platform_payload(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return tuple(_tupleize_platform_payload(child) for child in value)
    return value
