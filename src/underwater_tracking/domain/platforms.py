from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value
from math import pi
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from underwater_tracking.domain.models import StrictModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FiniteCoordinate = Annotated[float, Field(allow_inf_nan=False)]
PositionXY = tuple[FiniteCoordinate, FiniteCoordinate]


class PlatformModel(StrictModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class PlatformKind(StrEnum):
    USV = "usv"
    UUV = "uuv"


class MotionLimits(PlatformModel):
    min_speed_mps: NonNegativeFloat = 0.0
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_deceleration_mps2: PositiveFloat = 0.25
    max_turn_rate_rad_s: PositiveFloat

    @model_validator(mode="after")
    def speed_range_is_valid(self) -> MotionLimits:
        if self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("min_speed_mps must be below max_speed_mps")
        return self


class SubmarineMotionLimits(PlatformModel):
    """Bounded horizontal and vertical limits for a submarine target."""

    min_speed_mps: NonNegativeFloat = 0.0
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_deceleration_mps2: PositiveFloat = 0.25
    max_turn_rate_rad_s: PositiveFloat
    min_depth_m: NonNegativeFloat = 0.0
    max_depth_m: PositiveFloat = 1000.0
    max_vertical_speed_mps: PositiveFloat = 2.0
    max_vertical_acceleration_mps2: PositiveFloat = 0.2
    max_pitch_rad: float = Field(gt=0, le=pi / 2, allow_inf_nan=False)

    @model_validator(mode="after")
    def ranges_are_valid(self) -> SubmarineMotionLimits:
        if self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("min_speed_mps must be below max_speed_mps")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be below max_depth_m")
        return self


class SubmarineMotionState(PlatformModel):
    position_xy: PositionXY
    depth_m: NonNegativeFloat
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: NonNegativeFloat
    vertical_speed_mps: float = Field(allow_inf_nan=False)


class SubmarineMotionCommand(PlatformModel):
    desired_heading_rad: float = Field(allow_inf_nan=False)
    desired_speed_mps: NonNegativeFloat
    desired_depth_m: NonNegativeFloat


class SonarCapability(PlatformModel):
    passive_range_m: PositiveFloat
    passive_bearing_variance_rad2: PositiveFloat
    active_source_range_m: PositiveFloat
    active_receive_range_m: PositiveFloat
    active_range_sigma_m: PositiveFloat
    active_bearing_sigma_rad: PositiveFloat
    active_capable: bool
    ping_cooldown_s: int = Field(gt=0)
    ping_energy_cost_fraction: UnitFloat
    clutter_sensitivity: UnitFloat
    exposure_cost: UnitFloat


class CommunicationCapability(PlatformModel):
    surface_range_m: PositiveFloat
    acoustic_range_m: PositiveFloat


class PlatformCapability(PlatformModel):
    kind: PlatformKind
    motion: MotionLimits
    sonar: SonarCapability
    communications: CommunicationCapability


class MobilePlatformState(PlatformModel):
    platform_id: str = Field(min_length=1)
    platform_index: int = Field(ge=0)
    position_xy: PositionXY
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: NonNegativeFloat
    energy_fraction: UnitFloat
    deployment_state: Literal["onboard", "deployed", "returning", "failed"]
    capability: PlatformCapability
    group_id: str | None = None
    sensor_mode: Literal["passive", "active"] = "passive"


class USVPlatformState(MobilePlatformState):
    distance_to_carrier_m: NonNegativeFloat

    @model_validator(mode="after")
    def kind_is_usv(self) -> USVPlatformState:
        if self.capability.kind is not PlatformKind.USV:
            raise ValueError("USV state requires a USV capability")
        return self


class UUVPlatformState(MobilePlatformState):
    physically_exposed: bool = True
    is_group_leader: bool = False
    master_connected: bool = False
    group_instance_id: str | None = None
    deployment_revision: int | None = Field(default=None, ge=1)
    group_lifecycle: str | None = None

    @model_validator(mode="after")
    def kind_is_uuv(self) -> UUVPlatformState:
        if self.capability.kind is not PlatformKind.UUV:
            raise ValueError("UUV state requires a UUV capability")
        return self


class CarrierPlatformState(PlatformModel):
    carrier_id: str = Field(min_length=1)
    role: Literal["carrier", "mother_ship"] = "carrier"
    position_xy: PositionXY
    heading_rad: float = Field(allow_inf_nan=False)
    speed_mps: NonNegativeFloat
    support_radius_m: PositiveFloat
    onboard_platform_ids: tuple[str, ...]
    deployed_platform_ids: tuple[str, ...]
    returning_platform_ids: tuple[str, ...]

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierPlatformState:
        relationships = (
            self.onboard_platform_ids,
            self.deployed_platform_ids,
            self.returning_platform_ids,
        )
        if any(len(values) != len(set(values)) for values in relationships):
            raise ValueError("carrier platform relationship lists must be unique and disjoint")
        groups = tuple(set(values) for values in relationships)
        if any(
            left & right
            for index, left in enumerate(groups)
            for right in groups[index + 1 :]
        ):
            raise ValueError("carrier platform relationship lists must be unique and disjoint")
        return self


class PlatformRoster(PlatformModel):
    usvs: tuple[USVPlatformState, ...]
    uuvs: tuple[UUVPlatformState, ...]

    @model_validator(mode="after")
    def platform_ids_are_unique(self) -> PlatformRoster:
        ids = [platform.platform_id for platform in (*self.usvs, *self.uuvs)]
        if len(ids) != len(set(ids)):
            raise ValueError("platform IDs must be unique")
        for kind, platforms in (("USV", self.usvs), ("UUV", self.uuvs)):
            indices = [platform.platform_index for platform in platforms]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{kind} platform indices must be unique")
        return self


class CommunicationLink(PlatformModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    medium: Literal["surface", "acoustic"]
    distance_m: NonNegativeFloat


class PlatformSnapshot(PlatformModel):
    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    carrier: CarrierPlatformState
    carriers: tuple[CarrierPlatformState, ...] = ()
    roster: PlatformRoster
    communication_links: tuple[CommunicationLink, ...]

    @model_validator(mode="after")
    def carrier_ids_are_unique(self) -> PlatformSnapshot:
        carriers = self.carriers or (self.carrier,)
        ids = [carrier.carrier_id for carrier in carriers]
        if len(ids) != len(set(ids)):
            raise ValueError("carrier IDs must be unique")
        if self.carrier.carrier_id not in set(ids):
            raise ValueError("primary carrier must be present in carriers")
        listed: set[str] = set()
        for carrier in carriers:
            relationships = (
                *carrier.onboard_platform_ids,
                *carrier.deployed_platform_ids,
                *carrier.returning_platform_ids,
            )
            if listed.intersection(relationships):
                raise ValueError(
                    "carrier platform relationship lists must be disjoint across carriers"
                )
            listed.update(relationships)
        return self
