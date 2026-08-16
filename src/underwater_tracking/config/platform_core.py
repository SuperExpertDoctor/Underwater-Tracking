from __future__ import annotations

from math import hypot
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from underwater_tracking.domain.platforms import PlatformKind

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
CoordinateXY = tuple[FiniteFloat, FiniteFloat]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlatformCoreFiles(StrictConfig):
    environment: str = Field(min_length=1)
    platforms: str = Field(min_length=1)
    sensors: str = Field(min_length=1)
    communications: str = Field(min_length=1)


class RegionConfig(StrictConfig):
    region_id: str = Field(min_length=1)
    polygon_xy: tuple[CoordinateXY, ...] = Field(min_length=3)


class InitialPlatformConfig(StrictConfig):
    platform_id: str = Field(min_length=1)
    platform_index: int = Field(ge=0)
    kind: PlatformKind
    position_xy: CoordinateXY
    heading_rad: FiniteFloat
    energy_fraction: UnitFloat
    deployment_state: Literal["onboard", "deployed"]
    motion_profile: str = Field(min_length=1)
    sensor_profile: str = Field(min_length=1)
    communication_profile: str = Field(min_length=1)


class CarrierInitialConfig(StrictConfig):
    platform_id: str = Field(min_length=1)
    position_xy: CoordinateXY
    heading_rad: FiniteFloat
    speed_mps: NonNegativeFloat
    support_radius_m: PositiveFloat
    patrol_route_xy: tuple[CoordinateXY, ...] = Field(min_length=2)


class SubmarineInitialConfig(StrictConfig):
    target_id: str = Field(min_length=1)
    position_xy: CoordinateXY
    heading_rad: FiniteFloat
    speed_mps: PositiveFloat
    motion_profile: str = Field(min_length=1)
    task_region_id: str = Field(min_length=1)
    escape_region_ids: tuple[str, ...] = Field(min_length=1)


class EnvironmentConfig(StrictConfig):
    map_bounds_xy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    carrier: CarrierInitialConfig
    usvs: tuple[InitialPlatformConfig, ...]
    uuvs: tuple[InitialPlatformConfig, ...]
    submarines: tuple[SubmarineInitialConfig, ...]
    decoys: tuple[str, ...]
    task_regions: tuple[RegionConfig, ...]
    escape_regions: tuple[RegionConfig, ...]

    @model_validator(mode="after")
    def validate_roster(self) -> EnvironmentConfig:
        if len(self.usvs) != 4 or len(self.uuvs) != 12 or len(self.submarines) != 1:
            raise ValueError("explicit scenario requires 4 USVs, 12 UUVs, and 1 submarine")
        if self.decoys:
            raise ValueError("explicit single-target scenario does not allow decoys")
        platforms = (*self.usvs, *self.uuvs)
        ids = [self.carrier.platform_id, *(platform.platform_id for platform in platforms)]
        if len(ids) != len(set(ids)):
            raise ValueError("platform IDs must be unique")
        if any(platform.kind is not PlatformKind.USV for platform in self.usvs):
            raise ValueError("usvs must contain only USV entries")
        if any(platform.kind is not PlatformKind.UUV for platform in self.uuvs):
            raise ValueError("uuvs must contain only UUV entries")
        for usv in self.usvs:
            distance_to_carrier = hypot(
                usv.position_xy[0] - self.carrier.position_xy[0],
                usv.position_xy[1] - self.carrier.position_xy[1],
            )
            if distance_to_carrier > self.carrier.support_radius_m:
                raise ValueError(
                    f"USV {usv.platform_id!r} starts outside carrier support radius"
                )
        return self


class MotionProfileConfig(StrictConfig):
    max_speed_mps: PositiveFloat
    max_acceleration_mps2: PositiveFloat
    max_turn_rate_rad_s: PositiveFloat
    transit_energy_per_m: PositiveFloat
    hotel_energy_per_s: PositiveFloat


class PlatformCatalogConfig(StrictConfig):
    motion_profiles: dict[str, MotionProfileConfig]


class SensorProfileConfig(StrictConfig):
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


class SensorCatalogConfig(StrictConfig):
    profiles: dict[str, SensorProfileConfig]


class CommunicationProfileConfig(StrictConfig):
    surface_range_m: PositiveFloat
    acoustic_range_m: PositiveFloat


class CommunicationsConfig(StrictConfig):
    profiles: dict[str, CommunicationProfileConfig]
