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
    home_carrier_id: str | None = None
    position_xy: CoordinateXY
    heading_rad: FiniteFloat
    energy_fraction: UnitFloat
    deployment_state: Literal["onboard", "deployed"]
    motion_profile: str = Field(min_length=1)
    sensor_profile: str = Field(min_length=1)
    communication_profile: str = Field(min_length=1)


class CarrierInitialConfig(StrictConfig):
    platform_id: str = Field(min_length=1)
    role: Literal["carrier", "mother_ship"] = "carrier"
    formation_slot_offset_xy: CoordinateXY = (0.0, 0.0)
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
    detection_range_m: PositiveFloat = 1200.0
    motion_profile: str = Field(min_length=1)
    task_region_id: str = Field(min_length=1)
    escape_region_ids: tuple[str, ...] = Field(min_length=1)
    mission_route_xy: tuple[CoordinateXY, ...] = Field(min_length=2)


class EnvironmentConfig(StrictConfig):
    map_bounds_xy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    carrier: CarrierInitialConfig
    carriers: tuple[CarrierInitialConfig, ...] = ()
    rendezvous_tolerance_m: PositiveFloat = 250.0
    uuv_only: bool = False
    usvs: tuple[InitialPlatformConfig, ...]
    uuvs: tuple[InitialPlatformConfig, ...]
    submarines: tuple[SubmarineInitialConfig, ...]
    decoys: tuple[str, ...]
    task_regions: tuple[RegionConfig, ...]
    escape_regions: tuple[RegionConfig, ...]
    navigation_exclusion_regions: tuple[RegionConfig, ...] = ()

    @model_validator(mode="after")
    def validate_roster(self) -> EnvironmentConfig:
        min_x, max_x, min_y, max_y = self.map_bounds_xy
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("map_bounds_xy must define a positive area")
        if self.uuv_only:
            if self.usvs:
                raise ValueError("uuv-only environment must not contain USVs")
            if len(self.carriers) != 3:
                raise ValueError(
                    "uuv-only environment requires one carrier and three mother ships"
                )
            if len(self.uuvs) != 12 or len(self.submarines) != 1:
                raise ValueError("uuv-only scenario requires 12 UUVs and 1 submarine")
        elif len(self.usvs) != 4 or len(self.uuvs) != 12 or len(self.submarines) != 1:
            raise ValueError("explicit scenario requires 4 USVs, 12 UUVs, and 1 submarine")
        if self.decoys:
            raise ValueError("explicit single-target scenario does not allow decoys")
        carriers = (self.carrier, *self.carriers)
        if self.uuv_only:
            if self.carrier.platform_id != "carrier_01":
                raise ValueError("uuv-only environment requires one carrier and three mother ships")
            expected_mother_ids = ("carrier_02", "carrier_03", "carrier_04")
            if tuple(carrier.platform_id for carrier in self.carriers) != expected_mother_ids:
                raise ValueError("uuv-only environment requires one carrier and three mother ships")
            if self.carrier.role != "carrier" or any(
                carrier.role != "mother_ship" for carrier in self.carriers
            ):
                raise ValueError("uuv-only environment requires one carrier and three mother ships")
            slot_offsets = tuple(carrier.formation_slot_offset_xy for carrier in carriers)
            if len(slot_offsets) != len(set(slot_offsets)):
                raise ValueError("uuv-only carrier formation slots must be unique")
            mother_ids = set(expected_mother_ids)
            owner_ids = [uuv.home_carrier_id for uuv in self.uuvs]
            if any(owner_id is None for owner_id in owner_ids):
                raise ValueError("uuv-only UUV home_carrier_id is required")
            if any(owner_id == self.carrier.platform_id for owner_id in owner_ids):
                raise ValueError("carrier cannot own UUVs")
            if any(owner_id not in mother_ids for owner_id in owner_ids):
                raise ValueError("unknown UUV home carrier")
            if any(owner_ids.count(mother_id) != 4 for mother_id in expected_mother_ids):
                raise ValueError("uuv-only carrier inventory requires exactly four UUVs per mother ship")
            if any(uuv.deployment_state != "onboard" for uuv in self.uuvs):
                raise ValueError("uuv-only UUVs must start onboard")
            nearest_mother_distance = min(
                hypot(
                    self.submarines[0].position_xy[0] - carrier.position_xy[0],
                    self.submarines[0].position_xy[1] - carrier.position_xy[1],
                )
                for carrier in self.carriers
            )
            if not 2500.0 <= nearest_mother_distance <= 4000.0:
                raise ValueError(
                    "uuv-only target must start 2500-4000 m from the nearest mother ship"
                )
        for carrier in carriers:
            route_segments = tuple(
                hypot(end[0] - start[0], end[1] - start[1])
                for start, end in zip(
                    carrier.patrol_route_xy,
                    (*carrier.patrol_route_xy[1:], carrier.patrol_route_xy[0]),
                )
            )
            if not any(length > 0.0 for length in route_segments):
                raise ValueError(
                    "carrier patrol_route_xy must contain at least one valid non-zero segment"
                )
            if any(length == 0.0 for length in route_segments):
                raise ValueError(
                    "carrier patrol_route_xy cannot contain zero-length consecutive segments"
                )
        platforms = (*self.usvs, *self.uuvs)
        ids = [carrier.platform_id for carrier in carriers]
        ids.extend(platform.platform_id for platform in platforms)
        if len(ids) != len(set(ids)):
            raise ValueError("platform IDs must be unique")
        if any(platform.kind is not PlatformKind.USV for platform in self.usvs):
            raise ValueError("usvs must contain only USV entries")
        if any(platform.kind is not PlatformKind.UUV for platform in self.uuvs):
            raise ValueError("uuvs must contain only UUV entries")
        for kind, platforms in (("USV", self.usvs), ("UUV", self.uuvs)):
            indices = [platform.platform_index for platform in platforms]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{kind} platform indices must be unique")
        region_ids = [
            *(region.region_id for region in self.task_regions),
            *(region.region_id for region in self.escape_regions),
            *(region.region_id for region in self.navigation_exclusion_regions),
        ]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")
        for region in (
            *self.task_regions,
            *self.escape_regions,
            *self.navigation_exclusion_regions,
        ):
            if any(
                not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)
                for point in region.polygon_xy
            ):
                raise ValueError(f"region {region.region_id!r} is outside map bounds")
        task_region_ids = {region.region_id for region in self.task_regions}
        escape_region_ids = {region.region_id for region in self.escape_regions}
        for submarine in self.submarines:
            if submarine.task_region_id not in task_region_ids:
                raise ValueError(f"unknown task region {submarine.task_region_id!r}")
            if len(submarine.escape_region_ids) != len(set(submarine.escape_region_ids)):
                raise ValueError(
                    f"duplicate escape region IDs for submarine {submarine.target_id!r}"
                )
            for escape_region_id in submarine.escape_region_ids:
                if escape_region_id not in escape_region_ids:
                    raise ValueError(f"unknown escape region {escape_region_id!r}")
            if any(
                not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)
                for point in submarine.mission_route_xy
            ):
                raise ValueError(
                    f"mission route for {submarine.target_id!r} is outside map bounds"
                )
            for start, end in zip(
                submarine.mission_route_xy,
                submarine.mission_route_xy[1:],
            ):
                if start == end:
                    raise ValueError(
                        f"mission route for {submarine.target_id!r} contains a zero-length segment"
                    )
                if any(
                    _segment_intersects_polygon(start, end, region.polygon_xy)
                    for region in self.navigation_exclusion_regions
                ):
                    raise ValueError(
                        f"mission route for {submarine.target_id!r} intersects navigation exclusion"
                    )
        if not self.uuv_only:
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
    min_speed_mps: NonNegativeFloat = 0.0
    max_deceleration_mps2: PositiveFloat = 0.25
    max_turn_rate_rad_s: PositiveFloat
    transit_energy_per_m: PositiveFloat
    hotel_energy_per_s: PositiveFloat

    @model_validator(mode="after")
    def speed_range_is_valid(self) -> MotionProfileConfig:
        if self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("min_speed_mps must be below max_speed_mps")
        return self


def _orientation(a: CoordinateXY, b: CoordinateXY, c: CoordinateXY) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: CoordinateXY, b: CoordinateXY, point: CoordinateXY) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= point[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= point[1] <= max(a[1], b[1]) + 1e-9
    )


def _segments_intersect(
    first_start: CoordinateXY,
    first_end: CoordinateXY,
    second_start: CoordinateXY,
    second_end: CoordinateXY,
) -> bool:
    first = _orientation(first_start, first_end, second_start)
    second = _orientation(first_start, first_end, second_end)
    third = _orientation(second_start, second_end, first_start)
    fourth = _orientation(second_start, second_end, first_end)
    if ((first > 0.0 > second) or (first < 0.0 < second)) and (
        (third > 0.0 > fourth) or (third < 0.0 < fourth)
    ):
        return True
    return (
        abs(first) <= 1e-9
        and _on_segment(first_start, first_end, second_start)
        or abs(second) <= 1e-9
        and _on_segment(first_start, first_end, second_end)
        or abs(third) <= 1e-9
        and _on_segment(second_start, second_end, first_start)
        or abs(fourth) <= 1e-9
        and _on_segment(second_start, second_end, first_end)
    )


def _point_in_polygon(point: CoordinateXY, polygon: tuple[CoordinateXY, ...]) -> bool:
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if _on_segment(start, end, point) and abs(_orientation(start, end, point)) <= 1e-9:
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = (end[0] - start[0]) * (point[1] - start[1]) / (
                end[1] - start[1]
            ) + start[0]
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _segment_intersects_polygon(
    start: CoordinateXY,
    end: CoordinateXY,
    polygon: tuple[CoordinateXY, ...],
) -> bool:
    return _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon) or any(
        _segments_intersect(start, end, edge_start, edge_end)
        for edge_start, edge_end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def initial_route_join_distance(
    submarine: SubmarineInitialConfig,
    max_turn_rate_rad_s: float,
) -> float:
    """Return the distance from the target to its first route segment."""
    start, end = submarine.mission_route_xy[:2]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return float("inf")
    projection = (
        (submarine.position_xy[0] - start[0]) * dx
        + (submarine.position_xy[1] - start[1]) * dy
    ) / length_sq
    projection = min(1.0, max(0.0, projection))
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return hypot(
        submarine.position_xy[0] - nearest[0],
        submarine.position_xy[1] - nearest[1],
    )


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
