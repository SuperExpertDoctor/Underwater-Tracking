from math import inf, nan

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)


def capability(kind: PlatformKind) -> PlatformCapability:
    return PlatformCapability(
        kind=kind,
        motion=MotionLimits(
            max_speed_mps=6.0,
            max_acceleration_mps2=0.2,
            max_turn_rate_rad_s=0.03,
        ),
        sonar=SonarCapability(
            passive_range_m=5000.0,
            passive_bearing_variance_rad2=0.01,
            active_source_range_m=4000.0,
            active_receive_range_m=5000.0,
            active_range_sigma_m=12.0,
            active_bearing_sigma_rad=0.003,
            active_capable=True,
            ping_cooldown_s=30,
            ping_energy_cost_fraction=0.001,
            clutter_sensitivity=0.2,
            exposure_cost=0.4,
        ),
        communications=CommunicationCapability(
            surface_range_m=12000.0,
            acoustic_range_m=4500.0,
        ),
    )


def test_platform_snapshot_keeps_truth_out_of_public_contract() -> None:
    usv = USVPlatformState(
        platform_id="usv_00",
        platform_index=0,
        position_xy=(100.0, 0.0),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=0.9,
        deployment_state="deployed",
        capability=capability(PlatformKind.USV),
        distance_to_carrier_m=100.0,
    )
    uuv = UUVPlatformState(
        platform_id="uuv_00",
        platform_index=0,
        position_xy=(50.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.8,
        deployment_state="deployed",
        capability=capability(PlatformKind.UUV),
    )
    carrier = CarrierPlatformState(
        carrier_id="carrier_01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=3.0,
        support_radius_m=15000.0,
        onboard_platform_ids=(),
        deployed_platform_ids=("usv_00", "uuv_00"),
        returning_platform_ids=(),
    )
    snapshot = PlatformSnapshot(
        scenario_id="single-target-relay",
        sim_time_s=30,
        carrier=carrier,
        roster=PlatformRoster(usvs=(usv,), uuvs=(uuv,)),
        communication_links=(),
    )

    payload = snapshot.model_dump()
    assert payload["roster"]["usvs"][0]["platform_id"] == "usv_00"
    assert "truth" not in repr(payload).lower()
    assert "true_position" not in repr(payload).lower()


def test_carrier_rejects_duplicate_or_overlapping_relationships() -> None:
    with pytest.raises(ValidationError, match="unique and disjoint"):
        CarrierPlatformState(
            carrier_id="carrier_01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=3.0,
            support_radius_m=15000.0,
            onboard_platform_ids=("uuv_00", "uuv_00"),
            deployed_platform_ids=(),
            returning_platform_ids=(),
        )


def test_roster_rejects_duplicate_indices_within_platform_kind() -> None:
    first = UUVPlatformState(
        platform_id="uuv_00",
        platform_index=0,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=0.0,
        energy_fraction=1.0,
        deployment_state="onboard",
        capability=capability(PlatformKind.UUV),
    )
    second = first.model_copy(update={"platform_id": "uuv_01"})

    with pytest.raises(ValidationError, match="indices must be unique"):
        PlatformRoster(usvs=(), uuvs=(first, second))


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (MotionLimits, "max_speed_mps", inf),
        (SonarCapability, "passive_range_m", 0.0),
        (CommunicationCapability, "acoustic_range_m", -1.0),
    ],
)
def test_platform_capabilities_reject_non_finite_or_non_positive_values(
    model: type, field: str, value: float
) -> None:
    valid = {
        MotionLimits: {
            "max_speed_mps": 4.0,
            "max_acceleration_mps2": 0.1,
            "max_turn_rate_rad_s": 0.02,
        },
        SonarCapability: {
            "passive_range_m": 4000.0,
            "passive_bearing_variance_rad2": 0.01,
            "active_source_range_m": 3000.0,
            "active_receive_range_m": 4000.0,
            "active_range_sigma_m": 10.0,
            "active_bearing_sigma_rad": 0.003,
            "active_capable": True,
            "ping_cooldown_s": 30,
            "ping_energy_cost_fraction": 0.001,
            "clutter_sensitivity": 0.2,
            "exposure_cost": 0.3,
        },
        CommunicationCapability: {
            "surface_range_m": 10000.0,
            "acoustic_range_m": 4000.0,
        },
    }[model]
    with pytest.raises(ValidationError):
        model.model_validate({**valid, field: value})


@pytest.mark.parametrize("position", [(nan, 0.0), (0.0, inf)])
def test_platform_positions_reject_non_finite_coordinates(position: tuple[float, float]) -> None:
    with pytest.raises(ValidationError):
        USVPlatformState(
            platform_id="usv_00",
            platform_index=0,
            position_xy=position,
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            deployment_state="deployed",
            capability=capability(PlatformKind.USV),
            distance_to_carrier_m=100.0,
        )

    with pytest.raises(ValidationError):
        CarrierPlatformState(
            carrier_id="carrier_01",
            position_xy=position,
            heading_rad=0.0,
            speed_mps=3.0,
            support_radius_m=15000.0,
            onboard_platform_ids=(),
            deployed_platform_ids=(),
            returning_platform_ids=(),
        )


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (MotionLimits, "max_speed_mps", "4.0"),
        (CommunicationCapability, "surface_range_m", True),
    ],
)
def test_platform_contracts_reject_type_coercion(model: type, field: str, value: object) -> None:
    valid = {
        MotionLimits: {
            "max_speed_mps": 4.0,
            "max_acceleration_mps2": 0.1,
            "max_turn_rate_rad_s": 0.02,
        },
        CommunicationCapability: {
            "surface_range_m": 10000.0,
            "acoustic_range_m": 4000.0,
        },
    }[model]
    with pytest.raises(ValidationError):
        model.model_validate({**valid, field: value})


def test_platform_models_reject_field_assignment_after_construction() -> None:
    limits = MotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=0.1,
        max_turn_rate_rad_s=0.02,
    )

    with pytest.raises(ValidationError):
        limits.max_speed_mps = 5.0
