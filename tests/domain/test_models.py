import pytest
from pydantic import ValidationError
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    DeploymentState,
    SituationSnapshot,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.availability import is_deployable


def test_bearing_rejects_unknown_fields_and_normalizes_angle():
    observation = BearingObservation(
        observation_id="O1", scenario_id="S1", sim_time_s=30,
        uuv_id="U1", target_id="T1", azimuth_rad=7.0,
        variance_rad2=0.01, detection_confidence=0.9,
    )
    assert -3.141592653589793 <= observation.azimuth_rad < 3.141592653589793
    with pytest.raises(ValidationError):
        BearingObservation(**observation.model_dump(), target_truth=[1.0, 2.0])


def test_operational_snapshot_has_no_truth_field():
    assert "truth" not in SituationSnapshot.model_fields
    assert "true_targets" not in SituationSnapshot.model_fields


def test_carrier_and_deployment_state_round_trip():
    carrier = CarrierState(
        carrier_id="carrier-01",
        position_xy=(-3000.0, -3000.0),
        heading_rad=0.25,
        speed_mps=1.5,
        status="recovering",
        onboard_uuv_ids=("uuv_03",),
        deployed_uuv_ids=("uuv_01",),
        returning_uuv_ids=("uuv_02",),
    )
    restored = CarrierState.model_validate_json(carrier.model_dump_json())
    assert restored == carrier


def test_old_uuv_and_snapshot_payloads_get_compatible_defaults():
    uuv = UUVState.model_validate({
        "uuv_id": "uuv_01",
        "position_xy": [0.0, 0.0],
        "heading_rad": 0.0,
        "speed_mps": 1.0,
        "energy_fraction": 0.9,
        "status": "available",
    })
    assert uuv.deployment_state == DeploymentState.DEPLOYED
    assert SituationSnapshot.model_validate({
        "scenario_id": "scenario-1",
        "snapshot_revision": 1,
        "sim_time_s": 30,
        "uuvs": [uuv.model_dump()],
        "group_reports": [],
        "pending_events": [],
    }).carrier is None


@pytest.mark.parametrize(
    ("status", "expected_deployment_state"),
    [("returning", DeploymentState.RETURNING), ("failed", DeploymentState.FAILED)],
)
def test_old_uuv_status_normalizes_missing_deployment_state(
    status: str, expected_deployment_state: DeploymentState
) -> None:
    uuv = UUVState.model_validate(
        {
            "uuv_id": "uuv_01",
            "position_xy": [0.0, 0.0],
            "heading_rad": 0.0,
            "speed_mps": 1.0,
            "energy_fraction": 0.9,
            "status": status,
        }
    )
    assert uuv.deployment_state is expected_deployment_state


def test_uuv_rejects_returning_or_failed_status_with_deployed_state() -> None:
    base = {
        "uuv_id": "uuv_01",
        "position_xy": (0.0, 0.0),
        "heading_rad": 0.0,
        "speed_mps": 1.0,
        "energy_fraction": 0.9,
        "deployment_state": "deployed",
    }
    for status in ("returning", "failed"):
        with pytest.raises(ValidationError, match="deployment_state"):
            UUVState(status=status, **base)

    contradictory = UUVState.model_construct(status=UUVStatus.RETURNING, **base)
    assert is_deployable(contradictory) is False


@pytest.mark.parametrize(
    ("status", "deployment_state", "message"),
    [
        ("available", "returning", "returning deployment_state requires returning status"),
        ("available", "failed", "failed deployment_state requires failed status"),
        ("tracking", "onboard", "tracking status cannot be onboard"),
        ("tracking", "failed", "tracking status cannot be failed"),
    ],
)
def test_uuv_rejects_reverse_status_and_deployment_contradictions(
    status: str, deployment_state: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        UUVState(
            uuv_id="uuv_01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            energy_fraction=0.9,
            status=status,
            deployment_state=deployment_state,
        )


@pytest.mark.parametrize(
    ("status", "speed_mps", "onboard", "deployed", "returning", "message"),
    [
        ("transit", 1.0, (), (), ("uuv_01",), "returning UUVs require recovering status"),
        ("recovering", 1.0, (), ("uuv_01",), (), "recovering status requires returning UUVs"),
        ("deploying", 1.0, (), ("uuv_01",), (), "deploying status requires onboard and deployed UUVs"),
        ("transit", 0.0, (), (), (), "transit status requires movement"),
        ("standby", 1.0, (), (), (), "standby status requires zero speed"),
    ],
)
def test_carrier_rejects_status_list_and_speed_contradictions(
    status: str,
    speed_mps: float,
    onboard: tuple[str, ...],
    deployed: tuple[str, ...],
    returning: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        CarrierState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=speed_mps,
            status=status,
            onboard_uuv_ids=onboard,
            deployed_uuv_ids=deployed,
            returning_uuv_ids=returning,
        )


def test_carrier_lists_must_be_disjoint_and_match_snapshot_deployment_state():
    with pytest.raises(ValidationError, match="carrier relationship lists must be disjoint"):
        CarrierState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            onboard_uuv_ids=("uuv_01",),
            deployed_uuv_ids=("uuv_01",),
        )
    with pytest.raises(ValidationError, match="duplicate IDs"):
        CarrierState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            deployed_uuv_ids=("uuv_01", "uuv_01"),
        )

    uuv = UUVState(
        uuv_id="uuv_01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.9,
        status="available",
        deployment_state="onboard",
    )
    carrier = CarrierState(
        carrier_id="carrier-01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        status="transit",
        onboard_uuv_ids=(),
        deployed_uuv_ids=("uuv_01",),
        returning_uuv_ids=(),
    )
    with pytest.raises(ValidationError, match="deployment_state"):
        SituationSnapshot(
            scenario_id="scenario-1",
            snapshot_revision=1,
            sim_time_s=30,
            uuvs=(uuv,),
            carrier=carrier,
            group_reports=(),
            pending_events=(),
        )


def test_snapshot_rejects_duplicate_carrier_members_even_from_typed_carrier() -> None:
    uuv = UUVState(
        uuv_id="uuv_01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.9,
        status="available",
    )
    duplicate_carrier = CarrierState.model_construct(
        carrier_id="carrier-01",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        deployed_uuv_ids=("uuv_01", "uuv_01"),
        onboard_uuv_ids=(),
        returning_uuv_ids=(),
    )
    with pytest.raises(ValidationError, match="duplicate IDs"):
        SituationSnapshot(
            scenario_id="scenario-1",
            snapshot_revision=1,
            sim_time_s=30,
            uuvs=(uuv,),
            carrier=duplicate_carrier,
            group_reports=(),
            pending_events=(),
        )


def test_legacy_snapshot_carrier_normalizes_missing_deployment_relationships():
    snapshot = SituationSnapshot.model_validate(
        {
            "scenario_id": "scenario-1",
            "snapshot_revision": 1,
            "sim_time_s": 30,
            "uuvs": [
                {
                    "uuv_id": "uuv_01",
                    "position_xy": [0.0, 0.0],
                    "heading_rad": 0.0,
                    "speed_mps": 1.0,
                    "energy_fraction": 0.9,
                    "status": "available",
                }
            ],
            "carrier": {
                "carrier_id": "carrier-01",
                "position_xy": [0.0, 0.0],
                "heading_rad": 0.0,
                "speed_mps": 1.0,
            },
            "group_reports": [],
            "pending_events": [],
        }
    )
    assert snapshot.uuvs[0].deployment_state is DeploymentState.DEPLOYED
    assert snapshot.carrier is not None
    assert snapshot.carrier.deployed_uuv_ids == ("uuv_01",)


def test_legacy_returning_snapshot_and_typed_old_carrier_are_normalized() -> None:
    payload = {
        "scenario_id": "scenario-1",
        "snapshot_revision": 1,
        "sim_time_s": 30,
        "uuvs": [
            {
                "uuv_id": "uuv_01",
                "position_xy": [0.0, 0.0],
                "heading_rad": 0.0,
                "speed_mps": 1.0,
                "energy_fraction": 0.9,
                "status": "returning",
            }
        ],
        "carrier": {
            "carrier_id": "carrier-01",
            "position_xy": [0.0, 0.0],
            "heading_rad": 0.0,
            "speed_mps": 1.0,
        },
        "group_reports": [],
        "pending_events": [],
    }
    legacy = SituationSnapshot.model_validate(payload)
    assert legacy.uuvs[0].deployment_state is DeploymentState.RETURNING
    assert legacy.carrier is not None
    assert legacy.carrier.returning_uuv_ids == ("uuv_01",)

    typed_uuv = UUVState(**payload["uuvs"][0])
    typed_carrier = CarrierState(**payload["carrier"])
    typed = SituationSnapshot(
        **{**payload, "uuvs": (typed_uuv,), "carrier": typed_carrier}
    )
    assert typed.carrier is not None
    assert typed.carrier.returning_uuv_ids == ("uuv_01",)
