import pytest
from pydantic import ValidationError
import underwater_tracking.domain.models as domain_models
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    DeploymentState,
    SituationSnapshot,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.availability import is_deployable
from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    CommunicationLink,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    PlatformSnapshot,
    SonarCapability,
    USVPlatformState,
    UUVPlatformState,
)


def test_uuv_status_contract_uses_four_operational_modes() -> None:
    assert {status.value for status in UUVStatus} == {
        "active",
        "unavailable",
        "track",
        "scan",
    }


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


def test_surveillance_capability_validates_ranges_and_legacy_uuvs_get_default() -> None:
    capability = domain_models.SurveillanceCapability(
        passive_range_m=4500.0,
        active_range_m=3000.0,
        bearing_variance_rad2=0.01,
        active_sonar_available=True,
        passive_sonar_available=True,
        max_speed_mps=4.0,
        max_turn_rate_rad_s=0.05,
        endurance_s=28_800.0,
        availability=0.95,
    )
    assert capability.active_sonar_available is True
    assert capability.endurance_s == 28_800.0
    assert capability.availability == 0.95
    with pytest.raises(ValidationError):
        domain_models.SurveillanceCapability(
            **{**capability.model_dump(), "passive_range_m": 0.0}
        )

    legacy_uuv = UUVState.model_validate(
        {
            "uuv_id": "uuv_01",
            "position_xy": [0.0, 0.0],
            "heading_rad": 0.0,
            "speed_mps": 1.0,
            "energy_fraction": 0.9,
            "status": "available",
        }
    )
    assert legacy_uuv.capability == domain_models.SurveillanceCapability()


def test_operational_scheme_rejects_quality_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        domain_models.OperationalScheme(
            scheme_id="scheme-default",
            version=1,
            target_priorities={"target_00": 1.0},
            minimum_quality={"target_00": 1.01},
            valid_from_s=0,
            valid_until_s=28_800,
            constraints=("maintain_tracking",),
        )


def test_intelligence_report_validates_source_confidence_and_expiry() -> None:
    payload = {
        "report_id": "intel-001",
        "source": "sonar",
        "target_id": "target_00",
        "confidence": 0.8,
        "issued_at_s": 120,
        "valid_until_s": 300,
        "content_summary": "Passive acoustic evidence indicates evasive maneuvering.",
        "assessment": {"intent": "evade"},
    }
    report = domain_models.IntelligenceReport(**payload)
    assert report.source == domain_models.IntelligenceSource.SONAR
    for invalid in (
        {**payload, "source": "unknown"},
        {**payload, "confidence": 1.1},
        {**payload, "valid_until_s": 120},
    ):
        with pytest.raises(ValidationError):
            domain_models.IntelligenceReport(**invalid)


def test_intelligence_assessment_is_json_safe_and_excludes_truth_recursively() -> None:
    payload = {
        "report_id": "intel-001",
        "source": "sonar",
        "target_id": "target_00",
        "confidence": 0.8,
        "issued_at_s": 120,
        "valid_until_s": 300,
        "assessment": {
            "intent": "evade",
            "evidence": [{"bearing": 0.2}, None],
        },
    }
    report = domain_models.IntelligenceReport(**payload)
    assert report.model_dump(mode="json")["assessment"] == payload["assessment"]
    for invalid_assessment in (
        {"target_truth": {"position_xy": [1.0, 2.0]}},
        {"evidence": {"evaluation_state": "hidden"}},
        {"evidence": {"true_position": [1.0, 2.0]}},
        {"evidence": [{"true_targets": ["target_00"]}]},
        {"artifact": object()},
    ):
        with pytest.raises(ValidationError):
            domain_models.IntelligenceReport(
                **{**payload, "assessment": invalid_assessment}
            )

    ordinary_assessment = {
        "truthiness_indicator": "unverified",
        "evaluation_summary": "operator review pending",
    }
    assert domain_models.IntelligenceReport(
        **{**payload, "assessment": ordinary_assessment}
    ).assessment == ordinary_assessment


@pytest.mark.parametrize(
    "summary",
    (
        "ground truth: position=(1.0, 2.0)",
        "evaluation_state=hidden",
        '{"true_position": [1.0, 2.0]}',
        "真实位置：目标位于东南方",
        "评估结果: accuracy=0.99",
        "actual position: (1.0, 2.0)",
        "actual position is (1.0, 2.0)",
        "真实位置在东南方",
    ),
)
def test_intelligence_report_rejects_obvious_truth_or_evaluation_summary_payloads(
    summary: str,
) -> None:
    with pytest.raises(ValidationError):
        domain_models.IntelligenceReport(
            report_id="intel-001",
            source="sonar",
            target_id="target_00",
            confidence=0.8,
            issued_at_s=120,
            valid_until_s=300,
            content_summary=summary,
        )


def test_adaptive_nested_mappings_are_immutable_but_dumpable() -> None:
    scheme = domain_models.OperationalScheme(
        scheme_id="scheme-default",
        version=1,
        target_priorities={"target_00": 1.0},
        minimum_quality={"target_00": 0.75},
        valid_from_s=0,
        valid_until_s=28_800,
    )
    report = domain_models.IntelligenceReport(
        report_id="intel-001",
        source="sonar",
        target_id="target_00",
        confidence=0.8,
        issued_at_s=120,
        valid_until_s=300,
        assessment={"evidence": {"bearing": 0.2, "samples": [{"bearing": 0.2}]}},
    )

    assert not isinstance(scheme.target_priorities, dict)
    assert not isinstance(report.assessment, dict)
    assert not isinstance(report.assessment["evidence"]["samples"], list)
    with pytest.raises(TypeError):
        scheme.target_priorities["target_00"] = 2.0
    with pytest.raises(TypeError):
        dict.__setitem__(scheme.target_priorities, "target_01", 0.5)
    with pytest.raises(TypeError):
        report.assessment["evidence"]["bearing"] = 0.4
    with pytest.raises(TypeError):
        report.assessment["new"] = "value"
    with pytest.raises(TypeError):
        list.__setitem__(report.assessment["evidence"]["samples"], 0, {})

    assert scheme.model_dump(mode="json")["target_priorities"] == {"target_00": 1.0}
    assert report.model_dump(mode="json")["assessment"] == {
        "evidence": {"bearing": 0.2, "samples": [{"bearing": 0.2}]}
    }
    copied_scheme = scheme.model_copy(deep=True)
    copied_report = report.model_copy(deep=True)
    assert copied_scheme == scheme
    assert copied_report == report
    assert copied_scheme.target_priorities is not scheme.target_priorities
    assert copied_report.assessment is not report.assessment
    assert copied_report.assessment["evidence"] is not report.assessment["evidence"]
    assert copied_scheme.model_dump_json() == scheme.model_dump_json()
    assert copied_report.model_dump_json() == report.model_dump_json()
    assert domain_models.OperationalScheme.model_validate_json(
        scheme.model_dump_json()
    ) == scheme
    assert domain_models.IntelligenceReport.model_validate_json(
        report.model_dump_json()
    ) == report


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passive_range_m", float("inf")),
        ("active_range_m", float("nan")),
        ("bearing_variance_rad2", float("inf")),
        ("max_speed_mps", float("nan")),
        ("max_turn_rate_rad_s", float("inf")),
    ],
)
def test_surveillance_capability_rejects_non_finite_numeric_limits(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        domain_models.SurveillanceCapability(**{field: value})


@pytest.mark.parametrize(
    ("target_priorities", "minimum_quality"),
    [
        ({"target_00": -0.1}, {"target_00": 0.75}),
        ({"target_00": float("inf")}, {"target_00": 0.75}),
        ({"target_00": float("nan")}, {"target_00": 0.75}),
        ({"target_00": 1.0}, {"target_00": float("inf")}),
        ({"target_00": 1.0}, {"target_00": float("nan")}),
    ],
)
def test_operational_scheme_rejects_non_finite_or_negative_objectives(
    target_priorities: dict[str, float], minimum_quality: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        domain_models.OperationalScheme(
            scheme_id="scheme-default",
            version=1,
            target_priorities=target_priorities,
            minimum_quality=minimum_quality,
            valid_from_s=0,
            valid_until_s=28_800,
        )


@pytest.mark.parametrize("field", ["target_priorities", "minimum_quality"])
def test_operational_scheme_rejects_empty_target_mapping_ids(field: str) -> None:
    with pytest.raises(ValidationError):
        domain_models.OperationalScheme(
            scheme_id="scheme-default",
            version=1,
            target_priorities={"": 1.0} if field == "target_priorities" else {},
            minimum_quality={"": 0.75} if field == "minimum_quality" else {},
            valid_from_s=0,
            valid_until_s=28_800,
        )


def test_snapshot_round_trips_operational_scheme_and_intelligence() -> None:
    scheme = domain_models.OperationalScheme(
        scheme_id="scheme-default",
        version=1,
        target_priorities={"target_00": 1.0},
        minimum_quality={"target_00": 0.75},
        valid_from_s=0,
        valid_until_s=28_800,
        constraints=("maintain_tracking",),
    )
    intelligence = domain_models.IntelligenceReport(
        report_id="intel-001",
        source="sigint",
        target_id="target_00",
        confidence=0.8,
        issued_at_s=120,
        valid_until_s=300,
        content_summary="Technical reconnaissance reports intermittent propulsion activity.",
        assessment={"intent": "evade"},
    )
    snapshot = SituationSnapshot(
        scenario_id="scenario-1",
        snapshot_revision=1,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        operational_scheme=scheme,
        intelligence_reports=(intelligence,),
    )
    assert SituationSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_snapshot_round_trips_truth_safe_platform_snapshot_without_truth_payload() -> None:
    usv_capability = PlatformCapability(
        kind=PlatformKind.USV,
        motion=MotionLimits(
            max_speed_mps=8.0,
            max_acceleration_mps2=0.5,
            max_turn_rate_rad_s=0.1,
        ),
        sonar=SonarCapability(
            passive_range_m=5000.0,
            passive_bearing_variance_rad2=0.02,
            active_source_range_m=3500.0,
            active_receive_range_m=3000.0,
            active_range_sigma_m=20.0,
            active_bearing_sigma_rad=0.03,
            active_capable=True,
            ping_cooldown_s=30,
            ping_energy_cost_fraction=0.01,
            clutter_sensitivity=0.2,
            exposure_cost=0.3,
        ),
        communications=CommunicationCapability(surface_range_m=6000.0, acoustic_range_m=2500.0),
    )
    uuv_capability = usv_capability.model_copy(
        update={
            "kind": PlatformKind.UUV,
            "communications": CommunicationCapability(
                surface_range_m=1000.0, acoustic_range_m=4000.0
            ),
        }
    )
    platform_snapshot = PlatformSnapshot(
        scenario_id="scenario-1",
        sim_time_s=30,
        carrier=CarrierPlatformState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=2.0,
            support_radius_m=7000.0,
            onboard_platform_ids=("uuv-01",),
            deployed_platform_ids=("usv-01",),
            returning_platform_ids=(),
        ),
        roster=PlatformRoster(
            usvs=(
                USVPlatformState(
                    platform_id="usv-01",
                    platform_index=0,
                    position_xy=(100.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=4.0,
                    energy_fraction=0.9,
                    deployment_state="deployed",
                    capability=usv_capability,
                    distance_to_carrier_m=100.0,
                ),
            ),
            uuvs=(
                UUVPlatformState(
                    platform_id="uuv-01",
                    platform_index=0,
                    position_xy=(0.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=0.0,
                    energy_fraction=1.0,
                    deployment_state="onboard",
                    capability=uuv_capability,
                    is_group_leader=True,
                    master_connected=True,
                ),
            ),
        ),
        communication_links=(
            CommunicationLink(
                source_id="carrier-01",
                target_id="usv-01",
                medium="surface",
                distance_m=100.0,
            ),
        ),
    )
    snapshot = SituationSnapshot(
        scenario_id="scenario-1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        platform_snapshot=platform_snapshot,
    )

    restored = SituationSnapshot.model_validate_json(snapshot.model_dump_json())

    assert restored == snapshot
    assert restored.platform_snapshot == platform_snapshot
    dumped_platform = snapshot.model_dump(mode="json")["platform_snapshot"]
    assert isinstance(dumped_platform, dict)
    assert not any(
        forbidden in dumped_platform
        for forbidden in ("targets", "target_entities", "truth", "evaluation")
    )
    with pytest.raises(ValidationError):
        SituationSnapshot.model_validate(
            {
                **snapshot.model_dump(mode="json"),
                "platform_snapshot": {
                    **dumped_platform,
                    "target_entities": [{"target_id": "target-01"}],
                },
            }
        )


def test_adaptive_input_models_are_frozen_and_strictly_serializable() -> None:
    capability = domain_models.SurveillanceCapability()
    report = domain_models.IntelligenceReport(
        report_id="intel-001",
        source="technical_reconnaissance",
        target_id="target_00",
        confidence=0.8,
        issued_at_s=120,
        valid_until_s=300,
        content_summary="Contact report",
    )
    with pytest.raises(ValidationError):
        capability.endurance_s = 10.0
    with pytest.raises(ValidationError):
        report.confidence = 0.2
    for field, value in (
        ("endurance_s", 0.0),
        ("endurance_s", float("inf")),
        ("availability", -0.1),
        ("availability", float("nan")),
    ):
        with pytest.raises(ValidationError):
            domain_models.SurveillanceCapability(**{field: value})
    assert domain_models.IntelligenceReport.model_validate_json(
        report.model_dump_json()
    ) == report


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


def test_unavailable_uuv_can_remain_deployed_while_exiting() -> None:
    base = {
        "uuv_id": "uuv_01",
        "position_xy": (0.0, 0.0),
        "heading_rad": 0.0,
        "speed_mps": 1.0,
        "energy_fraction": 0.9,
        "deployment_state": "deployed",
    }
    unavailable = UUVState(status="unavailable", **base)
    assert unavailable.status is UUVStatus.UNAVAILABLE
    assert is_deployable(unavailable) is False


@pytest.mark.parametrize(
    ("status", "deployment_state", "message"),
    [
        ("active", "returning", "require unavailable status"),
        ("active", "failed", "require unavailable status"),
        ("track", "onboard", "require deployed deployment_state"),
        ("scan", "failed", "require deployed deployment_state"),
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
