from collections.abc import Mapping

import pytest

from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    TimeWindow,
    UUVRegionalPolicy,
    UUVRegionalStrategySet,
)
from underwater_tracking.planning.regional_plan_validator import (
    RegionalPlanError,
    ValidatedRegionalStrategy,
    validate_uuv_strategy,
)


def candidate(
    candidate_id: str = "T1:r1:square:0:0:1",
    *,
    predecessor_candidate_ids: tuple[str, ...] = (),
    successor_candidate_ids: tuple[str, ...] = (),
) -> RegionalMissionCandidate:
    return RegionalMissionCandidate(
        candidate_id=candidate_id,
        cell_ids=(f"{candidate_id}:cell:0:0",),
        time_window=TimeWindow(start_s=10, end_s=30),
        perimeter_points=((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)),
        predecessor_candidate_ids=predecessor_candidate_ids,
        successor_candidate_ids=successor_candidate_ids,
    )


def policy(
    candidate_id: str = "T1:r1:square:0:0:1",
    *,
    uuv_ids: tuple[str, ...] = ("U1",),
    tracking_mode: str = "active_scan",
    predecessor_candidate_id: str | None = None,
    successor_candidate_id: str | None = None,
) -> UUVRegionalPolicy:
    return UUVRegionalPolicy(
        candidate_id=candidate_id,
        coverage_mode="required",
        tracking_mode=tracking_mode,
        priority=1.0,
        required_quality=0.8,
        assigned_uuv_ids=uuv_ids,
        predecessor_candidate_id=predecessor_candidate_id,
        successor_candidate_id=successor_candidate_id,
        rationale="candidate is supported by the prediction evidence",
        evidence_ids=("belief:T1",),
    )


def test_validated_strategy_contains_only_known_uuv_candidate_assignments() -> None:
    result = validate_uuv_strategy(
        [candidate()],
        UUVRegionalStrategySet(policies=(policy(),)),
        {"U1"},
    )

    assert isinstance(result, ValidatedRegionalStrategy)
    assert result.policies[0].candidate_id == "T1:r1:square:0:0:1"


def test_strategy_rejects_unknown_uuv_and_region() -> None:
    with pytest.raises(RegionalPlanError, match="unknown region"):
        validate_uuv_strategy(
            [candidate()],
            UUVRegionalStrategySet(policies=(policy("T1:r9:square:0:0:1"),)),
            {"U1"},
        )

    with pytest.raises(RegionalPlanError, match="unknown UUV"):
        validate_uuv_strategy(
            [candidate()],
            UUVRegionalStrategySet(policies=(policy(uuv_ids=("U99",)),)),
            {"U1"},
        )


def test_strategy_rejects_legacy_surface_assignment_fields() -> None:
    raw_strategy: Mapping[str, object] = {
        "policies": [
            {
                **policy().model_dump(mode="json"),
                "assigned_usv_ids": ["S1"],
            }
        ]
    }

    with pytest.raises(RegionalPlanError, match="strict UUV strategy schema"):
        validate_uuv_strategy([candidate()], raw_strategy, {"U1"})


def test_active_scan_requires_an_active_capable_uuv() -> None:
    with pytest.raises(RegionalPlanError, match="active scan"):
        validate_uuv_strategy(
            [candidate()],
            UUVRegionalStrategySet(policies=(policy(),)),
            {"U1": {"active_capable": False}},
        )


def test_strategy_rejects_overlapping_assignments_and_bad_handoff_reference() -> None:
    first = candidate(
        "T1:r1:square:0:0:1",
        successor_candidate_ids=("T1:r1:square:1:0:1",),
    )
    second = candidate(
        "T1:r1:square:1:0:1",
        predecessor_candidate_ids=("T1:r1:square:0:0:1",),
    )
    with pytest.raises(RegionalPlanError, match="overlapping UUV"):
        validate_uuv_strategy(
            [first, second],
            UUVRegionalStrategySet(
                policies=(policy(), policy(second.candidate_id)),
            ),
            {"U1", "U2"},
        )

    with pytest.raises(RegionalPlanError, match="handoff"):
        validate_uuv_strategy(
            [first, second],
            UUVRegionalStrategySet(
                policies=(
                    policy(
                        uuv_ids=("U1",),
                        successor_candidate_id="T1:r1:square:9:9:1",
                    ),
                    policy(second.candidate_id, uuv_ids=("U2",)),
                )
            ),
            {"U1", "U2"},
        )
