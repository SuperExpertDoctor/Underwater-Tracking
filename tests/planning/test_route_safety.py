from __future__ import annotations

import pytest
import numpy as np

from underwater_tracking.planning.route_safety import (
    minimum_synchronous_separation_m,
    transition_separation_is_safe,
)
from underwater_tracking.planning.waypoints import _separated_mask


def test_crossing_routes_are_detected_between_safe_endpoints() -> None:
    minimum = minimum_synchronous_separation_m(
        (0.0, -650.0),
        (0.0, 650.0),
        (0.0, 650.0),
        (0.0, -650.0),
    )

    assert minimum == pytest.approx(0.0)
    assert not transition_separation_is_safe(
        (0.0, -650.0),
        (0.0, 650.0),
        (0.0, 650.0),
        (0.0, -650.0),
        min_separation_m=300.0,
    )


def test_parallel_routes_keep_their_separation() -> None:
    assert minimum_synchronous_separation_m(
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 400.0),
        (100.0, 400.0),
    ) == pytest.approx(400.0)
    assert transition_separation_is_safe(
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 400.0),
        (100.0, 400.0),
        min_separation_m=300.0,
    )


def test_initially_close_uuvs_may_spread_apart_without_first_converging() -> None:
    assert transition_separation_is_safe(
        (0.0, 0.0),
        (-200.0, 0.0),
        (0.0, 0.0),
        (200.0, 0.0),
        min_separation_m=300.0,
    )
    assert not transition_separation_is_safe(
        (-100.0, 0.0),
        (200.0, 0.0),
        (100.0, 0.0),
        (-200.0, 0.0),
        min_separation_m=300.0,
    )


def test_group_planner_candidate_filter_rejects_a_swept_crossing() -> None:
    mask = _separated_mask(
        np.asarray(((0.0, -650.0),)),
        np.asarray((0.0, 650.0)),
        np.asarray(((0.0, 650.0),)),
        np.asarray(((0.0, -650.0),)),
        300.0,
    )

    assert mask.tolist() == [False]
