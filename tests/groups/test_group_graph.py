# tests/groups/test_group_graph.py
from math import atan2, dist

import pytest

from underwater_tracking.domain.models import BearingObservation
from underwater_tracking.config.models import RuntimeRetentionConfig
from underwater_tracking.groups.graph import build_group_graph
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.groups.state import GroupState, PlanCommand


def _observation(
    obs_id: str, sim_time_s: int, uuv_id: str, target_id: str, azimuth_rad: float
) -> BearingObservation:
    return BearingObservation(
        observation_id=obs_id,
        scenario_id="S1",
        sim_time_s=sim_time_s,
        uuv_id=uuv_id,
        target_id=target_id,
        azimuth_rad=azimuth_rad,
        variance_rad2=1e-3,
        detection_confidence=1.0,
    )


@pytest.fixture
def two_uuv_observations() -> tuple[BearingObservation, BearingObservation]:
    """Two bearings from U1@(0, 0) and U2@(1000, 0), both crossing the (500, 500) prior."""
    return (
        _observation("O1", 30, "U1", "T1", atan2(500.0, 500.0)),
        _observation("O2", 30, "U2", "T1", atan2(500.0, 500.0 - 1000.0)),
    )


def test_group_graph_updates_belief_quality_and_report(
    two_uuv_observations: tuple[BearingObservation, BearingObservation],
) -> None:
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0)),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    output = graph.invoke(
        {"new_observations": two_uuv_observations},
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    assert output["belief"].target_id == "T1"
    assert 0 <= output["quality"].ewma <= 1
    assert output["report"].member_ids == ("U1", "U2")


def test_initialization_failure_emits_quality_guard_events() -> None:
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0)),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    guard_types = [
        event.event_type for event in output["emitted_events"] if event.event_type.startswith("quality_guard:")
    ]
    assert "quality_guard:no_accepted_observation" in guard_types
    assert "quality_guard:fim_degenerate" in guard_types
    assert output["quality"].instant == 0.0
    assert output["quality"].hard_guard_reasons


def test_two_group_beliefs_cannot_cross(
    two_uuv_observations: tuple[BearingObservation, BearingObservation],
) -> None:
    """Separate threads keep beliefs at their own crossing point, not the other group's."""
    positions = {"U1": (0.0, 0.0), "U2": (1000.0, 0.0)}
    t2_observations = (
        _observation("O3", 30, "U1", "T2", atan2(800.0, 500.0)),
        _observation("O4", 30, "U2", "T2", atan2(800.0, 500.0 - 1000.0)),
    )
    t2_later = (
        _observation("O5", 60, "U1", "T2", atan2(800.0, 500.0)),
        _observation("O6", 60, "U2", "T2", atan2(800.0, 500.0 - 1000.0)),
    )
    t1_later = (
        _observation("O7", 60, "U1", "T1", atan2(500.0, 500.0)),
        _observation("O8", 60, "U2", "T1", atan2(500.0, 500.0 - 1000.0)),
    )
    graph = build_group_graph()
    graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0), member_positions=positions),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    graph.invoke(
        GroupState.initial("S1", "G-T2", "T2", ("U1", "U2"), coarse_prior=(500.0, 500.0), member_positions=positions),
        config={"configurable": {"thread_id": "S1:T2"}},
    )
    graph.invoke(
        {"new_observations": two_uuv_observations}, config={"configurable": {"thread_id": "S1:T1"}}
    )
    graph.invoke(
        {"new_observations": t2_observations}, config={"configurable": {"thread_id": "S1:T2"}}
    )
    output_t1 = graph.invoke(
        {"new_observations": t1_later}, config={"configurable": {"thread_id": "S1:T1"}}
    )
    output_t2 = graph.invoke(
        {"new_observations": t2_later}, config={"configurable": {"thread_id": "S1:T2"}}
    )
    t1_xy = (output_t1["belief"].mean[0], output_t1["belief"].mean[1])
    t2_xy = (output_t2["belief"].mean[0], output_t2["belief"].mean[1])
    assert dist(t1_xy, (500.0, 500.0)) < dist(t1_xy, (500.0, 800.0))
    assert dist(t2_xy, (500.0, 800.0)) < dist(t2_xy, (500.0, 500.0))
    assert dist(t1_xy, t2_xy) > 100.0
    # Cross-feeding the other target's observations must not move the belief
    # (only float noise from the dt=0 mixing remains, at the micrometer scale).
    after_cross_feed = graph.invoke(
        {"new_observations": t2_later}, config={"configurable": {"thread_id": "S1:T1"}}
    )
    assert dist((after_cross_feed["belief"].mean[0], after_cross_feed["belief"].mean[1]), t1_xy) < 1e-3


def test_member_failure_plan_command_applies_replacement(
    two_uuv_observations: tuple[BearingObservation, BearingObservation],
) -> None:
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0)),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    output = graph.invoke(
        {"new_observations": two_uuv_observations}, config={"configurable": {"thread_id": "S1:T1"}}
    )
    command = PlanCommand(
        command_id="C1",
        scenario_id="S1",
        target_id="T1",
        sim_time_s=60,
        plan_revision=1,
        member_replacements={"U2": "U5"},
    )
    output = graph.invoke({"pending_command": command}, config={"configurable": {"thread_id": "S1:T1"}})
    assert output["report"].member_ids == ("U1", "U5")
    assert output["report"].plan_revision == 1
    assert output["plan_revision"] == 1
    assert "U2" not in output["member_positions"]
    assert any(
        event.event_type == "member_failed" and event.entity_id == "U2" for event in output["emitted_events"]
    )


def test_authoritative_plan_command_updates_roster_positions_and_events() -> None:
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial(
            "S1",
            "G-T1",
            "T1",
            ("U1", "U2"),
            coarse_prior=(500.0, 500.0),
            member_positions={"U1": (0.0, 0.0), "U2": (1000.0, 0.0)},
        ),
        config={"configurable": {"thread_id": "S1:T1:authoritative"}},
    )
    output = graph.invoke(
        {
            "pending_command": PlanCommand(
                command_id="grow",
                scenario_id="S1",
                target_id="T1",
                sim_time_s=30,
                plan_revision=1,
                desired_member_ids=("U1", "U2", "U3"),
                member_positions={"U1": (10.0, 0.0), "U2": (1010.0, 0.0), "U3": (500.0, 500.0)},
            )
        },
        config={"configurable": {"thread_id": "S1:T1:authoritative"}},
    )
    assert output["report"].member_ids == ("U1", "U2", "U3")
    assert output["member_positions"] == {"U1": (10.0, 0.0), "U2": (1010.0, 0.0), "U3": (500.0, 500.0)}
    assert output["plan_revision"] == 1
    assert [event.event_type for event in output["emitted_events"]][-1:] == ["member_added"]

    output = graph.invoke(
        {
            "pending_command": PlanCommand(
                command_id="replace",
                scenario_id="S1",
                target_id="T1",
                sim_time_s=60,
                plan_revision=2,
                desired_member_ids=("U1", "U3", "U4"),
                member_positions={"U1": (20.0, 0.0), "U3": (510.0, 500.0), "U4": (750.0, 500.0)},
            )
        },
        config={"configurable": {"thread_id": "S1:T1:authoritative"}},
    )
    assert output["report"].member_ids == ("U1", "U3", "U4")
    assert output["member_positions"] == {"U1": (20.0, 0.0), "U3": (510.0, 500.0), "U4": (750.0, 500.0)}
    assert [event.event_type for event in output["emitted_events"]][-1:] == ["member_replaced"]


def test_predict_only_cycle_ages_freshness_after_acceptance(
    two_uuv_observations: tuple[BearingObservation, BearingObservation],
) -> None:
    """A stale predict-only cycle must age the track, not report maxed freshness."""
    positions = {"U1": (0.0, 0.0), "U2": (1000.0, 0.0)}
    graph = build_group_graph()
    output = graph.invoke(
        GroupState.initial("S1", "G-T1", "T1", ("U1", "U2"), coarse_prior=(500.0, 500.0), member_positions=positions),
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    output = graph.invoke(
        {"new_observations": two_uuv_observations},
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    # Freshly accepted: zero age, freshness maxed.
    assert output["quality"].components["freshness"] == 1.0
    # Predict-only cycle: later-timestamped bearings, but no member positions
    # are supplied, so no update runs; the track must age instead of staying
    # max-fresh forever.
    stale = (
        _observation("O9", 60, "U1", "T1", atan2(500.0, 500.0)),
        _observation("O10", 60, "U2", "T1", atan2(500.0, 500.0 - 1000.0)),
    )
    output = graph.invoke(
        {"new_observations": stale, "member_positions": {}},
        config={"configurable": {"thread_id": "S1:T1"}},
    )
    assert output["belief"].sim_time_s == 60
    assert output["quality"].components["freshness"] < 1.0


def test_group_manager_creates_invokes_completes_and_lists(
    two_uuv_observations: tuple[BearingObservation, BearingObservation],
) -> None:
    manager = GroupManager()
    first = manager.create(
        "T1", scenario_id="S1", group_id="G-T1", member_ids=("U1", "U2"), coarse_prior=(500.0, 500.0)
    )
    assert first.target_id == "T1"
    assert first.member_ids == ("U1", "U2")
    report = manager.invoke("T1", observations=two_uuv_observations)
    assert report.belief.target_id == "T1"
    assert manager.list_groups() == ("T1",)
    manager.complete("T1")
    assert manager.list_groups() == ()


def test_group_manager_reapplies_event_retention_on_each_invoke() -> None:
    manager = GroupManager(
        retention=RuntimeRetentionConfig(event_history_limit=1)
    )
    report = manager.create(
        "T1",
        scenario_id="S1",
        group_id="G-T1",
        member_ids=("U1", "U2"),
        coarse_prior=(500.0, 500.0),
    )

    class RecordingGraph:
        def __init__(self) -> None:
            self.inputs: dict[str, object] | None = None

        def invoke(self, inputs: dict[str, object], *, config: object) -> dict[str, object]:
            del config
            self.inputs = inputs
            return {"report": report}

    graph = RecordingGraph()
    manager._graph = graph
    manager.invoke("T1")

    assert graph.inputs == {"event_history_limit": 1}
