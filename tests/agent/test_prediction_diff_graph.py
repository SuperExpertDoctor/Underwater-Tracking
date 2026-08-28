from types import SimpleNamespace

from underwater_tracking.agent.graphs.central import TrajectoryPredictionNode
from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth


TIMES = tuple(float(value) for value in range(30, 691, 30))
CONFIG = TrajectoryDiffConfig()


def _situation(sim_time_s: int) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id="S1",
        sim_time_s=sim_time_s,
        group_reports=(SimpleNamespace(target_id="T1"),),
        target_search_priors=(),
    )


def _prediction(sim_time_s: int, *, offset_m: float, regime: str = "short_history"):
    times = tuple(sim_time_s + value for value in TIMES)
    return PredictedTrackRef(
        prediction_id=f"P:{sim_time_s}",
        target_id="T1",
        sim_time_s=sim_time_s,
        horizon_s=660.0,
        sample_step_s=30.0,
        times_s=times,
        points_xy=tuple((time_s + offset_m, 0.0) for time_s in times),
        corridor_radius_m=(1.0,) * len(times),
        source_belief_history_ids=(f"O:{sim_time_s}",),
        fallback_used=regime != "bspline",
        prediction_regime=regime,
        imm_model_probabilities={"cv": 0.8, "left_turn": 0.1, "right_turn": 0.1},
    )


def _accepted(prediction: PredictedTrackRef) -> AcceptedPrediction:
    return AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="degraded" if prediction.prediction_regime != "imm" else "valid",
            regime=prediction.prediction_regime,
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=max(prediction.corridor_radius_m, default=0.0),
            raw_prediction_id=prediction.prediction_id,
        ),
    )


def test_prediction_node_checkpoints_two_cycle_gate_and_emits_once() -> None:
    situations = {f"R{index}": _situation(index * 30) for index in range(3)}
    offsets = {0: 0.0, 30: 300.0, 60: 600.0}
    node = TrajectoryPredictionNode(
        lambda situation, _target_id: _accepted(
            _prediction(
                situation.sim_time_s,
                offset_m=offsets[situation.sim_time_s],
            )
        ),
        situations.__getitem__,
        diff_config=CONFIG,
    )

    first = node({"scenario_id": "S1", "snapshot_ref": "R0"})
    assert first["prediction_diffs"]["T1"].status == "first_prediction"
    assert first["prediction_diff_gates"]["T1"].consecutive_count == 0

    second = node({**first, "scenario_id": "S1", "snapshot_ref": "R1"})
    assert second["prediction_diffs"]["T1"].gate_transition == "accumulating"
    assert second["prediction_diff_gates"]["T1"].consecutive_count == 1
    assert second["prediction_intent_verification_target_ids"] == ()

    third = node({**second, "scenario_id": "S1", "snapshot_ref": "R2"})
    diff = third["prediction_diffs"]["T1"]
    gate = third["prediction_diff_gates"]["T1"]
    assert diff.gate_transition == "suspected"
    assert diff.latched is True
    assert gate.consecutive_count == 2
    assert third["prediction_intent_verification_target_ids"] == ("T1",)
    assert [event.event_type for event in third["coalesced_events"]] == [
        "target_intent_change_suspected"
    ]
    event = third["coalesced_events"][0]
    assert event.event_id == "S1:target_intent_change_suspected:T1:60"
    assert event.payload["diff_id"] == diff.diff_id
    assert event.payload["exceeded"] is True
    assert event.payload["observation_ids"] == ("O:60",)
    assert gate.suspicion_event_id == event.event_id


def test_prediction_node_low_diff_and_regime_change_reset_accumulation() -> None:
    situations = {"R0": _situation(0), "R1": _situation(30), "R2": _situation(60)}
    predictions = {
        0: _prediction(0, offset_m=0.0),
        30: _prediction(30, offset_m=300.0),
        60: _prediction(60, offset_m=300.0, regime="bspline"),
    }
    node = TrajectoryPredictionNode(
        lambda situation, _target_id: _accepted(predictions[situation.sim_time_s]),
        situations.__getitem__,
        diff_config=CONFIG,
    )

    first = node({"scenario_id": "S1", "snapshot_ref": "R0"})
    accumulating = node({**first, "scenario_id": "S1", "snapshot_ref": "R1"})
    reset = node({**accumulating, "scenario_id": "S1", "snapshot_ref": "R2"})

    assert accumulating["prediction_diff_gates"]["T1"].consecutive_count == 1
    assert reset["prediction_diffs"]["T1"].status == "predictor_regime_reset"
    assert reset["prediction_diffs"]["T1"].gate_transition == "reset"
    assert reset["prediction_diff_gates"]["T1"].consecutive_count == 0
    assert reset["prediction_intent_verification_target_ids"] == ()


def test_prediction_node_keeps_last_evidence_when_target_is_temporarily_unobserved() -> None:
    situations = {
        "observed": _situation(0),
        "unobserved": SimpleNamespace(
            scenario_id="S1",
            sim_time_s=30,
            group_reports=(),
            target_search_priors=(),
        ),
    }
    node = TrajectoryPredictionNode(
        lambda situation, _target_id: _accepted(
            _prediction(
                situation.sim_time_s,
                offset_m=0.0,
            )
        ),
        situations.__getitem__,
        diff_config=CONFIG,
    )

    observed = node({"scenario_id": "S1", "snapshot_ref": "observed"})
    unobserved = node(
        {**observed, "scenario_id": "S1", "snapshot_ref": "unobserved"}
    )

    assert unobserved["predictions"] == observed["predictions"]
    assert unobserved["prediction_diffs"] == observed["prediction_diffs"]
    assert unobserved["prediction_diff_gates"] == observed["prediction_diff_gates"]
    assert unobserved["prediction_intent_verification_target_ids"] == ()


def test_prediction_node_does_not_publish_an_unavailable_candidate() -> None:
    unavailable = AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="boundary_recovery",
            reason_codes=("boundary_recovery_point_out_of_bounds",),
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=0.0,
            raw_prediction_id="raw-boundary-id",
        ),
    )
    node = TrajectoryPredictionNode(
        lambda _situation, _target_id: unavailable,
        lambda _ref: _situation(30),
        diff_config=CONFIG,
    )

    result = node({"scenario_id": "S1", "snapshot_ref": "R1"})

    assert result["predictions"] == {}
    assert result["prediction_diffs"] == {}
    assert result["prediction_diff_gates"] == {}
