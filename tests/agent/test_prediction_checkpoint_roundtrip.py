from __future__ import annotations

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.persistence.checkpoints import create_checkpointer


def test_accepted_prediction_survives_sqlite_checkpoint_reopen(tmp_path) -> None:
    prediction = PredictedTrackRef(
        prediction_id="S1:track:T1:1",
        target_id="T1",
        sim_time_s=60,
        horizon_s=120.0,
        sample_step_s=30.0,
        times_s=(90.0, 120.0),
        points_xy=((1.0, 2.0), (3.0, 4.0)),
        corridor_radius_m=(10.0, 12.0),
        prediction_regime="short_history",
    )
    accepted = AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="degraded",
            regime="short_history",
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=12.0,
            raw_prediction_id=prediction.prediction_id,
        ),
    )
    checkpoint = {
        "v": 1,
        "id": "accepted-prediction-c1",
        "ts": "2026-08-29T00:00:00Z",
        "channel_values": {"accepted_predictions": {"T1": accepted}},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    thread = {"configurable": {"thread_id": "S1", "checkpoint_ns": ""}}

    saver = create_checkpointer(tmp_path / "graph.db")
    try:
        saver.put(thread, checkpoint, {}, {})
    finally:
        saver.conn.close()

    reopened = create_checkpointer(tmp_path / "graph.db")
    try:
        restored = reopened.get_tuple(thread)
        assert restored is not None
        restored_accepted = restored.checkpoint["channel_values"][
            "accepted_predictions"
        ]["T1"]
        assert restored_accepted == accepted
        assert isinstance(restored_accepted, AcceptedPrediction)
    finally:
        reopened.conn.close()
