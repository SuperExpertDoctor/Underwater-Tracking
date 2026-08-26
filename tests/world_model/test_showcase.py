from fastapi.testclient import TestClient

from underwater_tracking.world_model.showcase import (
    build_showcase_frame,
    create_showcase_app,
)


def test_showcase_frame_exercises_the_real_frontend_contract() -> None:
    frame = build_showcase_frame("left_turn")

    assert len(frame.uuvs) == 3
    assert len(frame.target_estimates) == 1
    target = frame.target_estimates[0]
    assert target.prediction is not None
    assert target.world_model is not None
    assert target.world_model.control_authority is False
    assert target.world_model.data_status == "ready"
    assert target.world_model.events[0].event_type == "target_turn_left"
    assert "ground_truth" not in frame.model_dump_json()


def test_showcase_api_serves_snapshot_replay_and_websocket() -> None:
    client = TestClient(create_showcase_app("sprint"))

    snapshot = client.get("/api/operational/snapshot")
    replay = client.get("/api/replay?start_s=0&offset=0&limit=10")

    assert snapshot.status_code == 200
    assert snapshot.json()["target_estimates"][0]["world_model"]["events"][0][
        "event_type"
    ] == "high_speed_escape"
    assert replay.status_code == 200
    assert replay.json()["total_count"] == 1
    with client.websocket_connect("/ws/operational") as websocket:
        assert websocket.receive_json()["frame_id"] == 1
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"
