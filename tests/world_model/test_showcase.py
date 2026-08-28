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


def test_showcase_frame_advances_target_formation_and_prediction() -> None:
    first = build_showcase_frame("left_turn")
    advanced = build_showcase_frame("left_turn", elapsed_s=20.0, frame_id=5)

    assert advanced.frame_id == 5
    assert advanced.sim_time_s > first.sim_time_s
    assert advanced.target_estimates[0].mean != first.target_estimates[0].mean
    assert advanced.uuvs[0].position != first.uuvs[0].position
    assert advanced.target_estimates[0].prediction is not None
    assert advanced.target_estimates[0].world_model is not None
    assert advanced.target_estimates[0].world_model.events[0].event_type == ("target_turn_left")


def test_showcase_api_serves_snapshot_replay_and_websocket() -> None:
    client = TestClient(create_showcase_app("sprint"))

    snapshot = client.get("/api/operational/snapshot")
    replay = client.get("/api/replay?start_s=0&offset=0&limit=10")

    assert snapshot.status_code == 200
    assert (
        snapshot.json()["target_estimates"][0]["world_model"]["events"][0]["event_type"]
        == "high_speed_escape"
    )
    assert replay.status_code == 200
    assert replay.json()["total_count"] == 1
    with client.websocket_connect("/ws/operational") as websocket:
        assert websocket.receive_json()["frame_id"] == 1
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"


def test_animated_showcase_streams_new_frames() -> None:
    with TestClient(
        create_showcase_app("left_turn", animate=True, frame_interval_s=0.01, sim_step_s=5.0)
    ) as client:
        with client.websocket_connect("/ws/operational") as websocket:
            first = websocket.receive_json()
            second = websocket.receive_json()

        replay = client.get("/api/replay?start_s=0&offset=0&limit=10")
        run = client.get("/api/runs").json()["runs"][0]

    assert second["frame_id"] > first["frame_id"]
    assert second["sim_time_s"] > first["sim_time_s"]
    assert second["target_estimates"][0]["mean"] != first["target_estimates"][0]["mean"]
    assert replay.json()["total_count"] >= 2
    assert run["status"] == "running"
    assert run["effective_demo_speed"] == 500.0
