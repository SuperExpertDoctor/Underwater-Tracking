"""Capture-only non-interference: public observations fixed, evaluation truth varies."""

from math import atan2
from types import SimpleNamespace

from fastapi.testclient import TestClient
from tests.api.test_live_publisher import Ledger, Events
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.api.app import create_app
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.models import BearingObservation, SituationSnapshot
from underwater_tracking.domain.truth import TargetTruth
from underwater_tracking.domain.ui_models import EvaluationFrame
from underwater_tracking.groups.manager import GroupManager
from underwater_tracking.prediction.port import make_snapshot_predictor
from underwater_tracking.world_model.adapter import build_world_model_forecasts
from underwater_tracking.tracking.region_probability import public_region_probability


def public_pipeline(offset=0.0):
    positions = {"U1": (0.0, 0.0), "U2": (1000.0, 0.0)}
    manager = GroupManager()
    manager.create(
        "T",
        scenario_id="S",
        group_id="G",
        member_ids=tuple(positions),
        coarse_prior=(500.0, 500.0),
        member_positions=positions,
    )
    history = []
    for time in (30, 60, 90):
        point = (500.0 + time + offset, 500.0)
        observations = tuple(
            BearingObservation(
                observation_id=f"{u}:{time}",
                scenario_id="S",
                target_id="T",
                uuv_id=u,
                sim_time_s=time,
                azimuth_rad=atan2(point[1] - p[1], point[0] - p[0]),
                variance_rad2=0.001,
                detection_confidence=1.0,
                observer_position_xy=p,
            )
            for u, p in positions.items()
        )
        report = manager.invoke("T", observations=observations, sim_time_s=time)
        history.append((time, *report.belief.mean[:2]))
    snapshot = SituationSnapshot(
        scenario_id="S",
        snapshot_revision=3,
        sim_time_s=90,
        uuvs=(),
        group_reports=(report,),
        pending_events=(),
        map_bounds_xy=(-10000.0, 10000.0, -10000.0, 10000.0),
        region_probability_evidence={
            "R": public_region_probability(
                belief=report.belief,
                now_s=90,
                polygon_xy=((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)),
            )
        },
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda *_: history, horizon_s=1800.0, sample_step_s=30.0
    )
    accepted = predictor(snapshot, "T")
    assert accepted.prediction is not None  # A silent/empty pipeline cannot pass non-interference.
    forecasts = build_world_model_forecasts(
        snapshot, {"T": accepted.prediction}, accepted_predictions={"T": accepted}
    )
    assert forecasts["T"].data_status in {"ready", "degraded"}
    assert any(window.covered for window in forecasts["T"].horizons)
    return snapshot, accepted, forecasts, history


class CaptureLLM:
    def __init__(self):
        self.payloads = []

    def invoke_structured(self, operation, payload, response_model, **kwargs):
        self.payloads.append(payload)
        return IntentHypothesis(
            label="transit",
            confidence=0.8,
            evidence_ids=(payload["evidence_ids"][0],),
            model_id="capture-only",
            prompt_version="test",
        )


def test_changed_evaluation_truth_cannot_reach_events_prompt_http_ws_jsonl_or_replay(tmp_path):
    outcomes = []
    for index, point in enumerate(((8123.0, 7123.0), (-9234.0, -8234.0))):
        snapshot, accepted, forecasts, history = public_pipeline()
        evaluation = EvaluationFrame(
            frame_id=3,
            sim_time_s=90,
            scenario_id="S",
            run_id="truth-audit",
            plan_version=0,
            targets=(TargetTruth("T", point, (9.0, 8.0), "private-label"),),
        )
        # Deliberately place evaluation data alongside operational runtime state;
        # publication must select its public contracts, not serialize all state.
        state = {
            "accepted_predictions": {"T": accepted},
            "world_model_forecasts": forecasts,
            "evaluation_frame": evaluation,
        }
        runtime = SimpleNamespace(get_state=lambda state=state: state, active_plan=lambda: None)
        capture = CaptureLLM()
        node = IntentAnalysisNode(capture)
        node._invoke_intent(node.build_payload(snapshot, "T", belief_history=history))
        hub = OperationalHub()
        log = tmp_path / f"public-{index}.jsonl"
        publisher = OperationalFramePublisher(
            runtime=runtime, ledger=Ledger(), events=Events(), hub=hub, logger=FrameLogger(log)
        )
        frame = publisher.publish(snapshot)
        publisher.close()
        replay = ReplayService(log)
        with TestClient(create_app(runtime=runtime, hub=hub, replay=replay)) as client:
            http = client.get("/api/operational/snapshot").json()
            with client.websocket_connect("/ws/operational") as websocket:
                ws = websocket.receive_json()
            recorded = replay.range()[0].model_dump(mode="json")
            assert http == ws == recorded == frame.model_dump(mode="json")
        serialized = log.read_text(encoding="utf-8")
        assert "private-label" not in serialized and "evaluation_frame" not in serialized
        assert frame.region_probability_evidence["R"]["probability"] is not None
        for coordinate in point:
            assert str(coordinate) not in serialized
        outcomes.append((serialized, capture.payloads))
    assert outcomes[0] == outcomes[1]


def test_changed_public_observation_changes_estimate_prediction_and_prompt():
    first = public_pipeline(0.0)
    changed = public_pipeline(40.0)
    assert first[0].group_reports[0].belief.mean != changed[0].group_reports[0].belief.mean
    assert first[1].prediction.points_xy != changed[1].prediction.points_xy
    node = IntentAnalysisNode(CaptureLLM())
    assert node.build_payload(first[0], "T", belief_history=first[3]) != node.build_payload(
        changed[0], "T", belief_history=changed[3]
    )
