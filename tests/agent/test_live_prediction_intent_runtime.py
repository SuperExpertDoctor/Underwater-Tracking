from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth


class _EventStore:
    def __init__(self) -> None:
        self.appended: list[str] = []

    def append_if_absent(self, **payload: object) -> int:
        self.appended.append(str(payload["event_id"]))
        return 1


class _Predictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, situation: object, target_id: str) -> AcceptedPrediction:
        self.calls += 1
        offset = 500.0 * (self.calls - 1)
        prediction = PredictedTrackRef(
            prediction_id=f"prediction-{self.calls}",
            target_id=target_id,
            sim_time_s=int(situation.sim_time_s),
            horizon_s=300.0,
            sample_step_s=100.0,
            times_s=(100.0, 200.0, 300.0, 400.0),
            points_xy=((offset, 0.0),) * 4,
            corridor_radius_m=(1.0,) * 4,
            source_belief_history_ids=(f"belief-{self.calls}",),
        )
        return AcceptedPrediction(
            prediction=prediction,
            health=PredictionHealth(
                status="degraded",
                regime="short_history",
                source_track_age_s=0.0,
                clipped_point_fraction=0.0,
                maximum_radius_m=1.0,
                raw_prediction_id=prediction.prediction_id,
            ),
        )


def test_refresh_predictions_queues_intent_verification_without_calling_llm() -> None:
    predictor = _Predictor()
    event_store = _EventStore()
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = "S1"
    runtime._dependencies = SimpleNamespace(
        predictor=predictor,
        uuv_only=False,
        trajectory_diff_config=TrajectoryDiffConfig(
            minimum_overlap_s=100.0,
            minimum_samples=3,
        ),
        events=event_store,
        prediction_intent_monitor=object(),
        llm=object(),
        model_id="real-intent-model",
        belief_history=None,
        intent_change_confirmation=SimpleNamespace(),
    )
    runtime._state_cache = {}
    runtime._live_prediction_state = {}
    runtime._live_prediction_events = ()
    runtime._live_prediction_event_ids = set()
    runtime._live_prediction_pending_events = deque()
    runtime._live_prediction_snapshot_revision = -1
    runtime._live_prediction_lock = __import__("threading").RLock()

    def situation(revision: int, sim_time_s: int) -> object:
        return SimpleNamespace(
            scenario_id="S1",
            snapshot_revision=revision,
            sim_time_s=sim_time_s,
            group_reports=(SimpleNamespace(target_id="T1"),),
            target_search_priors=(),
        )

    runtime.refresh_predictions(situation(1, 30))
    runtime.refresh_predictions(situation(2, 60))
    state = runtime.refresh_predictions(situation(3, 90))

    assert state["prediction_intent_confirmed"] is False
    assert state["prediction_intent_verification_target_ids"] == ("T1",)
    assert event_store.appended == ["S1:target_intent_change_suspected:T1:90"]
