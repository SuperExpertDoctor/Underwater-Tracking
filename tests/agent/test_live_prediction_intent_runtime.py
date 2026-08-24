from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.models import EventLevel, RuntimeEvent


class _EventStore:
    def __init__(self) -> None:
        self.appended: list[str] = []

    def append_if_absent(self, **payload: object) -> int:
        self.appended.append(str(payload["event_id"]))
        return 1


class _Predictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, situation: object, target_id: str) -> PredictedTrackRef:
        self.calls += 1
        offset = 500.0 * (self.calls - 1)
        return PredictedTrackRef(
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


class _IntentWiring:
    instances: list[_IntentWiring] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.calls: list[dict[str, object]] = []
        self.instances.append(self)

    def __call__(self, state: dict[str, object]) -> dict[str, object]:
        self.calls.append(state)
        confirmation = RuntimeEvent(
            event_id="S1:target_intent_changed:T1:90",
            scenario_id="S1",
            sim_time_s=90,
            event_type="target_intent_changed",
            entity_id="T1",
            level=EventLevel.STRATEGIC,
            payload={"source": "real_intent_llm"},
        )
        return {
            "prediction_intent_confirmed": True,
            "prediction_intent_verification_target_ids": (),
            "confirmed_intent_labels": {"T1": "evade"},
            "intent_hypotheses": {},
            "llm_provenance": {},
            "prediction_diffs": state["prediction_diffs"],
            "prediction_diff_gates": state["prediction_diff_gates"],
            "coalesced_events": (*state["coalesced_events"], confirmation),
        }


def test_refresh_predictions_runs_intent_verification_for_latched_diff() -> None:
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

    with patch(
        "underwater_tracking.agent.runtime.PredictionIntentWiringNode",
        _IntentWiring,
    ):
        runtime.refresh_predictions(situation(1, 30))
        runtime.refresh_predictions(situation(2, 60))
        state = runtime.refresh_predictions(situation(3, 90))

    assert _IntentWiring.instances
    assert len(_IntentWiring.instances[-1].calls) == 1
    assert _IntentWiring.instances[-1].calls[0][
        "prediction_intent_verification_target_ids"
    ] == ("T1",)
    assert state["prediction_intent_confirmed"] is True
    assert state["prediction_intent_verification_target_ids"] == ()
    assert event_store.appended == [
        "S1:target_intent_change_suspected:T1:90",
        "S1:target_intent_changed:T1:90",
    ]
