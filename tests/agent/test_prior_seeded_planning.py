from __future__ import annotations

from math import floor, hypot

import pytest

from underwater_tracking.agent.graphs.central import (
    TrajectoryPredictionNode,
    _prior_seeded_planning_inputs,
)
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.agent.prompts import INTENT_PROMPT_VERSION, INTENT_SYSTEM_PROMPT
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
)
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.simulation.engine import SimulationEngine


class _FailClosedLLM:
    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[IntentHypothesis],
        *,
        prompt_version: str = "",
    ) -> IntentHypothesis:
        del operation, payload, response_model, prompt_version
        raise AssertionError("intent payload tests must not invoke any LLM")


def _intent_snapshot() -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=50,
        uuvs=(),
        group_reports=(
            GroupReport(
                group_id="G-T1",
                target_id="T1",
                sim_time_s=50,
                member_ids=("U1", "U2"),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=50,
                    mean=(50.0, 10.0, 1.0, 0.0),
                    covariance=(
                        (400.0, 0.0, 0.0, 0.0),
                        (0.0, 400.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                    model_probabilities={"cv": 1.0},
                    source_observation_ids=("bearing:T1:50",),
                ),
                quality=GroupQuality(
                    instant=0.8,
                    window_mean=0.75,
                    ewma=0.76,
                    components={"cov": 0.7},
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
    )


def test_intent_prompt_allows_only_observation_derived_estimated_history() -> None:
    prompt = INTENT_SYSTEM_PROMPT.lower()

    assert INTENT_PROMPT_VERSION == "intent-v4"
    assert "observation-derived estimated belief history" in prompt
    assert "globally observable" not in prompt
    assert "simulator-authorized" not in prompt


def test_intent_payload_declares_estimated_belief_history_source_without_network() -> None:
    snapshot = _intent_snapshot()
    history = tuple(
        (time_s, float(time_s), float(time_s // 5))
        for time_s in range(0, 51, 10)
    )
    node = IntentAnalysisNode(
        _FailClosedLLM(),
        belief_history=lambda _snapshot, _target_id: history,
    )

    payload = node.build_payload(snapshot, target_id="T1")

    assert payload["trajectory_history_source"] == "estimated_belief"
    assert payload["sampled_belief_history"][-1] == {
        "sim_time_s": 50,
        "x": 50.0,
        "y": 10.0,
    }


def test_public_prior_becomes_a_bounded_temporal_search_envelope() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(
        config,
        seed=7,
    )
    situation = engine.publication_situation()
    contact = next(
        contact
        for contact in situation.contacts
        if contact.classification.value == "submarine"
    )
    assert contact.estimated_position_xy is None
    engine.step()
    advanced_contact = next(
        item
        for item in engine.publication_situation().contacts
        if item.contact_id == contact.contact_id
    )
    assert advanced_contact.estimated_position_xy is None
    prior = situation.target_search_priors[0]

    state = _prior_seeded_planning_inputs(situation)
    prediction = state["predictions"][prior.target_id]

    assert prediction.fallback_used is True
    assert prediction.fallback_reason == "public_target_search_envelope"
    assert prediction.prediction_regime == "public_prior"
    assert prediction.imm_model_probabilities == {}
    assert prediction.source_belief_history_ids == ()
    assert prediction.times_s == tuple(sorted(prediction.times_s))
    assert prediction.horizon_s == config.timing.prediction_horizon_s == 1_800
    assert len(set(prediction.points_xy)) > 1
    assert prediction.points_xy[0] == prior.center_xy
    assert prediction.corridor_radius_m[0] >= max(
        prior.covariance_xy[0][0] ** 0.5,
        prior.covariance_xy[1][1] ** 0.5,
    )
    assert prediction.corridor_radius_m[-1] == pytest.approx(
        prediction.corridor_radius_m[0]
        + config.tracking.submarine_sprint_speed_mps
        * (prediction.times_s[-1] - prediction.times_s[0])
    )

    # The fallback may search outward, but it may not outrun the configured
    # public target-speed bound or smuggle a private target trajectory in.
    for time_s, point in zip(prediction.times_s, prediction.points_xy):
        elapsed_s = time_s - situation.sim_time_s
        displacement = hypot(
            point[0] - prior.center_xy[0],
            point[1] - prior.center_xy[1],
        )
        assert displacement <= config.tracking.submarine_sprint_speed_mps * elapsed_s + 1e-6


def test_public_prior_envelope_supports_four_in_bounds_task_cells() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    situation = SimulationEngine(config, seed=7).publication_situation()
    prediction = next(
        iter(_prior_seeded_planning_inputs(situation)["predictions"].values())
    )
    origin_x, origin_y = GridSpec().origin_xy
    map_min_x, map_max_x, map_min_y, map_max_y = config.environment.map_bounds_xy
    task_cells = {
        (
            floor((point[0] - origin_x) / 1_000.0),
            floor((point[1] - origin_y) / 1_000.0),
        )
        for point in prediction.points_xy
        if map_min_x <= point[0] <= map_max_x and map_min_y <= point[1] <= map_max_y
    }

    assert len(task_cells) >= 4


def test_public_prior_prediction_stops_at_its_valid_until_boundary() -> None:
    situation = SimulationEngine(
        load_app_config("configs/scenario/uuv_only_single_target.yaml"),
        seed=7,
    ).publication_situation()
    prior = situation.target_search_priors[0]
    near_expiry = situation.model_copy(
        update={
            "sim_time_s": prior.valid_until_s - 100,
            "snapshot_revision": prior.valid_until_s - 100,
        }
    )

    prediction = _prior_seeded_planning_inputs(near_expiry)["predictions"][prior.target_id]

    assert prediction.horizon_s == 100.0
    assert prediction.times_s[-1] == prior.valid_until_s
    assert max(prediction.times_s) <= prior.valid_until_s


def test_prediction_node_drops_cached_public_prior_when_prior_expires() -> None:
    situation = SimulationEngine(
        load_app_config("configs/scenario/uuv_only_single_target.yaml"),
        seed=7,
    ).publication_situation()
    prior = situation.target_search_priors[0]
    seeded = _prior_seeded_planning_inputs(situation)
    expired = situation.model_copy(
        update={
            "sim_time_s": prior.valid_until_s,
            "snapshot_revision": prior.valid_until_s,
            "target_search_priors": (),
        }
    )
    node = TrajectoryPredictionNode(
        lambda _snapshot, _target_id: (_ for _ in ()).throw(
            AssertionError("expired public prior must not call the predictor")
        ),
        lambda _snapshot_ref: expired,
        uuv_only=True,
    )

    result = node(
        {
            **seeded,
            "scenario_id": situation.scenario_id,
            "snapshot_ref": "expired",
            "prediction_snapshot_revision": expired.snapshot_revision,
        }
    )

    assert result["predictions"] == {}
    assert result["intent_hypotheses"] == {}


def test_prediction_node_preserves_non_prior_forecast_during_temporary_loss() -> None:
    situation = SimulationEngine(
        load_app_config("configs/scenario/uuv_only_single_target.yaml"),
        seed=7,
    ).publication_situation()
    prior = situation.target_search_priors[0]
    public_prediction = _prior_seeded_planning_inputs(situation)["predictions"][prior.target_id]
    observed_prediction = public_prediction.model_copy(
        update={
            "prediction_id": "observed:T1:0",
            "prediction_regime": "short_history",
            "fallback_used": False,
            "fallback_reason": None,
            "source_belief_history_ids": ("bearing:T1:0",),
        }
    )
    observed_intent = IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("bearing:T1:0",),
        model_id="observation-intent-model",
        prompt_version="intent-v1",
    )
    temporarily_unobserved = situation.model_copy(
        update={
            "sim_time_s": 30,
            "snapshot_revision": 30,
            "target_search_priors": (),
        }
    )
    node = TrajectoryPredictionNode(
        lambda _snapshot, _target_id: (_ for _ in ()).throw(
            AssertionError("temporary loss must not call the predictor")
        ),
        lambda _snapshot_ref: temporarily_unobserved,
        uuv_only=True,
    )

    result = node(
        {
            "scenario_id": situation.scenario_id,
            "snapshot_ref": "temporarily-unobserved",
            "predictions": {prior.target_id: observed_prediction},
            "intent_hypotheses": {prior.target_id: observed_intent},
            "prediction_snapshot_revision": situation.snapshot_revision,
        }
    )

    assert result["predictions"] == {prior.target_id: observed_prediction}
    assert result["intent_hypotheses"] == {prior.target_id: observed_intent}


def test_same_revision_cache_drops_only_inactive_public_prior_target() -> None:
    situation = SimulationEngine(
        load_app_config("configs/scenario/uuv_only_single_target.yaml"),
        seed=7,
    ).publication_situation()
    expired_prior = situation.target_search_priors[0]
    active_prior = expired_prior.model_copy(
        update={
            "prior_id": "intel-target-01-active",
            "target_id": "target_01",
        }
    )
    two_prior_situation = situation.model_copy(
        update={"target_search_priors": (expired_prior, active_prior)}
    )
    seeded = _prior_seeded_planning_inputs(two_prior_situation)
    observed_prediction = seeded["predictions"][active_prior.target_id].model_copy(
        update={
            "prediction_id": "observed:target_02:0",
            "target_id": "target_02",
            "prediction_regime": "short_history",
            "fallback_used": False,
            "fallback_reason": None,
            "source_belief_history_ids": ("bearing:target_02:0",),
        }
    )
    observed_intent = IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("bearing:target_02:0",),
        model_id="observation-intent-model",
        prompt_version="intent-v1",
    )
    cached_predictions = {
        **seeded["predictions"],
        observed_prediction.target_id: observed_prediction,
    }
    cached_intents = {
        **seeded["intent_hypotheses"],
        observed_prediction.target_id: observed_intent,
    }
    active_only = two_prior_situation.model_copy(
        update={
            "snapshot_revision": 1,
            "target_search_priors": (active_prior,),
        }
    )
    node = TrajectoryPredictionNode(
        lambda _snapshot, _target_id: (_ for _ in ()).throw(
            AssertionError("same-revision cache refresh must not call the predictor")
        ),
        lambda _snapshot_ref: active_only,
        uuv_only=True,
    )
    diffs = {target_id: f"diff:{target_id}" for target_id in cached_predictions}
    gates = {target_id: f"gate:{target_id}" for target_id in cached_predictions}

    result = node(
        {
            "scenario_id": situation.scenario_id,
            "snapshot_ref": "partial-expiry",
            "predictions": cached_predictions,
            "intent_hypotheses": cached_intents,
            "prediction_diffs": diffs,
            "prediction_diff_gates": gates,
            "prediction_intent_verification_target_ids": tuple(cached_predictions),
            "prediction_snapshot_revision": active_only.snapshot_revision,
        }
    )

    retained_ids = {active_prior.target_id, observed_prediction.target_id}
    assert set(result["predictions"]) == retained_ids
    assert set(result["intent_hypotheses"]) == retained_ids
    assert set(result["prediction_diffs"]) == retained_ids
    assert set(result["prediction_diff_gates"]) == retained_ids
    assert set(result["prediction_intent_verification_target_ids"]) == retained_ids
    assert result["predictions"][active_prior.target_id] == cached_predictions[
        active_prior.target_id
    ]
    assert result["predictions"][observed_prediction.target_id] == observed_prediction
    assert result["intent_hypotheses"][observed_prediction.target_id] == observed_intent
