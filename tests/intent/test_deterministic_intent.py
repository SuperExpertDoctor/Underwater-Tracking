from __future__ import annotations

from underwater_tracking.intent.deterministic import (
    DeterministicIntentClassifier,
    MotionIntentFeatures,
)


def _history(label: str) -> tuple[tuple[float, float, float], ...]:
    if label == "transit":
        return tuple((time, 10.0 * time, 0.0) for time in range(0, 41, 10))
    if label == "loiter":
        return (
            (0.0, 0.0, 0.0),
            (10.0, 2.0, 1.0),
            (20.0, -1.0, 2.0),
            (30.0, 1.0, -1.0),
            (40.0, 0.0, 0.0),
        )
    if label == "patrol":
        return (
            (0.0, 0.0, 0.0),
            (10.0, 100.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 100.0, 0.0),
            (40.0, 0.0, 0.0),
        )
    if label == "evade":
        return (
            (0.0, 0.0, 0.0),
            (10.0, 100.0, 0.0),
            (20.0, 100.0, 100.0),
            (30.0, 0.0, 100.0),
            (40.0, -100.0, 100.0),
        )
    if label == "approach":
        return tuple((time, 500.0 - 10.0 * time, 0.0) for time in range(0, 41, 10))
    if label == "withdraw":
        return tuple((time, 50.0 + 10.0 * time, 0.0) for time in range(0, 41, 10))
    raise AssertionError(label)


def test_classifier_covers_the_six_operational_motion_labels() -> None:
    classifier = DeterministicIntentClassifier(confirmation_cycles=1)

    labels = {
        label: classifier.classify_history(
            "T1",
            _history(label),
            prediction_revision=4,
            boundary=(0.0, 1000.0, -1000.0, 1000.0)
            if label in {"approach", "withdraw"}
            else None,
        ).intent_label
        for label in ("transit", "loiter", "patrol", "evade", "approach", "withdraw")
    }

    assert labels == {
        "transit": "transit",
        "loiter": "loiter",
        "patrol": "patrol",
        "evade": "evade",
        "approach": "approach",
        "withdraw": "withdraw",
    }


def test_intent_changes_require_consecutive_cycles_and_use_hysteresis() -> None:
    classifier = DeterministicIntentClassifier(confirmation_cycles=2)
    transit = classifier.classify_history("T1", _history("transit"), prediction_revision=1)
    noisy_evasion = MotionIntentFeatures(
        target_id="T1",
        sim_time_s=60.0,
        mean_speed_mps=8.0,
        max_speed_mps=9.0,
        acceleration_mps2=1.2,
        heading_change_rad=1.5,
        signed_turn_rate_rad_s=0.01,
        curvature_q75=0.01,
        path_efficiency=0.75,
        dwell_fraction=0.0,
        boundary_distance_m=500.0,
        boundary_approach_rate_mps=0.0,
        leading_model="CT_LEFT",
        leading_model_probability=0.52,
    )

    first = classifier.classify(noisy_evasion, prior=transit.latch_state, prediction_revision=2)
    second = classifier.classify(noisy_evasion, prior=first.latch_state, prediction_revision=3)

    assert first.intent_label == "transit"
    assert first.latch_state.candidate_label == "evade"
    assert second.intent_label == "evade"
    assert second.intent_revision == transit.intent_revision + 1


def test_small_imm_probability_jitter_does_not_change_semantic_intent() -> None:
    classifier = DeterministicIntentClassifier(confirmation_cycles=1)
    base = MotionIntentFeatures(
        target_id="T1",
        sim_time_s=30.0,
        mean_speed_mps=8.0,
        max_speed_mps=8.2,
        acceleration_mps2=0.05,
        heading_change_rad=0.02,
        signed_turn_rate_rad_s=0.0002,
        curvature_q75=0.0001,
        path_efficiency=0.98,
        dwell_fraction=0.0,
        boundary_distance_m=500.0,
        boundary_approach_rate_mps=0.0,
        leading_model="CV",
        leading_model_probability=0.51,
        model_probability_change=0.01,
    )
    first = classifier.classify(base, prediction_revision=1)
    jittered = base.model_copy(
        update={
            "leading_model": "CT_LEFT",
            "leading_model_probability": 0.52,
            "model_probability_change": 0.02,
        }
    )
    second = classifier.classify(
        jittered,
        prior=first.latch_state,
        prediction_revision=2,
    )

    assert first.intent_label == "transit"
    assert second.intent_label == "transit"


def test_llm_revision_requires_two_current_high_confidence_evidence_backed_calls() -> None:
    classifier = DeterministicIntentClassifier(confirmation_cycles=1)
    baseline = classifier.classify_history("T1", _history("transit"), prediction_revision=8)
    allowed = ("track:T1:8", "prediction:T1:8")

    rejected = classifier.accept_llm_revision(
        baseline,
        proposed_label="evade",
        confidence=0.91,
        evidence_ids=("outside",),
        allowed_evidence_ids=allowed,
        prediction_revision=8,
    )
    first = classifier.accept_llm_revision(
        baseline,
        proposed_label="evade",
        confidence=0.91,
        evidence_ids=allowed,
        allowed_evidence_ids=allowed,
        prediction_revision=8,
    )
    second = classifier.accept_llm_revision(
        first,
        proposed_label="evade",
        confidence=0.91,
        evidence_ids=allowed,
        allowed_evidence_ids=allowed,
        prediction_revision=8,
    )

    assert rejected.llm_revision_accepted is False
    assert first.intent_label == baseline.intent_label
    assert second.intent_label == "evade"
    assert second.llm_revision_accepted is True
    assert second.latch_state.llm_candidate_label == "evade"
