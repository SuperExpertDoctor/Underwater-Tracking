from __future__ import annotations

from tests.agent.test_prediction_intent_wiring import (
    _diff,
    _hypothesis,
    _state,
    _wiring,
)


def test_prediction_verifier_keeps_same_cycle_label_as_a_baseline() -> None:
    wiring, _llm = _wiring((_hypothesis("transit", 0.8),))
    state = _state()
    state["confirmed_intent_labels"] = {}
    state["intent_hypotheses"] = {"T1": _hypothesis("transit", 0.8)}

    result = wiring(state)

    assert result["prediction_intent_confirmed"] is False
    assert result["prediction_diff_gates"]["T1"].intent_baseline_label == "transit"
    assert result["prediction_intent_verification_target_ids"] == ()
