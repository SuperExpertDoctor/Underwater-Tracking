import pytest
from underwater_tracking.tracking.quality import QualityCalculator, QualityInputs


def _steady_inputs(**overrides):
    base = {
        "covariance_trace": 100.0,
        "fim_min_eigenvalue": 1e-3,
        "fim_condition": 1.0,
        "detection_rate": 1.0,
        "normalized_nis": 1.0,
        "age_s": 0.0,
    }
    base.update(overrides)
    return QualityInputs(**base)


def test_quality_is_bounded_and_hard_guard_bypasses_average():
    calculator = QualityCalculator(window_s=300, ewma_alpha=0.2)
    result = calculator.update(30, QualityInputs(
        covariance_trace=100.0, fim_min_eigenvalue=0.0, fim_condition=1e12,
        detection_rate=1.0, normalized_nis=1.0, age_s=0.0,
    ))
    assert 0 <= result.instant <= 1
    assert "fim_degenerate" in result.hard_guard_reasons


def test_healthy_inputs_produce_approved_weighted_score_without_guards():
    result = QualityCalculator().update(0.0, _steady_inputs())
    assert result.hard_guard_reasons == ()
    assert result.instant == pytest.approx(
        0.30 * result.components["covariance"]
        + 0.25 * result.components["fim"]
        + 0.20 * result.components["detection"]
        + 0.15 * result.components["nis"]
        + 0.10 * result.components["freshness"]
    )
    assert all(0.0 <= value <= 1.0 for value in result.components.values())


def test_each_hard_guard_reason_is_reported_and_pins_instant_to_zero():
    cases = [
        ("no_accepted_observation", {"detection_rate": 0.0}),
        ("fim_degenerate", {"fim_min_eigenvalue": 0.0}),
        ("fim_ill_conditioned", {"fim_condition": 1e8}),
        ("covariance_overflow", {"covariance_trace": 1e9}),
    ]
    for reason, overrides in cases:
        result = QualityCalculator().update(0.0, _steady_inputs(**overrides))
        assert reason in result.hard_guard_reasons
        assert result.instant == 0.0


def test_window_mean_ignores_stale_samples():
    good = _steady_inputs()
    bad = _steady_inputs(
        covariance_trace=9e5, fim_min_eigenvalue=1e-5, fim_condition=1e3,
        detection_rate=0.1, normalized_nis=0.1, age_s=300.0,
    )
    calculator = QualityCalculator(window_s=300, ewma_alpha=0.2)
    first = calculator.update(0, good)
    second = calculator.update(400, bad)
    # The t=0 sample is outside the 300 s window and must be ignored.
    assert second.window_mean == pytest.approx(second.instant)
    assert first.instant > second.instant
    # Samples within the window are averaged, so the mean sits between them.
    fresh_calculator = QualityCalculator(window_s=300, ewma_alpha=0.2)
    fresh_calculator.update(200, good)
    mixed = fresh_calculator.update(400, bad)
    assert first.instant > mixed.window_mean > second.instant


def test_ewma_converges_to_steady_instant_and_is_bounded():
    calculator = QualityCalculator(window_s=300, ewma_alpha=0.2)
    result = calculator.update(0.0, _steady_inputs())
    for time_s in range(30, 3000, 30):
        result = calculator.update(float(time_s), _steady_inputs())
    assert result.ewma == pytest.approx(result.instant, rel=1e-3)
    assert 0.0 <= result.ewma <= 1.0
