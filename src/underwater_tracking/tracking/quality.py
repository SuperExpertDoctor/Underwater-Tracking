"""Group-level tracking quality with hard guards.

The instantaneous group quality is the approved weighted mix

    Q = 0.30 q_cov + 0.25 q_FIM + 0.20 q_detect + 0.15 q_NIS + 0.10 q_fresh

with every component normalized into [0, 1] against explicit reference
scales from ``TrackingConfig``: covariance trace against
``covariance_reference_m2``, the FIM minimum eigenvalue, derived
log-determinant, and condition number against ``fim_min_eigenvalue_reference``
and ``fim_condition_reference``. The ``q_FIM`` component itself is the
weighted mix 0.5/0.3/0.2 of the minimum eigenvalue, determinant, and
condition sub-scores. Each component uses the same smooth ratio family,
``x / (x + reference)`` for "more is better" measures and
``reference / (reference + x)`` for "less is better" measures, which is
monotone, clamped only by input sanity, and never leaves [0, 1].
Freshness decays exponentially with the quality window: a belief updated
``window_s`` seconds ago scores ``exp(-1)``.

The calculator keeps every timestamped instant in a deque and maintains
both a time-bounded window mean (only samples within ``window_s`` of the
current time) and an EWMA with alpha ``ewma_alpha``.

Hard guards do not wait for the averages: when any of the four guard
conditions below holds, the reported instant quality is pinned to 0.0 so
downstream consumers treat the group as immediately critical while the
window mean and EWMA still reflect the honest statistics.

Guard reason strings:

- ``"no_accepted_observation"``: detection rate is zero in the window;
- ``"fim_degenerate"``: FIM minimum eigenvalue below 1% of the reference;
- ``"fim_ill_conditioned"``: FIM condition number above 100x the reference;
- ``"covariance_overflow"``: position covariance trace above 100x the reference.

All arithmetic is deterministic; no randomness is involved.
"""

from collections import deque
from dataclasses import dataclass
import math

from underwater_tracking.config.models import TrackingConfig
from underwater_tracking.domain.models import GroupQuality


@dataclass(frozen=True, slots=True)
class QualityInputs:
    """Per-update observations consumed by the quality calculator.

    ``normalized_nis`` and ``detection_rate`` are expected to already be
    normalized into [0, 1]; the calculator clamps them defensively.
    ``age_s`` is the seconds since the last accepted observation.
    """

    covariance_trace: float
    fim_min_eigenvalue: float
    fim_condition: float
    detection_rate: float
    normalized_nis: float
    age_s: float


# Approved instant-quality weights (spec section 13).
_WEIGHT_COVARIANCE = 0.30
_WEIGHT_FIM = 0.25
_WEIGHT_DETECTION = 0.20
_WEIGHT_NIS = 0.15
_WEIGHT_FRESHNESS = 0.10

# Sub-weights inside the q_FIM combination.
_FIM_MIN_EIGENVALUE_WEIGHT = 0.5
_FIM_DETERMINANT_WEIGHT = 0.3
_FIM_CONDITION_WEIGHT = 0.2

# Hard-guard thresholds expressed as multiples of the reference scales.
_DEGENERATE_FIM_RATIO = 0.01
_ILL_CONDITIONED_RATIO = 100.0
_COVARIANCE_OVERFLOW_RATIO = 100.0


class QualityCalculator:
    """Stateful calculator of instantaneous, window-mean, and EWMA quality."""

    def __init__(
        self,
        window_s: int = 300,
        ewma_alpha: float = 0.2,
        references: TrackingConfig | None = None,
    ) -> None:
        """Create a calculator with the given window and EWMA smoothing.

        ``references`` supplies the normalization reference scales;
        when omitted, the ``TrackingConfig`` defaults are used.
        """
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in the open interval (0, 1]")
        refs = references if references is not None else TrackingConfig()
        self._window_s = float(window_s)
        self._ewma_alpha = float(ewma_alpha)
        self._covariance_reference = refs.covariance_reference_m2
        self._fim_min_eigenvalue_reference = refs.fim_min_eigenvalue_reference
        self._fim_condition_reference = refs.fim_condition_reference
        self._samples: deque[tuple[float, float]] = deque()
        self._ewma: float | None = None

    def update(self, time: float, inputs: QualityInputs) -> GroupQuality:
        """Record one quality sample at ``time`` and return the summary.

        The returned ``GroupQuality`` carries the instantaneous score
        (pinned to 0.0 when a hard guard fires), the time-bounded window
        mean, the EWMA, the per-component scores, and the guard reasons.
        """
        components = self._components(inputs)
        guards = self._hard_guards(inputs)
        instant = _clamp01(
            _WEIGHT_COVARIANCE * components["covariance"]
            + _WEIGHT_FIM * components["fim"]
            + _WEIGHT_DETECTION * components["detection"]
            + _WEIGHT_NIS * components["nis"]
            + _WEIGHT_FRESHNESS * components["freshness"]
        )
        if guards:
            instant = 0.0
        self._samples.append((time, instant))
        cutoff = time - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        window_mean = sum(value for _, value in self._samples) / len(self._samples)
        if self._ewma is None:
            self._ewma = instant
        else:
            self._ewma = (
                (1.0 - self._ewma_alpha) * self._ewma + self._ewma_alpha * instant
            )
        return GroupQuality(
            instant=instant,
            window_mean=window_mean,
            ewma=self._ewma,
            components=components,
            hard_guard_reasons=tuple(guards),
        )

    def _components(self, inputs: QualityInputs) -> dict[str, float]:
        covariance_trace = max(0.0, inputs.covariance_trace)
        min_eigenvalue = max(0.0, inputs.fim_min_eigenvalue)
        condition = max(1.0, inputs.fim_condition)
        q_covariance = self._covariance_reference / (
            self._covariance_reference + covariance_trace
        )
        q_min_eigenvalue = min_eigenvalue / (
            min_eigenvalue + self._fim_min_eigenvalue_reference
        )
        reference_logdet = 2.0 * math.log(self._fim_min_eigenvalue_reference)
        q_determinant = _logistic(
            _estimate_logdet(min_eigenvalue, condition) - reference_logdet
        )
        q_condition = self._fim_condition_reference / (
            self._fim_condition_reference + condition
        )
        q_fim = (
            _FIM_MIN_EIGENVALUE_WEIGHT * q_min_eigenvalue
            + _FIM_DETERMINANT_WEIGHT * q_determinant
            + _FIM_CONDITION_WEIGHT * q_condition
        )
        q_detection = _clamp01(inputs.detection_rate)
        q_nis = _clamp01(inputs.normalized_nis)
        q_freshness = math.exp(-max(0.0, inputs.age_s) / self._window_s)
        return {
            "covariance": q_covariance,
            "fim": q_fim,
            "detection": q_detection,
            "nis": q_nis,
            "freshness": q_freshness,
        }

    def _hard_guards(self, inputs: QualityInputs) -> list[str]:
        guards: list[str] = []
        if inputs.detection_rate <= 0.0:
            guards.append("no_accepted_observation")
        if inputs.fim_min_eigenvalue < (
            self._fim_min_eigenvalue_reference * _DEGENERATE_FIM_RATIO
        ):
            guards.append("fim_degenerate")
        if inputs.fim_condition > (
            self._fim_condition_reference * _ILL_CONDITIONED_RATIO
        ):
            guards.append("fim_ill_conditioned")
        if inputs.covariance_trace > (
            self._covariance_reference * _COVARIANCE_OVERFLOW_RATIO
        ):
            guards.append("covariance_overflow")
        return guards


def _estimate_logdet(min_eigenvalue: float, condition: float) -> float:
    """Estimate log det(F) for a 2x2 FIM from its summary metrics.

    With eigenvalues ``min`` and ``max = condition * min``,
    ``log det(F) = log(condition) + 2 log(min)``. A zero minimum
    eigenvalue means the matrix is singular, so the log-determinant is
    negative infinity.
    """
    if min_eigenvalue <= 0.0:
        return float("-inf")
    return math.log(condition) + 2.0 * math.log(min_eigenvalue)


def _logistic(value: float) -> float:
    """Numerically stable logistic function, monotone into (0, 1)."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    positive = math.exp(value)
    return positive / (1.0 + positive)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
