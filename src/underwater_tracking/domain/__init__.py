# src/underwater_tracking/domain/__init__.py
"""Domain contracts for the underwater tracking assistant.

UI frame contracts (``ui_models``) are re-exported explicitly; truth
carriers stay confined to ``ui_models`` and the evaluation-only path.
"""

from underwater_tracking.domain.ui_models import (
    BearingRayView,
    CovarianceEllipse,
    EstimateQualityView,
    EvaluationFrame,
    EventView,
    GroupQualityView,
    GroupView,
    IntentView,
    LedgerView,
    MapBounds,
    MetricView,
    OperationalFrame,
    PlanView,
    Point2D,
    PredictionCorridorView,
    TargetEstimateView,
    UUVView,
)

__all__ = [
    "BearingRayView",
    "CovarianceEllipse",
    "EstimateQualityView",
    "EvaluationFrame",
    "EventView",
    "GroupQualityView",
    "GroupView",
    "IntentView",
    "LedgerView",
    "MapBounds",
    "MetricView",
    "OperationalFrame",
    "PlanView",
    "Point2D",
    "PredictionCorridorView",
    "TargetEstimateView",
    "UUVView",
]
