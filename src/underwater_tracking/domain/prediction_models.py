from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.execution_models import NonNegativeFloat, UnitFloat
from underwater_tracking.domain.models import StrictModel


PredictionHealthStatus = Literal["valid", "degraded", "unavailable"]
AcceptedPredictionRegime = Literal[
    "imm", "bspline", "short_history", "boundary_recovery"
]


class PredictionHealth(StrictModel):
    status: PredictionHealthStatus
    regime: AcceptedPredictionRegime
    reason_codes: tuple[str, ...] = ()
    source_track_age_s: NonNegativeFloat
    clipped_point_fraction: UnitFloat
    maximum_radius_m: NonNegativeFloat
    raw_prediction_id: str | None = None


class AcceptedPrediction(StrictModel):
    prediction: PredictedTrackRef | None = None
    health: PredictionHealth

    @model_validator(mode="after")
    def validate_payload(self) -> AcceptedPrediction:
        if self.health.status == "unavailable" and self.prediction is not None:
            raise ValueError("unavailable prediction cannot carry a payload")
        if self.health.status != "unavailable" and self.prediction is None:
            raise ValueError("valid prediction requires a payload")
        return self
