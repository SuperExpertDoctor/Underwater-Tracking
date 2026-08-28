import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  PredictionDiffView,
  TargetEstimateView,
} from "../types/frames";
import PredictionDiffPanel from "./PredictionDiffPanel";

const BASE_DIFF: PredictionDiffView = {
  diff_id: "diff-T1-2",
  state: "suspected",
  status: "active",
  reason: null,
  absolute_rms_m: 300,
  normalized_rms: 3,
  absolute_floor_m: 250,
  normalized_threshold: 2.45,
  consecutive_count: 2,
  confirmation_cycles: 2,
  previous_prediction_id: "prediction-T1-1",
  current_prediction_id: "prediction-T1-2",
  leading_model_changed: true,
  js_distance: 0.13,
  suspicion_event_id: "event-T1-2",
  confirmed_intent: null,
  resulting_plan_revision: null,
};

function targetWithDiff(
  overrides: Partial<PredictionDiffView> = {},
): TargetEstimateView {
  return {
    target_id: "T1",
    mean: { x: 10, y: 20 },
    covariance_ellipse: {
      semimajor_m: 10,
      semiminor_m: 5,
      rotation_rad: 0,
    },
    intent: { label: "transit", confidence: 0.8, alternatives: {} },
    prediction: {
      prediction_id: "prediction-T1-2",
      prediction_revision: 2,
      origin_sim_time_s: 0,
      health: {
        status: "valid",
        regime: "imm",
        reason_codes: [],
        source_track_age_s: 0,
        clipped_point_fraction: 0,
        maximum_radius_m: 0,
        raw_prediction_id: "prediction-T1-2",
      },
      horizon_s: 900,
      sample_step_s: 30,
      centerline_xy: [],
      radius_m: [],
      point_confidence: [],
      diff: { ...BASE_DIFF, ...overrides },
    },
    quality: {
      quality_score: 0.9,
      estimated_rmse_m: 8,
      fim_min_eigenvalue: 1,
      fim_condition: 2,
    },
    classification: "unknown",
    last_ping_s: null,
  };
}

describe("PredictionDiffPanel", () => {
  it("distinguishes suspected prediction divergence from confirmed intent", () => {
    render(
      <PredictionDiffPanel
        targets={[targetWithDiff({ state: "suspected" })]}
      />,
    );

    expect(screen.getByText("疑似行为变化")).toBeInTheDocument();
    expect(screen.getByText("300 m")).toBeInTheDocument();
    expect(screen.getByText("3.00 / 2.45")).toBeInTheDocument();
    expect(screen.queryByText("意图已改变")).not.toBeInTheDocument();
  });

  it("renders unavailable reasons without showing a zero score", () => {
    render(
      <PredictionDiffPanel
        targets={[
          targetWithDiff({
            state: "unavailable",
            reason: "insufficient_overlap",
            absolute_rms_m: null,
            normalized_rms: null,
          }),
        ]}
      />,
    );

    expect(screen.getByText("证据不足")).toBeInTheDocument();
    expect(screen.queryByText("0.00")).not.toBeInTheDocument();
  });

  it("renders nothing when no target has diff evidence", () => {
    const target = targetWithDiff();
    target.prediction = null;
    const { container } = render(<PredictionDiffPanel targets={[target]} />);

    expect(container).toBeEmptyDOMElement();
  });
});
