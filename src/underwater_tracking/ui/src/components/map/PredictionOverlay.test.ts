import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PredictionOverlay, { displayRadii } from "./PredictionOverlay";

function predictionFixture(overrides: Record<string, unknown> = {}) {
  return {
    prediction_id: "prediction-1",
    prediction_revision: 3,
    origin_sim_time_s: 120,
    health: {
      status: "valid" as const,
      regime: "imm" as const,
      reason_codes: [],
      source_track_age_s: 10,
      clipped_point_fraction: 0,
      maximum_radius_m: 400,
      raw_prediction_id: "raw-1",
    },
    horizon_s: 60,
    sample_step_s: 30,
    centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 0 }],
    radius_m: [100, 200],
    point_confidence: [0.8, 0.2],
    ...overrides,
  };
}

describe("IMM prediction confidence band", () => {
  it("renders backend radii without confidence inflation", () => {
    const prediction = predictionFixture({
      radius_m: [200, 300, 400],
      centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 200, y: 0 }],
      point_confidence: [0.9, 0.5, 0.1],
    });
    expect(displayRadii(prediction)).toEqual([200, 300, 400]);
  });

  it("uses the published covariance radius for legacy frames", () => {
    expect(displayRadii(predictionFixture())).toEqual([100, 200]);
  });

  it("does not draw a corridor for unavailable prediction health", () => {
    render(createElement(PredictionOverlay, {
      predictions: [{
        targetId: "T1",
        prediction: predictionFixture({
          health: { ...predictionFixture().health, status: "unavailable" },
        }),
      }],
      project: (point: { x: number; y: number }) => point,
      width: 200,
      height: 100,
    }));
    expect(screen.queryByTestId("prediction-corridor")).not.toBeInTheDocument();
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });

  it("uses restrained legacy styling and visible degraded styling", () => {
    for (const status of ["legacy_unknown", "degraded"] as const) {
      const view = render(createElement(PredictionOverlay, {
        predictions: [{
          targetId: "T1",
          prediction: predictionFixture({
            health: { ...predictionFixture().health, status },
          }),
        }],
        project: (point: { x: number; y: number }) => point,
        width: 200,
        height: 100,
      }));
      expect(view.container.querySelector(`[data-health-status="${status}"]`)).toBeInTheDocument();
      expect(view.container.querySelector(".prediction-health-status")).toBeInTheDocument();
      view.unmount();
    }
  });

  it("renders the confidence envelope above regions with one marker per estimate", () => {
    const prediction = predictionFixture({
      centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 20 }],
      radius_m: [10, 20],
      point_confidence: [0.9, 0.4],
    });

    render(createElement(PredictionOverlay, {
      predictions: [{ targetId: "T1", prediction }],
      project: (point: { x: number; y: number }) => point,
      width: 200,
      height: 100,
    }));

    const overlay = screen.getByLabelText("IMM 预测置信轨迹");
    expect(overlay.querySelectorAll("[data-testid=prediction-corridor]")).toHaveLength(1);
    expect(overlay.querySelectorAll(".imm-prediction-centerline")).toHaveLength(1);
    expect(overlay.querySelectorAll(".imm-prediction-point")).toHaveLength(2);
  });
});
