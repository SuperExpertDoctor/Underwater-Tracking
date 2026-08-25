import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PredictionOverlay, { confidenceAdjustedRadii } from "./PredictionOverlay";

describe("IMM prediction confidence band", () => {
  it("widens lower-confidence points while preserving the covariance radius", () => {
    expect(confidenceAdjustedRadii({
      horizon_s: 60,
      sample_step_s: 30,
      centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 0 }],
      radius_m: [100, 200],
      point_confidence: [0.8, 0.2],
    })).toEqual([110, 280]);
  });

  it("uses the published covariance radius for legacy frames", () => {
    expect(confidenceAdjustedRadii({
      horizon_s: 60,
      sample_step_s: 30,
      centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 0 }],
      radius_m: [100, 200],
    })).toEqual([100, 200]);
  });

  it("renders the confidence envelope above regions with one marker per estimate", () => {
    const prediction = {
      horizon_s: 60,
      sample_step_s: 30,
      centerline_xy: [{ x: 0, y: 0 }, { x: 100, y: 20 }],
      radius_m: [10, 20],
      point_confidence: [0.9, 0.4],
    };

    render(createElement(PredictionOverlay, {
      predictions: [{ targetId: "T1", prediction }],
      project: (point: { x: number; y: number }) => point,
      width: 200,
      height: 100,
    }));

    const overlay = screen.getByLabelText("IMM 预测置信轨迹");
    expect(overlay.querySelectorAll(".imm-confidence-band")).toHaveLength(1);
    expect(overlay.querySelectorAll(".imm-prediction-centerline")).toHaveLength(1);
    expect(overlay.querySelectorAll(".imm-prediction-point")).toHaveLength(2);
  });
});
