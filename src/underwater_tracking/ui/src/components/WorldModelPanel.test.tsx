import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TargetEstimateView, WorldModelForecastView } from "../types/frames";
import WorldModelPanel from "./WorldModelPanel";

const FORECAST: WorldModelForecastView = {
  model_kind: "rule_demo",
  model_version: "rule-event-v1",
  control_authority: false,
  as_of_s: 600,
  source_prediction_id: "scenario:track:T1:20",
  source_observation_ids: ["obs-20"],
  source_observability_event_ids: [],
  source_plan_revision: 4,
  data_status: "ready",
  trajectory_fallback_used: false,
  imm_model_probabilities: { cv: 0.22, left_turn: 0.78 },
  horizons: [
    { name: "H1", start_offset_s: 0, end_offset_s: 120, sample_count: 4, covered: true },
    { name: "H2", start_offset_s: 120, end_offset_s: 300, sample_count: 6, covered: true },
    { name: "H3", start_offset_s: 300, end_offset_s: 900, sample_count: 20, covered: true },
    { name: "H4", start_offset_s: 900, end_offset_s: 1800, sample_count: 30, covered: true },
  ],
  events: [
    {
      event_id: "event-turn",
      event_type: "target_turn_left",
      horizon: "H1",
      predicted_time_s: 720,
      time_to_event_s: 120,
      predicted_position: { x: 800, y: 120 },
      confidence: 0.78,
      level: "tactical",
      rule_id: "R-TURN-001",
      summary: "预测目标将在该时间段向左转向",
      evidence: [],
    },
  ],
  warnings: [],
};

function target(forecast: WorldModelForecastView | null): TargetEstimateView {
  return {
    target_id: "T1",
    mean: { x: 0, y: 0 },
    covariance_ellipse: { semimajor_m: 10, semiminor_m: 5, rotation_rad: 0 },
    intent: { label: "evade", confidence: 0.8, alternatives: {} },
    prediction: null,
    world_model: forecast,
    quality: {
      quality_score: 0.8,
      estimated_rmse_m: 20,
      fim_min_eigenvalue: 1,
      fim_condition: 2,
    },
    classification: "submarine",
    last_ping_s: null,
  };
}

describe("WorldModelPanel", () => {
  it("shows explainable H1-H4 events as read-only rule output", () => {
    render(<WorldModelPanel targets={[target(FORECAST)]} />);

    expect(screen.getByText("未来事件推演")).toBeInTheDocument();
    expect(screen.getByText("只读")).toBeInTheDocument();
    expect(screen.getByText("目标左转")).toBeInTheDocument();
    expect(screen.getByText("T+2 分钟")).toBeInTheDocument();
    expect(screen.getByText("规则置信度 78%")).toBeInTheDocument();
    expect(screen.getByText("方案 v4")).toBeInTheDocument();
    expect(screen.getByText(/不直接控制 UUV/)).toBeInTheDocument();
  });

  it("distinguishes a clear forecast from missing forecast data", () => {
    const clearForecast = { ...FORECAST, events: [], data_status: "degraded" as const };
    render(<WorldModelPanel targets={[target(clearForecast)]} />);

    expect(screen.getByText("当前规则未发现明显未来事件")).toBeInTheDocument();
    expect(screen.getByText("降级推演")).toBeInTheDocument();
  });

  it("renders nothing for legacy frames without world-model output", () => {
    const { container } = render(<WorldModelPanel targets={[target(null)]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it.each(["expired", "unavailable"] as const)("does not present %s as a clear forecast", (status) => {
    render(<WorldModelPanel targets={[target({ ...FORECAST, events: [], data_status: status,
      horizons: FORECAST.horizons.map((h) => ({ ...h, covered: false, sample_count: 0 })) })]} />);
    expect(screen.getByText("当前输入不足以判断未来事件")).toBeInTheDocument();
    expect(screen.queryByText("当前规则未发现明显未来事件")).not.toBeInTheDocument();
    expect(screen.queryByText("目标左转")).not.toBeInTheDocument();
    expect(screen.getAllByText("预测未覆盖")).toHaveLength(4);
  });
});
