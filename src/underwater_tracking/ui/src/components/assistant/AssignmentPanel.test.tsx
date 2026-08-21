import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TargetEstimateView, UUVView } from "../../types/frames";
import { isDeployableUuv } from "../../domain/availability";
import AssignmentPanel from "./AssignmentPanel";

const target: TargetEstimateView = {
  target_id: "T1",
  mean: { x: 10, y: 20 },
  covariance_ellipse: { semimajor_m: 10, semiminor_m: 5, rotation_rad: 0 },
  intent: { label: "transit", confidence: 0.8, alternatives: {} },
  prediction: null,
  quality: {
    quality_score: 0.9,
    estimated_rmse_m: 8,
    fim_min_eigenvalue: 1,
    fim_condition: 2,
  },
  classification: "unknown",
  last_ping_s: null,
};

const uuv = (id: string, reserved: boolean): UUVView => ({
  uuv_id: id,
  status: "tracking",
  deployment_state: "deployed",
  physically_exposed: true,
  position: { x: 0, y: 0 },
  heading_rad: 0,
  speed_mps: 2,
  energy_fraction: 0.8,
  group_id: null,
  current_waypoint: null,
  breadcrumb: [],
  sensor_mode: "passive",
  reserved,
});

const plan = {
  target_id: "T1",
  prediction_id: "prediction-1",
  revision: 2,
  cell_size_m: 125,
  regions: [
    {
      region_id: "T1:cell:0:0",
      display_name: "region_1",
      target_id: "T1",
      geometry: [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
      ],
      start_time_s: 0,
      end_time_s: 30,
      predecessor_region_ids: [],
      successor_region_ids: [],
      assigned_uuv_ids: ["uuv_01", "uuv_02"],
      tracking_mode: "heuristic_uuv" as const,
      group_id: "task_group_1",
      status: "active",
      effect: {
        status: "active" as const,
        coverage_ratio: 1,
        quality_score: 0.86,
        handoff_progress: 0.5,
        quality_source: "group_quality_proxy" as const,
        hard_guard_reasons: [],
        expert_feedback_ids: ["feedback-1"],
      },
    },
  ],
};

describe("AssignmentPanel", () => {
  it("shares the deployability contract with assignment filtering", () => {
    expect(isDeployableUuv(uuv("UUV-deployed", false))).toBe(true);
    expect(
      isDeployableUuv({
        ...uuv("UUV-onboard", false),
        deployment_state: "onboard",
      }),
    ).toBe(false);
    expect(
      isDeployableUuv({
        ...uuv("UUV-status-returning", false),
        status: "returning",
      }),
    ).toBe(false);
    expect(
      isDeployableUuv({ ...uuv("UUV-failed", false), status: "failed" }),
    ).toBe(false);
  });

  it("renders the LLM regional graph and tracking effect instead of fixed manual groups", () => {
    render(
      <AssignmentPanel
        targets={[target]}
        uuvs={[uuv("UUV-1", true)]}
        onAssign={vi.fn()}
        regionalPlans={{ T1: plan }}
      />,
    );
    expect(
      screen.getByRole("img", { name: "T1 区域接力知识图谱" }),
    ).toBeInTheDocument();
    expect(screen.getByText("启发式 UUV 协同")).toBeInTheDocument();
    expect(screen.getByText("跟踪覆盖 100% · 质量 86%")).toBeInTheDocument();
    expect(screen.queryByText("专家反馈 1 条")).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "指派跟踪" }),
    ).not.toBeInTheDocument();
  });

  it("switches graph and list while sharing the selected region", () => {
    const onSelectRegion = vi.fn();
    render(
      <AssignmentPanel
        targets={[target]}
        uuvs={[uuv("UUV-1", true)]}
        regionalPlans={{ T1: plan }}
        selectedRegionId="T1:cell:0:0"
        onSelectRegion={onSelectRegion}
      />,
    );

    expect(screen.getByRole("button", { name: "图谱" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "区域 R01" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    fireEvent.click(screen.getByRole("button", { name: "列表" }));
    fireEvent.click(screen.getByRole("button", { name: /region_1.*跟踪中/ }));
    expect(onSelectRegion).toHaveBeenCalledWith(null);
  });
});
