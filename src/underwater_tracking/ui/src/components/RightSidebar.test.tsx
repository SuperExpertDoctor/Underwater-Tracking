import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OperationalFrame } from "../types/frames";
import RightSidebar from "./RightSidebar";

const frame: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 4,
  sim_time_s: 120,
  plan_version: 3,
  map_bounds: { min_x: -4000, min_y: -4000, max_x: 4000, max_y: 4000 },
  uuvs: [
    {
      uuv_id: "UUV-01",
      status: "tracking",
      deployment_state: "deployed",
      physically_exposed: true,
      position: { x: 40, y: 20 },
      heading_rad: 0,
      speed_mps: 2.1,
      energy_fraction: 0.72,
      group_id: "G-1",
      current_waypoint: null,
      breadcrumb: [],
      sensor_mode: "passive",
      reserved: true,
      remaining_range_m: 4200,
      communication_status: "connected",
      tracked_target_id: "T1",
    },
  ],
  target_estimates: [
    {
      target_id: "T1",
      mean: { x: 0, y: 0 },
      covariance_ellipse: {
        semimajor_m: 120,
        semiminor_m: 60,
        rotation_rad: 0,
      },
      intent: { label: "evade", confidence: 0.8, alternatives: {} },
      prediction: null,
      quality: {
        quality_score: 0.8,
        estimated_rmse_m: 50,
        fim_min_eigenvalue: 1,
        fim_condition: 2,
      },
      classification: "submarine",
      last_ping_s: 100,
      detection_range_m: 600,
      detected_platform_ids: ["UUV-01"],
    },
  ],
  bearing_rays: [],
  groups: [],
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
  operational_stage_flags: ["task_execution", "dynamic_adjustment"],
  scheme: {
    scheme_id: "scheme-1",
    version: 3,
    valid_from_s: 0,
    valid_until_s: 900,
    target_priorities: { T1: 1 },
    minimum_quality: { T1: 0.8 },
    constraints: ["keep-passive"],
  },
  intelligence: [
    {
      report_id: "intel-1",
      source: "technical_reconnaissance",
      target_id: "T1",
      confidence: 0.85,
      issued_at_s: 90,
      valid_until_s: 300,
      content_summary: "Propulsion signature changed.",
    },
  ],
  adversary: {
    target_id: "T1",
    detection_range_m: 600,
    detected_platform_ids: ["UUV-01"],
    current_decision: {
      decision_id: "adv-2",
      target_id: "T1",
      sim_time_s: 120,
      intent: "规避跟踪",
      maneuver: "转入下一分段",
      segment: "未来水域 B",
      confidence: 0.86,
      rationale: "被动观测到近距平台，调整航向并降低暴露。",
      decision_summary: "检测到 UUV-01，执行分段转移。",
      trigger_event_ids: ["evt-1"],
      detected_platform_ids: ["UUV-01"],
      active_ping_risk: "中",
      communications_discipline: "静默",
    },
    decision_history: [
      {
        decision_id: "adv-1",
        target_id: "T1",
        sim_time_s: 90,
        intent: "潜伏",
        maneuver: "保持低速",
        segment: "当前水域",
        confidence: 0.72,
        rationale: "等待态势变化",
      },
    ],
  },
};

describe("RightSidebar operational cards", () => {
  it("renders multiple backend-selected operational stages without controls", () => {
    const { container } = render(
      <RightSidebar
        frame={frame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    const matrix = screen.getByLabelText("当前作业阶段");
    expect(matrix.querySelectorAll(".operational-stage-cell")).toHaveLength(4);
    expect(
      matrix.querySelectorAll(".operational-stage-cell.active"),
    ).toHaveLength(2);
    expect(screen.getByText("任务执行").parentElement).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(screen.getByText("动态调整").parentElement).toHaveAttribute(
      "aria-current",
      "step",
    );
    expect(
      container.querySelector(".operational-stage-matrix button"),
    ).toBeNull();
  });

  it("groups the sidebar into exactly three labelled command-center panels", () => {
    const { container } = render(
      <RightSidebar
        frame={frame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    const sidebar = container.querySelector("aside");
    const panels = Array.from(sidebar?.children ?? []).filter((child) =>
      child.matches("details.sidebar-collapsible"),
    );

    expect(panels).toHaveLength(3);
    expect(
      panels.map((panel) => panel.querySelector("summary > span")?.textContent),
    ).toEqual(["当前态势", "预测与接力", "智能助理"]);
    expect(screen.queryByText("方案约束")).not.toBeInTheDocument();
    expect(screen.queryByText("专家反馈")).not.toBeInTheDocument();
    expect(screen.queryByText("态势问答")).not.toBeInTheDocument();
  });

  it("keeps all command-center panels collapsed initially and the prediction panel at sidebar root", () => {
    const { container } = render(
      <RightSidebar
        frame={frame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    const sidebar = container.querySelector("aside.sidebar");
    const predictionPanel = sidebar?.querySelector(
      ":scope > details.prediction-panel",
    );
    const predictionContent = predictionPanel?.querySelector(
      ":scope > .sidebar-collapsible-content",
    );

    expect(sidebar).toHaveClass("open");
    expect(
      sidebar?.querySelectorAll("details.sidebar-collapsible[open]"),
    ).toHaveLength(0);
    expect(predictionPanel?.parentElement).toBe(sidebar);
    expect(predictionContent?.parentElement).toBe(predictionPanel);
  });

  it("renders intelligence without exposing a scheme constraints panel", () => {
    render(
      <RightSidebar
        frame={frame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText("技侦 1 / 情报 1")).toBeInTheDocument();
    expect(screen.queryByText("方案约束")).not.toBeInTheDocument();
  });

  it("renders lower-level UUV state and toggles target-brain detail from the adversary brain card", () => {
    const { container } = render(
      <RightSidebar
        frame={frame}
        selectedUuvId="UUV-01"
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText("剩余续航")).toBeInTheDocument();
    expect(screen.getByText("4.2 km")).toBeInTheDocument();
    expect(screen.getAllByText("已连通")).toHaveLength(2);
    expect(screen.getByText("负责目标")).toBeInTheDocument();
    const adversaryBrain = container.querySelector(
      "details.adversary-brain-card > summary",
    );
    if (!adversaryBrain) throw new Error("对手脑卡片未渲染");
    const adversaryDetail = adversaryBrain.parentElement;
    expect(adversaryDetail).not.toHaveAttribute("open");
    fireEvent.click(adversaryBrain);
    expect(adversaryDetail).toHaveAttribute("open");
    expect(screen.getByText("目标潜艇脑")).toBeInTheDocument();
    expect(
      screen.getByText("检测到 UUV-01，执行分段转移。"),
    ).toBeInTheDocument();
    expect(screen.getByText("已暴露 UUV-01")).toBeInTheDocument();
    expect(screen.getByText("反跟踪历史")).toBeInTheDocument();
    fireEvent.click(adversaryBrain);
    expect(adversaryDetail).not.toHaveAttribute("open");
    expect(
      container.querySelectorAll("details.sidebar-collapsible").length,
    ).toBeGreaterThan(0);
  });

  it("adapts the current plural API adversary summary and native link states", () => {
    const apiFrame: OperationalFrame = {
      ...frame,
      adversary: null,
      adversaries: [
        {
          target_id: "T1",
          sim_time_s: 120,
          detection_range_m: 600,
          detected_platform_ids: ["UUV-01"],
          trigger_event_ids: ["evt-2"],
          decision_id: "api-adv-1",
          intent: "静默规避",
          maneuver: "降低航速",
          segment: "当前水域",
          confidence: 0.7,
          rationale: "目标根据已探测平台调整航速。",
          communications_discipline: "静默",
          decision_status: "contact_maintained",
        },
      ],
      uuvs: frame.uuvs.map((uuv) => ({
        ...uuv,
        communication_status: "carrier",
      })),
    };

    render(
      <RightSidebar
        frame={apiFrame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText("母舰直连")).toBeInTheDocument();
    expect(screen.getAllByText("静默规避")).toHaveLength(2);
    expect(
      screen.getByText("目标根据已探测平台调整航速。"),
    ).toBeInTheDocument();
  });

  it("labels a belief-only adversary estimate as awaiting brain confirmation", () => {
    render(
      <RightSidebar
        frame={{
          ...frame,
          adversary: null,
          adversaries: [
            {
              target_id: "T1",
              sim_time_s: 120,
              detection_range_m: 600,
              intent: "evade",
              maneuver: "decoy_evasion",
              confidence: 0.65,
              rationale: "目标侧公开状态估计显示当前意图；等待对手脑复核。",
              decision_status: "inconclusive",
            },
          ],
        }}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText("目标侧估计 · 待对手脑确认")).toBeInTheDocument();
  });

  it("renders modern ready brains and permanent mother ownership without legacy synthesis", () => {
    const modernFrame: OperationalFrame = {
      ...frame,
      target_estimates: [],
      adversary: null,
      adversaries: [],
      brains: [
        {
          brain_id: "carrier-master",
          role: "master",
          status: "ready",
          last_update_s: null,
          message: "",
          connected_platform_ids: [],
        },
        {
          brain_id: "group-slave",
          role: "slave",
          status: "ready",
          last_update_s: null,
          message: "",
          connected_platform_ids: [],
        },
      ],
      uuvs: Array.from({ length: 12 }, (_, index) => ({
        ...frame.uuvs[0],
        uuv_id: `uuv_${String(index).padStart(2, "0")}`,
        status: "available" as const,
        deployment_state: "onboard" as const,
        physically_exposed: false,
        group_id: null,
      })),
      uuv_resources: Array.from({ length: 12 }, (_, index) => ({
        uuv_id: `uuv_${String(index).padStart(2, "0")}`,
        carrier_id: index < 4 ? "carrier_02" : index < 8 ? "carrier_03" : "carrier_04",
        mileage_m: 0,
        energy_fraction: 1,
        healthy: true,
        capability_active: true,
        deployment_state: "onboard",
        resource_episode: 0,
      })),
    };

    const { container } = render(
      <RightSidebar
        frame={modernFrame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(container.querySelector(".adversary-brain-card")).toBeNull();
    expect(screen.queryByText("目标潜艇脑")).not.toBeInTheDocument();
    expect(screen.getAllByText("待命").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/归属 carrier_02/)).toHaveLength(4);
    expect(screen.getAllByText(/归属 carrier_03/)).toHaveLength(4);
    expect(screen.getAllByText(/归属 carrier_04/)).toHaveLength(4);
    expect(screen.getAllByText(/计划分配/)).toHaveLength(12);
  });
});
