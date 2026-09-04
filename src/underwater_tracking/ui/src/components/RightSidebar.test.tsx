import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type {
  ExecutionRegionView,
  ExecutionView,
  OperationalFrame,
  TaskGroupInstanceView,
  UUVView,
} from "../types/frames";
import RightSidebar from "./RightSidebar";

const groupIds = Array.from({ length: 4 }, (_, index) => `group-${index + 1}`);

function runtimeExecution(): ExecutionView {
  const regions: ExecutionRegionView[] = groupIds.map((groupId, index) => {
    const regionId = `T1:task:${String(index + 1).padStart(2, "0")}`;
    const left = -4_000 + index * 2_000;
    return {
      region_id: regionId,
      target_id: "T1",
      slot_index: index + 1,
      execution_revision: 1,
      prediction_id: "prediction-1",
      geometry: [
        { x: left, y: 1_000 },
        { x: left + 2_000, y: 1_000 },
        { x: left + 2_000, y: -1_000 },
        { x: left, y: -1_000 },
      ],
      top_left_xy: { x: left, y: 1_000 },
      bottom_right_xy: { x: left + 2_000, y: -1_000 },
      start_s: index * 30,
      end_s: (index + 1) * 30,
      geometry_revision: 1,
      predecessor_region_id: index ? `T1:task:${String(index).padStart(2, "0")}` : null,
      successor_region_id: index < 3 ? `T1:task:${String(index + 2).padStart(2, "0")}` : null,
      handoff_start_s: null,
      handoff_end_s: null,
      status: index === 0 ? "active" : "prepositioning",
      task_group_id: groupId,
      evidence_ids: [],
    };
  });
  const taskGroups: TaskGroupInstanceView[] = regions.map((region, index) => ({
    group_instance_id: groupIds[index],
    target_id: "T1",
    region_id: region.region_id,
    deployment_revision: 1,
    member_uuv_ids: [
      `uuv-${index * 3}`,
      `uuv-${index * 3 + 1}`,
      `uuv-${index * 3 + 2}`,
    ],
    lifecycle: index === 0 ? "passive_track" : index === 1 ? "active_scan" : "entering",
    sensor_mode: index === 0 ? "passive" : "active",
    ownership_status: index === 0 ? "owner" : "candidate",
    reason: "sidebar_fixture",
    evidence_ids: [],
  }));
  return {
    target_id: "T1",
    execution_revision: 1,
    source_snapshot_revision: 1,
    prediction_revision: 1,
    intent_revision: 1,
    data_age_s: 0,
    valid_from_s: 0,
    valid_until_s: 900,
    health_status: "current",
    health_reasons: [],
    region_generation_mode: "imm",
    plan_source: "deterministic",
    current_region_id: regions[0].region_id,
    next_region_id: regions[1].region_id,
    evidence_ids: [],
    regions,
    task_groups: taskGroups,
    tracking_policy: {
      region_count: 4,
      task_group_size: 3,
      task_region_side_m: 2_000,
      target_detection_radius_m: 1_000,
      uuv_active_detection_radius_m: 600,
      uuv_passive_detection_radius_m: 600,
      region_entry_probability_threshold: 0.7,
      region_transition_confirm_cycles: 2,
      max_uuv_mileage_m: 50_000,
      dedicated_release_remaining_mileage_m: 7_000,
    },
    tracking_control: {
      mode: "regional",
      tracking_owner_group_id: groupIds[0],
      pending_successor_group_id: null,
      dedicated_release_triggered_at_m: null,
      dedicated_release_reason: null,
      source_event_ids: [],
    },
    replacements: [],
    degraded: false,
    degradation_reasons: [],
    active_plan_preserved: false,
  };
}

function runtimeUuv(id: string, index: number): UUVView {
  const groupIndex = Math.floor(index / 3);
  const groupId = groupIds[groupIndex];
  const passive = groupIndex === 0;
  return {
    uuv_id: id,
    status: passive ? "track" : "active",
    deployment_state: "deployed",
    physically_exposed: true,
    position: { x: -3_500 + index * 600, y: 0 },
    heading_rad: 0,
    speed_mps: 2.1,
    energy_fraction: 0.72,
    group_id: "T1",
    group_instance_id: groupId,
    deployment_revision: 1,
    group_lifecycle: passive ? "passive_track" : groupIndex === 1 ? "active_scan" : "entering",
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: passive ? "passive" : "active",
    reserved: true,
    remaining_range_m: 4_200,
    communication_status: "connected",
    tracked_target_id: "T1",
  };
}

function runtimeFrame(): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: 4,
    sim_time_s: 120,
    plan_version: 1,
    map_bounds: { min_x: -5_000, min_y: -4_000, max_x: 5_000, max_y: 4_000 },
    execution: runtimeExecution(),
    uuvs: Array.from({ length: 12 }, (_, index) => runtimeUuv(`uuv-${index}`, index)),
    target_estimates: [{
      target_id: "T1",
      mean: { x: 0, y: 0 },
      covariance_ellipse: { semimajor_m: 120, semiminor_m: 60, rotation_rad: 0 },
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
      detection_range_m: 1_000,
      detected_platform_ids: ["uuv-0"],
    }],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
    brains: [
      {
        brain_id: "master",
        role: "master",
        status: "ready",
        last_update_s: 120,
        message: "master ready",
        connected_platform_ids: ["uuv-0"],
      },
      {
        brain_id: "adversary",
        role: "adversary",
        status: "online",
        last_update_s: 120,
        message: "adversary online",
        connected_platform_ids: [],
      },
    ],
    intelligence: [],
    adversary: {
      target_id: "T1",
      sim_time_s: 120,
      detection_range_m: 1_000,
      detected_platform_ids: ["uuv-0"],
      current_decision: {
        decision_id: "decision-1",
        target_id: "T1",
        sim_time_s: 120,
        intent: "evade",
        maneuver: "turn",
        segment: "region-1",
        confidence: 0.86,
        rationale: "decision rationale",
        decision_summary: "decision summary",
        trigger_event_ids: ["event-1"],
        detected_platform_ids: ["uuv-0"],
        active_ping_risk: "high",
        communications_discipline: "passive",
      },
      decision_history: [],
    },
    operational_stage_flags: ["task_execution", "dynamic_adjustment"],
  };
}

function renderSidebar(frame: OperationalFrame, selectedUuvId: string | null = null) {
  return render(
    <RightSidebar
      frame={frame}
      selectedUuvId={selectedUuvId}
      onSelectUuv={() => undefined}
      open
      onClose={() => undefined}
    />,
  );
}

describe("RightSidebar runtime contract", () => {
  it("renders all four authoritative groups and their twelve exposed members", () => {
    const { container } = renderSidebar(runtimeFrame());

    expect(container.querySelectorAll(".uuv-row")).toHaveLength(12);
    expect(container.querySelector('[data-visible-uuv-count="12"]')).toBeTruthy();
    expect(container.querySelector('[data-entering-group-count="2"]')).toBeTruthy();
    expect(container.querySelectorAll(".brain-card")).toHaveLength(2);
  });

  it("renders deployment-aware identity and selected UUV runtime details", () => {
    const { container, getByText } = renderSidebar(runtimeFrame(), "uuv-0");

    expect(container.querySelector(".uuv-row.selected")).toHaveTextContent("uuv-0");
    expect(getByText("group-1")).toBeInTheDocument();
    expect(getByText("4.2 km")).toBeInTheDocument();
    expect(container.querySelector(".selected-detail")).toHaveTextContent("T1");
  });

  it("toggles the explicit adversary brain without synthesizing a legacy brain", () => {
    const { container, getByText } = renderSidebar(runtimeFrame());
    const adversaryBrain = container.querySelector(
      "details.adversary-brain-card > summary",
    );
    if (!adversaryBrain) throw new Error("missing explicit adversary brain");

    expect(container.querySelectorAll(".brain-card")).toHaveLength(2);
    expect(adversaryBrain.parentElement).not.toHaveAttribute("open");
    fireEvent.click(adversaryBrain);
    expect(adversaryBrain.parentElement).toHaveAttribute("open");
    expect(container.querySelector(".target-submarine-brain")).toBeTruthy();
    expect(getByText("decision summary")).toBeInTheDocument();
    expect(getByText("已暴露 uuv-0")).toBeInTheDocument();
  });

  it("does not expose UUVs when the authoritative runtime execution is absent", () => {
    const frame = runtimeFrame();
    frame.execution = null;
    const { container } = renderSidebar(frame);

    expect(container.querySelectorAll(".uuv-row")).toHaveLength(0);
    expect(container.querySelector('[data-visible-uuv-count="0"]')).toBeTruthy();
  });
});
