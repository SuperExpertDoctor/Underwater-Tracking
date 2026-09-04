import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type {
  ExecutionRegionView,
  ExecutionView,
  OperationalFrame,
  RegionTimelineView,
  TaskGroupInstanceView,
} from "../types/frames";
import RegionTimelinePanel from "./RegionTimelinePanel";
import { offsetPercent, sortRegionTimeline, timelineRowsForFrame, timelineWindow } from "./regionTimeline";

function row(regionId: string, start: number, status: RegionTimelineView["status"] = "active"): RegionTimelineView {
  return {
    region_id: regionId,
    target_id: "T1",
    center: { x: 50, y: 50 },
    bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
    start_offset_s: start,
    end_offset_s: start + 30,
    status,
    coverage_mode: "required",
    priority: 0.8,
    occupancy_likelihood: 0.7,
    uuv_assignments: [
      { platform_id: "uuv-1", platform_kind: "uuv", role: "passive_tracker", start_offset_s: start, end_offset_s: start + 30, sonar_mode: "passive" },
    ],
    communication_links: [],
    handoff_from: null,
    handoff_to: "T1:task:02",
    evidence_ids: [],
    degraded_reasons: status === "degraded" ? ["insufficient_coverage"] : [],
    plan_revision: 1,
  };
}

function runtimeFrame(): OperationalFrame {
  const regions: ExecutionRegionView[] = Array.from({ length: 4 }, (_, index) => {
    const regionId = `T1:task:${String(index + 1).padStart(2, "0")}`;
    return {
      region_id: regionId,
      target_id: "T1",
      slot_index: index + 1,
      execution_revision: 3,
      prediction_id: "imm:T1:3",
      geometry: [{ x: index * 2_000, y: 2_000 }, { x: index * 2_000 + 2_000, y: 2_000 }, { x: index * 2_000 + 2_000, y: 0 }, { x: index * 2_000, y: 0 }],
      top_left_xy: { x: index * 2_000, y: 2_000 },
      bottom_right_xy: { x: index * 2_000 + 2_000, y: 0 },
      start_s: 100 + index * 30,
      end_s: 130 + index * 30,
      geometry_revision: 3,
      predecessor_region_id: index ? `T1:task:${String(index).padStart(2, "0")}` : null,
      successor_region_id: index < 3 ? `T1:task:${String(index + 2).padStart(2, "0")}` : null,
      handoff_start_s: null,
      handoff_end_s: null,
      status: index === 0 ? "active" : "planned",
      task_group_id: `${regionId}:deploy:000003`,
      evidence_ids: [`e-${index}`],
    };
  });
  const taskGroups: TaskGroupInstanceView[] = regions.map((region, index) => ({
    group_instance_id: `${region.region_id}:deploy:000003`,
    target_id: "T1",
    region_id: region.region_id,
    deployment_revision: 3,
    member_uuv_ids: [`uuv-${index * 3}`, `uuv-${index * 3 + 1}`, `uuv-${index * 3 + 2}`] as [string, string, string],
    lifecycle: index === 0 ? "active_scan" : "entering",
    sensor_mode: "active",
    ownership_status: index === 0 ? "candidate" : "candidate",
    reason: "timeline_fixture",
    evidence_ids: [region.evidence_ids[0]],
  }));
  const execution: ExecutionView = {
    target_id: "T1",
    execution_revision: 3,
    source_snapshot_revision: 10,
    prediction_revision: 3,
    intent_revision: 3,
    data_age_s: 0,
    valid_from_s: 0,
    valid_until_s: 1_000,
    health_status: "current",
    health_reasons: [],
    region_generation_mode: "imm",
    plan_source: "deterministic",
    current_region_id: regions[0].region_id,
    next_region_id: regions[1].region_id,
    evidence_ids: ["execution"],
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
      tracking_owner_group_id: null,
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
  return {
    schema_version: "1.0",
    frame_id: 3,
    sim_time_s: 100,
    plan_version: 3,
    map_bounds: { min_x: -1_000, min_y: -1_000, max_x: 10_000, max_y: 4_000 },
    execution,
    uuvs: [],
    target_estimates: [],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
  };
}

describe("RegionTimelinePanel", () => {
  it("sorts rows by start offset then region id", () => {
    expect(sortRegionTimeline([row("R1", 20), row("R0", 20), row("R2", -5)]).map((item) => item.region_id)).toEqual(["R2", "R0", "R1"]);
  });

  it("calculates bounded timeline positions and windows", () => {
    expect(offsetPercent(50, 0, 100)).toBe(50);
    expect(offsetPercent(-10, 0, 100)).toBe(0);
    expect(offsetPercent(110, 0, 100)).toBe(100);
    expect(timelineWindow([row("R1", 20)])).toEqual({ start: 0, end: 600 });
  });

  it("renders three UUV assignments for each authoritative runtime region", () => {
    const { container } = render(<RegionTimelinePanel frame={runtimeFrame()} />);
    const rows = timelineRowsForFrame(runtimeFrame());
    expect(rows).toHaveLength(4);
    expect(rows.every((item) => item.uuv_assignments.length === 3)).toBe(true);
    expect(container.querySelectorAll(".region-assignment-chip")).toHaveLength(3);
    expect(screen.getAllByRole("button")).toHaveLength(4);
  });

  it("uses a controlled selection and clears it when the selected row is clicked", () => {
    const onSelectRegion = vi.fn();
    render(<RegionTimelinePanel frame={runtimeFrame()} selectedRegionId="T1:task:02" onSelectRegion={onSelectRegion} />);
    const selectedRow = screen.getByRole("button", { name: /T1:task:02/ });
    expect(selectedRow).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(selectedRow);
    expect(onSelectRegion).toHaveBeenCalledWith(null);
  });

  it("derives exactly four rows from the runtime execution chain", () => {
    const rows = timelineRowsForFrame(runtimeFrame());
    expect(rows).toHaveLength(4);
    expect(rows.map((item) => item.region_id)).toEqual([
      "T1:task:01",
      "T1:task:02",
      "T1:task:03",
      "T1:task:04",
    ]);
    expect(rows[0].uuv_assignments).toHaveLength(3);
    expect(rows[0].uuv_assignments.every((item) => item.sonar_mode === "active")).toBe(true);
  });

  it("does not synthesize a timeline from a frame without runtime execution", () => {
    render(<RegionTimelinePanel frame={{ sim_time_s: 100 } as OperationalFrame} />);
    expect(screen.getByText("当前暂无区域任务")).toBeInTheDocument();
  });
});
