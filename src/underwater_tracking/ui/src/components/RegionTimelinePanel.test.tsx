import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OperationalFrame, RegionTimelineView } from "../types/frames";
import RegionTimelinePanel from "./RegionTimelinePanel";
import { offsetPercent, sortRegionTimeline } from "./regionTimeline";

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
    uuv_assignments: [{ platform_id: "uuv-1", platform_kind: "uuv", role: "passive_tracker", start_offset_s: start, end_offset_s: start + 30, sonar_mode: "passive" }],
    usv_assignments: [{ platform_id: "USV-01", platform_kind: "usv", role: "surface_relay", start_offset_s: start, end_offset_s: start + 30, sonar_mode: "passive" }],
    communication_links: [],
    handoff_from: null,
    handoff_to: "T1:cell:1:0",
    evidence_ids: [],
    degraded_reasons: status === "degraded" ? ["insufficient_uuv"] : [],
    plan_revision: 1,
  };
}

const frameWithTimeline = (timeline: RegionTimelineView[]): OperationalFrame => ({ region_timeline: timeline, sim_time_s: 100 } as OperationalFrame);

describe("RegionTimelinePanel", () => {
  it("sorts rows by start offset then region id", () => {
    expect(sortRegionTimeline([row("R1", 20), row("R0", 20), row("R2", -5)]).map((item) => item.region_id)).toEqual(["R2", "R0", "R1"]);
  });

  it("calculates a bounded position percentage", () => {
    expect(offsetPercent(50, 0, 100)).toBe(50);
    expect(offsetPercent(-10, 0, 100)).toBe(0);
    expect(offsetPercent(110, 0, 100)).toBe(100);
  });

  it("renders assignments, relay and degraded reason after selection", () => {
    render(<RegionTimelinePanel frame={frameWithTimeline([row("T1:cell:0:0", 0, "degraded")])} />);
    expect(screen.getByText("T1:cell:0:0")).toBeInTheDocument();
    expect(screen.getByText(/uuv-1.*passive_tracker/)).toBeInTheDocument();
    expect(screen.getByText(/USV-01.*surface_relay/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /T1:cell:0:0/ }));
    expect(screen.getByText("insufficient_uuv")).toBeInTheDocument();
  });

  it("shows an empty state for old frames", () => {
    render(<RegionTimelinePanel frame={{ sim_time_s: 100 } as OperationalFrame} />);
    expect(screen.getByText("当前暂无区域任务")).toBeInTheDocument();
  });
});
