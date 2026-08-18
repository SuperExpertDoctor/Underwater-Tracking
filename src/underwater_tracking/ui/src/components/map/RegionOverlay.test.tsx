import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RegionalPlanView, RegionTimelineView } from "../../types/frames";
import RegionOverlay, { regionOverlayEntries } from "./RegionOverlay";

const effect = {
  status: "active" as const,
  coverage_ratio: 0.7,
  quality_score: 0.8,
  handoff_progress: 0.2,
  quality_source: "group_quality_proxy" as const,
  hard_guard_reasons: [],
  expert_feedback_ids: [],
};

const regionStates = ["active", "handoff_ready", "degraded", "uncovered"] as const;

const plan: RegionalPlanView = {
  target_id: "T1",
  prediction_id: "prediction-1",
  revision: 3,
  cell_size_m: 100,
  regions: regionStates.map((status, index) => ({
    region_id: `T1:cell:${index}:0`,
    display_name: `region_${index + 1}`,
    target_id: "T1",
    geometry: [{ x: index * 20, y: 0 }, { x: index * 20 + 16, y: 0 }, { x: index * 20 + 16, y: 16 }, { x: index * 20, y: 16 }],
    start_time_s: index * 30,
    end_time_s: index * 30 + 30,
    predecessor_region_ids: index ? [`T1:cell:${index - 1}:0`] : [],
    successor_region_ids: index === 3 ? [] : [`T1:cell:${index + 1}:0`],
    assigned_uuv_ids: [],
    assigned_usv_ids: [],
    tracking_mode: "heuristic_uuv" as const,
    relay_usv_ids: [],
    group_id: null,
    status,
    effect: { ...effect, status },
  })),
};

const timeline: RegionTimelineView[] = plan.regions.map((region, index) => ({
  region_id: region.region_id,
  target_id: region.target_id,
  center: { x: index * 20 + 8, y: 8 },
  bounds: { min_x: index * 20, min_y: 0, max_x: index * 20 + 16, max_y: 16 },
  start_offset_s: index * 30,
  end_offset_s: index * 30 + 30,
  status: index === 1 ? "handed_off" : region.effect.status === "handoff_ready" ? "planned" : region.effect.status,
  coverage_mode: "required",
  priority: 0.9 - index * 0.1,
  occupancy_likelihood: 0.8 - index * 0.1,
  uuv_assignments: [],
  usv_assignments: [],
  communication_links: [],
  handoff_from: index ? `T1:cell:${index - 1}:0` : null,
  handoff_to: index === 3 ? null : `T1:cell:${index + 1}:0`,
  evidence_ids: [],
  degraded_reasons: [],
  plan_revision: 3,
}));

describe("RegionOverlay", () => {
  it("joins operator-safe probability and priority with active, handoff, degraded, and uncovered geometry", () => {
    const entries = regionOverlayEntries([plan], timeline);
    expect(entries).toHaveLength(4);
    expect(entries.map((entry) => entry.label)).toEqual(["R01", "R02", "R03", "R04"]);
    expect(entries[0]).toMatchObject({ probability: 0.8, priority: 0.9, state: "active" });
    expect(entries[1].handoff).toBe(true);
    expect(entries.map((entry) => entry.state)).toEqual(["active", "handoff", "degraded", "uncovered"]);
  });

  it("renders state, probability, priority, and controlled region selection without backend truth fields", () => {
    const onSelectRegion = vi.fn();
    render(<RegionOverlay plans={[plan]} timeline={timeline} selectedRegionId="T1:cell:0:0" onSelectRegion={onSelectRegion} project={(point) => point} />);
    expect(screen.getByRole("button", { name: /R01.*概率 80%.*优先级 0.90.*当前覆盖/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("接力")).toBeInTheDocument();
    expect(screen.getByText("降级")).toBeInTheDocument();
    expect(screen.getByText("未覆盖")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /R02/ }));
    expect(onSelectRegion).toHaveBeenCalledWith("T1:cell:1:0");
  });
});
