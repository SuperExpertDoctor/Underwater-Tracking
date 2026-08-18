import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RegionalPlanView } from "../../types/frames";
import RegionTaskGraph, {
  ENTITY_NODE_RADIUS,
  REGION_NODE_HEIGHT,
  REGION_NODE_WIDTH,
  buildRegionGraphLayout,
} from "./RegionTaskGraph";

const effect = {
  status: "active" as const,
  coverage_ratio: 1,
  quality_score: 0.9,
  handoff_progress: 0.5,
  quality_source: "group_quality_proxy" as const,
  hard_guard_reasons: [],
  expert_feedback_ids: [],
};

const plan: RegionalPlanView = {
  target_id: "T1",
  prediction_id: "pred-1",
  revision: 2,
  cell_size_m: 100,
  regions: [
    {
      region_id: "T1:cell:0:0",
      display_name: "region_1",
      target_id: "T1",
      geometry: [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }],
      start_time_s: 0,
      end_time_s: 30,
      predecessor_region_ids: [],
      successor_region_ids: ["T1:cell:1:0"],
      assigned_uuv_ids: ["uuv_01", "uuv_02"],
      assigned_usv_ids: ["usv_01"],
      tracking_mode: "uuv_primary_usv_relay",
      relay_usv_ids: ["usv_01"],
      group_id: "G1",
      status: "active",
      effect,
    },
    {
      region_id: "T1:cell:1:0",
      display_name: "region_2",
      target_id: "T1",
      geometry: [{ x: 100, y: 0 }, { x: 200, y: 0 }, { x: 200, y: 100 }, { x: 100, y: 100 }],
      start_time_s: 30,
      end_time_s: 60,
      predecessor_region_ids: ["T1:cell:0:0"],
      successor_region_ids: ["T1:cell:2:0"],
      assigned_uuv_ids: ["uuv_03"],
      assigned_usv_ids: ["usv_02"],
      tracking_mode: "heuristic_usv",
      relay_usv_ids: [],
      group_id: "G2",
      status: "planned",
      effect: { ...effect, status: "planned" },
    },
    {
      region_id: "T1:cell:2:0",
      display_name: "region_3",
      target_id: "T1",
      geometry: [{ x: 200, y: 0 }, { x: 300, y: 0 }, { x: 300, y: 100 }, { x: 200, y: 100 }],
      start_time_s: 60,
      end_time_s: 90,
      predecessor_region_ids: ["T1:cell:1:0"],
      successor_region_ids: ["T1:cell:3:0"],
      assigned_uuv_ids: ["uuv_01"],
      assigned_usv_ids: [],
      tracking_mode: "heuristic_uuv",
      relay_usv_ids: [],
      group_id: "G1",
      status: "handoff_ready",
      effect: { ...effect, status: "handoff_ready" },
    },
    {
      region_id: "T1:cell:3:0",
      display_name: "region_4",
      target_id: "T1",
      geometry: [{ x: 300, y: 0 }, { x: 400, y: 0 }, { x: 400, y: 100 }, { x: 300, y: 100 }],
      start_time_s: 90,
      end_time_s: 120,
      predecessor_region_ids: ["T1:cell:2:0"],
      successor_region_ids: [],
      assigned_uuv_ids: [],
      assigned_usv_ids: ["usv_02"],
      tracking_mode: "heuristic_usv",
      relay_usv_ids: [],
      group_id: "G2",
      status: "degraded",
      effect: { ...effect, status: "degraded" },
    },
  ],
};

describe("RegionTaskGraph", () => {
  it("lays out four R01-R04 regions, three UUVs, two USVs, temporal arrows, and distinct responsibilities", () => {
    const layout = buildRegionGraphLayout(plan);
    expect(layout.nodes.filter((node) => node.shape === "square")).toHaveLength(4);
    expect(layout.nodes.filter((node) => node.shape === "circle" && node.platform === "uuv")).toHaveLength(3);
    expect(layout.nodes.filter((node) => node.shape === "circle" && node.platform === "usv")).toHaveLength(2);
    expect(layout.nodes.filter((node) => node.kind === "region").map((node) => node.label)).toEqual(["R01", "R02", "R03", "R04"]);
    expect(layout.edges.filter((edge) => edge.kind === "temporal")).toHaveLength(3);
    expect(layout.edges.find((edge) => edge.relay)?.source).toBe("entity:usv_01");
    expect(layout.edges.find((edge) => edge.kind === "responsibility" && !edge.relay)?.responsibility).toBe("active_tracking");
    expect(REGION_NODE_WIDTH).toBeGreaterThanOrEqual(96);
    expect(REGION_NODE_HEIGHT).toBeGreaterThanOrEqual(44);
    expect(ENTITY_NODE_RADIUS * 2).toBeGreaterThanOrEqual(24);
  });

  it("expands horizontally for 64 regions instead of compressing the temporal sequence", () => {
    const largePlan = { ...plan, regions: Array.from({ length: 64 }, (_, index) => ({
      ...plan.regions[0],
      region_id: `T1:cell:${index}:0`,
      display_name: `region_${index + 1}`,
      start_time_s: index * 30,
      end_time_s: index * 30 + 30,
      successor_region_ids: index === 63 ? [] : [`T1:cell:${index + 1}:0`],
    })) };
    const layout = buildRegionGraphLayout(largePlan);
    expect(layout.nodes.filter((node) => node.kind === "region")).toHaveLength(64);
    expect(layout.width).toBeGreaterThan(720);
    expect(layout.height).toBeLessThan(400);
  });

  it("uses arrows for temporal and responsibility edges, and separates relay from active tracking", () => {
    const view = render(<RegionTaskGraph plan={plan} />);
    expect(view.container.querySelectorAll('[data-edge-kind="temporal"][marker-end]')).toHaveLength(3);
    expect(view.container.querySelectorAll('[data-responsibility="active_tracking"][data-relay="false"][marker-end]')).not.toHaveLength(0);
    expect(view.container.querySelectorAll('[data-responsibility="relay"][data-relay="true"][marker-end]')).toHaveLength(1);
  });

  it("selects regions and entities from accessible SVG nodes", () => {
    const onRegion = vi.fn();
    const onEntity = vi.fn();
    const view = render(<RegionTaskGraph plan={plan} onSelectRegion={onRegion} onSelectEntity={onEntity} />);
    fireEvent.click(view.getByRole("button", { name: "区域 R02" }));
    fireEvent.click(view.getByRole("button", { name: "实体 UUV_1" }));
    expect(onRegion).toHaveBeenCalledWith("T1:cell:1:0");
    expect(onEntity).toHaveBeenCalledWith("uuv_01");
    expect(view.container.querySelectorAll('[data-edge-kind="temporal"]')).toHaveLength(3);
    expect(view.container.querySelectorAll('[data-relay="true"]')).toHaveLength(1);
  });

  it("clears a controlled region selection when the selected node is chosen again", () => {
    const onRegion = vi.fn();
    const view = render(<RegionTaskGraph plan={plan} selectedRegionId="T1:cell:1:0" onSelectRegion={onRegion} />);

    const selectedNode = view.getByRole("button", { name: "区域 R02" });
    expect(selectedNode).toHaveClass("selected");
    fireEvent.click(selectedNode);

    expect(onRegion).toHaveBeenCalledWith(null);
  });

  it("shows an operator-safe empty state before a regional plan exists", () => {
    const view = render(<RegionTaskGraph plan={null} />);
    expect(view.getByRole("status")).toHaveTextContent("等待目标预测区域和编组任务");
  });
});
