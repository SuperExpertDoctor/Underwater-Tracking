import { render } from "@testing-library/react";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";
import type {
  ExecutionRegionView,
  ExecutionView,
  OperationalFrame,
  RegionTaskView,
  TargetEstimateView,
  TaskGroupInstanceView,
  TrackingPolicyView,
  UUVView,
} from "../types/frames";
import { DEFAULT_VIEW_CONFIG } from "../types/viewConfig";
import { MAP_DISPLAY_CONFIG } from "../../configs/map_display";
import CanvasMap, {
  CANVAS_LAYER_ORDER,
  DETECTION_LABEL_LAYER,
  GRID_DIVISIONS,
  TARGET_DETECTION_STYLE,
  TARGET_MARKER_SIZE_RANGE_PX,
  UUV_MARKER_SIZE_RANGE_PX,
  UUV_SENSOR_FOOTPRINT_SPAN_RAD,
  cameraBoundsForFrame,
  currentTaskUuvIds,
  deploymentAwareUuvKey,
  detectedPlatformIds,
  detectionZoneLabels,
  displayCovarianceEllipse,
  displayRegionalPlans,
  focusRegionForCanvas,
  highlightedUuvIds,
  hitTestRegion,
  mapScaleForView,
  nextRegionFocusZoom,
  regionLabelForZoom,
  regionLayerStyle,
  screenSpriteSize,
  sensorLayerStyle,
  spatialExecutionUuvs,
  targetDetectionRange,
  uuvDetectionFootprint,
  uuvSensorFootprint,
  uuvSpriteAppearance,
} from "./CanvasMap";

const policy: TrackingPolicyView = {
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
};

function target(targetId = "T1"): TargetEstimateView {
  return {
    target_id: targetId,
    mean: { x: 0, y: 0 },
    covariance_ellipse: { semimajor_m: 100, semiminor_m: 50, rotation_rad: 0 },
    intent: { label: "transit", confidence: 0.9, alternatives: {} },
    prediction: null,
    quality: {
      quality_score: 0.9,
      estimated_rmse_m: 20,
      fim_min_eigenvalue: 1,
      fim_condition: 1,
    },
    classification: "submarine",
    last_ping_s: 0,
    detection_range_m: 250,
  };
}

function region(index: number): ExecutionRegionView {
  const x = -4_000 + index * 2_000;
  const regionId = `T1:task:${String(index + 1).padStart(2, "0")}`;
  return {
    region_id: regionId,
    target_id: "T1",
    slot_index: index + 1,
    execution_revision: 7,
    prediction_id: "prediction:7",
    geometry: [
      { x, y: 1_000 },
      { x: x + 2_000, y: 1_000 },
      { x: x + 2_000, y: -1_000 },
      { x, y: -1_000 },
    ],
    top_left_xy: { x, y: 1_000 },
    bottom_right_xy: { x: x + 2_000, y: -1_000 },
    start_s: index * 100,
    end_s: (index + 1) * 100,
    geometry_revision: 7,
    predecessor_region_id: index ? `T1:task:${String(index).padStart(2, "0")}` : null,
    successor_region_id: index < 3 ? `T1:task:${String(index + 2).padStart(2, "0")}` : null,
    handoff_start_s: index * 100 + 80,
    handoff_end_s: (index + 1) * 100,
    status: index === 0 ? "passive" : "active",
    task_group_id: regionId + ":deploy:000007",
    evidence_ids: [`evidence:${index + 1}`],
  };
}

function group(index: number, lifecycle?: TaskGroupInstanceView["lifecycle"]): TaskGroupInstanceView {
  const groupInstanceId = `T1:task:${String(index + 1).padStart(2, "0")}:deploy:000007`;
  const selectedLifecycle = lifecycle ?? (index === 0 ? "passive_track" : "active_scan");
  const member_uuv_ids = [
    `uuv-${index * 3}`,
    `uuv-${index * 3 + 1}`,
    `uuv-${index * 3 + 2}`,
  ] as [string, string, string];
  return {
    group_instance_id: groupInstanceId,
    target_id: "T1",
    region_id: `T1:task:${String(index + 1).padStart(2, "0")}`,
    deployment_revision: 7,
    member_uuv_ids,
    lifecycle: selectedLifecycle,
    sensor_mode: selectedLifecycle === "active_scan" ? "active" : "passive",
    ownership_status: index === 0 ? "owner" : "candidate",
    entry_boundary_point: null,
    exit_boundary_point: null,
    source_group_instance_id: null,
    reason: "test_fixture",
    evidence_ids: [`evidence:${index + 1}`],
  };
}

function execution(
  groups: TaskGroupInstanceView[] = Array.from({ length: 4 }, (_, index) => group(index)),
): ExecutionView {
  const regions = Array.from({ length: 4 }, (_, index) => region(index));
  return {
    target_id: "T1",
    execution_revision: 7,
    source_snapshot_revision: 7,
    prediction_revision: 7,
    intent_revision: 7,
    data_age_s: 0,
    valid_from_s: 0,
    valid_until_s: 1_000,
    health_status: "current",
    health_reasons: [],
    region_generation_mode: "imm",
    plan_source: "deterministic",
    current_region_id: regions[0].region_id,
    next_region_id: regions[1].region_id,
    evidence_ids: ["execution:7"],
    regions,
    task_groups: groups,
    tracking_policy: policy,
    tracking_control: {
      mode: "regional",
      tracking_owner_group_id: groups.find((item) => item.ownership_status === "owner")?.group_instance_id ?? null,
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

function uuv(
  id: string,
  groupInstanceId: string | null,
  sensorMode: "active" | "passive",
  index: number,
  physicallyExposed = true,
): UUVView {
  return {
    uuv_id: id,
    status: sensorMode === "active" ? "scan" : "track",
    deployment_state: physicallyExposed ? "deployed" : "onboard",
    physically_exposed: physicallyExposed,
    position: { x: -3_500 + index * 600, y: 0 },
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 1,
    group_id: "T1",
    group_instance_id: groupInstanceId,
    deployment_revision: groupInstanceId ? 7 : null,
    group_lifecycle: groupInstanceId ? (sensorMode === "active" ? "active_scan" : "passive_track") : null,
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: sensorMode,
    reserved: false,
    active_range_m: 600,
    passive_range_m: 600,
    active_capable: true,
    tracked_target_id: "T1",
  };
}

function frame(
  groups: TaskGroupInstanceView[] = Array.from({ length: 4 }, (_, index) => group(index)),
): OperationalFrame {
  const groupUuvs = groups.flatMap((item, groupIndex) =>
    item.member_uuv_ids.map((id, memberIndex) =>
      uuv(id, item.group_instance_id, item.sensor_mode === "active" ? "active" : "passive", groupIndex * 3 + memberIndex),
    ),
  );
  return {
    schema_version: "1.0",
    frame_id: 7,
    sim_time_s: 100,
    plan_version: 7,
    map_bounds: { min_x: -8_000, min_y: -4_000, max_x: 8_000, max_y: 4_000 },
    execution: execution(groups),
    uuvs: groupUuvs,
    target_estimates: [target()],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
  };
}

describe("CanvasMap runtime contract", () => {
  it("keeps the semantic layer order stable", () => {
    expect(CANVAS_LAYER_ORDER).toEqual([
      "map/grid",
      "regions/handoffs",
      "prediction corridor",
      "prediction centerline/samples",
      "target detection circle",
      "UUV sonar fans",
      "labels",
      "selection/errors",
    ]);
  });

  it("uses policy radii and published task-group geometry", () => {
    const liveFrame = frame();
    const liveTarget = liveFrame.target_estimates[0];
    expect(targetDetectionRange(liveFrame)).toBe(1_000);
    expect(detectionZoneLabels(liveFrame, liveTarget)).toMatchObject({
      radiusM: 1_000,
      detectedCount: 3,
      rangeText: "1 km",
    });
    expect(displayRegionalPlans(liveFrame)[0].regions).toHaveLength(4);
    expect(displayRegionalPlans(liveFrame)[0].regions.every((item) => item.assigned_uuv_ids.length === 3)).toBe(true);
    expect(displayRegionalPlans(liveFrame)[0].regions[0].authoritative_geometry).toBe(true);
  });

  it("renders runtime counts without a reserve or legacy role projection", () => {
    const liveFrame = frame();
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    try {
      const view = render(createElement(CanvasMap, {
        frame: liveFrame,
        selectedUuvId: null,
        onSelectUuv: vi.fn(),
        showGrid: false,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: true,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      const canvas = view.container.querySelector("canvas");
      expect(canvas).toHaveAttribute("data-visible-uuv-count", "12");
      expect(canvas).toHaveAttribute("data-task-group-count", "4");
      expect(canvas).toHaveAttribute("data-task-group-size", "3");
      expect(canvas).toHaveAttribute("data-region-side-m", "2000");
      expect(canvas).toHaveAttribute("data-target-radius-m", "1000");
      expect(canvas).toHaveAttribute("data-uuv-radius-m", "600");
      expect(canvas).toHaveAttribute("data-tracking-mode", "regional");
    } finally {
      getContext.mockRestore();
    }
  });

  it("selects all three members from the authoritative group instance", () => {
    const liveFrame = frame();
    expect(spatialExecutionUuvs(liveFrame)).toHaveLength(12);
    expect(currentTaskUuvIds(liveFrame)).toEqual(new Set(["uuv-0", "uuv-1", "uuv-2"]));
    expect([...highlightedUuvIds(liveFrame, "uuv-1")]).toEqual(["uuv-0", "uuv-1", "uuv-2"]);
  });

  it("uses deployment identity for animated UUV labels and footprints", () => {
    const liveFrame = frame();
    const selected = liveFrame.uuvs[0];
    expect(deploymentAwareUuvKey(selected)).toBe("uuv-0:T1:task:01:deploy:000007");
    expect(uuvSensorFootprint(selected, liveFrame)).toMatchObject({
      radiusM: 600,
      spanAngleRad: UUV_SENSOR_FOOTPRINT_SPAN_RAD,
    });
    expect(uuvDetectionFootprint(selected, liveFrame)?.mode).toBe("passive");
  });

  it("excludes disappeared groups from visible execution entities", () => {
    const groups = [
      group(0, "passive_track"),
      group(1, "active_scan"),
      group(2, "exiting"),
      group(3, "disappeared"),
    ];
    const liveFrame = frame(groups);
    expect(spatialExecutionUuvs(liveFrame).map((item) => item.uuv_id)).toEqual([
      ...groups.slice(0, 3).flatMap((item) => item.member_uuv_ids),
    ]);
  });

  it("uses exact square corners for hit tests and focus", () => {
    const region: RegionTaskView = {
      region_id: "T1:task:01",
      display_name: "T1:task:01",
      target_id: "T1",
      geometry: [{ x: 20, y: 20 }, { x: 80, y: 20 }, { x: 80, y: 80 }, { x: 20, y: 80 }],
      top_left_xy: { x: 0, y: 100 },
      bottom_right_xy: { x: 100, y: 0 },
      start_time_s: 0,
      end_time_s: 100,
      predecessor_region_ids: [],
      successor_region_ids: [],
      assigned_uuv_ids: ["uuv-0", "uuv-1", "uuv-2"],
      task_group_ids: ["T1:task:01:deploy:000007"],
      tracking_mode: "heuristic_uuv",
      uuv_roles: ["passive_tracker", "passive_tracker", "passive_tracker"],
      authoritative_geometry: true,
      group_id: "T1:task:01:deploy:000007",
      status: "active",
      effect: {
        status: "active",
        coverage_ratio: 1,
        quality_score: 1,
        handoff_progress: 0,
        quality_source: "region_telemetry",
        hard_guard_reasons: [],
        expert_feedback_ids: [],
      },
    };
    expect(hitTestRegion({ x: 50, y: 50 }, [region])?.region_id).toBe("T1:task:01");
    expect(hitTestRegion({ x: 110, y: 50 }, [region])).toBeNull();
    const focused = focusRegionForCanvas(
      { min_x: -200, min_y: -200, max_x: 200, max_y: 200 },
      { width: 400, height: 300 },
      region,
      2,
    );
    expect(focused.zoom).toBe(2);
    expect(nextRegionFocusZoom(1)).toBe(2);
  });

  it("keeps visual-only style transforms separate from policy data", () => {
    expect(regionLayerStyle("planned")).not.toEqual(regionLayerStyle("active"));
    expect(sensorLayerStyle("active").stroke).toContain("247, 189, 69");
    expect(sensorLayerStyle("passive").stroke).toContain("33, 208, 195");
    expect(TARGET_DETECTION_STYLE.lineDash).toEqual([4, 7]);
    expect(DETECTION_LABEL_LAYER).toBe("labels");
    expect(GRID_DIVISIONS).toBe(24);
    expect(MAP_DISPLAY_CONFIG.uuvSensorSpanRad).toBe(UUV_SENSOR_FOOTPRINT_SPAN_RAD);
  });

  it("keeps camera, marker and uncertainty rendering bounded", () => {
    const liveFrame = frame();
    const bounds = cameraBoundsForFrame(liveFrame, DEFAULT_VIEW_CONFIG, true);
    expect(bounds.min_x).toBeGreaterThanOrEqual(liveFrame.map_bounds.min_x);
    expect(bounds.max_x).toBeLessThanOrEqual(liveFrame.map_bounds.max_x);
    expect(mapScaleForView(bounds, 800, 600, 1).label).toBeTruthy();
    expect(regionLabelForZoom({ region_id: "T1:task:01", display_name: "region_01" } as RegionTaskView, 1.5)).toBe("R01");
    expect(screenSpriteSize(null, 100, TARGET_MARKER_SIZE_RANGE_PX.min, TARGET_MARKER_SIZE_RANGE_PX.max)).toEqual({ width: 32, height: 32 });
    expect(screenSpriteSize(null, 1, UUV_MARKER_SIZE_RANGE_PX.min, UUV_MARKER_SIZE_RANGE_PX.max)).toEqual({ width: 22, height: 22 });
    expect(uuvSpriteAppearance(liveFrame.uuvs[0], null, 1, false).size).toEqual({ width: 30, height: 30 });
    const source = { semimajor_m: 6_000, semiminor_m: 200, rotation_rad: 0.4 };
    expect(displayCovarianceEllipse(source).semimajor_m).toBe(MAP_DISPLAY_CONFIG.estimateEllipseMaxSemimajorM);
    expect(source.semimajor_m).toBe(6_000);
  });

  it("filters explicit detections to the visible runtime UUV set", () => {
    const liveFrame = frame();
    const estimate = liveFrame.target_estimates[0];
    liveFrame.adversary = {
      target_id: "T1",
      detected_platform_ids: ["uuv-0", "unknown-uuv"],
    };
    expect(detectedPlatformIds(liveFrame, estimate)).toEqual(["uuv-0"]);
  });
});
