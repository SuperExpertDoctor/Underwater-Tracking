import { fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";
import * as CanvasMapModule from "./CanvasMap";
import type { UUVView } from "../types/frames";
import {
  DEFAULT_SUBMARINE_DETECTION_RANGE_M,
  cameraBoundsForFrame,
  clampedMarkerPixels,
  communicationRangeForUuv,
  carrierAssetRotation,
  displayRegionalPlans,
  detectedPlatformIds,
  GRID_DIVISIONS,
  highlightedUuvIds,
  hitTestRegion,
  currentTaskUuvIds,
  regionLabelForZoom,
  mapScaleForView,
  spatialExecutionUuvs,
  focusRegionForCanvas,
  nextRegionFocusZoom,
  shouldDrawDetectionRange,
  submarineAssetRotation,
  targetDetectionRange,
  uuvSpriteAppearance,
  uuvDisplayOpacity,
  warshipAssetRotation,
  waterborneUuvs,
  CANVAS_LAYER_ORDER,
  regionLayerStyle,
  sensorLayerStyle,
  TARGET_DETECTION_STYLE,
  detectionZoneLabels,
  DETECTION_LABEL_LAYER,
  stableLabelCandidatesForFrame,
  type RegionLayerStatus,
} from "./CanvasMap";
import CanvasMap from "./CanvasMap";
import { worldToScreen } from "./map/geometry";
import { semanticCameraForFrame } from "./map/camera";
import { stableLabelPlacements } from "./map/camera";
import type {
  CarrierView,
  ExecutionView,
  OperationalFrame,
  Point2D,
  RegionTaskView,
  TaskGroupView,
  TargetEstimateView,
} from "../types/frames";
import { DEFAULT_VIEW_CONFIG } from "../types/viewConfig";

const uuv: UUVView = {
  uuv_id: "uuv_01",
  status: "active",
  deployment_state: "deployed",
  position: { x: 0, y: 0 },
  heading_rad: 0,
  speed_mps: 1,
  energy_fraction: 1,
  group_id: null,
  current_waypoint: null,
  breadcrumb: [],
  sensor_mode: "passive",
  reserved: false,
  physically_exposed: true,
};

function targetEstimateFixture(
  overrides: Partial<TargetEstimateView> = {},
): TargetEstimateView {
  return {
    target_id: "T1",
    mean: { x: 0, y: 0 },
    covariance_ellipse: {
      semimajor_m: 10,
      semiminor_m: 5,
      rotation_rad: 0,
    },
    intent: { label: "unknown", confidence: 0, alternatives: {} },
    prediction: null,
    quality: {
      quality_score: 1,
      estimated_rmse_m: 0,
      fim_min_eigenvalue: 1,
      fim_condition: 1,
    },
    classification: "unknown",
    last_ping_s: null,
    ...overrides,
  };
}

function operationalFrameFixture(
  overrides: Partial<OperationalFrame> = {},
): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: 1,
    sim_time_s: 0,
    plan_version: 1,
    map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
    uuvs: [uuv],
    target_estimates: [targetEstimateFixture()],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
    ...overrides,
  };
}

describe("CanvasMap semantic layer contract", () => {
  it("draws semantic layers in the operator-facing order", () => {
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

  it("uses distinct region status styles", () => {
    const statuses: RegionLayerStatus[] = [
      "planned",
      "active",
      "handoff",
      "degraded",
      "uncovered",
    ];
    const styles = statuses.map(regionLayerStyle);
    expect(new Set(styles.map((style) => `${style.fill}:${style.stroke}`)).size).toBe(5);
  });

  it("uses amber active and cyan passive sonar styles", () => {
    expect(sensorLayerStyle("active").stroke).toContain("247, 189, 69");
    expect(sensorLayerStyle("passive").stroke).toContain("33, 208, 195");
  });

  it("keeps target detection styling red and dashed", () => {
    expect(TARGET_DETECTION_STYLE.stroke).toBe("#ff7882");
    expect(TARGET_DETECTION_STYLE.lineDash).toEqual([4, 7]);
  });

  it("keeps target detection labels in the labels layer and uses backend range", () => {
    const target = targetEstimateFixture({ detection_range_m: 275 });
    const frame = operationalFrameFixture({
      target_estimates: [target],
      uuvs: [{ ...uuv, uuv_id: "UUV-1", position: { x: 100, y: 0 } }],
      adversary: { target_id: target.target_id, detected_platform_ids: ["UUV-1"] },
    });

    expect(DETECTION_LABEL_LAYER).toBe("labels");
    expect(detectionZoneLabels(frame, target)).toEqual({
      radiusM: 275,
      detectedCount: 1,
      rangeText: "275 m",
      detectedText: "1 DETECTED",
    });
  });
});

it("clamps UUV boundary-transition opacity", () => {
  expect(uuvDisplayOpacity(uuv)).toBe(1);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: 0.35 })).toBe(0.35);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: -1 })).toBe(0);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: 2 })).toBe(1);
});

it("doubles the local region focus zoom and centres the selected region", () => {
  const bounds = { min_x: 0, min_y: 0, max_x: 1000, max_y: 1000 };
  const region = {
    geometry: [
      { x: 700, y: 300 },
      { x: 800, y: 300 },
      { x: 800, y: 400 },
      { x: 700, y: 400 },
    ],
  } as RegionTaskView;

  expect(nextRegionFocusZoom(1)).toBe(2);
  expect(nextRegionFocusZoom(2)).toBe(4);
  expect(nextRegionFocusZoom(5)).toBe(8);

  const view = focusRegionForCanvas(bounds, { width: 800, height: 600 }, region, 2);
  expect(view.zoom).toBe(2);
  expect(
    worldToScreen({ x: 750, y: 350 }, bounds, 800, 600, view),
  ).toEqual({ x: 400, y: 300 });
});

it("keeps onboard and onboard-failed UUVs out of spatial map inputs", () => {
  const onboard = { ...uuv, uuv_id: "UUV-onboard", physically_exposed: false };
  const failedOnboard = {
    ...onboard,
    uuv_id: "UUV-failed-onboard",
    status: "unavailable" as const,
    deployment_state: "failed" as const,
  };
  const returning = { ...uuv, uuv_id: "UUV-returning", deployment_state: "returning" as const, status: "unavailable" as const };
  const visible = waterborneUuvs({ uuvs: [onboard, failedOnboard, returning, uuv] } as unknown as OperationalFrame);
  expect(visible.map((item) => item.uuv_id)).toEqual(["UUV-returning", "uuv_01"]);

  const target = { target_id: "T1", mean: { x: 0, y: 0 } } as TargetEstimateView;
  const explicitDetections = detectedPlatformIds(
    {
      uuvs: [onboard, failedOnboard, returning, uuv],
      adversary: {
        detected_platform_ids: [
          "UUV-onboard",
          "UUV-failed-onboard",
          "UUV-returning",
          "uuv_01",
        ],
      },
    } as unknown as OperationalFrame,
    target,
  );
  expect(explicitDetections).toEqual(["UUV-returning", "uuv_01"]);
});

describe("CanvasMap sprite semantics", () => {
  it("uses only the authoritative four regions and eight spatial execution UUVs", () => {
    const taskRegions = Array.from({ length: 4 }, (_, index) => ({
      region_id: `target_00:task:${String(index + 1).padStart(2, "0")}`,
      target_id: "target_00",
      slot_index: index + 1,
      execution_revision: 4,
      prediction_id: "imm:target_00:4",
      geometry: [
        { x: index * 20, y: 0 },
        { x: index * 20 + 10, y: 0 },
        { x: index * 20 + 10, y: 10 },
      ],
      start_s: index * 60,
      end_s: (index + 1) * 60,
      geometry_revision: 4,
      predecessor_region_id: index ? `target_00:task:${String(index).padStart(2, "0")}` : null,
      successor_region_id: index < 3 ? `target_00:task:${String(index + 2).padStart(2, "0")}` : null,
      handoff_start_s: index * 60 + 50,
      handoff_end_s: index * 60 + 60,
      status: index === 0 ? "active" : "planned",
      task_group_id: `TG-${String(index + 1).padStart(2, "0")}`,
      evidence_ids: [`evidence-${index + 1}`],
    }));
    const taskGroups = taskRegions.map((region, index) => ({
      task_group_id: region.task_group_id,
      target_id: "target_00",
      region_id: region.region_id,
      execution_revision: 4,
      member_uuv_ids: [`uuv_${index * 2}`, `uuv_${index * 2 + 1}`],
      active_verifier_uuv_id: `uuv_${index * 2}`,
      passive_tracker_uuv_id: `uuv_${index * 2 + 1}`,
      status: index === 0 ? "active" : "prepositioning",
      evidence_ids: [region.evidence_ids[0]],
    }));
    const execution = {
      target_id: "target_00",
      execution_revision: 4,
      source_snapshot_revision: 20,
      prediction_revision: 4,
      intent_revision: 4,
      data_age_s: 0,
      data_status: "current",
      plan_source: "deterministic",
      current_region_id: taskRegions[0].region_id,
      next_region_id: taskRegions[1].region_id,
      evidence_ids: ["execution-evidence"],
      regions: taskRegions,
      task_groups: taskGroups,
      reserve_uuv_ids: ["uuv_08", "uuv_09", "uuv_10", "uuv_11"],
      degraded: false,
      degradation_reasons: [],
      active_plan_preserved: false,
    };
    const frame = {
      sim_time_s: 30,
      execution,
      regional_plans: {
        target_00: {
          target_id: "target_00",
          prediction_id: "candidate",
          revision: 99,
          cell_size_m: 250,
          regions: Array.from({ length: 36 }, (_, index) => ({
            region_id: `target_00:cell:${index}:0`,
            display_name: `cell_${index}`,
            target_id: "target_00",
            geometry: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }],
            start_time_s: 0,
            end_time_s: 60,
            predecessor_region_ids: [],
            successor_region_ids: [],
            assigned_uuv_ids: [],
            tracking_mode: "heuristic_uuv",
            group_id: null,
            status: "planned",
            effect: {
              status: "planned",
              coverage_ratio: 0,
              quality_score: 0,
              handoff_progress: 0,
              quality_source: "group_quality_proxy",
              hard_guard_reasons: [],
              expert_feedback_ids: [],
            },
          })),
        },
      },
      uuvs: [
        ...taskGroups.flatMap((group, index) => group.member_uuv_ids.map((uuvId) => ({
          ...uuv,
          uuv_id: uuvId,
          group_id: group.task_group_id,
          position: { x: index * 20, y: index * 20 },
        }))),
        ...["uuv_08", "uuv_09", "uuv_10", "uuv_11"].map((uuvId) => ({
          ...uuv,
          uuv_id: uuvId,
          group_id: null,
          physically_exposed: true,
          position: { x: 500, y: 500 },
        })),
      ],
    } as unknown as OperationalFrame;

    const plans = displayRegionalPlans(frame);

    expect(plans).toHaveLength(1);
    expect(plans[0].regions).toHaveLength(4);
    expect(plans[0].regions.every((region) => region.region_id.includes(":task:"))).toBe(true);
    expect(spatialExecutionUuvs(frame)).toHaveLength(8);
    expect(currentTaskUuvIds(frame)).toEqual(new Set(["uuv_0", "uuv_1"]));
  });

  it("uses prediction-corridor camera bounds for the rendered map", () => {
    const frame = {
      map_bounds: { min_x: -10_000, min_y: -10_000, max_x: 10_000, max_y: 10_000 },
      uuvs: [],
      target_estimates: [{
        target_id: "T1",
        mean: { x: 0, y: 0 },
        covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
        intent: { label: "transit", confidence: 0.8, alternatives: {} },
        prediction: {
          horizon_s: 60,
          sample_step_s: 30,
          centerline_xy: [{ x: 0, y: 0 }, { x: 1_000, y: 200 }],
          radius_m: [100, 200],
          point_confidence: [0.9, 0.7],
        },
        quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
        classification: "submarine",
        last_ping_s: null,
      }],
      regional_plans: {},
      groups: [],
      carriers: [],
    } as unknown as OperationalFrame;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);

    const view = render(createElement(CanvasMap, {
      frame,
      selectedUuvId: null,
      onSelectUuv: vi.fn(),
      showGrid: true,
      showPredictedRegions: true,
      showRegionHandoffs: true,
      showDetectionRange: false,
      trailMode: "tail",
      viewConfig: DEFAULT_VIEW_CONFIG,
    }));
    const map = view.container.querySelector(".canvas-area");
    const expected = semanticCameraForFrame(frame, { width: 1, height: 1 }).worldBounds;

    expect(map).toHaveAttribute("data-visible-bounds", JSON.stringify(expected));
    expect(map).not.toHaveAttribute("data-visible-bounds", JSON.stringify(frame.map_bounds));
  });

  it("does not reset user pan or zoom when the container resizes", () => {
    let resize: (() => void) | undefined;
    let clientWidth = 400;
    let clientHeight = 300;
    class FakeResizeObserver {
      constructor(callback: ResizeObserverCallback) {
        resize = () => callback([], this);
      }

      observe(_target: Element): void {}
      unobserve(_target: Element): void {}
      disconnect(): void {}
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null);
    const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
    const height = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get: () => clientWidth,
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get: () => clientHeight,
    });
    try {
      const view = render(createElement(CanvasMap, {
        frame: operationalFrameFixture(),
        selectedUuvId: null,
        onSelectUuv: vi.fn(),
        showGrid: true,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      const canvas = view.container.querySelector("canvas");
      const map = view.container.querySelector(".canvas-area");
      if (!canvas || !map) throw new Error("Canvas map did not render");
      vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
        x: 0,
        y: 0,
        width: 400,
        height: 300,
        top: 0,
        right: 400,
        bottom: 300,
        left: 0,
        toJSON: () => ({}),
      });
      Object.defineProperty(canvas, "setPointerCapture", { value: vi.fn() });
      Object.defineProperty(canvas, "releasePointerCapture", { value: vi.fn() });
      fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 200, clientY: 150 });
      fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 230, clientY: 175 });
      fireEvent.pointerUp(canvas, { pointerId: 1, clientX: 230, clientY: 175 });
      fireEvent.wheel(canvas, { deltaY: -100, clientX: 200, clientY: 150 });
      const boundsBeforeResize = map.getAttribute("data-visible-bounds");
      const panBeforeResize = map.getAttribute("data-camera-pan");
      const zoomBeforeResize = map.querySelector(".map-tools span")?.textContent;
      expect(map).toHaveAttribute("data-camera-dirty", "true");
      expect(panBeforeResize).not.toBe(JSON.stringify({ x: 0, y: 0 }));
      expect(zoomBeforeResize).not.toBe("1.0×");

      clientWidth = 390;
      clientHeight = 844;
      resize?.();

      expect(map).toHaveAttribute("data-camera-dirty", "true");
      expect(map.getAttribute("data-visible-bounds")).toBe(boundsBeforeResize);
      expect(map.getAttribute("data-camera-pan")).toBe(panBeforeResize);
      expect(map.querySelector(".map-tools span")?.textContent).toBe(zoomBeforeResize);
    } finally {
      getContext.mockRestore();
      vi.unstubAllGlobals();
      if (width) Object.defineProperty(HTMLElement.prototype, "clientWidth", width);
      if (height) Object.defineProperty(HTMLElement.prototype, "clientHeight", height);
    }
  });

  it("includes remaining assigned UUVs in the stable label candidate set", () => {
    const taskGroups: TaskGroupView[] = [
      {
        task_group_id: "TG-1",
        target_id: "T1",
        region_id: "T1:task:01",
        execution_revision: 1,
        member_uuv_ids: ["UUV-1", "UUV-2"],
        active_verifier_uuv_id: "UUV-1",
        passive_tracker_uuv_id: "UUV-2",
        status: "active",
        evidence_ids: [],
      },
      {
        task_group_id: "TG-2",
        target_id: "T1",
        region_id: "T1:task:02",
        execution_revision: 1,
        member_uuv_ids: ["UUV-3"],
        active_verifier_uuv_id: "UUV-3",
        passive_tracker_uuv_id: "UUV-3",
        status: "prepositioning",
        evidence_ids: [],
      },
    ];
    const execution: ExecutionView = {
      target_id: "T1",
      execution_revision: 1,
      source_snapshot_revision: 1,
      prediction_revision: 1,
      intent_revision: 1,
      data_age_s: 0,
      valid_from_s: 0,
      valid_until_s: 100,
      health_status: "current",
      health_reasons: [],
      region_generation_mode: "imm",
      plan_source: "deterministic",
      current_region_id: "T1:task:01",
      next_region_id: "T1:task:02",
      evidence_ids: [],
      regions: [],
      task_groups: taskGroups,
      reserve_uuv_ids: [],
      degraded: false,
      degradation_reasons: [],
      active_plan_preserved: false,
    };
    const frame = operationalFrameFixture({
      execution,
      uuvs: [
        { ...uuv, uuv_id: "UUV-1" },
        { ...uuv, uuv_id: "UUV-2" },
        { ...uuv, uuv_id: "UUV-3" },
      ],
    });
    const candidates = stableLabelCandidatesForFrame(
      frame,
      (point) => point,
      { selectedUuvId: null, showDetectionRange: false },
      frame.uuvs,
    );
    const uuvCandidates = candidates.filter((candidate) => candidate.id.startsWith("uuv:"));
    expect(uuvCandidates.map((candidate) => candidate.id)).toEqual([
      "uuv:UUV-1",
      "uuv:UUV-2",
      "uuv:UUV-3",
    ]);
    expect(uuvCandidates.find((candidate) => candidate.id === "uuv:UUV-1")?.priority).toBe(4);
    expect(uuvCandidates.find((candidate) => candidate.id === "uuv:UUV-3")?.priority).toBe(5);
  });

  it("keeps carrier labels in the stable layout with deterministic suppression", () => {
    const carriers: CarrierView[] = ["carrier-1", "carrier-2"].map((carrierId) => ({
      carrier_id: carrierId,
      role: "carrier",
      position: { x: 100, y: 30 },
      heading_rad: 0,
      speed_mps: 0,
      status: "standby",
      onboard_uuv_ids: [],
      deployed_uuv_ids: [],
      returning_uuv_ids: [],
    }));
    const frame = operationalFrameFixture({
      target_estimates: [targetEstimateFixture({ mean: { x: 100, y: 30 } })],
      carrier: null,
      carriers,
    });
    const candidates = stableLabelCandidatesForFrame(
      frame,
      (point) => point,
      { selectedUuvId: null, showDetectionRange: false },
      [],
    );
    expect(candidates.map((candidate) => candidate.id)).toEqual([
      "target:T1",
      "carrier:carrier-1",
      "carrier:carrier-2",
    ]);
    expect(candidates.slice(1)).toMatchObject([
      { priority: 6, text: "CARRIER carrier-1" },
      { priority: 6, text: "CARRIER carrier-2" },
    ]);

    const first = stableLabelPlacements(candidates, { width: 200, height: 48 });
    const second = stableLabelPlacements(candidates, { width: 200, height: 48 });
    expect(first).toEqual(second);
    expect(first.find((placement) => placement.id === "target:T1")?.suppressed).toBe(false);
    expect(first.find((placement) => placement.id === "carrier:carrier-1")?.suppressed).toBe(true);
    expect(first.find((placement) => placement.id === "carrier:carrier-2")?.suppressed).toBe(true);
  });

  it("draws stable labels with a top baseline at the placement origin", () => {
    type DrawStableLabels = (
      context: CanvasRenderingContext2D,
      frame: OperationalFrame,
      transform: (point: Point2D) => Point2D,
      options: { selectedUuvId: string | null; showDetectionRange: boolean },
      visibleUuvs: UUVView[],
      viewport: { width: number; height: number },
    ) => void;
    const drawStableLabels = Reflect.get(
      CanvasMapModule,
      "drawStableLabels",
    ) as DrawStableLabels;
    const frame = operationalFrameFixture({
      target_estimates: [targetEstimateFixture({ mean: { x: 190, y: 30 } })],
      uuvs: [],
    });
    let textBaseline = "alphabetic";
    const baselineValues: string[] = [];
    const fillCalls: Array<{ text: string; x: number; y: number; baseline: string }> = [];
    const context = {
      save: vi.fn(),
      restore: vi.fn(),
      measureText: vi.fn((_text: string) => ({
        width: 80,
        actualBoundingBoxLeft: 0,
        actualBoundingBoxRight: 80,
      })),
      fillText: vi.fn((text: string, x: number, y: number) => {
        fillCalls.push({ text, x, y, baseline: textBaseline });
      }),
      fillStyle: "",
      font: "",
      get textBaseline() {
        return textBaseline;
      },
      set textBaseline(value: string) {
        textBaseline = value;
        baselineValues.push(value);
      },
    };

    drawStableLabels(
      context as unknown as CanvasRenderingContext2D,
      frame,
      (point) => point,
      { selectedUuvId: null, showDetectionRange: false },
      [],
      { width: 200, height: 100 },
    );

    expect(baselineValues).toEqual(["top"]);
    expect(context.measureText).toHaveBeenCalledWith("target");
    expect(fillCalls).toEqual([
      { text: "target", x: 100, y: 20, baseline: "top" },
    ]);
    expect(fillCalls[0].x + 80).toBeLessThanOrEqual(200);
  });

  it("uses executing regional missions when a planning overlay is absent", () => {
    const frame = {
      regional_plans: {},
      regional_missions: [{
        region_id: "target_00:task:01",
        target_id: "target_00",
        cell_ids: ["target_00:cell:01"],
        geometry: [
          { x: -8000, y: -7000 },
          { x: -6000, y: -7000 },
          { x: -6000, y: -6000 },
          { x: -8000, y: -6000 },
        ],
        entry_s: 1200,
        exit_s: 1800,
        lifecycle: "ACTIVE_SCAN",
        active_scan_uuv_ids: ["uuv_00"],
        passive_track_uuv_ids: ["uuv_01", "uuv_02"],
        reserve_uuv_ids: [],
        coverage: 0.8,
        tracking_quality: 0.75,
        handoff_from: null,
        handoff_to: null,
        carrier_task_id: "carrier_01:deploy:0",
        carrier_id: "carrier_01",
        degraded_reasons: [],
        plan_revision: 1,
      }],
    } as unknown as OperationalFrame;

    const plans = displayRegionalPlans(frame);

    expect(plans).toHaveLength(1);
    expect(plans[0].regions[0]).toMatchObject({
      region_id: "target_00:task:01",
      assigned_uuv_ids: ["uuv_00", "uuv_01", "uuv_02"],
      effect: { status: "active", coverage_ratio: 0.8, quality_score: 0.75 },
    });
  });

  it("treats live task regions as authoritative over cell-level planning regions", () => {
    const plannedCell = {
      region_id: "target_00:cell:0:0",
      display_name: "cell_1",
      target_id: "target_00",
      geometry: [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }],
      start_time_s: 0,
      end_time_s: 300,
      predecessor_region_ids: [],
      successor_region_ids: [],
      assigned_uuv_ids: [],
      tracking_mode: "heuristic_uuv",
      group_id: null,
      status: "planned",
      effect: {
        status: "planned",
        coverage_ratio: 0,
        quality_score: 0,
        handoff_progress: 0,
        quality_source: "group_quality_proxy",
        hard_guard_reasons: [],
        expert_feedback_ids: [],
      },
    } as RegionTaskView;
    const frame = {
      regional_plans: {
        target_00: {
          target_id: "target_00",
          prediction_id: "prediction-1",
          revision: 1,
          cell_size_m: 250,
          regions: Array.from({ length: 36 }, (_, index) => ({
            ...plannedCell,
            region_id: `target_00:cell:${index}:0`,
          })),
        },
      },
      regional_missions: Array.from({ length: 4 }, (_, index) => ({
        region_id: `target_00:task:${String(index + 1).padStart(2, "0")}`,
        target_id: "target_00",
        cell_ids: [`target_00:cell:${index}:0`],
        geometry: [{ x: index * 20, y: 0 }, { x: index * 20 + 10, y: 0 }, { x: index * 20 + 10, y: 10 }],
        entry_s: index * 300,
        exit_s: (index + 1) * 300,
        lifecycle: index === 0 ? "ACTIVE_SCAN" : "PLANNED",
        active_scan_uuv_ids: index === 0 ? ["uuv_00"] : [],
        passive_track_uuv_ids: [],
        reserve_uuv_ids: [],
        coverage: 0,
        tracking_quality: 0,
        handoff_from: index === 0 ? null : `target_00:task:${String(index).padStart(2, "0")}`,
        handoff_to: index === 3 ? null : `target_00:task:${String(index + 2).padStart(2, "0")}`,
        carrier_task_id: null,
        carrier_id: null,
        degraded_reasons: [],
        plan_revision: 2,
      })),
    } as unknown as OperationalFrame;

    const plans = displayRegionalPlans(frame);

    expect(plans).toHaveLength(1);
    expect(plans[0].regions).toHaveLength(4);
    expect(plans[0].regions.every((region) => region.region_id.includes(":task:"))).toBe(true);
  });

  it("uses the current fitted view to label the scale bar", () => {
    const bounds = { min_x: -12000, min_y: -12000, max_x: 12000, max_y: 12000 };
    const overview = mapScaleForView(bounds, 960, 960, 1);
    const zoomed = mapScaleForView(bounds, 960, 960, 4);

    expect(overview.label).toBe("2 km");
    expect(overview.widthPx).toBeCloseTo(80, 0);
    expect(zoomed.label).toBe("500 m");
    expect(zoomed.widthPx).toBeCloseTo(80, 0);
  });

  it("aligns carrier and warship assets with the shared world heading convention", () => {
    expect(carrierAssetRotation(0)).toBeCloseTo(0);
    expect(carrierAssetRotation(Math.PI / 2)).toBeCloseTo(-Math.PI / 2);
    expect(warshipAssetRotation(0)).toBeCloseTo(Math.PI / 2);
    expect(warshipAssetRotation(Math.PI / 2)).toBeCloseTo(0);
  });

  it("keeps active, failed, reserved, and selected cues when a UUV image is loaded", () => {
    const image = {
      naturalWidth: 1536,
      naturalHeight: 1024,
    } as HTMLImageElement;

    expect(
      uuvSpriteAppearance({ ...uuv, sensor_mode: "active" }, image, 1, false)
        .cueColors,
    ).toContain("#f7bd45");
    expect(
      uuvSpriteAppearance({ ...uuv, status: "unavailable" }, image, 1, false)
        .cueColors,
    ).toContain("#ff7882");
    expect(
      uuvSpriteAppearance({ ...uuv, reserved: true }, image, 1, false)
        .cueColors,
    ).toContain("#c4b4ff");
    expect(uuvSpriteAppearance(uuv, image, 1, true).cueColors).toContain(
      "#f8fdff",
    );
  });

  it("highlights only the selected UUV and peers from its assigned group", () => {
    const frame = {
      uuvs: [
        { ...uuv, uuv_id: "UUV-01", group_id: "G-1" },
        { ...uuv, uuv_id: "UUV-02", group_id: "G-1" },
        { ...uuv, uuv_id: "UUV-03", group_id: "G-2" },
      ],
      groups: [{ group_id: "G-1", member_ids: ["UUV-01", "UUV-02"] }],
    } as unknown as OperationalFrame;

    expect([...highlightedUuvIds(frame, null)]).toEqual([]);
    expect([...highlightedUuvIds(frame, "UUV-01")]).toEqual([
      "UUV-01",
      "UUV-02",
    ]);
    expect([...highlightedUuvIds(frame, "UUV-03")]).toEqual(["UUV-03"]);
  });

  it("aligns the submarine asset and exposes range-driven platform visibility", () => {
    expect(submarineAssetRotation(0)).toBeCloseTo(Math.PI);
    const target = {
      target_id: "T1",
      mean: { x: 0, y: 0 },
      covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
      intent: { label: "unknown", confidence: 0, alternatives: {} },
      prediction: null,
      quality: {
        quality_score: 0.8,
        estimated_rmse_m: 20,
        fim_min_eigenvalue: 1,
        fim_condition: 1,
      },
      classification: "submarine",
      last_ping_s: null,
      detection_range_m: 100,
    } as TargetEstimateView;
    const frame = {
      map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
      uuvs: [
        { ...uuv, uuv_id: "UUV-NEAR", position: { x: 80, y: 0 } },
        { ...uuv, uuv_id: "UUV-FAR", position: { x: 160, y: 0 } },
      ],
      communication_links: [
        {
          source_id: "UUV-NEAR",
          target_id: "CARRIER-01",
          medium: "surface",
          distance_m: 400,
          limit_m: 900,
          status: "connected",
          relay: true,
        },
      ],
    } as unknown as OperationalFrame;

    expect(communicationRangeForUuv(frame, "UUV-NEAR")).toBe(900);
    expect(targetDetectionRange(target)).toBe(100);
    expect(detectedPlatformIds(frame, target)).toEqual(["UUV-NEAR"]);
    expect(DEFAULT_SUBMARINE_DETECTION_RANGE_M).toBe(5000);
  });

  it("aims a passive footprint with its sensor heading independently from its hull", () => {
    const passive = CanvasMapModule.uuvSensorFootprint({
      ...uuv,
      heading_rad: 0,
      sensor_heading_rad: Math.PI / 2,
    });
    const active = CanvasMapModule.uuvSensorFootprint({
      ...uuv,
      heading_rad: Math.PI / 2,
      sensor_mode: "active",
    });

    expect(passive).toMatchObject({
      radiusM: 2000,
      centerAngleRad: -Math.PI / 2,
      spanAngleRad: Math.PI / 2,
      strokeStyle: "rgba(33, 208, 195, 0.82)",
    });
    expect(active).toMatchObject({
      radiusM: 2000,
      centerAngleRad: -Math.PI / 2,
      spanAngleRad: Math.PI / 2,
      strokeStyle: "rgba(247, 189, 69, 0.88)",
    });
    expect(CanvasMapModule.uuvSensorFootprint({
      ...uuv,
      sensor_mode: "active",
      active_range_m: 750,
    }).radiusM).toBe(750);
    expect(CanvasMapModule.uuvSensorFootprint({
      ...uuv,
      sensor_mode: "passive",
      passive_range_m: 1250,
    }).radiusM).toBe(1250);
  });

  it("keeps detection range opt-in while using a fine base grid", () => {
    expect(shouldDrawDetectionRange(false)).toBe(false);
    expect(shouldDrawDetectionRange(true)).toBe(true);
    expect(GRID_DIVISIONS).toBe(24);
  });

  it("focuses the default camera on the prediction corridor without hidden detection bounds", () => {
    const frame = {
      map_bounds: { min_x: -5000, min_y: -5000, max_x: 5000, max_y: 5000 },
      target_estimates: [
        {
          target_id: "T1",
          mean: { x: 0, y: 0 },
          covariance_ellipse: {
            semimajor_m: 20,
            semiminor_m: 10,
            rotation_rad: 0,
          },
          intent: { label: "unknown", confidence: 0, alternatives: {} },
          prediction: {
            horizon_s: 120,
            sample_step_s: 30,
            centerline_xy: [
              { x: 0, y: 0 },
              { x: 1000, y: 0 },
            ],
            radius_m: [100, 200],
          },
          quality: {
            quality_score: 0.8,
            estimated_rmse_m: 20,
            fim_min_eigenvalue: 1,
            fim_condition: 1,
          },
          classification: "submarine",
          last_ping_s: null,
          detection_range_m: 1800,
        },
      ],
      regional_plans: {
        T1: {
          target_id: "T1",
          prediction_id: "pred-1",
          revision: 1,
          cell_size_m: 100,
          regions: [
            {
              region_id: "T1:cell:0:0",
              display_name: "region_1",
              target_id: "T1",
              geometry: [
                { x: 300, y: 100 },
                { x: 500, y: 100 },
                { x: 500, y: 300 },
                { x: 300, y: 300 },
              ],
              start_time_s: 0,
              end_time_s: 10,
              predecessor_region_ids: [],
              successor_region_ids: [],
              assigned_uuv_ids: [],
              tracking_mode: "heuristic_uuv",
              group_id: null,
              status: "planned",
              effect: {
                status: "planned",
                coverage_ratio: 0,
                quality_score: 0,
                handoff_progress: 0,
                quality_source: "group_quality_proxy",
                hard_guard_reasons: [],
                expert_feedback_ids: [],
              },
            },
          ],
        },
      },
    } as unknown as OperationalFrame;

    expect(cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false)).toEqual({
      min_x: -150,
      min_y: -275,
      max_x: 1150,
      max_y: 375,
    });
    expect(cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, true)).toEqual({
      min_x: -2340,
      min_y: -2340,
      max_x: 2340,
      max_y: 2340,
    });
    expect(
      cameraBoundsForFrame(
        frame,
        { ...DEFAULT_VIEW_CONFIG, focusMode: "full_area" },
        false,
      ),
    ).toEqual(frame.map_bounds);
  });

  it("frames carrier positions and a known submarine before deployment", () => {
    const frame = {
      map_bounds: { min_x: -12000, min_y: -12000, max_x: 12000, max_y: 12000 },
      carriers: [
        {
          carrier_id: "carrier_01",
          role: "carrier",
          position: { x: -8000, y: -8000 },
          heading_rad: 0,
          speed_mps: 4,
          status: "transit",
          onboard_uuv_ids: [],
          deployed_uuv_ids: [],
          returning_uuv_ids: [],
        },
        {
          carrier_id: "carrier_03",
          role: "mother_ship",
          position: { x: -7000, y: -8000 },
          heading_rad: 0,
          speed_mps: 8,
          status: "transit",
          onboard_uuv_ids: ["uuv_04"],
          deployed_uuv_ids: [],
          returning_uuv_ids: [],
        },
      ],
      uuvs: [
        { ...uuv, uuv_id: "uuv_04", physically_exposed: false, deployment_state: "onboard" },
      ],
      target_estimates: [{
        target_id: "T1",
        mean: { x: -4200, y: -6200 },
        covariance_ellipse: { semimajor_m: 25, semiminor_m: 12, rotation_rad: 0 },
        intent: { label: "unknown", confidence: 0, alternatives: {} },
        prediction: null,
        quality: { quality_score: 1, estimated_rmse_m: 0, fim_min_eigenvalue: 1, fim_condition: 1 },
        classification: "submarine",
        last_ping_s: null,
      }],
      regional_plans: {},
    } as unknown as OperationalFrame;

    const bounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false);
    expect(bounds.min_x).toBeLessThanOrEqual(-8000);
    expect(bounds.max_x).toBeGreaterThanOrEqual(-4200);
    expect(bounds.min_y).toBeLessThanOrEqual(-8000);
    expect(bounds.max_y).toBeGreaterThanOrEqual(-6200);
  });

  it("keeps a usable local camera span for a lone target without entering its hidden detection range", () => {
    const frame = {
      map_bounds: { min_x: -5000, min_y: -5000, max_x: 5000, max_y: 5000 },
      uuvs: [{ ...uuv, position: { x: 400, y: 0 } }],
      target_estimates: [
        {
          target_id: "T1",
          mean: { x: 0, y: 0 },
          covariance_ellipse: {
            semimajor_m: 20,
            semiminor_m: 10,
            rotation_rad: 0,
          },
          intent: { label: "unknown", confidence: 0, alternatives: {} },
          prediction: null,
          quality: {
            quality_score: 0.8,
            estimated_rmse_m: 20,
            fim_min_eigenvalue: 1,
            fim_condition: 1,
          },
          classification: "submarine",
          last_ping_s: null,
          detection_range_m: 1800,
        },
      ],
      regional_plans: {},
    } as unknown as OperationalFrame;

    const bounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false);

    expect(bounds.max_x - bounds.min_x).toBeGreaterThanOrEqual(1000);
    expect(bounds.max_y - bounds.min_y).toBeGreaterThanOrEqual(1000);
    expect(bounds.min_x).toBeLessThanOrEqual(400);
    expect(bounds.max_x).toBeGreaterThanOrEqual(400);
    expect(bounds.max_x).toBeLessThan(1800);
  });

  it("frames future event positions only while prediction overlays are visible", () => {
    const frame = {
      map_bounds: { min_x: -10000, min_y: -10000, max_x: 10000, max_y: 10000 },
      uuvs: [],
      target_estimates: [
        {
          target_id: "T1",
          mean: { x: 0, y: 0 },
          covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
          intent: { label: "unknown", confidence: 0, alternatives: {} },
          prediction: null,
          world_model: {
            events: [{ predicted_position: { x: 5000, y: 0 } }],
          },
          quality: {
            quality_score: 0.8,
            estimated_rmse_m: 20,
            fim_min_eigenvalue: 1,
            fim_condition: 1,
          },
          classification: "submarine",
          last_ping_s: null,
        },
      ],
      regional_plans: {},
    } as unknown as OperationalFrame;

    const hidden = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, false);
    const visible = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, true);

    expect(hidden.max_x).toBeLessThan(5000);
    expect(visible.max_x).toBeGreaterThanOrEqual(5000);
  });

  it("keeps marker dimensions clamped in screen pixels", () => {
    expect(clampedMarkerPixels(14, 18, 42)).toBe(18);
    expect(clampedMarkerPixels(84, 18, 42)).toBe(42);
    expect(clampedMarkerPixels(30, 18, 42)).toBe(30);
  });

  it("retains detailed regions for hit tests while adapting labels to zoom", () => {
    const region = {
      region_id: "T1:cell:0:1",
      display_name: "region_2",
      target_id: "T1",
      geometry: [
        { x: 100, y: 100 },
        { x: 200, y: 100 },
        { x: 200, y: 200 },
        { x: 100, y: 200 },
      ],
    } as RegionTaskView;

    expect(regionLabelForZoom(region, 0.75)).toBe("区域");
    expect(regionLabelForZoom(region, 1.5)).toBe("R02");
    expect(hitTestRegion({ x: 150, y: 150 }, [region])?.region_id).toBe(
      "T1:cell:0:1",
    );
    expect(hitTestRegion({ x: 250, y: 150 }, [region])).toBeNull();
  });

  it("clears regional selection while hidden, ignores hidden clicks, and preserves UUV selection", () => {
    const frame = {
      map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
      uuvs: [{ ...uuv, uuv_id: "UUV-1", position: { x: 100, y: 100 } }],
      target_estimates: [
        {
          target_id: "T1",
          mean: { x: 100, y: 100 },
          covariance_ellipse: {
            semimajor_m: 20,
            semiminor_m: 10,
            rotation_rad: 0,
          },
          intent: { label: "unknown", confidence: 0, alternatives: {} },
          prediction: null,
          quality: {
            quality_score: 0.8,
            estimated_rmse_m: 20,
            fim_min_eigenvalue: 1,
            fim_condition: 1,
          },
          classification: "submarine",
          last_ping_s: null,
        },
      ],
      regional_plans: {
        T1: {
          target_id: "T1",
          prediction_id: "pred-1",
          revision: 1,
          cell_size_m: 100,
          regions: [
            {
              region_id: "T1:cell:0:1",
              display_name: "region_2",
              target_id: "T1",
              geometry: [
                { x: 300, y: 300 },
                { x: 500, y: 300 },
                { x: 500, y: 500 },
                { x: 300, y: 500 },
              ],
              start_time_s: 0,
              end_time_s: 10,
              predecessor_region_ids: [],
              successor_region_ids: [],
              assigned_uuv_ids: [],
              tracking_mode: "heuristic_uuv",
              group_id: null,
              status: "planned",
              effect: {
                status: "planned",
                coverage_ratio: 0,
                quality_score: 0,
                handoff_progress: 0,
                quality_source: "group_quality_proxy",
                hard_guard_reasons: [],
                expert_feedback_ids: [],
              },
            },
          ],
        },
      },
    } as unknown as OperationalFrame;
    const bounds = semanticCameraForFrame(frame, { width: 400, height: 300 }).worldBounds;
    const hiddenBounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, false);
    const regionScreenPoint = worldToScreen(
      { x: 400, y: 400 },
      bounds,
      400,
      300,
      { zoom: 1, pan: { x: 0, y: 0 } },
    );
    const hiddenRegionScreenPoint = worldToScreen(
      { x: 400, y: 400 },
      hiddenBounds,
      400,
      300,
      { zoom: 1, pan: { x: 0, y: 0 } },
    );
    const focusedView = focusRegionForCanvas(bounds, { width: 400, height: 300 }, {
      geometry: [
        { x: 300, y: 300 },
        { x: 500, y: 300 },
        { x: 500, y: 500 },
        { x: 300, y: 500 },
      ],
    }, 2);
    const focusedUuvScreenPoint = worldToScreen(
      { x: 100, y: 100 },
      bounds,
      400,
      300,
      focusedView,
    );
    const onSelectUuv = vi.fn();
    const getContext = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null);
    const width = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "clientWidth",
    );
    const height = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "clientHeight",
    );
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get: () => 400,
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get: () => 300,
    });

    try {
      const view = render(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: true,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      const canvas = view.container.querySelector("canvas");
      if (!canvas) throw new Error("Canvas map did not render a canvas");
      vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
        x: 0,
        y: 0,
        width: 400,
        height: 300,
        top: 0,
        right: 400,
        bottom: 300,
        left: 0,
        toJSON: () => ({}),
      });

      fireEvent.click(canvas, {
        clientX: regionScreenPoint.x,
        clientY: regionScreenPoint.y,
      });
      expect(screen.getByText("区域 region_2")).toBeInTheDocument();
      expect(onSelectUuv).not.toHaveBeenCalled();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          selectedRegionId: null,
          showGrid: true,
          showPredictedRegions: true,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: true,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: false,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: true,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: false,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      fireEvent.click(canvas, {
        clientX: hiddenRegionScreenPoint.x,
        clientY: hiddenRegionScreenPoint.y,
      });
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(
        createElement(CanvasMap, {
          frame,
          selectedUuvId: null,
          onSelectUuv,
          showGrid: true,
          showPredictedRegions: true,
          showRegionHandoffs: true,
          showDetectionRange: false,
          trailMode: "tail",
          viewConfig: DEFAULT_VIEW_CONFIG,
        }),
      );
      fireEvent.doubleClick(canvas, {
        clientX: regionScreenPoint.x,
        clientY: regionScreenPoint.y,
      });
      fireEvent.click(canvas, {
        clientX: focusedUuvScreenPoint.x,
        clientY: focusedUuvScreenPoint.y,
      });
      expect(onSelectUuv).toHaveBeenCalledWith("UUV-1");
    } finally {
      getContext.mockRestore();
      if (width)
        Object.defineProperty(HTMLElement.prototype, "clientWidth", width);
      if (height)
        Object.defineProperty(HTMLElement.prototype, "clientHeight", height);
    }
  });
});
