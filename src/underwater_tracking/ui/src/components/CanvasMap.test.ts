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
  regionLabelForZoom,
  mapScaleForView,
  shouldDrawDetectionRange,
  submarineAssetRotation,
  targetDetectionRange,
  uuvSpriteAppearance,
  uuvDisplayOpacity,
  warshipAssetRotation,
  waterborneUuvs,
} from "./CanvasMap";
import CanvasMap from "./CanvasMap";
import { worldToScreen } from "./map/geometry";
import type {
  OperationalFrame,
  RegionTaskView,
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

it("clamps UUV boundary-transition opacity", () => {
  expect(uuvDisplayOpacity(uuv)).toBe(1);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: 0.35 })).toBe(0.35);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: -1 })).toBe(0);
  expect(uuvDisplayOpacity({ ...uuv, display_opacity: 2 })).toBe(1);
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
    const expected = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false);

    expect(map).toHaveAttribute("data-visible-bounds", JSON.stringify(expected));
    expect(map).not.toHaveAttribute("data-visible-bounds", JSON.stringify(frame.map_bounds));
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
    const bounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, true);
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
    const uuvScreenPoint = worldToScreen({ x: 100, y: 100 }, bounds, 400, 300, {
      zoom: 1,
      pan: { x: 0, y: 0 },
    });
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
      fireEvent.click(canvas, {
        clientX: uuvScreenPoint.x,
        clientY: uuvScreenPoint.y,
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
