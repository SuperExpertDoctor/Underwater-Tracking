import { fireEvent, render, screen } from "@testing-library/react";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";
import type { UUVView } from "../types/frames";
import {
  CARRIER_ASSET_HEADING_OFFSET,
  DEFAULT_SUBMARINE_DETECTION_RANGE_M,
  cameraBoundsForFrame,
  clampedMarkerPixels,
  communicationRangeForUsv,
  carrierAssetRotation,
  detectedPlatformIds,
  GRID_DIVISIONS,
  hitTestRegion,
  regionLabelForZoom,
  mapScaleForView,
  shouldDrawDetectionRange,
  submarineAssetRotation,
  targetDetectionRange,
  usvSpriteAppearance,
  uuvSpriteAppearance,
} from "./CanvasMap";
import CanvasMap from "./CanvasMap";
import { worldToScreen } from "./map/geometry";
import type { OperationalFrame, RegionTaskView, TargetEstimateView, USVView } from "../types/frames";
import { DEFAULT_VIEW_CONFIG } from "../types/viewConfig";

const uuv: UUVView = {
  uuv_id: "uuv_01",
  status: "available",
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
};

describe("CanvasMap sprite semantics", () => {
  it("uses the current fitted view to label the scale bar", () => {
    const bounds = { min_x: -12000, min_y: -12000, max_x: 12000, max_y: 12000 };
    const overview = mapScaleForView(bounds, 960, 960, 1);
    const zoomed = mapScaleForView(bounds, 960, 960, 4);

    expect(overview.label).toBe("2 km");
    expect(overview.widthPx).toBeCloseTo(80, 0);
    expect(zoomed.label).toBe("500 m");
    expect(zoomed.widthPx).toBeCloseTo(80, 0);
  });

  it("aligns the upward-facing carrier asset with the vector heading convention", () => {
    expect(CARRIER_ASSET_HEADING_OFFSET).toBeCloseTo(Math.PI / 2);
    expect(carrierAssetRotation(0)).toBeCloseTo(Math.PI / 2);
    expect(carrierAssetRotation(Math.PI / 2)).toBeCloseTo(0);
  });

  it("keeps active, failed, reserved, and selected cues when a UUV image is loaded", () => {
    const image = { naturalWidth: 1536, naturalHeight: 1024 } as HTMLImageElement;

    expect(uuvSpriteAppearance({ ...uuv, sensor_mode: "active" }, image, 1, false).cueColors)
      .toContain("#f7bd45");
    expect(uuvSpriteAppearance({ ...uuv, status: "failed" }, image, 1, false).cueColors)
      .toContain("#ff7882");
    expect(uuvSpriteAppearance({ ...uuv, reserved: true }, image, 1, false).cueColors)
      .toContain("#c4b4ff");
    expect(uuvSpriteAppearance(uuv, image, 1, true).cueColors)
      .toContain("#f8fdff");
  });

  it("keeps a marker ring visible for unselected UUVs and USVs", () => {
    const image = { naturalWidth: 1536, naturalHeight: 1024 } as HTMLImageElement;
    const uuvAppearance = uuvSpriteAppearance(uuv, image, 1, false);
    const usvAppearance = usvSpriteAppearance({ sensor_mode: "passive" } as USVView, image, 1);

    expect(uuvAppearance.markerRing.color).toBe("#21d0c3");
    expect(uuvAppearance.markerRing.highlightColor).toBeNull();
    expect(usvAppearance.markerRing.color).toBe("#66e0ad");
  });

  it("aligns the submarine asset and exposes range-driven platform visibility", () => {
    expect(submarineAssetRotation(0)).toBeCloseTo(Math.PI);
    const target = {
      target_id: "T1",
      mean: { x: 0, y: 0 },
      covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
      intent: { label: "unknown", confidence: 0, alternatives: {} },
      prediction: null,
      quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
      classification: "submarine",
      last_ping_s: null,
      detection_range_m: 100,
    } as TargetEstimateView;
    const frame = {
      map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
      uuvs: [{ ...uuv, uuv_id: "UUV-NEAR", position: { x: 80, y: 0 } }, { ...uuv, uuv_id: "UUV-FAR", position: { x: 160, y: 0 } }],
      usvs: [],
      communication_links: [{ source_id: "USV-01", target_id: "CARRIER-01", medium: "surface", distance_m: 400, limit_m: 900, status: "connected", relay: true }],
    } as unknown as OperationalFrame;

    expect(communicationRangeForUsv(frame, "USV-01")).toBe(900);
    expect(targetDetectionRange(target)).toBe(100);
    expect(detectedPlatformIds(frame, target)).toEqual(["UUV-NEAR"]);
    expect(DEFAULT_SUBMARINE_DETECTION_RANGE_M).toBeGreaterThan(0);
  });

  it("keeps detection range opt-in while using a fine base grid", () => {
    expect(shouldDrawDetectionRange(false)).toBe(false);
    expect(shouldDrawDetectionRange(true)).toBe(true);
    expect(GRID_DIVISIONS).toBe(16);
  });

  it("focuses the default camera on the prediction corridor without hidden detection bounds", () => {
    const frame = {
      map_bounds: { min_x: -5000, min_y: -5000, max_x: 5000, max_y: 5000 },
      target_estimates: [{
        target_id: "T1",
        mean: { x: 0, y: 0 },
        covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
        intent: { label: "unknown", confidence: 0, alternatives: {} },
        prediction: { horizon_s: 120, sample_step_s: 30, centerline_xy: [{ x: 0, y: 0 }, { x: 1000, y: 0 }], radius_m: [100, 200] },
        quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
        classification: "submarine",
        last_ping_s: null,
        detection_range_m: 1800,
      }],
      regional_plans: {
        T1: {
          target_id: "T1", prediction_id: "pred-1", revision: 1, cell_size_m: 100,
          regions: [{
            region_id: "T1:cell:0:0", display_name: "region_1", target_id: "T1",
            geometry: [{ x: 300, y: 100 }, { x: 500, y: 100 }, { x: 500, y: 300 }, { x: 300, y: 300 }],
            start_time_s: 0, end_time_s: 10, predecessor_region_ids: [], successor_region_ids: [], assigned_uuv_ids: [], assigned_usv_ids: [],
            tracking_mode: "heuristic_uuv", relay_usv_ids: [], group_id: null, status: "planned",
            effect: { status: "planned", coverage_ratio: 0, quality_score: 0, handoff_progress: 0, quality_source: "group_quality_proxy", hard_guard_reasons: [], expert_feedback_ids: [] },
          }],
        },
      },
    } as unknown as OperationalFrame;

    expect(cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false)).toEqual({ min_x: -150, min_y: -275, max_x: 1150, max_y: 375 });
    expect(cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, true)).toEqual({ min_x: -2340, min_y: -2340, max_x: 2340, max_y: 2340 });
    expect(cameraBoundsForFrame(frame, { ...DEFAULT_VIEW_CONFIG, focusMode: "full_area" }, false)).toEqual(frame.map_bounds);
  });

  it("keeps a usable local camera span for a lone target without entering its hidden detection range", () => {
    const frame = {
      map_bounds: { min_x: -5000, min_y: -5000, max_x: 5000, max_y: 5000 },
      uuvs: [{ ...uuv, position: { x: 400, y: 0 } }],
      target_estimates: [{
        target_id: "T1",
        mean: { x: 0, y: 0 },
        covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
        intent: { label: "unknown", confidence: 0, alternatives: {} },
        prediction: null,
        quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
        classification: "submarine",
        last_ping_s: null,
        detection_range_m: 1800,
      }],
      regional_plans: {},
    } as unknown as OperationalFrame;

    const bounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false);

    expect(bounds.max_x - bounds.min_x).toBeGreaterThanOrEqual(1000);
    expect(bounds.max_y - bounds.min_y).toBeGreaterThanOrEqual(1000);
    expect(bounds.min_x).toBeLessThanOrEqual(400);
    expect(bounds.max_x).toBeGreaterThanOrEqual(400);
    expect(bounds.max_x).toBeLessThan(1800);
  });

  it("keeps marker dimensions clamped in screen pixels", () => {
    expect(clampedMarkerPixels(14, 18, 42)).toBe(18);
    expect(clampedMarkerPixels(84, 18, 42)).toBe(42);
    expect(clampedMarkerPixels(30, 18, 42)).toBe(30);
  });

  it("retains detailed regions for hit tests while adapting labels to zoom", () => {
    const region = {
      region_id: "T1:cell:0:1", display_name: "region_2", target_id: "T1",
      geometry: [{ x: 100, y: 100 }, { x: 200, y: 100 }, { x: 200, y: 200 }, { x: 100, y: 200 }],
    } as RegionTaskView;

    expect(regionLabelForZoom(region, 0.75)).toBe("区域");
    expect(regionLabelForZoom(region, 1.5)).toBe("R02");
    expect(hitTestRegion({ x: 150, y: 150 }, [region])?.region_id).toBe("T1:cell:0:1");
    expect(hitTestRegion({ x: 250, y: 150 }, [region])).toBeNull();
  });

  it("clears regional selection while hidden, ignores hidden clicks, and preserves UUV selection", () => {
    const frame = {
      map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
      uuvs: [{ ...uuv, uuv_id: "UUV-1", position: { x: 100, y: 100 } }],
      target_estimates: [{
        target_id: "T1",
        mean: { x: 100, y: 100 },
        covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
        intent: { label: "unknown", confidence: 0, alternatives: {} },
        prediction: null,
        quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
        classification: "submarine",
        last_ping_s: null,
      }],
      regional_plans: {
        T1: {
          target_id: "T1", prediction_id: "pred-1", revision: 1, cell_size_m: 100,
          regions: [{
            region_id: "T1:cell:0:1", display_name: "region_2", target_id: "T1",
            geometry: [{ x: 300, y: 300 }, { x: 500, y: 300 }, { x: 500, y: 500 }, { x: 300, y: 500 }],
            start_time_s: 0, end_time_s: 10, predecessor_region_ids: [], successor_region_ids: [], assigned_uuv_ids: [], assigned_usv_ids: [],
            tracking_mode: "heuristic_uuv", relay_usv_ids: [], group_id: null, status: "planned",
            effect: { status: "planned", coverage_ratio: 0, quality_score: 0, handoff_progress: 0, quality_source: "group_quality_proxy", hard_guard_reasons: [], expert_feedback_ids: [] },
          }],
        },
      },
    } as unknown as OperationalFrame;
    const bounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, true);
    const hiddenBounds = cameraBoundsForFrame(frame, DEFAULT_VIEW_CONFIG, false, false);
    const regionScreenPoint = worldToScreen({ x: 400, y: 400 }, bounds, 400, 300, { zoom: 1, pan: { x: 0, y: 0 } });
    const hiddenRegionScreenPoint = worldToScreen({ x: 400, y: 400 }, hiddenBounds, 400, 300, { zoom: 1, pan: { x: 0, y: 0 } });
    const uuvScreenPoint = worldToScreen({ x: 100, y: 100 }, bounds, 400, 300, { zoom: 1, pan: { x: 0, y: 0 } });
    const onSelectUuv = vi.fn();
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
    const height = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 400 });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => 300 });

    try {
      const view = render(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      const canvas = view.container.querySelector("canvas");
      if (!canvas) throw new Error("Canvas map did not render a canvas");
      vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
        x: 0, y: 0, width: 400, height: 300, top: 0, right: 400, bottom: 300, left: 0, toJSON: () => ({}),
      });

      fireEvent.click(canvas, { clientX: regionScreenPoint.x, clientY: regionScreenPoint.y });
      expect(screen.getByText("区域 region_2")).toBeInTheDocument();
      expect(onSelectUuv).not.toHaveBeenCalled();

      view.rerender(createElement(CanvasMap, {
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
      }));
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: false,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();
      expect(screen.queryByRole("status")).not.toBeInTheDocument();

      view.rerender(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: false,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      fireEvent.click(canvas, { clientX: hiddenRegionScreenPoint.x, clientY: hiddenRegionScreenPoint.y });
      expect(screen.queryByText("区域 region_2")).not.toBeInTheDocument();

      view.rerender(createElement(CanvasMap, {
        frame,
        selectedUuvId: null,
        onSelectUuv,
        showGrid: true,
        showPredictedRegions: true,
        showRegionHandoffs: true,
        showDetectionRange: false,
        trailMode: "tail",
        viewConfig: DEFAULT_VIEW_CONFIG,
      }));
      fireEvent.click(canvas, { clientX: uuvScreenPoint.x, clientY: uuvScreenPoint.y });
      expect(onSelectUuv).toHaveBeenCalledWith("UUV-1");
    } finally {
      getContext.mockRestore();
      if (width) Object.defineProperty(HTMLElement.prototype, "clientWidth", width);
      if (height) Object.defineProperty(HTMLElement.prototype, "clientHeight", height);
    }
  });
});
