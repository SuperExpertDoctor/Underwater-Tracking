import { describe, expect, it } from "vitest";
import type {
  ExecutionRegionView,
  ExecutionView,
  OperationalFrame,
  Point2D,
  TargetEstimateView,
  TaskGroupView,
  UUVView,
} from "../../types/frames";
import {
  semanticCameraCandidates,
  semanticCameraForFrame,
  stableLabelPlacements,
  type CameraViewport,
} from "./camera";
import { MAP_DISPLAY_CONFIG } from "../../../configs/map_display";

const mapBounds = { min_x: -10_000, min_y: -8_000, max_x: 10_000, max_y: 8_000 };

function uuv(id: string, position: Point2D): UUVView {
  return {
    uuv_id: id,
    status: "active",
    deployment_state: "deployed",
    physically_exposed: true,
    position,
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 1,
    group_id: `group-${id}`,
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: "passive",
    reserved: false,
  };
}

function targetEstimate(
  predictionStatus: "valid" | "unavailable",
  targetId = "T1",
): TargetEstimateView {
  return {
    target_id: targetId,
    mean: { x: 0, y: 0 },
    covariance_ellipse: {
      semimajor_m: 100,
      semiminor_m: 60,
      rotation_rad: 0,
    },
    intent: { label: "transit", confidence: 0.9, alternatives: {} },
    prediction: {
      prediction_id: "prediction-1",
      prediction_revision: 7,
      origin_sim_time_s: 120,
      health: {
        status: predictionStatus,
        regime: "imm",
        reason_codes: predictionStatus === "valid" ? [] : ["prediction_unavailable"],
        source_track_age_s: 10,
        clipped_point_fraction: 0,
        maximum_radius_m: 300,
        raw_prediction_id: "raw-1",
      },
      horizon_s: 60,
      sample_step_s: 30,
      centerline_xy: predictionStatus === "unavailable"
        ? [{ x: 60_000, y: 60_000 }, { x: 62_000, y: 60_000 }]
        : [{ x: -1_000, y: 0 }, { x: 1_000, y: 0 }],
      radius_m: [200, 200],
      point_confidence: [0.9, 0.8],
    },
    quality: {
      quality_score: 0.9,
      estimated_rmse_m: 20,
      fim_min_eigenvalue: 1,
      fim_condition: 1,
    },
    classification: "submarine",
    last_ping_s: 120,
    detection_range_m: 2_000,
  };
}

function execution(targetId = "T1"): ExecutionView {
  const regions: ExecutionRegionView[] = Array.from({ length: 4 }, (_, index) => ({
    region_id: `T1:task:${String(index + 1).padStart(2, "0")}`,
    target_id: targetId,
    slot_index: index + 1,
    execution_revision: 7,
    prediction_id: "prediction-1",
    geometry: [
      { x: -1_500 + index * 900, y: -500 },
      { x: -800 + index * 900, y: -500 },
      { x: -800 + index * 900, y: 500 },
      { x: -1_500 + index * 900, y: 500 },
    ],
    start_s: index * 100,
    end_s: (index + 1) * 100,
    geometry_revision: 7,
    predecessor_region_id: null,
    successor_region_id: null,
    handoff_start_s: null,
    handoff_end_s: null,
    status: index === 0 ? "active" : "prepositioning",
    task_group_id: `group-${index}`,
    evidence_ids: [],
  }));
  const taskGroups: TaskGroupView[] = regions.map((region, index) => ({
    task_group_id: region.task_group_id,
    target_id: targetId,
    region_id: region.region_id,
    execution_revision: 7,
    member_uuv_ids: [`uuv-${index * 2}`, `uuv-${index * 2 + 1}`],
    active_verifier_uuv_id: `uuv-${index * 2}`,
    passive_tracker_uuv_id: `uuv-${index * 2 + 1}`,
    status: index === 0 ? "active" : "prepositioning",
    evidence_ids: [],
  }));
  return {
    target_id: targetId,
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
    evidence_ids: [],
    regions,
    task_groups: taskGroups,
    reserve_uuv_ids: [],
    degraded: false,
    degradation_reasons: [],
    active_plan_preserved: false,
  };
}

function extremeLiveFrame(
  predictionStatus: "valid" | "unavailable" = "valid",
): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: 1,
    sim_time_s: 100,
    plan_version: 1,
    map_bounds: mapBounds,
    uuvs: Array.from({ length: 8 }, (_, index) =>
      uuv(`uuv-${index}`, { x: -1_800 + index * 500, y: 1_500 }),
    ),
    target_estimates: [targetEstimate(predictionStatus)],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
    execution: execution(),
  };
}

function contains(bounds: typeof mapBounds, point: Point2D): boolean {
  return point.x >= bounds.min_x && point.x <= bounds.max_x
    && point.y >= bounds.min_y && point.y <= bounds.max_y;
}

describe("semantic camera", () => {
  it.each([
    { width: 1600, height: 1000 },
    { width: 390, height: 844 },
  ])("keeps semantic bounds inside the map at $width x $height", (viewport: CameraViewport) => {
    const camera = semanticCameraForFrame(extremeLiveFrame(), viewport);
    semanticCameraCandidates(extremeLiveFrame()).forEach((candidate) => {
      expect(contains(camera.worldBounds, candidate)).toBe(true);
    });
    expect(contains(mapBounds, { x: camera.worldBounds.min_x, y: camera.worldBounds.min_y })).toBe(true);
    expect(contains(mapBounds, { x: camera.worldBounds.max_x, y: camera.worldBounds.max_y })).toBe(true);
    expect(camera.targetDetectionDiameterPx).toBeGreaterThanOrEqual(160);
    expect(camera.minimumRegionDimensionPx).toBeGreaterThanOrEqual(48);
    expect(camera.twoKilometerSegmentPx).toBeGreaterThanOrEqual(120);
  });

  it("ignores unavailable prediction geometry when fitting", () => {
    const frame = extremeLiveFrame("unavailable");
    expect(semanticCameraCandidates(frame)).not.toContainEqual({ x: 60_000, y: 60_000 });
  });

  it("keeps the large detection boundary out of the default camera until requested", () => {
    const frame = extremeLiveFrame();
    const target = frame.target_estimates[0];
    expect(semanticCameraCandidates(frame)).not.toContainEqual({
      x: target.mean.x + MAP_DISPLAY_CONFIG.targetDetectionRadiusM,
      y: target.mean.y,
    });
    expect(semanticCameraCandidates(frame, true)).toContainEqual({
      x: target.mean.x + MAP_DISPLAY_CONFIG.targetDetectionRadiusM,
      y: target.mean.y,
    });
  });

  it("fits the same published square corners that the region overlay renders", () => {
    const frame = extremeLiveFrame();
    const region = frame.execution?.regions[0];
    if (!region) throw new Error("missing execution region fixture");
    region.top_left_xy = { x: -5_000, y: 5_000 };
    region.bottom_right_xy = { x: -3_000, y: 3_000 };

    const candidates = semanticCameraCandidates(frame);
    [
      { x: -5_000, y: 3_000 },
      { x: -3_000, y: 3_000 },
      { x: -3_000, y: 5_000 },
      { x: -5_000, y: 5_000 },
    ].forEach((point) => expect(candidates).toContainEqual(point));
  });

  it("fits the IMM confidence band and the independent B-spline line", () => {
    const frame = extremeLiveFrame();
    const prediction = frame.target_estimates[0].prediction;
    if (!prediction) throw new Error("missing prediction fixture");
    prediction.imm_centerline_xy = [{ x: -1_000, y: 0 }, { x: 1_000, y: 0 }];
    prediction.imm_radius_m = [200, 200];
    prediction.bspline_centerline_xy = [{ x: -900, y: 300 }, { x: 1_200, y: 800 }];

    const candidates = semanticCameraCandidates(frame);
    expect(candidates).toContainEqual({ x: -1_000, y: -200 });
    expect(candidates).toContainEqual({ x: -1_000, y: 200 });
    expect(candidates).toContainEqual({ x: 1_200, y: 800 });
  });

  it("does not fall back to a non-current target when execution target is missing", () => {
    const frame = extremeLiveFrame();
    const target = frame.target_estimates[0];
    target.mean = { x: 7_500, y: 6_500 };
    target.prediction = targetEstimate("valid").prediction;
    if (!target.prediction) throw new Error("missing prediction fixture");
    target.prediction.centerline_xy = [
      { x: 7_600, y: 6_500 },
      { x: 7_800, y: 6_500 },
    ];
    target.detection_range_m = 500;
    frame.execution = execution("missing-target");
    const candidates = semanticCameraCandidates(frame);
    [
      { x: 7_500, y: 6_500 },
      { x: 7_600, y: 6_500 },
      { x: 7_800, y: 6_500 },
      { x: 7_000, y: 6_500 },
      { x: 8_000, y: 6_500 },
      { x: 7_500, y: 6_000 },
      { x: 7_500, y: 7_000 },
    ].forEach((point) => expect(candidates).not.toContainEqual(point));
    expect(candidates).toContainEqual({ x: -1_650, y: -500 });
    expect(candidates).toContainEqual({ x: -1_800, y: 1_500 });
  });

  it("excludes non-current target prediction and detection geometry", () => {
    const frame = extremeLiveFrame();
    frame.target_estimates = [
      targetEstimate("valid", "T1"),
      targetEstimate("valid", "T2"),
    ];
    frame.target_estimates[1].mean = { x: 60_000, y: 60_000 };
    frame.target_estimates[1].detection_range_m = 60_000;
    const candidates = semanticCameraCandidates(frame);
    expect(candidates).not.toContainEqual({ x: 10_000, y: 8_000 });
  });

  it("keeps an impossible wide candidate span and reports readability limits", () => {
    const frame = extremeLiveFrame();
    frame.target_estimates[0].prediction = targetEstimate("valid").prediction;
    if (!frame.target_estimates[0].prediction) throw new Error("missing prediction fixture");
    frame.target_estimates[0].prediction.centerline_xy = [
      { x: 60_000, y: 60_000 },
      { x: 62_000, y: 60_000 },
    ];
    const camera = semanticCameraForFrame(frame, { width: 390, height: 844 });
    semanticCameraCandidates(frame).forEach((candidate) => {
      expect(contains(camera.worldBounds, candidate)).toBe(true);
    });
    expect(camera.readabilityWarnings).toContain(
      "semantic candidate span prevents all minimum readability constraints",
    );
  });

  it("keeps every label candidate in deterministic priority order", () => {
    const placements = stableLabelPlacements([
      { id: "remaining", anchor: { x: 0, y: 0 }, width: 1_000, height: 40, priority: 5 },
      { id: "selected", anchor: { x: 0, y: 0 }, width: 1_000, height: 40, priority: 0 },
      { id: "active", anchor: { x: 0, y: 0 }, width: 1_000, height: 40, priority: 4 },
    ], { width: 2_000, height: 200 });
    expect(placements.map((placement) => placement.id)).toEqual([
      "selected",
      "active",
      "remaining",
    ]);
    expect(placements[0].suppressed).toBe(false);
    expect(placements[1].suppressed).toBe(true);
    expect(placements[2].suppressed).toBe(true);
  });

  it("keeps visible label rectangles inside the viewport", () => {
    const [placement] = stableLabelPlacements([
      { id: "edge", anchor: { x: 198, y: 2 }, width: 40, height: 14, priority: 0 },
    ], { width: 200, height: 100 });
    expect(placement.suppressed).toBe(false);
    expect(placement.x).toBeGreaterThanOrEqual(0);
    expect(placement.y).toBeGreaterThanOrEqual(0);
    expect(placement.x + placement.width).toBeLessThanOrEqual(200);
    expect(placement.y + placement.height).toBeLessThanOrEqual(100);
  });
});
