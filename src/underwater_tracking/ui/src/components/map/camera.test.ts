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
  type CameraViewport,
} from "./camera";

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

function targetEstimate(predictionStatus: "valid" | "unavailable"): TargetEstimateView {
  return {
    target_id: "T1",
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
      centerline_xy: [{ x: 60_000, y: 60_000 }, { x: 62_000, y: 60_000 }],
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

function execution(): ExecutionView {
  const regions: ExecutionRegionView[] = Array.from({ length: 4 }, (_, index) => ({
    region_id: `T1:task:${String(index + 1).padStart(2, "0")}`,
    target_id: "T1",
    slot_index: index + 1,
    execution_revision: 7,
    prediction_id: "prediction-1",
    geometry: [
      { x: -8_000 + index * 2_000, y: -1_000 },
      { x: -6_500 + index * 2_000, y: -1_000 },
      { x: -6_500 + index * 2_000, y: 1_000 },
      { x: -8_000 + index * 2_000, y: 1_000 },
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
    target_id: "T1",
    region_id: region.region_id,
    execution_revision: 7,
    member_uuv_ids: [`uuv-${index * 2}`, `uuv-${index * 2 + 1}`],
    active_verifier_uuv_id: `uuv-${index * 2}`,
    passive_tracker_uuv_id: `uuv-${index * 2 + 1}`,
    status: index === 0 ? "active" : "prepositioning",
    evidence_ids: [],
  }));
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
      uuv(`uuv-${index}`, { x: -7_500 + index * 2_000, y: 3_000 }),
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
});
