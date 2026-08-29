/** Fast synthetic design regression; the strict live acceptance is separate. */
import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const regionStarts = [
  [-8_000, -5_000],
  [-5_000, -4_500],
  [-2_000, -3_000],
  [1_000, -1_500],
] as const;

const predictionPoints = [
  { x: -7_500, y: -3_500 },
  { x: -5_200, y: -3_200 },
  { x: -2_800, y: -2_300 },
  { x: -500, y: -1_100 },
  { x: 1_800, y: 100 },
  { x: 4_400, y: 1_500 },
];

const uuvPositions = [
  [-7_000, -3_900], [-6_100, -2_700],
  [-4_300, -3_100], [-3_200, -1_900],
  [-1_200, -1_700], [-100, -600],
  [1_900, 100], [3_200, 1_000],
] as const;

const passiveSensorHeading = (x: number, y: number, regionIndex: number) => {
  const aimPoint = predictionPoints[Math.min(regionIndex + 1, predictionPoints.length - 1)];
  return Math.atan2(aimPoint.y - y, aimPoint.x - x);
};

const regions = regionStarts.map(([x, y], index) => ({
  region_id: `target_00:task:${String(index + 1).padStart(2, "0")}`,
  display_name: `region_${index + 1}`,
  target_id: "target_00",
  geometry: [
    { x, y },
    { x: x + 4_000, y },
    { x: x + 4_000, y: y + 4_000 },
    { x, y: y + 4_000 },
  ],
  start_time_s: index * 450,
  end_time_s: (index + 1) * 450 + 120,
  predecessor_region_ids: index === 0 ? [] : [`target_00:task:${String(index).padStart(2, "0")}`],
  successor_region_ids: index === 3 ? [] : [`target_00:task:${String(index + 2).padStart(2, "0")}`],
  assigned_uuv_ids: index === 0
    ? ["uuv_08", "uuv_01"]
    : [`uuv_${String(index * 2).padStart(2, "0")}`, `uuv_${String(index * 2 + 1).padStart(2, "0")}`],
  tracking_mode: "heuristic_uuv",
  uuv_roles: ["active_verifier", "passive_tracker"],
  group_id: `group_${index + 1}`,
  status: index === 0 ? "active" : "planned",
  effect: {
    status: index === 0 ? "active" : index === 1 ? "handoff_ready" : "planned",
    coverage_ratio: index === 0 ? 0.82 : 0.35,
    quality_score: index === 0 ? 0.79 : 0.5,
    handoff_progress: index === 1 ? 0.58 : 0,
    quality_source: "region_telemetry",
    hard_guard_reasons: [],
    expert_feedback_ids: [],
  },
}));

const frame = {
  schema_version: "1.0",
  scenario_id: "uuv-only-single-target",
  frame_id: 120,
  sim_time_s: 600,
  physics_step_s: 5,
  plan_version: 8,
  uuv_only: true,
  map_bounds: { min_x: -13_000, min_y: -9_000, max_x: 7_000, max_y: 7_000 },
  uuvs: [...uuvPositions.map(([x, y], index) => ({
    uuv_id: `uuv_${String(index).padStart(2, "0")}`,
    status: index === 0 ? "unavailable" : "track",
    deployment_state: index === 0 ? "returning" : "deployed",
    physically_exposed: true,
    position: { x, y },
    heading_rad: -0.25,
    ...(index % 2 === 1
      ? { sensor_heading_rad: passiveSensorHeading(x, y, Math.floor(index / 2)) }
      : {}),
    speed_mps: 2.2,
    energy_fraction: index === 0 ? 0.12 : 0.92 - index * 0.02,
    display_opacity: index === 0 ? 0.32 : 1,
    group_id: index === 0 ? null : `group_${Math.floor(index / 2) + 1}`,
    current_waypoint: predictionPoints[Math.min(Math.floor(index / 2) + 1, predictionPoints.length - 1)],
    breadcrumb: [{ x: x - 180, y: y - 80 }, { x, y }],
    sensor_mode: index % 2 === 0 ? "active" : "passive",
    reserved: false,
    active_range_m: 3_500,
    passive_range_m: 4_500,
  })), {
    uuv_id: "uuv_08",
    status: "track",
    deployment_state: "deployed",
    physically_exposed: true,
    position: { x: -7_950, y: -3_050 },
    heading_rad: 0.18,
    speed_mps: 2.0,
    energy_fraction: 1,
    display_opacity: 0.46,
    group_id: "group_1",
    current_waypoint: { x: -6_650, y: -3_250 },
    breadcrumb: [{ x: -8_000, y: -3_100 }, { x: -7_950, y: -3_050 }],
    sensor_mode: "active",
    reserved: false,
    active_range_m: 3_500,
    passive_range_m: 4_500,
  }],
  target_estimates: [{
    target_id: "target_00",
    mean: predictionPoints[0],
    heading_rad: -0.15,
    covariance_ellipse: { semimajor_m: 260, semiminor_m: 110, rotation_rad: -0.15 },
    intent: { label: "transit", confidence: 0.86, alternatives: { evade: 0.14 } },
    prediction: {
      horizon_s: 1_800,
      sample_step_s: 360,
      centerline_xy: predictionPoints,
      radius_m: [260, 340, 420, 510, 620, 760],
      point_confidence: [0.86, 0.72, 0.58, 0.44, 0.30, 0.18],
    },
    quality: { quality_score: 0.79, estimated_rmse_m: 145, fim_min_eigenvalue: 0.18, fim_condition: 6.2 },
    classification: "submarine",
    last_ping_s: 570,
    detection_range_m: 5_000,
  }],
  bearing_rays: [],
  groups: regions.map((region, index) => ({
    group_id: `group_${index + 1}`,
    target_id: "target_00",
    member_ids: region.assigned_uuv_ids,
    quality: { instant: 0.75, window_mean: 0.72, ewma: 0.73, components: { fim: 0.74 }, hard_guard_reasons: [] },
  })),
  regional_plans: {
    target_00: {
      target_id: "target_00",
      prediction_id: "IMM:target_00:600",
      revision: 8,
      cell_size_m: 1_000,
      current_handoff_region_id: regions[0].region_id,
      next_handoff_region_id: regions[1].region_id,
      regions,
    },
  },
  execution: {
    target_id: "target_00",
    execution_revision: 8,
    source_snapshot_revision: 120,
    prediction_revision: 8,
    intent_revision: 8,
    data_age_s: 0,
    data_status: "current",
    plan_source: "deterministic",
    current_region_id: regions[0].region_id,
    next_region_id: regions[1].region_id,
    evidence_ids: ["IMM:target_00:600"],
    regions: regions.map((region, index) => ({
      region_id: region.region_id,
      target_id: region.target_id,
      slot_index: index + 1,
      execution_revision: 8,
      prediction_id: "IMM:target_00:600",
      geometry: region.geometry,
      start_s: region.start_time_s,
      end_s: region.end_time_s,
      geometry_revision: 8,
      predecessor_region_id: region.predecessor_region_ids[0] ?? null,
      successor_region_id: region.successor_region_ids[0] ?? null,
      handoff_start_s: region.start_time_s + 360,
      handoff_end_s: region.end_time_s,
      status: index === 0 ? "active" : "planned",
      task_group_id: region.group_id,
      evidence_ids: ["IMM:target_00:600"],
    })),
    task_groups: regions.map((region, index) => ({
      task_group_id: region.group_id,
      target_id: region.target_id,
      region_id: region.region_id,
      execution_revision: 8,
      member_uuv_ids: region.assigned_uuv_ids,
      active_verifier_uuv_id: region.assigned_uuv_ids[0],
      passive_tracker_uuv_id: region.assigned_uuv_ids[1],
      status: index === 0 ? "active" : "prepositioning",
      evidence_ids: ["IMM:target_00:600"],
    })),
    reserve_uuv_ids: [],
    degraded: false,
    degradation_reasons: [],
    active_plan_preserved: false,
  },
  region_timeline: regions.map((region, index) => ({
    region_id: region.region_id,
    target_id: "target_00",
    center: { x: regionStarts[index][0] + 2_000, y: regionStarts[index][1] + 2_000 },
    bounds: {
      min_x: regionStarts[index][0],
      min_y: regionStarts[index][1],
      max_x: regionStarts[index][0] + 4_000,
      max_y: regionStarts[index][1] + 4_000,
    },
    start_offset_s: index * 450,
    end_offset_s: (index + 1) * 450 + 120,
    status: index === 0 ? "active" : index === 1 ? "handed_off" : "planned",
    coverage_mode: "required",
    priority: 1 - index * 0.1,
    occupancy_likelihood: 0.88 - index * 0.12,
    uuv_assignments: [],
    communication_links: [],
    handoff_from: index === 0 ? null : regions[index - 1].region_id,
    handoff_to: index === 3 ? null : regions[index + 1].region_id,
    evidence_ids: ["IMM:target_00:600"],
    degraded_reasons: [],
    plan_revision: 8,
  })),
  carriers: [],
  carrier: null,
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    class FakeWebSocket {
      static OPEN = 1;
      readyState = 0;
      onopen: (() => void) | null = null;
      onmessage: ((event: { data: string }) => void) | null = null;
      onclose: (() => void) | null = null;
      constructor(_url: string) {
        window.setTimeout(() => {
          this.readyState = FakeWebSocket.OPEN;
          this.onopen?.();
        }, 0);
      }
      send(_message: string) {}
      close() {
        this.readyState = 3;
        this.onclose?.();
      }
    }
    Object.defineProperty(window, "WebSocket", { configurable: true, value: FakeWebSocket });
  });
  await page.route("**/api/operational/snapshot", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(frame),
  }));
});

test("renders the synthetic overlapping four-region UUV handoff effect", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/");

  const map = page.locator(".canvas-area");
  await expect(map.locator(".region-map-overlay polygon")).toHaveCount(4);
  await expect(map.locator(".imm-confidence-band")).toHaveCount(1);
  await expect(map.locator(".imm-prediction-point")).toHaveCount(6);
  await expect(map.locator("canvas")).toHaveAttribute("data-waterborne-uuv-count", "8");
  await expect(map.locator("canvas")).toHaveAttribute("data-execution-uuv-count", "8");
  await expect(map.locator("canvas")).toHaveAttribute("data-task-group-count", "4");
  await expect(map.locator("canvas")).toHaveAttribute("data-carrier-count", "0");
  await map.screenshot({
    path: resolve(process.cwd(), "../../../outputs/uuv-four-state-tracking-effect.png"),
    animations: "disabled",
  });
});
