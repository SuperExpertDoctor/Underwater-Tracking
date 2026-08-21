import type { OperationalFrame } from "./frames";
import { DEFAULT_VIEW_CONFIG, toPlanningPayload } from "./viewConfig";

test("uses local view defaults and removes display settings from planning payloads", () => {
  expect(DEFAULT_VIEW_CONFIG).toMatchObject({
    focusMode: "prediction_corridor",
    showDetectionRange: false,
    gridDivisions: 16,
    predictionPadding: 0.15,
  });

  expect(toPlanningPayload({
    target_id: "T1",
    uuv_ids: ["UUV-1"],
    expected_plan_version: 3,
    ...DEFAULT_VIEW_CONFIG,
  })).toEqual({ target_id: "T1", uuv_ids: ["UUV-1"], expected_plan_version: 3 });
});

test("accepts regional plan graph and tracking effect fields", () => {
  const frame: OperationalFrame = {
    schema_version: "1.0",
    frame_id: 1,
    sim_time_s: 100,
    plan_version: 3,
    map_bounds: { min_x: -100, min_y: -100, max_x: 100, max_y: 100 },
    carrier: null,
    uuvs: [],
    target_estimates: [],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    regional_plans: {
      T1: {
        target_id: "T1",
        prediction_id: "prediction-1",
        revision: 3,
        cell_size_m: 100,
        regions: [
          {
            region_id: "T1:cell:0:0",
            display_name: "region_1",
            target_id: "T1",
            geometry: [
              { x: 0, y: 0 },
              { x: 100, y: 0 },
              { x: 100, y: 100 },
              { x: 0, y: 100 },
            ],
            start_time_s: 100,
            end_time_s: 110,
            predecessor_region_ids: [],
            successor_region_ids: ["T1:cell:1:0"],
            assigned_uuv_ids: ["UUV-1"],
            tracking_mode: "heuristic_uuv",
            group_id: "G1",
            status: "active",
            effect: {
              status: "active",
              coverage_ratio: 1,
              quality_score: 0.9,
              handoff_progress: 0.5,
              quality_source: "group_quality_proxy",
              hard_guard_reasons: [],
              expert_feedback_ids: [],
            },
          },
        ],
      },
    },
  };

  expect(frame.regional_plans?.T1.regions[0].display_name).toBe("region_1");
});
