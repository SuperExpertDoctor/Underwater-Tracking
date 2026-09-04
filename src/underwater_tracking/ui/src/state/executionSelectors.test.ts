import { describe, expect, it } from "vitest";
import type {
  ExecutionView,
  OperationalFrame,
  TaskGroupInstanceView,
  UUVView,
} from "../types/frames";
import {
  executionCounts,
  groupForUuv,
  groupInstanceId,
  groupsByRegionSlot,
  ownerGroup,
  visibleExecutionUuvs,
} from "./executionSelectors";

function group(
  index: number,
  lifecycle: TaskGroupInstanceView["lifecycle"],
  sensorMode: TaskGroupInstanceView["sensor_mode"] = "passive",
): TaskGroupInstanceView {
  const groupInstanceId = `T1:task:${String((index % 4) + 1).padStart(2, "0")}:deploy:${String(index + 1).padStart(6, "0")}`;
  return {
    group_instance_id: groupInstanceId,
    target_id: "T1",
    region_id: `T1:task:${String((index % 4) + 1).padStart(2, "0")}`,
    deployment_revision: index + 1,
    member_uuv_ids: [
      `${groupInstanceId}:member:01`,
      `${groupInstanceId}:member:02`,
      `${groupInstanceId}:member:03`,
    ],
    lifecycle,
    sensor_mode: sensorMode,
    ownership_status: index === 0 ? "owner" : "candidate",
    reason: "test",
    evidence_ids: [`${groupInstanceId}:created`],
  };
}

function uuv(groupInstanceId: string, index: number): UUVView {
  return {
    uuv_id: `${groupInstanceId}:member:${String(index + 1).padStart(2, "0")}`,
    status: "active",
    deployment_state: "deployed",
    physically_exposed: true,
    position: { x: index, y: index },
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 1,
    group_id: groupInstanceId,
    group_instance_id: groupInstanceId,
    deployment_revision: 1,
    group_lifecycle: "active_scan",
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: "active",
    reserved: false,
  };
}

function frameWithGroups(
  groups: TaskGroupInstanceView[],
  uuvs: UUVView[],
): OperationalFrame {
  return {
    schema_version: "1.0",
    frame_id: 1,
    sim_time_s: 0,
    plan_version: 1,
    map_bounds: { min_x: -10_000, min_y: -10_000, max_x: 10_000, max_y: 10_000 },
    uuvs,
    target_estimates: [],
    bearing_rays: [],
    groups: [],
    events: [],
    plans: [],
    ledger: [],
    metrics: [],
    carrier: null,
    execution: {
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
      current_region_id: groups[0]?.region_id ?? "T1:task:01",
      next_region_id: "T1:task:02",
      evidence_ids: ["execution:test"],
      regions: [],
      task_groups: groups,
      reserve_uuv_ids: null,
      tracking_policy: {
        region_count: 4,
        task_group_size: 3,
        task_region_side_m: 2_000,
        target_detection_radius_m: 1_000,
        uuv_active_detection_radius_m: 600,
        uuv_passive_detection_radius_m: 600,
        region_entry_probability_threshold: 0.8,
        region_transition_confirm_cycles: 2,
        max_uuv_mileage_m: 10_000,
        dedicated_release_remaining_mileage_m: 700,
      },
      tracking_control: {
        mode: "regional",
        tracking_owner_group_id: groups[0]?.group_instance_id ?? null,
        pending_successor_group_id: null,
        source_event_ids: [],
      },
      replacements: [],
      degraded: false,
      degradation_reasons: [],
      active_plan_preserved: false,
    } as ExecutionView,
  };
}

describe("execution selectors", () => {
  it("keeps incoming and outgoing instances visible for the same region", () => {
    const groups = Array.from({ length: 8 }, (_, index) => group(
      index,
      index === 0 ? "exiting" : index === 1 ? "entering" : "active_scan",
      index === 0 ? "passive" : "active",
    ));
    const uuvs = groups.flatMap((item) => item.member_uuv_ids.map((_, index) => uuv(item.group_instance_id, index)));
    const frame = frameWithGroups(groups, uuvs);

    const visible = visibleExecutionUuvs(frame);

    expect(visible).toHaveLength(24);
    expect(new Set(visible.map((item) => item.group_instance_id)).size).toBe(8);
    expect(groupsByRegionSlot(frame).get(groups[0].region_id)).toHaveLength(2);
    expect(groupForUuv(frame, uuvs[0].uuv_id)
      ? groupInstanceId(groupForUuv(frame, uuvs[0].uuv_id)!)
      : null).toBe(groups[0].group_instance_id);
  });

  it("derives counts from lifecycle without assuming twelve", () => {
    const groups = [
      ...Array.from({ length: 3 }, (_, index) => group(index, "active_scan", "active")),
      group(3, "passive_track"),
      ...Array.from({ length: 4 }, (_, index) => group(index + 4, "entering", "active")),
      group(8, "exiting"),
    ];
    const uuvs = groups.slice(0, 5).flatMap((item) => item.member_uuv_ids.map((_, index) => uuv(item.group_instance_id, index)));
    const frame = frameWithGroups(groups, uuvs);

    expect(executionCounts(frame)).toEqual({
      visibleUuvs: 15,
      enteringGroups: 4,
      exitingGroups: 1,
      activeScanGroups: 3,
      passiveTrackGroups: 1,
    });
    expect(ownerGroup(frame)
      ? groupInstanceId(ownerGroup(frame)!)
      : null).toBe(groups[0].group_instance_id);
  });
});
