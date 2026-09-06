import type {
  ExecutionRegionView,
  ExecutionView,
  EventView,
  OperationalFrame,
  Point2D,
  PredictionCorridorView,
  ScanTelemetryView,
  TaskGroupInstanceView,
  TargetEstimateView,
  UUVView,
} from "../types/frames";

const MAP_BOUNDS = { min_x: -1_400, min_y: -1_100, max_x: 9_800, max_y: 3_100 } as const;
const TARGET_ID = "T1";
const TARGET_CENTER = { x: 3_900, y: 960 };
const REGION_SIZE = 2_000;
const REGION_ORIGIN_X = -500;
const REGION_ORIGIN_Y = -500;

type Lifecycle = TaskGroupInstanceView["lifecycle"];

const EVENT_CATALOG: EventView[] = [
  { event_id: "mock-target-found", sim_time_s: 100, event_type: "target_found", level: "strategic", entity_id: TARGET_ID, message: "目标 T1 已由多艇观测确认" },
  { event_id: "mock-entry-1", sim_time_s: 120, event_type: "region_entry_confirmed", level: "tactical", entity_id: "T1:task:01", message: "R01 入区确认 1/2，等待连续确认" },
  { event_id: "mock-entry-2", sim_time_s: 150, event_type: "region_entry_confirmed", level: "tactical", entity_id: "T1:task:01", message: "R01 入区确认 2/2，允许被动跟踪" },
  { event_id: "mock-passive-owner", sim_time_s: 150, event_type: "passive_track_started", level: "strategic", entity_id: "T1:task:01:deploy:000001", message: "TG-01 被动跟踪组成为当前负责组" },
  { event_id: "mock-handoff-pending", sim_time_s: 180, event_type: "handoff_ready", level: "tactical", entity_id: "T1:task:02:deploy:000001", message: "后继组已收到 2/3 个有效观测" },
  { event_id: "mock-handoff-complete", sim_time_s: 210, event_type: "handoff", level: "strategic", entity_id: "T1:task:02:deploy:000001", message: "交接证据 3/3，责任切换至 TG-02" },
  { event_id: "mock-replacement", sim_time_s: 210, event_type: "region_replacement", level: "tactical", entity_id: "T1:task:03", message: "R03 几何修订 v1→v2，替补组进入" },
  { event_id: "mock-dedicated", sim_time_s: 270, event_type: "dedicated_release_pending", level: "strategic", entity_id: TARGET_ID, message: "专用跟踪释放待定，等待剩余里程阈值" },
  { event_id: "mock-estimate-expired", sim_time_s: 320, event_type: "estimate_expired", level: "critical", entity_id: TARGET_ID, message: "目标估计超过 valid_until，等待刷新" },
];

const trackingPolicy: ExecutionView["tracking_policy"] = {
  region_count: 4,
  task_group_size: 3,
  task_region_side_m: REGION_SIZE,
  target_detection_radius_m: 650,
  uuv_active_detection_radius_m: 420,
  uuv_passive_detection_radius_m: 520,
  region_entry_probability_threshold: 0.7,
  region_transition_confirm_cycles: 2,
  max_uuv_mileage_m: 50_000,
  dedicated_release_remaining_mileage_m: 7_000,
};

function regionId(index: number): string {
  return `${TARGET_ID}:task:${String(index).padStart(2, "0")}`;
}

function groupId(index: number): string {
  return `${regionId(index)}:deploy:000001`;
}

function replacementGroupId(): string {
  return `${regionId(3)}:deploy:000002`;
}

function regionGeometry(index: number): Point2D[] {
  const column = (index - 1) % 2;
  const row = Math.floor((index - 1) / 2);
  const x = REGION_ORIGIN_X + column * REGION_SIZE * 2;
  const y = REGION_ORIGIN_Y + row * REGION_SIZE * 2;
  return [
    { x, y },
    { x: x + REGION_SIZE, y },
    { x: x + REGION_SIZE, y: y + REGION_SIZE },
    { x, y: y + REGION_SIZE },
  ];
}

function centerOf(points: Point2D[]): Point2D {
  return points.reduce((sum, point) => ({ x: sum.x + point.x / points.length, y: sum.y + point.y / points.length }), { x: 0, y: 0 });
}

function telemetry(
  routeProgress: number,
  activeCoverageRatio: number,
  sourceBackedPingCount: number,
  round: number,
  scanCompleted = false,
): ScanTelemetryView {
  return {
    route_progress: routeProgress,
    scan_round: round,
    active_coverage_ratio: activeCoverageRatio,
    source_backed_ping_count: sourceBackedPingCount,
    scan_completed: scanCompleted,
    scan_completion_threshold: 0.85,
    evidence_ids: [`mock-scan-round-${round}`],
  };
}

function prediction(simTime: number, revision: number): PredictionCorridorView {
  const centerline = [
    { x: TARGET_CENTER.x, y: TARGET_CENTER.y },
    { x: TARGET_CENTER.x + 600, y: TARGET_CENTER.y + 160 },
    { x: TARGET_CENTER.x + 1_250, y: TARGET_CENTER.y + 370 },
    { x: TARGET_CENTER.x + 1_900, y: TARGET_CENTER.y + 620 },
  ];
  return {
    prediction_id: `mock-imm:${revision}`,
    prediction_revision: revision,
    origin_sim_time_s: simTime,
    health: {
      status: "valid",
      regime: "imm",
      reason_codes: [],
      source_track_age_s: 0.4,
      clipped_point_fraction: 0,
      maximum_radius_m: 480,
      raw_prediction_id: `mock-raw:${revision}`,
    },
    horizon_s: 90,
    sample_step_s: 30,
    centerline_xy: centerline,
    radius_m: [180, 250, 340, 450],
    imm_centerline_xy: centerline,
    imm_radius_m: [180, 250, 340, 450],
    bspline_centerline_xy: [
      { x: TARGET_CENTER.x, y: TARGET_CENTER.y },
      { x: TARGET_CENTER.x + 580, y: TARGET_CENTER.y + 130 },
      { x: TARGET_CENTER.x + 1_300, y: TARGET_CENTER.y + 410 },
      { x: TARGET_CENTER.x + 1_900, y: TARGET_CENTER.y + 590 },
    ],
    point_confidence: [0.94, 0.84, 0.7, 0.58],
  };
}

function targetEstimate(simTime: number, revision: number, expired = false): TargetEstimateView {
  // An expired frame keeps the last published estimate at 270s; the current
  // simulation clock advances independently so the UI can show the age gap.
  const estimateTime = expired ? 270 : simTime;
  const mean = {
    x: TARGET_CENTER.x + Math.max(0, estimateTime - 100) * 2.4,
    y: TARGET_CENTER.y + Math.max(0, estimateTime - 100) * 0.55,
  };
  return {
    target_id: TARGET_ID,
    mean,
    covariance_ellipse: { semimajor_m: expired ? 420 : 160, semiminor_m: expired ? 250 : 90, rotation_rad: 0.22 },
    intent: { label: "transit", confidence: expired ? 0.38 : 0.86, alternatives: { evade: 0.14 } },
    prediction: prediction(estimateTime, revision),
    estimate_freshness: {
      status: expired ? "expired" : "live",
      estimate_time_s: expired ? 270 : simTime,
      valid_until_s: 300,
      data_age_s: expired ? simTime - 270 : 0.4,
      track_revision: revision,
      source_observation_ids: expired ? ["mock-obs-270"] : [`mock-obs-${simTime}`],
      reason: expired ? "valid_until 已过，估计等待刷新" : null,
    },
    quality: {
      quality_score: expired ? 0.31 : 0.88,
      estimated_rmse_m: expired ? 390 : 46,
      fim_min_eigenvalue: expired ? 0.09 : 1.8,
      fim_condition: expired ? 48 : 4.2,
    },
    classification: "submarine",
    last_ping_s: expired ? 270 : simTime - 1,
    estimated_depth_m: 175,
    depth_uncertainty_m: expired ? 60 : 18,
    heading_rad: 0.15,
    detection_range_m: trackingPolicy.target_detection_radius_m,
    detected_platform_ids: ["uuv-01", "uuv-02"],
    detected_platform_count: 2,
  };
}

function lifecycleFor(index: number, scenario: number): Lifecycle {
  if (index === 1) {
    if (scenario >= 5) return "disappeared";
    if (scenario === 4) return "exiting";
    if (scenario >= 2) return "passive_track";
  }
  if (index === 2 && scenario >= 6) return "dedicated_track";
  if (index === 2 && scenario >= 3) return "passive_track";
  if (index === 3) {
    if (scenario >= 5) return "disappeared";
    if (scenario === 4) return "exiting";
  }
  return index === 4 && scenario === 4 ? "entering" : "active_scan";
}

function groupMembers(index: number): [string, string, string] {
  const base = (index - 1) * 3 + 1;
  return [`uuv-${String(base).padStart(2, "0")}`, `uuv-${String(base + 1).padStart(2, "0")}`, `uuv-${String(base + 2).padStart(2, "0")}`];
}

function groupFor(index: number, scenario: number): TaskGroupInstanceView {
  const lifecycle = lifecycleFor(index, scenario);
  const members = groupMembers(index);
  const isOwner = index === 1 && scenario < 4 || index === 2 && scenario >= 4;
  const passive = lifecycle === "passive_track" || lifecycle === "dedicated_track" || lifecycle === "dedicated_release_pending" || lifecycle === "exiting";
  const deployed = lifecycle === "entering" ? members.slice(0, 1) : lifecycle === "disappeared" ? [] : [...members];
  return {
    group_instance_id: groupId(index),
    target_id: TARGET_ID,
    region_id: regionId(index),
    deployment_revision: 1,
    member_uuv_ids: members,
    lifecycle,
    sensor_mode: lifecycle === "disappeared" ? "off" : passive ? "passive" : "active",
    ownership_status: isOwner ? "owner" : index === 2 && scenario === 3 ? "successor" : "candidate",
    entry_boundary_point: regionGeometry(index)[0],
    exit_boundary_point: regionGeometry(index)[2],
    source_group_instance_id: index === 3 && scenario >= 4 ? groupId(3) : null,
    reason: lifecycle === "exiting" ? "handoff evidence published; retain until DISAPPEARED" : lifecycle === "disappeared" ? "runtime removed after exit" : "mock evidence-backed runtime group",
    evidence_ids: [`mock-group-${index}-${scenario}`],
    deployed_member_uuv_ids: deployed,
    passive_observation_uuv_ids: passive ? [...members] : [],
  };
}

function replacementGroup(scenario: number): TaskGroupInstanceView {
  const members: [string, string, string] = ["uuv-13", "uuv-14", "uuv-15"];
  return {
    group_instance_id: replacementGroupId(),
    target_id: TARGET_ID,
    region_id: regionId(3),
    deployment_revision: 2,
    member_uuv_ids: members,
    lifecycle: scenario === 4 ? "entering" : scenario === 5 ? "active_scan" : "dedicated_release_pending",
    sensor_mode: scenario >= 6 ? "passive" : "active",
    ownership_status: "candidate",
    entry_boundary_point: regionGeometry(3)[0],
    exit_boundary_point: regionGeometry(3)[2],
    source_group_instance_id: groupId(3),
    reason: "replacement incoming; old group remains until DISAPPEARED",
    evidence_ids: ["mock-replacement-incoming"],
    deployed_member_uuv_ids: scenario === 4 ? ["uuv-13"] : [...members],
    passive_observation_uuv_ids: [],
  };
}

function regionFor(index: number, scenario: number): ExecutionRegionView {
  let status: ExecutionRegionView["status"] = "planned";
  if (index === 1) status = scenario >= 2 && scenario <= 3 ? "passive" : scenario === 4 ? "handoff_completed" : scenario >= 5 ? "active" : "active";
  else if (index === 2) status = scenario >= 3 ? "passive" : "active";
  else if (index === 3) status = scenario === 4 ? "degraded" : scenario >= 5 ? "active" : "planned";
  else status = scenario >= 5 ? "active" : "planned";
  const route = index === 1 ? 1 : scenario >= 2 ? 0.7 : 0.35;
  const coverage = index === 1
    ? scenario === 0 ? 0.25 : scenario === 1 ? 0.32 : scenario >= 6 ? 0.9 : scenario >= 2 ? 0.46 : 0
    : index === 2 && scenario >= 3 ? 0.3 : 0;
  const pings = index === 1 ? (scenario === 0 ? 3 : scenario === 1 ? 4 : scenario >= 6 ? 18 : 6) : index === 2 && scenario >= 3 ? 2 : 0;
  const geometry = regionGeometry(index);
  return {
    region_id: regionId(index),
    target_id: TARGET_ID,
    // Backend execution slots are zero-based; the human-facing region ID is
    // still task:01…task:04.
    slot_index: index - 1,
    execution_revision: scenario + 1,
    prediction_id: `mock-imm:${scenario + 1}`,
    geometry,
    top_left_xy: geometry[0],
    bottom_right_xy: geometry[2],
    start_s: 100 + (index - 1) * 65,
    end_s: 165 + (index - 1) * 65,
    geometry_revision: index === 3 && scenario >= 4 ? 2 : 1,
    predecessor_region_id: index > 1 ? regionId(index - 1) : null,
    successor_region_id: index < 4 ? regionId(index + 1) : null,
    handoff_start_s: index < 4 ? 145 + (index - 1) * 65 : null,
    handoff_end_s: index < 4 ? 165 + (index - 1) * 65 : null,
    status,
    task_group_id: index === 3 && scenario >= 4 ? replacementGroupId() : groupId(index),
    evidence_ids: [`mock-region-${index}-${scenario}`],
    scan_telemetry: telemetry(route, coverage, pings, scenario >= 5 ? 2 : 1, scenario >= 6 && index === 1),
  };
}

function executionFor(scenario: number, simTime: number): ExecutionView {
  const regions = [1, 2, 3, 4].map((index) => regionFor(index, scenario));
  const groups = [1, 2, 3, 4].map((index) => groupFor(index, scenario));
  if (scenario >= 4) groups.push(replacementGroup(scenario));
  const owner = scenario >= 4 ? groupId(2) : scenario >= 2 ? groupId(1) : null;
  const handoff = scenario < 3
    ? null
    : {
        predecessor_group_id: groupId(1),
        successor_group_id: groupId(2),
        successor_region_id: regionId(2),
        observation_cycle_s: 12,
        required_uuv_ids: [...groupMembers(2)],
        valid_observation_uuv_ids: scenario === 3 ? ["uuv-04", "uuv-05"] : [...groupMembers(2)],
        status: scenario === 3 ? "pending" as const : "completed" as const,
        blocking_reasons: scenario === 3 ? ["仍缺少 uuv-06 的同周期有效观测"] : [],
        evidence_ids: [scenario === 3 ? "mock-handoff-2-of-3" : "mock-handoff-3-of-3"],
      };
  const entryCounts: Record<string, number> = {
    [regionId(1)]: scenario === 0 ? 0 : scenario === 1 ? 1 : 2,
    [regionId(2)]: scenario >= 3 ? 1 : 0,
    [regionId(3)]: 0,
    [regionId(4)]: 0,
  };
  const entryProbabilities: Record<string, number> = {
    [regionId(1)]: scenario === 0 ? 0.61 : scenario === 1 ? 0.78 : 0.86,
    [regionId(2)]: scenario >= 3 ? 0.74 : 0.48,
    [regionId(3)]: 0.22,
    [regionId(4)]: 0.18,
  };
  return {
    target_id: TARGET_ID,
    execution_revision: scenario + 1,
    source_snapshot_revision: 100 + scenario,
    prediction_revision: scenario + 1,
    intent_revision: scenario + 1,
    data_age_s: scenario === 7 ? 50 : 0.4,
    valid_from_s: 90,
    valid_until_s: 300,
    health_status: scenario === 7 ? "expired" : "current",
    health_reasons: scenario === 7 ? ["target_estimate_expired", "waiting_for_refresh"] : [],
    region_generation_mode: "imm",
    plan_source: "deterministic",
    current_region_id: regionId(1),
    next_region_id: regionId(2),
    evidence_ids: ["mock-execution-contract"],
    regions,
    task_groups: groups,
    tracking_policy: trackingPolicy,
    tracking_control: {
      mode: scenario >= 6 ? "dedicated" : "regional",
      tracking_owner_group_id: owner,
      pending_successor_group_id: scenario === 3 ? groupId(2) : scenario === 4 ? replacementGroupId() : null,
      dedicated_release_triggered_at_m: scenario >= 6 ? 6_850 : null,
      dedicated_release_reason: scenario >= 6 ? "remaining_mileage_below_threshold_pending_release" : null,
      source_event_ids: scenario >= 3 ? [scenario === 3 ? "mock-handoff-pending" : "mock-handoff-complete"] : [],
    },
    replacements: scenario >= 4 ? [{
      region_id: regionId(3),
      source_geometry_revision: 1,
      target_geometry_revision: 2,
      outgoing_group_id: groupId(3),
      incoming_group_id: replacementGroupId(),
      latest_pending_geometry_revision: scenario === 4 ? 2 : null,
      batch_id: `mock-batch-${scenario >= 6 ? 2 : 1}`,
    }] : [],
    region_entry_probabilities: entryProbabilities,
    entry_confirmation_counts: entryCounts,
    entry_confirmation_required_cycles: 2,
    entry_blocking_reasons: {
      [regionId(1)]: scenario === 0 ? ["entry_probability_below_threshold"] : [],
      [regionId(2)]: scenario === 3 ? ["entry_confirmation_interrupted"] : [],
    },
    region_entry_evidence: Object.fromEntries(regions.map((region) => [region.region_id, {
      probability: entryProbabilities[region.region_id],
      confirmation_count: entryCounts[region.region_id],
      required_cycles: 2,
      status: entryCounts[region.region_id] >= 2 ? "confirmed" as const : "pending" as const,
      blocking_reasons: scenario === 0 && region.region_id === regionId(1) ? ["entry_probability_below_threshold"] : [],
      evidence_ids: [`mock-entry-evidence-${scenario}`],
      source_track_revision: scenario + 1,
      source_observation_ids: [`mock-obs-${simTime}`],
    }])),
    handoff_evidence: handoff,
    blocking_reasons: scenario === 7 ? [
      { code: "stale_prediction", message: "目标估计已过期，等待新鲜观测", scope: TARGET_ID, evidence_ids: ["mock-estimate-expired"] },
      { code: "successor_not_deployed", message: "后继编组部署证据不足", scope: regionId(3), evidence_ids: ["mock-replacement-incoming"] },
    ] : scenario === 3 ? [{ code: "missing_effective_successor_observations", message: "后继组尚缺同周期有效观测", scope: TARGET_ID, evidence_ids: ["mock-handoff-2-of-3"] }] : [],
    degraded: scenario === 4 || scenario === 7,
    degradation_reasons: scenario === 4 ? ["replacement_geometry_pending"] : scenario === 7 ? ["target_estimate_expired"] : [],
    active_plan_preserved: scenario >= 4,
  };
}

function uuvFor(id: string, group: TaskGroupInstanceView, simTime: number): UUVView {
  const memberIndex = group.member_uuv_ids.indexOf(id);
  const regionNumber = Number(group.region_id.match(/(\d+)$/)?.[1] ?? 1);
  const center = centerOf(regionGeometry(regionNumber));
  const offset = memberIndex * 150 - 150;
  const physical = group.lifecycle !== "disappeared";
  const remainingRange = group.lifecycle === "dedicated_track"
    ? 6_850
    : group.lifecycle === "dedicated_release_pending"
      ? 6_950
      : 18_000 - memberIndex * 500;
  return {
    uuv_id: id,
    status: group.lifecycle === "passive_track" || group.lifecycle === "dedicated_track" || group.lifecycle === "dedicated_release_pending" ? "track" : group.lifecycle === "disappeared" ? "unavailable" : "scan",
    deployment_state: group.lifecycle === "disappeared" ? "returning" : "deployed",
    physically_exposed: physical,
    display_opacity: physical ? 1 : 0.28,
    position: { x: center.x + offset + Math.max(0, simTime - 100) * 0.7, y: center.y + offset * 0.25 },
    heading_rad: memberIndex * 0.18,
    sensor_heading_rad: memberIndex * 0.18,
    speed_mps: group.lifecycle === "exiting" ? 3.4 : 2.1,
    energy_fraction: group.lifecycle === "exiting" ? 0.41 : 0.78 - memberIndex * 0.04,
    group_id: group.region_id,
    group_instance_id: group.group_instance_id,
    deployment_revision: group.deployment_revision,
    group_lifecycle: group.lifecycle,
    current_waypoint: center,
    breadcrumb: [
      { x: center.x - 260, y: center.y - 80 },
      { x: center.x - 80, y: center.y - 20 },
      { x: center.x + offset, y: center.y + offset * 0.25 },
    ],
    sensor_mode: group.sensor_mode === "off" ? "passive" : group.sensor_mode,
    reserved: group.ownership_status === "owner",
    passive_range_m: trackingPolicy.uuv_passive_detection_radius_m,
    active_range_m: trackingPolicy.uuv_active_detection_radius_m,
    active_capable: true,
    is_group_leader: memberIndex === 0,
    master_connected: true,
    connected_peer_ids: [...group.member_uuv_ids].filter((candidate) => candidate !== id),
    remaining_range_m: remainingRange,
    endurance_remaining_m: remainingRange,
    communication_status: "connected",
    link_state: "connected",
    tracked_target_id: TARGET_ID,
    tracked_target: TARGET_ID,
  };
}

function scenarioFrame(scenario: number): OperationalFrame {
  const simTime = [100, 120, 150, 180, 210, 240, 270, 320][scenario] ?? 100;
  const execution = executionFor(scenario, simTime);
  const groups = execution.task_groups;
  const uuvs = groups.flatMap((group) => group.member_uuv_ids.map((id) => uuvFor(id, group, simTime)));
  const eventLimit = scenario === 0 ? 1 : scenario === 1 ? 2 : scenario === 2 ? 4 : scenario === 3 ? 5 : scenario === 4 ? 7 : scenario === 5 ? 7 : scenario === 6 ? 8 : 9;
  const events = EVENT_CATALOG.slice(0, eventLimit);
  const target = targetEstimate(simTime, scenario + 1, scenario === 7);
  return {
    schema_version: "mock-ui-1.0",
    scenario_id: "three-uuv-mock-seed-42",
    frame_id: scenario + 1,
    sim_time_s: simTime,
    physics_step_s: 1,
    plan_version: scenario >= 4 ? 2 : 1,
    run_phase: "running",
    planning_snapshot_revision: 100 + scenario,
    planning_sim_time_s: simTime,
    planning_data_age_s: scenario === 7 ? 40 : 0.3,
    planning_data_status: scenario === 7 ? "stale" : "current",
    planning: {
      status: scenario === 7 ? "degraded" : "committed",
      epoch_id: `mock-epoch-${scenario >= 4 ? 2 : 1}`,
      base_physics_revision: 100,
      current_physics_revision: 100 + scenario,
      latest_physics_revision: 100 + scenario,
      base_sim_time_s: 100,
      current_sim_time_s: simTime,
      latest_sim_time_s: simTime,
      data_age_s: scenario === 7 ? 40 : 0.3,
      node: "mock-planner",
      attempt: 1,
      queued_event_count: scenario >= 3 ? 1 : 0,
      last_result_status: scenario === 7 ? "degraded" : "committed",
      last_error: scenario === 7 ? "等待目标估计刷新" : null,
    },
    execution,
    map_bounds: MAP_BOUNDS,
    uuvs,
    planned_assignments: [],
    execution_groups: [],
    target_estimates: [target],
    bearing_rays: [],
    groups: [],
    regional_plans: {},
    regional_missions: [],
    events,
    plans: [],
    ledger: [],
    metrics: [
      { metric_id: "coverage", label: "source-backed active coverage", value: execution.regions.reduce((sum, region) => sum + (region.scan_telemetry?.active_coverage_ratio ?? 0), 0) / execution.regions.length, unit: "ratio", threshold: 0.85, window_s: 30, series: [], reason: "mock_scan_telemetry" },
      { metric_id: "ping_count", label: "source-backed ping count", value: execution.regions.reduce((sum, region) => sum + (region.scan_telemetry?.source_backed_ping_count ?? 0), 0), unit: "count", threshold: null, window_s: 30, series: [], reason: "mock_scan_telemetry" },
    ],
    carrier: {
      carrier_id: "carrier-01",
      role: "carrier",
      position: { x: -1_000, y: 2_600 },
      heading_rad: 0,
      speed_mps: 0,
      status: "standby",
      onboard_uuv_ids: [],
      deployed_uuv_ids: uuvs.filter((uuv) => uuv.physically_exposed).map((uuv) => uuv.uuv_id),
      returning_uuv_ids: uuvs.filter((uuv) => uuv.deployment_state === "returning").map((uuv) => uuv.uuv_id),
      support_radius_m: 1_200,
    },
    carriers: [],
    uuv_resources: uuvs.map((uuv) => ({
      uuv_id: uuv.uuv_id,
      carrier_id: "carrier-01",
      mileage_m: 1_500 + scenario * 480,
      energy_fraction: uuv.energy_fraction,
      healthy: uuv.status !== "unavailable",
      capability_active: true,
      deployment_state: uuv.deployment_state,
      resource_episode: uuv.deployment_revision ?? 1,
    })),
    communication_links: [],
    brains: [
      { brain_id: "master", role: "master", status: "running", last_update_s: simTime, message: "按当前 OperationalFrame 执行", connected_platform_ids: ["uuv-01", "uuv-04"] },
      { brain_id: "slave", role: "slave", status: "ready", last_update_s: simTime - 1, message: "区域扫描证据已同步", connected_platform_ids: ["uuv-07", "uuv-10"] },
    ],
    intelligence: [{ report_id: "mock-intel-1", source: "sonar", target_id: TARGET_ID, confidence: 0.82, issued_at_s: simTime - 8, valid_until_s: simTime + 45, content_summary: "目标保持东南向转移" }],
    operational_stage_flags: scenario >= 3 ? ["task_execution", "event_trigger", "dynamic_adjustment"] : ["task_execution", "event_trigger"],
    llm_thinking: scenario === 3 ? "入区已达到 2/2；后继组仍缺少 1 个同周期有效观测，暂不切换 owner。" : scenario === 4 ? "交接证据已满足 3/3；R03 几何修订进入替补部署阶段。" : scenario === 7 ? "目标估计已过期；保持物理执行状态并等待刷新，不把旧估计当作实时真值。" : "当前帧仅展示后端已发布的执行证据，扫描完成度与路线进度分开统计。",
    llm_thinking_trigger: scenario === 7 ? "estimate_expired" : scenario >= 3 ? "handoff_evidence" : "scan_telemetry",
    llm_thinking_epoch_id: `mock-epoch-${scenario >= 4 ? 2 : 1}`,
    llm_thinking_source_event_ids: events.slice(-2).map((event) => event.event_id),
  };
}

/** Deterministic seed-42 frames used only by the frontend mock mode. */
export const MOCK_FRAMES: OperationalFrame[] = Array.from({ length: 8 }, (_, index) => scenarioFrame(index));

export function mockFrameAt(index: number): OperationalFrame {
  return MOCK_FRAMES[Math.max(0, Math.min(MOCK_FRAMES.length - 1, Math.round(index)))] ?? MOCK_FRAMES[0];
}
