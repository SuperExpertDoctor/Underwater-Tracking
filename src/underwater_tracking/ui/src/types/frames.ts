/**
 * Authoritative browser mirror of domain.ui_models.  Operational frames carry
 * estimator-visible state only; evaluation truth has a separate build and
 * route and is intentionally absent from these types.
 */

export type Point2D = { x: number; y: number };

export type UUVStatus = "active" | "unavailable" | "track" | "scan";
export type CommunicationStatus =
  | "connected"
  | "degraded"
  | "disconnected"
  | "unknown"
  | "carrier"
  | "relay"
  | "mesh";
export type CarrierStatus = "standby" | "transit" | "deploying" | "recovering";
export type DeploymentState = "onboard" | "deployed" | "returning" | "failed";
export type EventLevel = "critical" | "strategic" | "tactical" | "informational";
export type IntentLabel =
  | "transit"
  | "patrol"
  | "loiter"
  | "evade"
  | "approach"
  | "withdraw"
  | "unknown";
export type IntentMotiveLabel =
  | "persistent_straight_transit"
  | "hard_turn_evasion"
  | "sprint_escape"
  | "weaving_evasion"
  | "speed_deception";
export type PlanStatus =
  | "draft"
  | "validating"
  | "active"
  | "superseded"
  | "completed"
  | "rejected"
  | "degraded";
export type Concept =
  "quality_first" | "balanced" | "resource_saving" | "hold_current";
/** Read-only operating phase highlighted in the command-center sidebar. */
export type OperationalStage =
  "task_execution" | "event_trigger" | "human_feedback" | "dynamic_adjustment";

export interface MapBounds {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
}

export interface CovarianceEllipse {
  semimajor_m: number;
  semiminor_m: number;
  rotation_rad: number;
}

export interface UUVView {
  uuv_id: string;
  status: UUVStatus;
  deployment_state: DeploymentState;
  physically_exposed: boolean;
  display_opacity?: number;
  position: Point2D;
  heading_rad: number;
  sensor_heading_rad?: number | null;
  speed_mps: number;
  energy_fraction: number;
  group_id: string | null;
  current_waypoint: Point2D | null;
  breadcrumb: Point2D[];
  sensor_mode: "active" | "passive";
  reserved: boolean;
  passive_range_m?: number | null;
  active_range_m?: number | null;
  active_capable?: boolean;
  is_group_leader?: boolean;
  master_connected?: boolean;
  connected_peer_ids?: string[];
  /** Lower-level control telemetry. Optional until the runtime publisher exposes it. */
  remaining_range_m?: number | null;
  endurance_remaining_m?: number | null;
  communication_status?: CommunicationStatus | null;
  link_state?: CommunicationStatus | null;
  tracked_target_id?: string | null;
  tracked_target?: string | null;
}

export interface UUVResourceView {
  uuv_id: string;
  carrier_id: string | null;
  mileage_m: number;
  energy_fraction: number;
  healthy: boolean;
  capability_active: boolean;
  deployment_state: string;
  resource_episode: number;
}

export interface CarrierView {
  carrier_id: string;
  /** Optional for legacy replay frames; new UUV-only frames always provide it. */
  role?: "carrier" | "mother_ship";
  position: Point2D;
  heading_rad: number;
  speed_mps: number;
  status: CarrierStatus;
  onboard_uuv_ids: string[];
  deployed_uuv_ids: string[];
  returning_uuv_ids: string[];
  support_radius_m?: number | null;
}

export interface CommunicationLinkView {
  source_id: string;
  target_id: string;
  medium: "surface" | "acoustic";
  distance_m: number;
  limit_m: number;
  status: "connected" | "disconnected";
  relay: boolean;
}

export interface RegionAssignmentView {
  platform_id: string;
  platform_kind: "uuv";
  role: string;
  start_offset_s: number;
  end_offset_s: number;
  sonar_mode: "passive" | "active";
}

export interface RegionTimelineView {
  region_id: string;
  target_id: string;
  center: Point2D;
  bounds: MapBounds;
  start_offset_s: number;
  end_offset_s: number;
  status: "planned" | "active" | "handed_off" | "degraded" | "uncovered";
  coverage_mode: "required" | "reserve" | "optional";
  priority: number;
  occupancy_likelihood: number;
  uuv_assignments: RegionAssignmentView[];
  communication_links: CommunicationLinkView[];
  handoff_from: string | null;
  handoff_to: string | null;
  evidence_ids: string[];
  degraded_reasons: string[];
  plan_revision: number;
  task_group_id?: string | null;
}

export interface BrainView {
  brain_id: string;
  role: "master" | "slave" | "adversary";
  status: BrainStatus;
  last_update_s: number | null;
  message: string;
  connected_platform_ids: string[];
  operation?: string | null;
  evidence_platform_ids?: string[];
}

export type BrainStatus =
  | "unconfigured"
  | "ready"
  | "running"
  | "succeeded"
  | "degraded"
  | "failed"
  | "online"
  | "paused"
  | "unknown";

export interface PlannedAssignmentView {
  target_id: string;
  region_id: string;
  uuv_ids: string[];
  carrier_id: string;
  plan_version: number;
  status: "planned" | "transporting" | "ready_to_deploy";
}

export interface ExecutionGroupView {
  group_id: string;
  target_id: string;
  region_id: string;
  member_ids: string[];
  mode: "active_scan" | "passive_track" | "returning";
}

export type ExecutionPlanSource =
  | "deterministic"
  | "llm_optimized"
  | "human_revised";

export type ExecutionRegionStatus =
  | "planned"
  | "prepositioning"
  | "active"
  | "passive"
  | "handoff_pending"
  | "handoff_completed"
  | "monitoring_complete"
  | "degraded"
  | "uncovered";

export interface ExecutionRegionView {
  region_id: string;
  target_id: string;
  slot_index: number;
  execution_revision: number;
  prediction_id: string;
  geometry: Point2D[];
  start_s: number;
  end_s: number;
  geometry_revision: number;
  predecessor_region_id: string | null;
  successor_region_id: string | null;
  handoff_start_s: number | null;
  handoff_end_s: number | null;
  status: ExecutionRegionStatus;
  task_group_id: string;
  evidence_ids: string[];
}

export type ExecutionTaskGroupStatus =
  | "prepositioning"
  | "active"
  | "handoff_pending"
  | "replacing"
  | "degraded"
  | "complete";

export interface TaskGroupView {
  task_group_id: string;
  target_id: string;
  region_id: string;
  execution_revision: number;
  member_uuv_ids: string[];
  active_verifier_uuv_id: string;
  passive_tracker_uuv_id: string;
  status: ExecutionTaskGroupStatus;
  evidence_ids: string[];
}

export interface ExecutionView {
  target_id: string;
  execution_revision: number;
  source_snapshot_revision: number;
  prediction_revision: number;
  intent_revision: number;
  data_age_s: number;
  data_status: "current" | "stale" | "unavailable";
  plan_source: ExecutionPlanSource;
  current_region_id: string;
  next_region_id: string;
  evidence_ids: string[];
  regions: ExecutionRegionView[];
  task_groups: TaskGroupView[];
  reserve_uuv_ids: string[];
  degraded: boolean;
  degradation_reasons: string[];
  active_plan_preserved: boolean;
}

export interface FrameConsistencyReport {
  valid: boolean;
  execution_revision?: number | null;
  source_snapshot_revision?: number | null;
  errors?: string[];
}

export interface AdversaryDecisionView {
  decision_id?: string;
  target_id: string;
  sim_time_s: number;
  intent: string;
  maneuver: string;
  segment?: string | null;
  confidence?: number | null;
  rationale: string;
  decision_summary?: string | null;
  trigger_event_ids?: string[];
  detected_platform_ids?: string[];
  active_ping_risk?: string | null;
  communications_discipline?: string | null;
  speed_mps?: number | null;
  heading_rad?: number | null;
  decoy_count?: number;
  decision_status?: string | null;
  escape_region_id?: string | null;
  decision_source?: "llm" | "mission_route" | "boundary_avoidance" | "safe_hold" | null;
  guidance_id?: string | null;
  guidance_waypoint_xy?: Point2D | null;
  guidance_speed_mps?: number | null;
  guidance_heading_rad?: number | null;
  guidance_valid_until_s?: number | null;
  degraded_reason?: string | null;
}

export interface AdversaryView {
  target_id: string;
  sim_time_s?: number;
  detection_range_m?: number | null;
  detected_platform_ids?: string[];
  trigger_event_ids?: string[];
  decision_id?: string | null;
  maneuver?: string | null;
  intent?: string | null;
  segment?: string | null;
  speed_mps?: number | null;
  heading_rad?: number | null;
  decoy_count?: number;
  confidence?: number | null;
  rationale?: string | null;
  communications_discipline?: string | null;
  decision_status?: string | null;
  escape_region_id?: string | null;
  decision_source?: "llm" | "mission_route" | "boundary_avoidance" | "safe_hold" | null;
  guidance_id?: string | null;
  guidance_waypoint_xy?: Point2D | null;
  guidance_speed_mps?: number | null;
  guidance_heading_rad?: number | null;
  guidance_valid_until_s?: number | null;
  degraded_reason?: string | null;
  current_decision?: AdversaryDecisionView | null;
  decision_history?: AdversaryDecisionView[];
}

export interface IntentView {
  label: IntentLabel;
  confidence: number;
  alternatives: Partial<Record<IntentLabel, number>>;
  ranked_motives?: Array<{ label: IntentMotiveLabel; probability: number; rationale: string }>;
}

export interface PredictionDiffView {
  diff_id: string;
  state:
    | "stable"
    | "accumulating"
    | "suspected"
    | "verifying"
    | "confirmed"
    | "reset"
    | "unavailable";
  status: string;
  reason: string | null;
  absolute_rms_m: number | null;
  normalized_rms: number | null;
  absolute_floor_m: number;
  normalized_threshold: number;
  consecutive_count: number;
  confirmation_cycles: number;
  previous_prediction_id: string | null;
  current_prediction_id: string;
  leading_model_changed: boolean;
  js_distance: number | null;
  suspicion_event_id: string | null;
  confirmed_intent: string | null;
  resulting_plan_revision: number | null;
}

export interface PredictionCorridorView {
  horizon_s: number;
  sample_step_s: number;
  centerline_xy: Point2D[];
  radius_m: number[];
  point_confidence?: number[];
  diff?: PredictionDiffView | null;
}

export type WorldModelHorizon = "H1" | "H2" | "H3" | "H4";

export interface WorldModelEvidenceView {
  key: string;
  source:
    | "imm"
    | "bspline"
    | "tracking_context"
    | "uuv_projection"
    | "map_bounds"
    | "observability";
  value: number;
  threshold: number | null;
  unit: string;
  description: string;
}

export interface WorldModelEventView {
  event_id: string;
  event_type: string;
  horizon: WorldModelHorizon;
  predicted_time_s: number;
  time_to_event_s: number;
  predicted_position: Point2D;
  confidence: number;
  level: EventLevel;
  rule_id: string;
  summary: string;
  evidence: WorldModelEvidenceView[];
}

export interface WorldModelHorizonView {
  name: WorldModelHorizon;
  start_offset_s: number;
  end_offset_s: number;
  sample_count: number;
  covered: boolean;
}

export interface WorldModelForecastView {
  model_kind: "rule_demo";
  model_version: string;
  control_authority: false;
  as_of_s: number;
  source_prediction_id: string;
  source_observation_ids: string[];
  source_observability_event_ids: string[];
  source_plan_revision: number | null;
  data_status: "ready" | "degraded";
  trajectory_fallback_used: boolean;
  imm_model_probabilities: Record<string, number>;
  horizons: WorldModelHorizonView[];
  events: WorldModelEventView[];
  warnings: string[];
}

export interface EstimateQualityView {
  quality_score: number;
  estimated_rmse_m: number;
  fim_min_eigenvalue: number;
  fim_condition: number;
}

export interface TargetEstimateView {
  target_id: string;
  mean: Point2D;
  covariance_ellipse: CovarianceEllipse;
  intent: IntentView;
  prediction: PredictionCorridorView | null;
  world_model?: WorldModelForecastView | null;
  quality: EstimateQualityView;
  classification: "submarine" | "decoy" | "unknown";
  last_ping_s: number | null;
  estimated_depth_m?: number | null;
  depth_uncertainty_m?: number | null;
  heading_rad?: number | null;
  detection_range_m?: number | null;
  detected_platform_ids?: string[];
  detected_platform_count?: number;
}

export interface BearingRayView {
  observation_id: string;
  uuv_id: string;
  target_id: string;
  origin: Point2D;
  azimuth_rad: number;
  variance_rad2: number;
  confidence: number;
}

export interface GroupQualityView {
  instant: number;
  window_mean: number;
  ewma: number;
  components: Record<string, number>;
  hard_guard_reasons: string[];
}

export interface GroupView {
  group_id: string;
  target_id: string;
  member_ids: string[];
  quality: GroupQualityView;
}

export type TrackingEffectStatus =
  "planned" | "active" | "handoff_ready" | "degraded" | "uncovered";

export interface TrackingEffectView {
  status: TrackingEffectStatus;
  coverage_ratio: number;
  quality_score: number;
  handoff_progress: number;
  quality_source: "group_quality_proxy" | "region_telemetry";
  hard_guard_reasons: string[];
  expert_feedback_ids: string[];
}

export interface RegionTaskView {
  region_id: string;
  display_name: string;
  target_id: string;
  geometry: Point2D[];
  grid_x?: number | null;
  grid_y?: number | null;
  start_time_s: number;
  end_time_s: number;
  visit_window_index?: number;
  visit_window?: { start_s: number; end_s: number } | null;
  predecessor_region_ids: string[];
  successor_region_ids: string[];
  assigned_uuv_ids: string[];
  tracking_mode: "heuristic_uuv";
  uuv_roles?: Array<"passive_tracker" | "active_verifier" | "handoff_reserve">;
  sonar_policy?: {
    passive_required: boolean;
    active_allowed: boolean;
    active_mode: "none" | "probe" | "continuous";
    active_cooldown_s: number;
  } | null;
  communication?: {
    carrier_to_uuv: boolean;
    acoustic_link_required: boolean;
  } | null;
  communication_links?: string[];
  group_id: string | null;
  status: string;
  revision?: number;
  effect: TrackingEffectView;
}

export interface RegionalPlanView {
  target_id: string;
  prediction_id: string;
  revision: number;
  cell_size_m: number;
  grid_spec?: {
    origin_xy: [number, number];
    map_coordinate_convention: string;
    target_grid_cells: number;
    min_cell_size_m: number;
    max_cell_size_m: number;
    cell_size_rounding_m: number;
    lateral_half_width_cells: number;
    max_uncertainty_margin_cells: number;
    require_uuv_per_region: boolean;
  } | null;
  evidence_ids?: string[];
  current_handoff_region_id?: string | null;
  next_handoff_region_id?: string | null;
  causal_event_ids?: string[];
  llm_hashes?: [string, string] | null;
  regions: RegionTaskView[];
}

export interface RegionalMissionView {
  region_id: string;
  target_id: string;
  cell_ids: string[];
  geometry: Point2D[];
  entry_s: number;
  exit_s: number;
  lifecycle: "PLANNED" | "CARRIER_DEPLOYING" | "ACTIVE_SCAN" | "PASSIVE_TRACK" | "HANDOFF_PENDING" | "TRACKING_COMPLETED" | "CARRIER_RECOVERY" | "RECOVERED" | "DEGRADED" | "UNCOVERED";
  active_scan_uuv_ids: string[];
  passive_track_uuv_ids: string[];
  reserve_uuv_ids: string[];
  coverage: number;
  tracking_quality: number;
  handoff_from: string | null;
  handoff_to: string | null;
  carrier_task_id: string | null;
  carrier_id: string | null;
  degraded_reasons: string[];
  plan_revision: number;
}

export interface EventView {
  event_id: string;
  sim_time_s: number;
  event_type: string;
  level: EventLevel;
  entity_id: string | null;
  message: string;
}

export interface OperationalSchemeView {
  scheme_id: string;
  version: number;
  valid_from_s: number;
  valid_until_s: number;
  target_priorities: Record<string, number>;
  minimum_quality: Record<string, number>;
  constraints: string[];
}

export type IntelligenceSource =
  "technical_reconnaissance" | "sigint" | "elint" | "humint" | "sonar";

export interface IntelligenceView {
  report_id: string;
  source: IntelligenceSource;
  target_id: string;
  confidence: number;
  issued_at_s: number;
  valid_until_s: number;
  content_summary: string | null;
}

export interface PlanView {
  plan_id: string;
  version: number;
  status: PlanStatus;
  concept: Concept;
  reason: string;
  affected_targets: string[];
  group_changes: string[];
  valid_from_s: number;
  valid_until_s: number | null;
  segment_plan: string[];
}

export interface LedgerView {
  decision_id: string;
  sim_time_s: number;
  outcome: "committed" | "degraded" | "rejected";
  trigger_event_ids: string[];
  evidence_ids: string[];
  final_plan_id: string | null;
  final_plan_version: number | null;
}

export interface TimelineFactorView {
  kind: "event" | "evidence" | "directive" | "knowledge";
  ref_id: string;
  label: string;
  detail: string;
}

export interface TimelinePlanView {
  plan_id: string;
  version: number;
  status: PlanStatus;
  summary: string;
  group_changes: string[];
}

export interface PlanTimelineView {
  adjustment_id: string;
  sim_time_s: number;
  factors: TimelineFactorView[];
  plan: TimelinePlanView | null;
}

export interface MetricView {
  metric_id: string;
  label: string;
  value: number;
  unit: string;
  threshold: number | null;
  window_s: number;
  series: number[];
  status?: string;
  mean_window?: number | null;
  worst_window?: number | null;
  trend_per_sec?: number | null;
  valid_fraction?: number | null;
  reason?: string;
}

export type SuggestionCategory =
  | "tracking_quality"
  | "segmented_handoff"
  | "resource_rotation"
  | "commander_preference";

export interface PlanAdjustmentSuggestionView {
  suggestion_id: string;
  category: SuggestionCategory;
  title: string;
  rationale: string;
  proposed_feedback: string;
  target_ids: string[];
  evidence_ids: string[];
  confidence: number;
}

export interface PlanningHealthView {
  status:
    | "idle"
    | "queued"
    | "running"
    | "committed"
    | "invalidated"
    | "rejected"
    | "failed"
    | "awaiting_retry"
    | "degraded";
  epoch_id?: string | null;
  base_physics_revision?: number | null;
  current_physics_revision?: number | null;
  latest_physics_revision?: number | null;
  base_sim_time_s?: number | null;
  current_sim_time_s?: number | null;
  latest_sim_time_s?: number | null;
  data_age_s?: number | null;
  deadline_utc_ms?: number | null;
  node?: string | null;
  attempt?: number;
  planning_epoch_invariant_failures?: number;
  queued_event_count?: number;
  last_result_status?: string | null;
  last_error?: string | null;
}

export interface OperationalFrame {
  schema_version: string;
  scenario_id?: string | null;
  frame_id: number;
  sim_time_s: number;
  physics_step_s?: number;
  plan_version: number;
  run_phase?:
    | "created"
    | "bootstrap_planning"
    | "awaiting_retry"
    | "running"
    | "completed"
    | "stopping"
    | "stopped"
    | "failed";
  planning_snapshot_revision?: number | null;
  planning_sim_time_s?: number | null;
  planning_data_age_s?: number | null;
  planning_data_status?: "current" | "stale" | "unavailable";
  planning?: PlanningHealthView | null;
  execution?: ExecutionView | null;
  execution_consistency?: FrameConsistencyReport | null;
  uuv_only?: boolean;
  map_bounds: MapBounds;
  uuvs: UUVView[];
  planned_assignments?: PlannedAssignmentView[];
  execution_groups?: ExecutionGroupView[];
  target_estimates: TargetEstimateView[];
  bearing_rays: BearingRayView[];
  groups: GroupView[];
  regional_plans?: Record<string, RegionalPlanView>;
  regional_missions?: RegionalMissionView[];
  events: EventView[];
  plans: PlanView[];
  ledger: LedgerView[];
  metrics: MetricView[];
  carrier: CarrierView | null;
  carriers?: CarrierView[];
  uuv_resources?: UUVResourceView[];
  communication_links?: CommunicationLinkView[];
  brains?: BrainView[];
  /** Current API publishes one operator-safe summary per target. */
  adversaries?: AdversaryView[];
  adversary?: AdversaryView | null;
  adversary_decision?: AdversaryDecisionView | null;
  adversary_history?: AdversaryDecisionView[];
  scheme?: OperationalSchemeView | null;
  intelligence?: IntelligenceView[];
  plan_timeline?: PlanTimelineView[];
  region_timeline?: RegionTimelineView[];
  plan_adjustment_suggestions?: PlanAdjustmentSuggestionView[];
  /** Backend-selected active flags for the read-only command-center status matrix. */
  operational_stage_flags?: OperationalStage[];
  /** One operator-facing LLM thinking paragraph for the current frame. */
  llm_thinking?: string | null;
  /** Backend-supplied factor that triggered the current LLM thinking update. */
  llm_thinking_trigger?: string | null;
  llm_thinking_epoch_id?: string | null;
  llm_thinking_source_event_ids?: string[];
}

export type StreamMessage =
  OperationalFrame | { type: "heartbeat"; sim_time_s: number | null };
