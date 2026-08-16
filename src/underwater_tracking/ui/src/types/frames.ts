/**
 * Authoritative browser mirror of domain.ui_models.  Operational frames carry
 * estimator-visible state only; evaluation truth has a separate build and
 * route and is intentionally absent from these types.
 */

export type Point2D = { x: number; y: number };

export type UUVStatus = "available" | "tracking" | "returning" | "failed";
export type CarrierStatus = "standby" | "transit" | "deploying" | "recovering";
export type DeploymentState = "onboard" | "deployed" | "returning" | "failed";
export type EventLevel = "strategic" | "tactical" | "informational";
export type IntentLabel =
  | "transit"
  | "patrol"
  | "loiter"
  | "evade"
  | "approach"
  | "withdraw"
  | "unknown";
export type PlanStatus =
  | "draft"
  | "validating"
  | "active"
  | "superseded"
  | "completed"
  | "rejected"
  | "degraded";
export type Concept = "quality_first" | "balanced" | "resource_saving" | "hold_current";

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
  deployment_state?: DeploymentState;
  position: Point2D;
  heading_rad: number;
  speed_mps: number;
  energy_fraction: number;
  group_id: string | null;
  current_waypoint: Point2D | null;
  breadcrumb: Point2D[];
  sensor_mode: "active" | "passive";
  reserved: boolean;
}

export interface CarrierView {
  carrier_id: string;
  position: Point2D;
  heading_rad: number;
  speed_mps: number;
  status: CarrierStatus;
  onboard_uuv_ids: string[];
  deployed_uuv_ids: string[];
  returning_uuv_ids: string[];
}

export interface IntentView {
  label: IntentLabel;
  confidence: number;
  alternatives: Partial<Record<IntentLabel, number>>;
}

export interface PredictionCorridorView {
  horizon_s: number;
  sample_step_s: number;
  centerline_xy: Point2D[];
  radius_m: number[];
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
  quality: EstimateQualityView;
  classification: "submarine" | "decoy" | "unknown";
  last_ping_s: number | null;
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

export interface EventView {
  event_id: string;
  sim_time_s: number;
  event_type: string;
  level: EventLevel;
  entity_id: string | null;
  message: string;
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

export interface MetricView {
  metric_id: string;
  label: string;
  value: number;
  unit: string;
  threshold: number | null;
  window_s: number;
  series: number[];
}

export interface OperationalFrame {
  schema_version: string;
  frame_id: number;
  sim_time_s: number;
  plan_version: number;
  map_bounds: MapBounds;
  uuvs: UUVView[];
  target_estimates: TargetEstimateView[];
  bearing_rays: BearingRayView[];
  groups: GroupView[];
  events: EventView[];
  plans: PlanView[];
  ledger: LedgerView[];
  metrics: MetricView[];
  carrier?: CarrierView | null;
}

export type StreamMessage = OperationalFrame | { type: "heartbeat"; sim_time_s: number | null };
