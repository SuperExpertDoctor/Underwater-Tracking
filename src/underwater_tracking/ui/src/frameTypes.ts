/**
 * Shell-level frame contract for the migrated command UI.
 *
 * These types mirror the plan's versioned frame contracts
 * (docs/superpowers/plans/2026-08-14-underwater-tracking-ui-plan.md,
 * Task 2: OperationalFrame, UUVView, TargetEstimateView, GroupView,
 * EventView, PlanView) using the UUV/target/belief/group/plan semantics
 * of this project.  Later tasks (Task 2/6) define the authoritative
 * contracts and the full frame type; this file is the migrated shell's
 * minimal standing-in typing.
 */

/** A [col, row] grid position, matching the reference data flow. */
export type Point2D = [number, number];

export type UUVState =
  | "idle"
  | "transit"
  | "searching"
  | "tracking"
  | "returning"
  | "refueling"
  | "holding";

/** One UUV as seen by the operator (estimator-visible state only). */
export interface UUVView {
  id: string;
  status: UUVState;
  position: Point2D;
  heading_deg: number;
  /** Normalized [0, 1] remaining energy budget. */
  energy_remaining_pct: number;
  /** Current group assignment, or null when unassigned. */
  group_id: string | null;
  /** Currently active plan id, or null when not executing a plan. */
  active_plan_id: string | null;
  time_to_available_min: number;
  /** Recent breadcrumb trail (grid positions). */
  breadcrumb: Point2D[];
}

/** Estimated target state; never contains hidden truth. */
export interface TargetEstimateView {
  id: string;
  group_id: string | null;
  /** Estimated mean position (grid coordinates). */
  mean: Point2D;
  /** Latest intent label, when the estimator has one. */
  intent: string | null;
  /** Tracking quality in [0, 1], when available. */
  quality: number | null;
  /** Predicted future corridor, when available. */
  prediction_corridor: Point2D[] | null;
}

/** A UUV group and its members. */
export interface GroupView {
  id: string;
  member_ids: string[];
  quality: number | null;
}

/** A discrete operator-relevant event. */
export interface EventView {
  time: number;
  type: string;
  data: Record<string, unknown>;
}

/** Latest plan-decision cycle (renamed from the reference's llm_cycle). */
export interface PlanCycleView {
  plan_version: number;
  model: string;
  success: boolean;
  attempts: number;
  response: string | null;
}

/** Versioned operational frame broadcast to the command UI. */
export interface OperationalFrame {
  schema_version: string;
  frame_id: number;
  /** Simulation time in seconds. */
  sim_time_s: number;
  plan_version: number;
  /** HH:MM:SS rendering of sim_time_s, as in the reference. */
  timestamp: string;
  total_steps: number;
  mode: "live" | "replay";
  coverage_pct: number | null;
  uuvs: UUVView[];
  targets: TargetEstimateView[];
  groups: GroupView[];
  events: EventView[];
  plan_cycle: PlanCycleView | null;
}
