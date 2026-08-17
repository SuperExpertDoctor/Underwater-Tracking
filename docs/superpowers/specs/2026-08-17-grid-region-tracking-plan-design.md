# Grid-Region Tracking Plan Design

## 1. Purpose

The current plan represents a target track as time segments with one
intercept point and one target-level group. That representation cannot answer
the operational question: which square water areas must be covered, which
platform roles are required in each area, how sonar should be used, and how
coverage is handed off between adjacent areas.

This design makes the spatial task area the primary planning object. A target
prediction and its LLM-derived intent hypothesis produce an ordered set of
non-overlapping, axis-aligned square cells. Each cell has an explicit
coverage task, a time window, UUV/USV role requirements, sonar policy, and
communication requirements. Concrete platform IDs are selected only after
the area-level strategy has been produced.

## 2. Goals

- Derive a stable, deterministic square-cell representation from the
  estimated prediction corridor and the target intent hypothesis.
- Size the grid from the actual predicted task-area scale while keeping the
  number of cells manageable.
- Make region strategy, not target-level membership, the primary plan
  contract.
- Represent UUV and USV roles separately from their concrete IDs.
- Make passive sonar continuous and make active sonar an explicit,
  validated regional policy.
- Make USV relay responsibilities and carrier-radius constraints explicit.
- Preserve real LangGraph and real LLM decision ownership:
  mathematical prediction owns geometry, the LLM owns semantic regional
  policy, and deterministic optimization owns feasibility and concrete
  platform allocation.
- Persist region geometry, policies, allocations, evidence, and revisions for
  replay and human review.
- Keep the existing target-level and segment-level execution APIs working
  during migration through derived compatibility views.

## 3. Non-goals

- A hydrodynamic submarine model or a full ocean propagation simulator.
- A learned end-to-end allocation policy.
- Allowing the LLM to invent arbitrary coordinates, platform IDs, or
  infeasible communication paths.
- Removing the existing target-level aggregate quality metrics.

## 4. Domain Model

### 4.1 Grid specification

Add a configuration model for regionalization:

- grid origin and map coordinate convention: the existing global map frame;
- target_grid_cells: desired number of cells per target task area;
- min_cell_size_m and max_cell_size_m;
- cell_size_rounding_m;
- lateral_half_width_cells: default 2;
- max_uncertainty_margin_cells;
- require_uuv_per_region and require_usv_per_region;
- relay_overlap_policy.

The cell size is adaptive, not a universal hard-coded 1000 m value. A
continuous envelope area is estimated from the predicted centerline,
prediction corridor, and configured operational boundary. The initial cell
size is:

    sqrt(envelope_area / target_grid_cells)

It is clamped to the configured minimum and maximum and rounded to the
configured engineering step. The resulting grid is then rasterized. The
target cell count is a balance objective, not an exact cardinality, because
the fixed lateral band, map boundary, turns, and disconnected predicted
branches can change the final count.

### 4.2 Region cell

Add a strict RegionCell model with:

- region_id, target_id;
- integer grid_x and grid_y;
- axis-aligned bounds: min_x, max_x, min_y, max_y;
- center_xy and cell_size_m;
- first_entry_s and last_exit_s;
- one or more visit windows;
- occupancy or coverage likelihood;
- intent labels and prediction/evidence references;
- ordered predecessor and successor region IDs.

The region ID is deterministic, for example:

    T1:cell:<grid_x>:<grid_y>

The same grid cell cannot overlap another cell. A cell may have multiple
visit windows if the predicted path loops back through it; spatial identity
and temporal visits are separate fields.

### 4.3 Region task

Add a RegionTask model containing:

- region_id and target_id;
- active time window and visit window reference;
- regional priority and required quality;
- required UUV count and required USV count;
- UUV role requirements;
- USV role requirement;
- sonar policy;
- communication and relay requirements;
- handoff predecessor and successor;
- assigned UUV and USV IDs after allocation;
- assignment status: planned, active, handed_off, degraded, or uncovered;
- evidence IDs and plan revision.

The region task is the primary operational unit. The target remains the
mission object, but a target-level group is an aggregate over its regional
tasks.

### 4.4 Regional roles

UUV roles:

- passive_tracker: continuous passive observation and track maintenance;
- active_verifier: passive by default, active only when the regional policy
  authorizes an active probe;
- handoff_reserve: pre-positioning or replacement resource.

USV roles:

- surface_relay: maintains carrier-to-underwater connectivity;
- active_tracker: uses the USV active sonar capability for regional support;
- relay_and_tracker: performs both roles when communication and exposure
  constraints permit;
- handoff_reserve: moves toward the next regional task without claiming
  current coverage.

An assigned platform may have one primary role per time window. A USV may
relay for adjacent regions in overlapping windows only when it remains within
the carrier support radius and every required link is connected. It may not
simultaneously claim incompatible active-tracking tasks.

### 4.5 Compatibility views

Keep the existing Segment and SegmentPlan models during migration. A
compatibility segment is derived from an ordered RegionTask sequence:

- group_id becomes a derived regional group identifier;
- start_s and end_s come from the region task window;
- intercept_xy is the region center or the validated standoff point;
- segment_plan never becomes the source of truth.

Existing target-level member_ids_by_target and waypoints_by_member remain
available as aggregates until all execution and UI consumers use regional
fields.

## 5. Region Generation

### 5.1 Inputs

The regionalization node consumes:

- the estimated target history and current TargetBelief;
- PredictedTrackRef centerline, times, and uncertainty corridor;
- IntentHypothesis label, confidence, alternatives, planning effects, and
  evidence IDs;
- map bounds and GridSpec;
- operational scheme and valid intelligence summaries.

It never consumes simulator truth.

### 5.2 Rasterization algorithm

1. Build a continuous task envelope from the predicted centerline and
   uncertainty corridor. Clip it to the configured map bounds.
2. Compute and configure the adaptive square-cell side length.
3. Use a global axis-aligned grid. Map every prediction sample to its grid
   cell.
4. Estimate the local tangent from adjacent prediction samples. Compute the
   local normal. For each centerline cell/sample, add the cells reached by
   offsets -2, -1, 0, 1, and 2 cell widths along the local normal.
5. Add any grid cell intersected by the prediction corridor when it is inside
   the configured uncertainty margin. This can widen coverage but never
   removes the mandatory two-cell lateral band.
6. Deduplicate cells by grid coordinates.
7. Assign each cell its first and last predicted visit time, visit windows,
   occupancy score, confidence, and source evidence.
8. Sort cells by first entry time, then grid coordinates. Derive predecessor
   and successor relationships only between temporally adjacent cells.
9. Emit a TargetRegionPlan. Every spatial cell is disjoint; adjacent cells
   may share an edge but never an area.

If a prediction fallback is used, the plan records fallback_used and expands
the confidence corridor according to the predictor output. It does not
silently replace regional planning with target-level allocation.

### 5.3 Intent influence

The mathematical predictor remains responsible for the centerline. Intent
influences regional planning in two bounded ways:

- the intent label and alternatives affect regional priority and coverage
  urgency through the LLM-generated regional policy;
- the uncertainty and behavior evidence determine which candidate cells need
  mandatory coverage versus reserve coverage.

The LLM receives the generated region IDs and their structured geometry; it
does not create new coordinates. An unknown or low-confidence intent may
cause the LLM to retain more candidate cells as high-priority coverage, but
the validator still requires the output to reference only known regions.

## 6. LangGraph Flow

### 6.1 Strategic route

The carrier graph becomes:

    ingest
      -> event_monitor
      -> build_snapshot
      -> intent_analysis (real LLM)
      -> trajectory_prediction (deterministic mathematics)
      -> region_generation (deterministic mathematics)
      -> regional_strategy_generation (real LLM)
      -> verify_strategy
      -> resource_optimizer
      -> verify_plan
      -> commit_plan
      -> record_decision

Regional strategy generation returns exactly one policy for every generated
region. It may prioritize, require, or reserve a region, but may not emit
platform IDs or arbitrary coordinates. It must cite current region, intent,
prediction, estimator, and operational evidence.

### 6.2 Tactical route

Every observation cycle recomputes regional geometry and regional health.
When the current regional policy remains valid, the tactical branch reuses
the policy and optimizes only platform movement, sonar mode, and link
continuity. It does not invoke the carrier strategy LLM on every ordinary
cycle.

The full strategic route is required when:

- an intent change is confirmed;
- target detection is lost or reacquired;
- a high-priority region becomes unreachable;
- covariance or observability crosses a hard guard;
- a required USV relay leaves the carrier support radius;
- UUV endurance or rotation constraints invalidate coverage;
- an applied human directive changes regional requirements;
- valid intelligence or operational scheme changes.

### 6.3 Slave and adversary interaction

The slave context is extended with the current RegionTask, regional roles,
handoff windows, and connectivity requirements. The slave LLM chooses
immediate passive/active sonar actions and local handoff behavior within the
committed regional policy.

The adversary graph continues to receive target-owned observations and
threat summaries. Its output is applied to simulation, and the next regional
prediction incorporates the resulting estimated observations. The target
side remains separated from carrier regional truth.

## 7. Regional Allocation

### 7.1 Candidate filtering

For every RegionTask, the deterministic allocator filters candidate UUVs and
USVs using:

- platform kind and declared capability;
- time-to-region and bounded kinematics;
- remaining range and return reserve;
- energy and endurance;
- deployment state;
- carrier support radius for USVs;
- current and projected communication links;
- active sonar availability and cooldown;
- existing regional reservations and handoff conflicts.

Candidate scores favor coverage quality and continuity, then travel cost,
energy, resource churn, and relay stability. The exact scoring remains
deterministic and auditable.

### 7.2 Role assignment

The allocator fills role requirements in this order:

1. Preserve the current region's passive coverage.
2. Pre-position the next region's handoff reserve.
3. Establish the required carrier/USV/UUV communication path.
4. Assign active verification only when the LLM regional policy authorizes it
   and hard sensor constraints pass.
5. Rotate or return resources that violate endurance or reserve limits.

If requirements cannot be met, the plan is degraded with explicit uncovered
regions and reasons. It is not converted to a target-only plan.

### 7.3 Waypoints

Waypoints are derived from RegionTask geometry and platform standoff rules.
They are not generated by the LLM. Each platform receives a sequence that
may include:

- transit to the regional staging point;
- regional standoff/coverage points;
- handoff transit;
- return or recovery route.

The validator checks map bounds, motion limits, standoff, and inter-platform
separation.

## 8. Human Interaction and Persistence

Human feedback can reference region IDs and regional constraints, including:

- raise or lower regional priority;
- require covert passive-only coverage;
- authorize selective active verification in a region;
- require a USV relay in a region;
- return or release a specific platform from a regional task;
- replace regional role requirements.

The free-text directive is still parsed by the real LLM and explicitly
confirmed before application. Direct assignment controls gain a regional
scope while retaining the existing target-level compatibility path.

Persist in every plan revision:

- GridSpec and computed cell size;
- TargetRegionPlan and RegionTask geometry;
- intent and prediction references;
- regional LLM policy request/response hashes and evidence;
- selected platform roles and IDs;
- communication links;
- sensor mode decisions;
- degraded/uncovered reasons;
- triggering events and applied human directives.

Replay frames expose the same regional data. Timeline rows show the causal
factor on the left and the affected region/policy/allocation on the right.

## 9. UI

The tactical map adds:

- a global square-grid overlay clipped to the current task area;
- cell coloring for predicted probability, priority, current coverage,
  handoff, reserve, degraded, and uncovered;
- cell labels with region ID and time window;
- predicted centerline and uncertainty corridor;
- visible USV communication and target detection ranges;
- platform markers with heading and current regional role.

The regional operations panel shows one row per RegionTask:

- region geometry and time window;
- assigned UUV/USV IDs and roles;
- passive/active sonar policy and current mode;
- relay chain and connectivity status;
- arrival, handoff, and coverage health;
- LLM rationale and evidence IDs.

Clicking a region or policy suggestion scopes the human feedback to that
region. The submitted feedback remains the exact clicked text and enters
the real LLM directive preview path.

## 10. Validation and Failure Handling

Regional strategy validation rejects:

- unknown or duplicate region IDs;
- missing region policies;
- overlapping or out-of-bounds cells;
- invalid time windows or broken temporal ordering;
- arbitrary coordinates or platform IDs in the LLM policy;
- passive sonar disabled;
- active sonar outside policy or capability;
- USV outside carrier support radius;
- absent required communication path;
- platform double booking except an explicitly allowed relay overlap;
- missing required UUV/USV coverage without a degraded reason.

Transport failures use the existing retry/reconnect/pause behavior. Structured
LLM failures receive bounded correction requests. A second failure pauses
the decision loop; no deterministic strategy is substituted for a failed
real LLM decision.

## 11. Tests and Acceptance Criteria

Unit tests:

- adaptive cell-size calculation and clamping;
- axis-aligned cell ID stability;
- centerline and +/-2 normal-cell rasterization;
- corridor intersection and boundary clipping;
- deduplication and non-overlap;
- visit windows and temporal ordering;
- intent/prediction evidence propagation;
- exact one-policy-per-region LLM schema validation;
- regional role and sonar policy validation;
- USV radius and communication-path checks;
- platform conflict and relay-overlap rules.

Integration tests:

- real/fake structured LLM ports drive intent and regional strategy without
  rule replacement;
- a strategy revision changes RegionTask policies and concrete allocations;
- target maneuver or intent event causes a new regional plan;
- a disconnected relay produces an explicit degraded region;
- human regional feedback is persisted and changes the next plan revision;
- replay reconstructs region geometry, roles, sensor decisions, and handoff.

UI tests:

- grid cells and region status render from an operational frame;
- regional role and relay details are visible;
- clicking a region suggestion submits exact feedback text and region scope;
- current and next handoff cells update after a new frame;
- mobile and desktop layouts do not overlap.

Acceptance is reached when a single-target run can show, in sequence:

1. estimated centerline and intent hypothesis;
2. a non-overlapping ordered square-cell task area;
3. a regional LLM policy for every cell;
4. concrete UUV/USV role allocation with real links;
5. passive/active sonar decisions;
6. a handoff from one region to the next;
7. a plan revision caused by observation, intent, resource, link, or human
   feedback;
8. a complete replay of those changes.
