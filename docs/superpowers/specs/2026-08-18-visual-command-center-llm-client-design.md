# Visual Command Center And Unified LLM Client Design

## 1. Purpose

The current command center renders the whole operational search area at once.
The radar envelope therefore dominates the viewport, predicted regions become
small, and UUV/USV markers are hard to inspect. The right sidebar also exposes
several related controls as separate cards, while expert correction and
evidence questions travel through different interaction paths.

This design keeps the simulation and planning coordinates unchanged, but adds
a presentation layer that focuses the map on the single target's predicted
corridor, renders fine-grained subregions at readable sizes, and organizes the
right side as a small command workspace. Human feedback and evidence
questions share one LLM Client conversation; the backend classifies each turn
and preserves the existing preview/apply safety gate for plan changes.

## 2. Goals

- Make the target prediction corridor the default map focus.
- Keep the detection range hidden by default and show it only as an optional
  context layer.
- Support finer regional cells without shrinking platform markers below a
  readable pixel size.
- Make UUV/USV role rings and current responsibility visible without requiring
  a click.
- Show movement, coverage, handoff, and plan revision as a time-based tracking
  story rather than a static assignment picture.
- Consolidate expert feedback and evidence questions into one multi-turn LLM
  Client panel.
- Let the LLM classify a user turn as plan revision, evidence query, mixed, or
  clarification while deterministic backend gates prevent unsafe side effects.
- Preserve the two collaboration modes already accepted by the project:
  `uuv_primary_usv_relay`, `heuristic_uuv`, and `heuristic_usv`; no fixed team
  size is introduced by the UI.
- Reuse the useful organization patterns from `Maritime-Surveillance-main`:
  map-first layout, dedicated bottom timeline/drawer, compact status metrics,
  and stable square-preserving map geometry.

## 3. Non-goals

- Changing world coordinates, target motion, prediction mathematics, or LLM
  regional membership decisions solely for visual reasons.
- Showing the entire radar envelope as the primary map subject.
- Allowing an evidence question to mutate the active plan.
- Applying a plan correction directly from free text without preview and human
  confirmation.
- Replacing the existing directive/question persistence model in one step.

## 4. Presentation Scale Contract

The backend continues to emit metres, seconds, confidence, and region
geometry in their existing units. The UI introduces a separate display
configuration:

```ts
export interface ViewConfig {
  focusMode: "prediction_corridor" | "full_area";
  showDetectionRange: boolean;
  radarScale: number;
  predictionPadding: number;
  gridDivisions: number;
  markerPixels: { target: number; uuv: number; usv: number };
  playbackRate: number;
}
```

Initial defaults are `prediction_corridor`, `false`, `0.35`, `1.15`, `16`,
`{ target: 30, uuv: 24, usv: 28 }`, and `1`. These are display defaults, not
  planning constraints. Marker sizes are clamped in screen pixels so map zoom
  cannot make operational symbols disappear.

The camera fit algorithm uses the union of target mean, predicted centerline,
and visible regional geometry, then adds a 15% padding. It does not include
the detection circle unless `showDetectionRange` is true or the operator
chooses `full_area`. A small context indicator may show the full-area extent,
but it must not change the main camera fit.

## 5. Map And Tracking Visualization

The map has four independent layers:

1. Prediction corridor: centerline, uncertainty band, and target forecast.
2. Fine-grained subregions: square cells with confidence/coverage state.
3. Regional handoff: temporal arrows and responsibility edges.
4. Detection range: red dashed context circle, disabled by default.

The default grid is 16x16 for the local task viewport. At lower zoom levels,
labels collapse to region groups; at higher zoom levels, individual `R01`,
`R02`, ... labels appear. The data remains fine-grained even when the renderer
uses level-of-detail aggregation.

Platform rings are always drawn. UUV, USV relay, USV tracking, target, active
sonar, and degraded states use distinct colors and line styles. A selected
marker gets an additional focus ring, but selection is no longer required to
see responsibility or communication status.

The bottom playback bar is a simulation-time axis. It displays current time,
start/end seconds, event ticks, and the active handoff interval. It does not
use frame counts as the primary progress indicator.

Tracking effect is represented by frame-derived facts:

- active region and next handoff region;
- coverage ratio and quality score, with proxy/measured source;
- target deviation from the predicted corridor;
- handoff state and plan revision;
- degraded or uncovered reason;
- latency from target manoeuvre to blue-side regional response.

## 6. Right-Side Workspace

The right sidebar is a fixed-width command workspace on desktop and a drawer
on narrow screens. It has three primary collapsible sections:

### 6.1 Situation

Compact metrics for simulation time, plan revision, target status, coverage,
quality, active tracking platforms, and communications. Platform health is a
roster within this section rather than a separate top-level card.

### 6.2 Prediction And Handoff

This is the dominant card. It has three tabs: `graph`, `timeline`, and `list`.
The graph uses square region nodes, circular UUV/USV nodes, temporal directed
edges, and responsibility/relay edges. Selecting a region synchronizes the
map, timeline, and details; it does not change the plan.

### 6.3 LLM Client

One conversation surface replaces the separate expert feedback and situation
question cards. Each assistant turn may contain:

- classification badge: 方案修正, 证据质询, 混合处理, or 需要澄清;
- ordinary answer text;
- evidence chips that focus the timeline or event drawer;
- a plan-diff preview when a revision is proposed;
- an explicit `应用修正` action only when the backend says confirmation is
  required.

Carrier/brain/adversary details remain available in compact expandable rows or
  the bottom event drawer, so they do not compete with the primary regional
  task story.

## 7. Unified Conversation Contract

The UI sends a generic message instead of choosing a directive or question
mode:

```json
{
  "conversation_id": "scenario:operator",
  "text": "region_4 的交接质量为什么下降？下一窗口是否需要调整编组？",
  "target_id": "target_01",
  "selected_region_ids": ["region_4"],
  "expected_plan_version": 7,
  "sim_time_s": 120
}
```

The backend returns a structured turn:

```json
{
  "turn_id": "turn-...",
  "classification": "mixed",
  "answer": "...",
  "evidence_ids": ["event-...", "quality-..."],
  "proposal": { "status": "preview", "directive_id": "directive-..." },
  "needs_confirmation": true,
  "plan_version": 7
}
```

The classifier is an LLM call with a strict structured schema, but the
following rules are deterministic:

- `evidence_query` is read-only and cannot enqueue a strategic event.
- `plan_revision` must become a typed `ExpertDirective` preview and requires
  an explicit apply call before the next strategy cycle can use it.
- `mixed` returns evidence first and an independently reviewable plan proposal.
- `clarification` asks one bounded follow-up question and has no side effect.
- Evidence IDs must belong to the bounded planning snapshot or stored frame.
- Plan mutations carry `expected_plan_version`; stale versions return a
  conflict instead of silently applying to a new plan.
- Conversation messages, classification, evidence references, and applied
  directives remain auditable separately.

The first implementation may route the new endpoint internally to the current
`/api/directives` and `/api/questions` handlers. The old endpoints stay
available for compatibility and tests until the unified path is verified.

## 8. Acceptance Criteria

- On the default desktop viewport, the target prediction corridor and labels
  occupy the main visual focus; the red detection circle is absent until
  enabled.
- Fine-grained regions remain readable, and UUV/USV rings are visible before
  selection.
- The time axis shows simulation seconds and handoff/event markers.
- The right side has exactly one LLM Client conversation section; no separate
  expert-feedback and situation-question cards are presented.
- A user evidence question produces cited evidence without changing plan
  version.
- A plan correction produces a preview/diff and changes the plan only after
  explicit confirmation.
- A mixed turn renders both an evidence answer and a revision proposal in one
  conversation turn.
- Existing regional modes, LLM-selected membership, replay, and `main.py`
  startup continue to work.

