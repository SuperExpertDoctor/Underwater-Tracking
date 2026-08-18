# Segment Tracking Gantt Design

## Goal

Make the project-root `main.py` start the complete underwater-tracking
algorithm and command center, then add a compact segment-tracking window that
shows regional handoff logic for UUVs and USVs.

The visualization uses the existing React/Vite command center. It is not a
second desktop application. The selected layout is region swimlanes: one row
per predicted region, with platform assignments and relay handoffs rendered
inside each row.

## Confirmed Product Decisions

- Primary layout: region swimlanes.
- Time axis: relative to the current frame, with the current simulation time
  rendered as `T+0`.
- Live mode: the panel updates on every WebSocket operational frame.
- Replay mode: the panel follows the selected replay frame and playback bar.
- Zoom: application-level 80%-125% scaling in 10% steps, plus reset to 100%.
- Responsive behavior: desktop side-by-side command center, medium-width
  compressed panels with horizontal timeline scrolling, and narrow-screen
  drawers/compact cards.
- The panel displays operational estimates and plans only; it must not expose
  simulator target truth.

## Backend Contract

Add strict operator-safe view models beside the existing UI contracts:

- `RegionAssignmentView`: platform ID, platform kind, regional role, start and
  end offsets, sonar mode, and assignment status.
- `RegionTimelineView`: region ID, target ID, square geometry, relative start
  and end offsets, priority, occupancy likelihood, coverage mode, status,
  assigned UUV/USV views, relay links, predecessor/successor handoffs,
  evidence IDs, plan revision, and degraded reasons.

Add optional `region_timeline` to `OperationalFrame` and its TypeScript mirror.
The field remains optional so old persisted frames and clients deserialize
without migration. `build_operational_frame` derives entries from
`TrackingPlan.regional_plans` and `TrackingPlan.region_tasks`; it computes
relative offsets from the current frame `sim_time_s` and keeps task times
absolute only inside the server-side plan models.

The builder sorts by `(start_offset_s, grid_y, grid_x, region_id)` and sorts
platform IDs within each assignment. It preserves degraded and uncovered
regions. Missing regional data produces an empty tuple, never fabricated
assignments.

## Frontend Surface

Add a `SegmentTrackingPanel` component to the existing `BottomDrawer` under a
new `分段跟踪` tab. The panel contains:

- a relative time ruler beginning at `T+0`;
- a current-time marker;
- one row per region;
- compact bars for passive tracking, active verification, reserve, and USV
  relay coverage;
- handoff nodes and connectors between predecessor/successor regions;
- status colors for active, handed off, degraded, and uncovered regions;
- a click target that opens the selected region details in the same drawer;
- a details area showing time window, square bounds, platform roles, sonar,
  relay chain, evidence, and degradation reasons.

The component consumes only `OperationalFrame.region_timeline` and the current
frame time. It does not reconstruct regional policy from target/group fields.
The selected region is retained by ID across live frame updates and replay
seeks; if the region disappears, selection clears.

On narrow screens the timeline scrolls horizontally while region metadata
stays visible. Text must remain inside its row and buttons must have accessible
labels. Existing target/group tabs remain available as compatibility views.

## Application Zoom

Add zoom-in, zoom-out, and reset controls to the top toolbar using the existing
Lucide icon dependency. Store the scale in React state and `localStorage`.
Clamp it to 0.8 through 1.25 in 0.1 increments. Apply the scale to the full
command-center surface, including the map, sidebars, drawer, and segment
timeline. Keep the existing responsive breakpoints active so scaling does not
create clipped or overlapping controls.

## `main.py` Lifecycle

`main.py` must validate all required backend imports and frontend prerequisites
before starting Vite. The preflight must identify the missing package and the
exact interpreter/installation command in a concise error. Once Vite starts,
the backend import and serve call must be inside the same `try/finally` that
stops Vite, so an algorithm startup failure cannot leave an orphan frontend.

The normal command remains:

```text
python main.py
```

The implementation must preserve `--config`, `--steps`, `--seed`, `--host`,
and `--port`. The project-root entrypoint is the supported launch path; the
feature worktree is only an isolation mechanism for development.

## Data Flow

```text
regional TargetRegionPlan + RegionTask
        -> build_operational_frame
        -> OperationalFrame.region_timeline
        -> WebSocket / replay JSON
        -> SegmentTrackingPanel
```

Live and replay use the same frame schema. Relative offsets are recomputed for
each frame, so seeking does not require client-side plan mutation.

## Failure Handling

- Missing `langgraph` or other backend dependency: fail before Vite starts,
  with a clear diagnostic and non-zero exit.
- Backend failure after Vite starts: stop the child process and return the
  backend exit status.
- Missing regional plans: show an explicit empty state in the panel.
- Degraded/uncovered region: render it with reasons; never drop it.
- Malformed timeline data: strict backend validation rejects the frame and the
  UI shows the existing connection/data error state.

## Verification

Backend tests cover model round trips, relative offset calculation, stable
ordering, degraded preservation, and `main.py` preflight/cleanup. Frontend
tests cover live/replay rendering, handoff connectors, region selection,
responsive overflow, and zoom controls. The final acceptance command starts
the project-root `main.py`, checks the Web UI and API health endpoints, and
captures desktop and narrow viewport screenshots.

The current remote environment is known to be incomplete: the project-root
`.venv` currently raises `ModuleNotFoundError` for `langgraph`, and earlier
dependency installation was blocked by its 32-bit Python interpreter on an
x86_64 host. This is an environment repair item, not a reason to weaken the
project dependency bounds or replace the algorithm with mock data.
