# Target Count, Historical Replay, and Tactical Map Design

## Goal

Make the real `main.py` command center easier to operate and read during a
single-target adversarial tracking run. The default run starts with one target;
the operator can choose a target count before starting a new run, review any
previous live run, and focus the map on segmented tracking rather than large
decorative overlays.

## Decisions

- Target count is a run-start setting. The UI does not add or remove targets
  inside an active simulation.
- A new run keeps the current API and UI process alive, but replaces the
  simulation loop, runtime, hub, replay source, and output directory as one
  atomic controller operation. The old run is closed and remains read-only in
  the run catalog.
- The default scenario uses one synthetic target. The selected count is
  validated against the scenario's configured synthetic target capacity; an
  explicit platform-core scenario may only select targets present in its
  configured submarine roster.
- Historical replay is identified by `run_id`, never by an implicit “latest
  file” choice. The current run and completed runs are both catalog entries.
- Operational frames remain truth-safe. Radar heading is an estimator-visible
  heading derived from the belief/prediction or adversary operational summary,
  not target truth.

## Architecture

### Run controller

Add a small controller around the existing `_AgentLoop`, `SimulationEngine`,
`OperationalHub`, and `ReplayService` construction. The controller owns the
current run bundle and exposes:

- current run metadata and health;
- a validated `start_run(target_count, seed)` operation;
- a safe stop/close operation;
- the current live runtime, hub, and replay service to the API layer.

The FastAPI app receives a stable controller-facing port rather than a fixed
runtime object. Live snapshot and WebSocket reads resolve the current bundle;
the simulation worker publishes only to the current bundle. Starting a run
first validates the request and constructs a candidate bundle in a unique
`serve-*` directory without starting its worker. Only after construction
succeeds does the controller stop/close the previous bundle and atomically
install/start the candidate. If validation or construction fails, the
previous bundle remains installed and the API returns a validation or startup
error.

`main.py` continues to launch one backend and one Vite process. The target
count control therefore changes the run inside the existing command center,
without spawning nested `main.py` processes or changing ports.

### Configuration and target selection

The API exposes the allowed target count and current run metadata. The UI
posts a target-count request with a seed and receives the new `run_id` and
status. The controller clones the loaded scenario configuration for the new
run, validates the requested count, and preserves all other timing, sensor,
LLM, and map settings. Explicit platform-core rosters are sliced only when
the requested count is within the configured roster and their scenario
contract remains valid.

### Historical replay

Add a run catalog that scans `outputs/serve-*` directories and reads each
manifest plus its `operational_frames.jsonl` metadata. The API provides:

- `GET /api/runs`: sorted run summaries, including current/live status;
- `POST /api/runs`: start a new run with validated target count and seed;
- `GET /api/replay?run_id=...&start_s=...&end_s=...`: range replay for one
  catalog entry.

The replay service validates the requested run ID against the catalog and
returns a clear 404/422 for unknown or corrupt logs. The UI's replay mode
loads the selected run before loading its time range, clears the old frame
buffer on run changes, and keeps playback speed based on each frame's physical
step.

## Tactical map and segmented tracking presentation

### Radar sector

Extend the operational target estimate with an optional estimator-visible
`heading_rad`. The frame builder derives it in this order: current operational
adversary heading, prediction corridor direction, then a stable zero heading.
The map renders a fixed-width bounded sector, approximately 70 degrees wide,
from the target estimate to its detection range. It draws a filled sector with
one arc and two radial edges; it never draws a full detection circle. Exposure
badges and the platform-in-sector calculation use the same angle and range
predicate, so what the operator sees matches the displayed detection logic.

### Map hierarchy

- Reduce UUV, USV, carrier, and target marker sizes to approximately 65-75%
  of their current screen size while preserving minimum hit areas.
- Keep the active target, current region, and current handoff path at full
  contrast.
- Render planned/future regions with lower fill opacity and thinner strokes;
  keep only the current and next handoff regions visually dominant.
- Keep the dynamic scale bar and map controls unobstructed in the corners.
- Make “分段跟踪” the default task-detail view when a regional timeline exists,
  showing current region, time offset, assignee, relay, and degraded reason.

The map remains interactive: target/UUV selection, region selection, fit,
wheel zoom, and drag pan must continue to work on desktop and mobile.

## Error handling and lifecycle

- Invalid target counts are rejected before stopping the current run.
- A failed new-run construction leaves the current live run serving.
- Historical logs are never mutated by replay requests.
- A live run may be selected for replay while it is still appending frames;
  the index refreshes on file changes.
- LLM outages do not stop physical stepping or operational frame publication;
  the current run displays the paused/degraded brain state and can continue
  into replay.
- Shutdown closes the active run, publisher, runtime, repositories, and LLM
  clients without deleting historical output directories.

## Testing and acceptance

Backend tests cover:

- default target count is one;
- valid/invalid target-count run requests;
- controller replacement and failed-start rollback;
- run catalog ordering and unknown-run rejection;
- replay isolation between two run directories;
- estimator-safe heading derivation and sector membership;
- continuous physical frames while a carrier LLM cycle is blocked.

Frontend tests cover:

- run target-count control and new-run status;
- historical run selection and range loading;
- replay buffer reset and physical-step playback;
- sector geometry and marker sizing;
- segmented-tracking tab prominence and region selection.

Verification uses `lang_py310` for backend execution:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lang_py310 python -m pytest tests -m 'not real_llm' -q
cd src/underwater_tracking/ui && npm test -- --run && npm run build
```

The final acceptance run starts the real `lang_py310 python main.py`, creates
at least two runs with different target counts, verifies their separate
`outputs/serve-*` logs, loads one historical run in the browser, and checks
desktop/mobile screenshots for nonblank canvas pixels, a bounded radar sector,
compact markers, visible segmented tracking, and working selection/zoom/replay
controls.
