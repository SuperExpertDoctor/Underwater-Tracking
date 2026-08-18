# Segment Region Gantt Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Expose estimator-safe regional handoff timelines through the operational frame and render them as a live/replay region-swimlane Gantt window that starts through main.py.

**Architecture:** Keep TargetRegionPlan.region_tasks as the source of truth. Add a pure Python adapter that converts regional tasks into an optional OperationalFrame.region_timeline; the React UI consumes only that frame view and renders a resizable BottomDrawer tab in live and replay modes. Preserve all existing map, group, segment, API, and frame fields.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI/WebSocket, React 18, TypeScript, Vite, Vitest, Playwright.

---

## Task 1: Add strict operational timeline contracts

**Files:**
- Modify: src/underwater_tracking/domain/ui_models.py beside Point2D, CommunicationLinkView, and OperationalFrame
- Modify: src/underwater_tracking/ui/src/types/frames.ts beside the matching frame types
- Create: tests/api/test_region_timeline_contracts.py

- [ ] Step 1: Write failing Python contract tests

Add tests for strict validation, JSON round trips, and old-frame compatibility:

~~~python
def test_region_timeline_round_trip_keeps_assignments_and_offsets() -> None:
    item = RegionTimelineView(
        region_id="T1:cell:0:0",
        target_id="T1",
        center=Point2D(x=50.0, y=50.0),
        bounds=MapBounds(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0),
        start_offset_s=0.0,
        end_offset_s=30.0,
        status="active",
        coverage_mode="required",
        priority=0.8,
        occupancy_likelihood=0.7,
        uuv_assignments=(
            RegionAssignmentView(
                platform_id="uuv-1",
                platform_kind="uuv",
                role="passive_tracker",
                start_offset_s=0.0,
                end_offset_s=30.0,
            ),
        ),
    )
    frame = make_operational_frame(region_timeline=(item,))
    assert OperationalFrame.model_validate_json(frame.model_dump_json()).region_timeline == (item,)


def test_old_operational_frame_without_region_timeline_is_compatible() -> None:
    payload = make_operational_frame().model_dump(mode="json")
    payload.pop("region_timeline", None)
    assert OperationalFrame.model_validate(payload).region_timeline == ()


def test_timeline_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError, match="end_offset_s"):
        RegionTimelineView(**timeline_payload(start_offset_s=20.0, end_offset_s=10.0))
~~~

Define test helpers by reusing the smallest existing OperationalFrame fixture in tests/api/test_frame_contracts.py; do not add truth or simulator-only fields to the operational fixture.

- [ ] Step 2: Run the focused contract tests and confirm the expected failure

Run from the worktree:

~~~bash
.venv/bin/python -m pytest tests/api/test_region_timeline_contracts.py -q
~~~

Expected failure: RegionTimelineView and OperationalFrame.region_timeline do not exist yet.

- [ ] Step 3: Implement the Python and TypeScript contracts

Add these strict Python models:

~~~python
class RegionAssignmentView(StrictModel):
    platform_id: str = Field(min_length=1)
    platform_kind: Literal["uuv", "usv"]
    role: str = Field(min_length=1)
    start_offset_s: float = Field(allow_inf_nan=False)
    end_offset_s: float = Field(allow_inf_nan=False)
    sonar_mode: Literal["passive", "active"] = "passive"

    @model_validator(mode="after")
    def ordered_offsets(self) -> RegionAssignmentView:
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("end_offset_s must not precede start_offset_s")
        return self


class RegionTimelineView(StrictModel):
    region_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    center: Point2D
    bounds: MapBounds
    start_offset_s: float = Field(allow_inf_nan=False)
    end_offset_s: float = Field(allow_inf_nan=False)
    status: Literal["planned", "active", "handed_off", "degraded", "uncovered"]
    coverage_mode: Literal["required", "reserve", "optional"] = "required"
    priority: float = Field(ge=0, allow_inf_nan=False)
    occupancy_likelihood: float = Field(ge=0, le=1, allow_inf_nan=False)
    uuv_assignments: tuple[RegionAssignmentView, ...] = ()
    usv_assignments: tuple[RegionAssignmentView, ...] = ()
    communication_links: tuple[CommunicationLinkView, ...] = ()
    handoff_from: str | None = None
    handoff_to: str | None = None
    evidence_ids: tuple[str, ...] = ()
    degraded_reasons: tuple[str, ...] = ()
    plan_revision: int = Field(default=1, ge=1)
~~~

Add region_timeline: tuple[RegionTimelineView, ...] = () to OperationalFrame and matching TypeScript interfaces. Keep the TypeScript frame field optional so old JSON remains accepted.

- [ ] Step 4: Run focused contract tests and static checks

~~~bash
.venv/bin/python -m pytest tests/api/test_region_timeline_contracts.py -q
.venv/bin/python -m py_compile src/underwater_tracking/domain/ui_models.py
cd src/underwater_tracking/ui
npm test -- --run src/components/RegionTimelinePanel.test.tsx
~~~

Expected supported-environment result: focused tests pass. If the remote interpreter lacks pytest or pydantic, record the exact dependency failure and continue with compile/type checks without weakening project bounds.

- [ ] Step 5: Commit the contract slice

~~~bash
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/ui/src/types/frames.ts tests/api/test_region_timeline_contracts.py
git commit -m "feat: add operational region timeline contracts"
~~~

## Task 2: Derive relative regional timelines in the frame builder

**Files:**
- Modify: src/underwater_tracking/api/frame_builder.py
- Create: tests/api/test_region_timeline_builder.py
- Modify: tests/api/test_frame_contracts.py only for shared fixture fields if required

- [ ] Step 1: Write failing builder tests

Cover absolute-to-relative conversion, stable ordering, handoff metadata, assignments, links, and degraded preservation:

~~~python
def test_build_region_timeline_uses_current_frame_as_t_plus_zero() -> None:
    plan = make_plan_with_regions(
        sim_time_s=100,
        region_windows=(("T1:cell:1:0", 110, 160), ("T1:cell:0:0", 80, 120)),
    )
    timeline = build_region_timeline(plan, sim_time_s=100)
    assert [item.region_id for item in timeline] == ["T1:cell:0:0", "T1:cell:1:0"]
    assert timeline[0].start_offset_s == -20.0
    assert timeline[0].end_offset_s == 20.0


def test_degraded_region_is_kept_with_reason_and_handoff() -> None:
    tracking_plan = make_tracking_plan_with_degraded_regions()
    item = build_region_timeline(
        tracking_plan,
        sim_time_s=tracking_plan.valid_from_s,
    )[0]
    assert item.status == "degraded"
    assert item.degraded_reasons == ("insufficient_uuv",)
    assert item.handoff_to == plan.tasks[0].successor_region_id
~~~

- [ ] Step 2: Run the builder tests to verify they fail

~~~bash
.venv/bin/python -m pytest tests/api/test_region_timeline_builder.py -q
~~~

Expected failure: build_region_timeline is not defined.

- [ ] Step 3: Implement the pure adapter

Add one pure function:

~~~python
def build_region_timeline(
    plan: TrackingPlan | None,
    sim_time_s: int,
) -> tuple[RegionTimelineView, ...]:
    if plan is None:
        return ()
    rows: list[RegionTimelineView] = []
    for regional_plan in plan.regional_plans.values():
        cells = {cell.region_id: cell for cell in regional_plan.cells}
        for task in regional_plan.tasks:
            cell = cells.get(task.region_id)
            if cell is None:
                continue
            rows.append(_region_timeline_row(regional_plan, task, cell, sim_time_s))
    return tuple(sorted(rows, key=lambda row: (row.start_offset_s, row.region_id)))
~~~

The row helper maps UUV roles in stable assigned-ID order, maps the assigned USV role, copies task communication links into public link views, and converts TimeWindow seconds to offsets without clamping. If a region has no valid task/cell pair, skip only that malformed row and leave remaining regions visible.

Pass the result into build_operational_frame and the OperationalFrame constructor. Keep the existing PlanView.segment_plan path unchanged.

- [ ] Step 4: Run Python frame and truth-boundary tests

~~~bash
.venv/bin/python -m pytest tests/api/test_region_timeline_builder.py tests/api/test_frame_contracts.py tests/domain/test_truth_boundary.py -q
.venv/bin/python -m py_compile src/underwater_tracking/api/frame_builder.py
~~~

Expected result: regional timeline round trips, old frame contracts, and truth-boundary tests pass.

- [ ] Step 5: Commit the adapter slice

~~~bash
git add src/underwater_tracking/api/frame_builder.py tests/api/test_region_timeline_builder.py tests/api/test_frame_contracts.py
git commit -m "feat: expose relative regional tracking timeline"
~~~

## Task 3: Build the region-swimlane Gantt component

**Files:**
- Create: src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx
- Create: src/underwater_tracking/ui/src/components/RegionTimelineRow.tsx
- Create: src/underwater_tracking/ui/src/components/regionTimeline.ts
- Create: src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx
- Modify: src/underwater_tracking/ui/src/App.css

- [ ] Step 1: Write failing Vitest tests for pure timeline math and rendering

Test stable ordering, relative-axis percentage, status classes, assignment labels, handoff details, and empty data:

~~~tsx
it("sorts rows by start offset then region id", () => {
  expect(sortRegionTimeline([row("R1", 20), row("R0", 20), row("R2", -5)])
    .map((item) => item.region_id)).toEqual(["R2", "R0", "R1"]);
});

it("renders assignments, relay and degraded reason", () => {
  render(<RegionTimelinePanel frame={frameWithTimeline()} />);
  expect(screen.getByText("T1:cell:0:0")).toBeInTheDocument();
  expect(screen.getByText(/uuv-1.*passive_tracker/)).toBeInTheDocument();
  expect(screen.getByText(/USV-01.*relay/)).toBeInTheDocument();
  expect(screen.getByText("insufficient_uuv")).toBeInTheDocument();
});

it("shows an empty state for old frames", () => {
  render(<RegionTimelinePanel frame={frameWithoutTimeline()} />);
  expect(screen.getByText("当前暂无区域任务")).toBeInTheDocument();
});
~~~

- [ ] Step 2: Run the UI tests and confirm failure

~~~bash
cd src/underwater_tracking/ui
npm test -- --run src/components/RegionTimelinePanel.test.tsx
~~~

Expected failure: the component and timeline helpers do not exist.

- [ ] Step 3: Implement deterministic timeline helpers

In regionTimeline.ts, use an explicit visible window and avoid division by zero:

~~~ts
export function timelineWindow(rows: RegionTimelineView[], horizon = 600) {
  const maxEnd = rows.reduce((max, row) => Math.max(max, row.end_offset_s), horizon);
  return {
    start: Math.min(0, ...rows.map((row) => row.start_offset_s)),
    end: Math.max(horizon, maxEnd),
  };
}

export function offsetPercent(offset: number, start: number, end: number): number {
  if (end <= start) return 0;
  return Math.max(0, Math.min(100, ((offset - start) / (end - start)) * 100));
}
~~~

Use a stable status-to-class map and sort by start_offset_s then region_id. Do not derive rows from uuvs or groups; only consume frame.region_timeline ?? [].

- [ ] Step 4: Implement the visual component

RegionTimelinePanel renders a header, relative axis, current T+0 line, rows, and a detail section for the selected region. RegionTimelineRow renders the region bar, assignment chips, relay chips, and handoff node. Use semantic buttons for selectable rows, aria-label on the timeline and status, and CSS horizontal overflow on narrow screens.

Keep selected state keyed by region_id; if a new frame removes it, clear the selection. Use the exact labels and colors defined in the spec: active cyan, relay amber, reserve violet, degraded amber, uncovered red, standby gray.

- [ ] Step 5: Run component tests and build

~~~bash
cd src/underwater_tracking/ui
npm test -- --run src/components/RegionTimelinePanel.test.tsx
npm run build
~~~

Expected result: focused component tests pass and TypeScript/Vite build exits with status 0.

- [ ] Step 6: Commit the component slice

~~~bash
git add src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx src/underwater_tracking/ui/src/components/RegionTimelineRow.tsx src/underwater_tracking/ui/src/components/regionTimeline.ts src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx src/underwater_tracking/ui/src/App.css
git commit -m "feat: render regional tracking gantt window"
~~~

## Task 4: Integrate the panel into live and replay UI

**Files:**
- Modify: src/underwater_tracking/ui/src/components/BottomDrawer.tsx
- Modify: src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx
- Modify: src/underwater_tracking/ui/src/App.css only for tab and drawer layout rules

- [ ] Step 1: Add the failing integration test

Extend BottomDrawer.test.tsx:

~~~tsx
it("opens the 分段跟踪 tab and renders regional rows", async () => {
  const user = userEvent.setup();
  render(<BottomDrawer frame={frameWithTimeline()} visible onToggle={vi.fn()} />);
  await user.click(screen.getByRole("tab", { name: "分段跟踪" }));
  expect(screen.getByRole("region", { name: "区域分段跟踪甘特图" })).toBeInTheDocument();
  expect(screen.getByText("T1:cell:0:0")).toBeInTheDocument();
});
~~~

- [ ] Step 2: Run the integration test to confirm failure

~~~bash
cd src/underwater_tracking/ui
npm test -- --run src/components/BottomDrawer.test.tsx
~~~

Expected failure: the drawer has no 分段跟踪 tab.

- [ ] Step 3: Add the tab without changing existing tabs

Add the Route icon and a tab entry:

~~~tsx
{ label: "分段跟踪", icon: Route }
~~~

Render RegionTimelinePanel when its tab index is active and pass the current frame. Keep 时间线, 方案, 事件, 决策台账, and 指标 behavior unchanged. The panel owns selection and details; the parent does not need a second state channel.

- [ ] Step 4: Run UI regression tests and build

~~~bash
cd src/underwater_tracking/ui
npm test -- --run src/components/BottomDrawer.test.tsx src/components/RegionTimelinePanel.test.tsx
npm run build
~~~

Expected result: existing drawer tests and new regional tests pass.

- [ ] Step 5: Commit UI integration

~~~bash
git add src/underwater_tracking/ui/src/components/BottomDrawer.tsx src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx src/underwater_tracking/ui/src/App.css
git commit -m "feat: add regional tracking tab to command center"
~~~

## Task 5: Verify main.py startup and end-to-end frame flow

**Files:**
- Modify: main.py only when the startup smoke test identifies a lifecycle defect
- Modify: tests/main/test_main.py with a bounded startup helper test when the lifecycle test identifies a defect
- Create: tests/integration/test_region_timeline_acceptance.py
- Modify: README.md with the exact startup command and UI tab description

- [ ] Step 1: Write the acceptance test before changing startup code

The Python acceptance test constructs a regional plan, calls build_operational_frame, and asserts the public contract:

~~~python
def test_region_timeline_acceptance_preserves_live_and_replay_offsets() -> None:
    frame_at_start = build_frame_for_region_plan(sim_time_s=100)
    frame_later = build_frame_for_region_plan(sim_time_s=120)
    assert frame_at_start.region_timeline[0].start_offset_s == 0
    assert frame_later.region_timeline[0].start_offset_s == -20
    assert frame_at_start.region_timeline[0].uuv_assignments[0].platform_id == "uuv-1"
    assert frame_at_start.region_timeline[0].usv_assignments[0].platform_id == "usv-1"
~~~

Add a frontend acceptance fixture that loads the same frame JSON in live and replay component renders and checks the same region IDs appear.

- [ ] Step 2: Run the acceptance test to establish its result

~~~bash
.venv/bin/python -m pytest tests/integration/test_region_timeline_acceptance.py tests/main/test_main.py -q
~~~

Expected supported-environment result: PASS. On the current remote environment, first report missing pytest or Python dependency failures instead of changing pyproject.toml bounds or committing generated environment files.

- [ ] Step 3: Verify the current main.py lifecycle

Keep the existing check_frontend_prereqs, spawn_vite, stop_vite, and cli.main flow. The smoke check must cover both child startup and cleanup. If a child-process leak or missing output is observed, add a regression test before changing the lifecycle. The expected lifecycle test shape is:

~~~python
def test_main_stops_vite_when_cli_exits(monkeypatch, main_script) -> None:
    events: list[str] = []
    monkeypatch.setattr(main_script, "spawn_vite", lambda *_: FakeVite(events))
    monkeypatch.setattr(main_script.cli, "main", lambda *_: 0)
    monkeypatch.setattr(main_script.shutil, "which", lambda _: "npm")
    assert main_script.main(["--steps", "1"]) == 0
    assert events == ["terminate", "wait"]
~~~

Do not add a second GUI process; the existing Vite URL is the demonstration window.

- [ ] Step 4: Run main.py with a finite demo horizon

After supported dependencies are installed, run:

~~~bash
.venv/bin/python main.py --config configs/scenario/default.yaml --steps 120 --seed 42 --host 127.0.0.1 --port 8000
~~~

Confirm the output contains Web UI at port 5173 and API/WS at port 8000, then open the Web UI and select 分段跟踪. Capture a screenshot showing at least one region row, one UUV assignment, one USV relay, and one handoff state. Stop the finite run cleanly and verify no Vite process remains.

- [ ] Step 5: Run final verification

~~~bash
.venv/bin/python -m pytest -q
cd src/underwater_tracking/ui
npm test -- --run
npm run build
cd ../..
git diff --check master..HEAD
git status --short --branch
~~~

Expected result: Python tests, UI tests, build, and diff check pass; the feature branch is clean. If the remote interpreter remains the known 32-bit build without required wheels, record that exact blocker and report the static checks and UI checks that did run.

- [ ] Step 6: Commit documentation and acceptance

~~~bash
git add tests/integration/test_region_timeline_acceptance.py tests/main/test_main.py README.md
git commit -m "test: verify regional tracking gantt startup flow"
~~~

## Self-review checklist

- Every spec section maps to a task: contracts (Task 1), relative frame derivation (Task 2), region-swimlane UI (Task 3), drawer/live/replay integration (Task 4), startup/error/acceptance (Task 5).
- The single source of truth is consistent: Python derives only from TrackingPlan.region_tasks; React reads only from OperationalFrame.region_timeline.
- The same field and function names are used throughout: RegionTimelineView, RegionAssignmentView, OperationalFrame.region_timeline, build_region_timeline, and RegionTimelinePanel.
- Old frames and old UI fields remain optional or unchanged.
- No truth-only model is added to the operational frame.
- All commands use the remote worktree root and no generated dependency lock or environment file is committed.
