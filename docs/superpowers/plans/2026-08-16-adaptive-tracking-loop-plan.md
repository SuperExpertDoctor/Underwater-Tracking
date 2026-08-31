# Adaptive Tracking Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six audit gaps that prevent the production LangGraph loop from using preplanned operations, technical intelligence, heterogeneous UUV capabilities, forwarded runtime events, automatic rotation, and target-level quality constraints.

**Architecture:** Keep `SituationSnapshot` as the only live planning input and add typed operational-scheme, intelligence, and surveillance-capability fields with compatibility defaults. The engine publishes those fields and its pending events to the runtime adapter; the adapter queues periodic review and battery events. Deterministic allocation/commit remains authoritative for members and waypoints, while strategy receives bounded summaries and can only propose priorities/quality/policies.

**Tech Stack:** Python 3.11-compatible Pydantic v2, LangGraph, FastAPI, SciPy MILP/fallback allocator, React/TypeScript/Vitest/Playwright.

## Global Constraints

- Preserve operational truth isolation: no target truth or evaluation-only state may enter `SituationSnapshot`, `OperationalFrame`, prompts, or API responses.
- Preserve preview -> explicit apply -> next strategic LangGraph cycle for human directives; external input may enqueue events but never replace the active plan directly.
- Preserve legacy snapshots/configuration by defaulting missing scheme, intelligence, capability, and deployment fields.
- Preserve deterministic ordering, original event IDs, bounded prompt payloads, and hard-constraint rejection/degradation semantics.
- Every production behavior change must have a regression test that was observed failing before the implementation.

---

### Task 1: Add typed operational inputs and capability/intelligence contracts

**Files:**
- Modify: `src/underwater_tracking/domain/models.py`
- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/scenario/default.yaml`
- Modify: `configs/tracking.yaml`
- Test: `tests/domain/test_models.py`, `tests/agent/test_agent_loader.py`

**Interfaces:**
- Produces `OperationalScheme`, `IntelligenceReport`, and `SurveillanceCapability` domain models.
- `UUVState.capability` defaults to a standard passive/active capability.
- `SituationSnapshot.operational_scheme` and `SituationSnapshot.intelligence_reports` default to `None` and `()`.
- `TrackingConfig.uuv_capabilities` is an optional ID-to-capability mapping; `ScenarioConfig.operational_scheme` is optional.

- [ ] **Step 1: Write failing tests** for capability range validation/legacy defaulting, scheme quality bounds, intelligence source/confidence/expiry validation, and snapshot round-trip.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the new contracts/fields do not exist.
- [ ] **Step 3: Implement the strict models and config fields** with compatibility defaults and no hidden-state fields.
- [ ] **Step 4: Add the default scenario scheme and two example capability profiles** while keeping all omitted config sections valid.
- [ ] **Step 5: Run the focused tests and the existing domain/config tests**; commit as `feat: add adaptive tracking input contracts`.

### Task 2: Wire engine inputs, event forwarding, periodic review, and battery rotation

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/simulation/uuv.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Modify: `src/underwater_tracking/cli.py`
- Test: `tests/simulation/test_engine.py`, `tests/agent/test_event_monitor.py`, `tests/integration/test_agent_loop.py`

**Interfaces:**
- `SimulationEngine.submit_intelligence(report: IntelligenceReport) -> None` and `set_operational_scheme(scheme: OperationalScheme) -> None` queue input for subsequent snapshots.
- `CarrierRuntime.submit_events(events: Sequence[RuntimeEvent]) -> None` preserves IDs and de-duplicates.
- `_AgentLoop.on_situation` forwards `SituationSnapshot.pending_events`, emits one `strategic_review` per configured interval, and emits deduplicated `battery_rotation` events for low-energy deployed members.

- [ ] **Step 1: Write failing engine/adapter tests** proving an `active_ping`, lifecycle event, quality guard, review event, and low-energy rotation event reach the runtime queue/classifier.
- [ ] **Step 2: Run those tests and observe the current missing-forwarding/unknown-event failures.**
- [ ] **Step 3: Add engine storage/filtering for active scheme, valid intelligence, and per-UUV capabilities; use capability values in sensor observations and UUV state snapshots.
- [ ] **Step 4: Add runtime batch event submission with stable-ID de-duplication and wire the loop to forward source events before ticking.
- [ ] **Step 5: Add EventMonitor routes for `strategic_review`, lifecycle/report events, scheme/intelligence updates, and `quality_guard:*`.
- [ ] **Step 6: Add interval/cooldown bookkeeping for review and battery events, then run focused integration tests and commit as `feat: close runtime event feedback loop`.

### Task 3: Make strategy and optimizer use scheme, intelligence, capabilities, quality, and rotation

**Files:**
- Modify: `src/underwater_tracking/agent/prompts.py`
- Modify: `src/underwater_tracking/agent/nodes/strategy.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Modify: `src/underwater_tracking/agent/nodes/commit.py`
- Modify: `src/underwater_tracking/planning/allocation.py`
- Modify: `src/underwater_tracking/domain/agent_models.py`
- Test: `tests/agent/test_strategy.py`, `tests/agent/test_plan_pipeline.py`, `tests/planning/test_allocation.py`, `tests/agent/test_commit.py`

**Interfaces:**
- `AllocationInput.required_quality_by_target` and `AllocationInput.target_priority_by_target` carry deterministic target-level objectives.
- `PlanningConfig.rotation_threshold` is explicit and used by allocation/commit.
- `TrackingPlan.rotation_uuv_ids` contains only members below the rotation threshold; `PlanCommand.actions` rotates only those IDs.
- `CandidateMetrics` includes quality deficit/priority loss in candidate ordering.

- [ ] **Step 1: Write failing tests** for scheme minimum quality overriding a weak proposal, capability-specific feasible pairs, quality-aware candidate ordering, and selective member rotation/replacement.
- [ ] **Step 2:** Run focused tests and verify they fail under the current global-capability/current-EWMA implementation.
- [ ] **Step 3: Add bounded scheme/intelligence/capability summaries to decision factors and update the strategy prompt’s allowed evidence and required decision checklist.
- [ ] **Step 4: Extend allocator bounds/objective and optimizer problem construction with effective target quality/priority, capability range/variance/speed, and healthy replacement eligibility.
- [ ] **Step 5: Add projected quality/deficit metrics, persist rotation IDs, and make commit validation enforce required quality without trusting candidate metrics.
- [ ] **Step 6:** Run focused agent/planning tests and commit as `feat: make adaptive planning constraints executable`.

### Task 4: Expose and render the continuous information flow

**Files:**
- Modify: `src/underwater_tracking/api/dependencies.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Test: `tests/api/test_app.py`, `tests/api/test_frame_contracts.py`, `src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx`, `src/underwater_tracking/ui/src/hooks/useWebSocket.test.tsx`

**Interfaces:**
- Add `POST /api/intelligence` and `PUT /api/operational-scheme`; return `202` after queuing and `501` when the injected runtime lacks the optional input port.
- Add operational-frame `scheme` and bounded `intelligence` views with compatibility defaults.
- Show scheme version/minimum quality, intelligence source counts/recency, review events, and rotation events in the existing sidebar/drawer without changing Canvas truth boundaries.

- [ ] **Step 1: Write failing API/frame/UI tests** for accepted input, optional-port fallback, frame serialization, and visible summary/event labels.
- [ ] **Step 2:** Run focused Python/Vitest tests and observe the missing route/field/render failures.
- [ ] **Step 3: Add typed API request handling and delegate to the runtime input ports.
- [ ] **Step 4: Add frame mapping and compact UI summaries using existing design tokens/components.
- [ ] **Step 5: Run API tests, all Vitest tests, build, and the command-center Playwright path; commit as `feat: surface adaptive tracking inputs in command center`.

### Task 5: Whole-branch verification and review

**Files:**
- Modify: `docs/superpowers/audits/2026-08-16-underwater-tracking-implementation-audit.md`
- Modify: `README.md` only if command/API behavior needs documentation.

- [ ] **Step 1:** Run focused regressions for all four tasks.
- [ ] **Step 2:** Run Python non-real-LLM tests, Ruff, Mypy, UI Vitest, UI build, and Playwright using the project-compatible environment; capture exact counts.
- [ ] **Step 3:** Inspect a 1440×900 command-center screenshot and verify the new scheme/intelligence/rotation information remains readable.
- [ ] **Step 4:** Dispatch the final whole-branch review against this plan and resolve all Critical/Important findings.
- [ ] **Step 5:** Update the audit with evidence, commit verification/docs, and report the remaining formal-evaluation scope explicitly.
