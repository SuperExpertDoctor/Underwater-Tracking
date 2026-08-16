# Carrier lifecycle and consistency remediation plan

> This follow-up plan addresses the merge-gate findings from the full-branch
> review of the carrier scene implementation. It preserves the existing
> LangGraph human-in-the-loop and plan-version flow while making deployment
> state executable and making the browser rendering contract testable.

## Goals

- Make `onboard -> deployed -> returning -> onboard` a real, deterministic
  engine lifecycle driven by explicit recovery/deployment commands and plan
  actions, with carrier-owned relationship lists derived from one source.
- Reject or exclude onboard/returning/failed UUVs consistently in human
  assignment, optimizer, commit validation, and simulation observation pools.
- Keep operational frames truth-safe and validate that UUV deployment states
  agree with carrier relationship lists.
- Fix carrier heading and UUV state visual cues, enlarge selection hit testing
  to the rendered sprite, and make Playwright exercise visible carrier and
  recovery paths plus image-failure fallback.
- Verify old JSONL frames, LangGraph re-planning, prompts/decision factors,
  Python 3.11 compatibility, and the complete frontend suite before merge.

## Task 1: Implement the authoritative deployment lifecycle

Modify `src/underwater_tracking/simulation/engine.py` and, only if needed,
`src/underwater_tracking/simulation/carrier.py`. Add focused tests under
`tests/simulation/` and an engine/frame/log integration test under
`tests/integration/`.

The engine owns a deployment-state map initialized to `deployed` for the
existing deterministic scenario. Add explicit public operations
`request_uuv_recovery(uuv_id, reason=...)` and
`request_uuv_deployment(uuv_id, reason=...)`; reject unknown IDs and illegal
transitions with deterministic errors. A `rotate` or `return` action in a
committed `PlanCommand` must call the recovery operation. An incoming `track`
command for an onboard member must call the deployment operation. While
returning, the UUV steers toward the current carrier position; when within a
small recovery radius it snaps onboard, stops, clears waypoints, and emits an
observable runtime event. Deployment clears the onboard state, resumes the
commanded waypoint, and emits a deployment event. Failed UUVs never enter
either transition. `_uuv_state`, observations, group membership and carrier
aggregation must use this same state map. Add tests for deployed -> returning
-> onboard -> deployed, illegal/failed transitions, plan-command actions, and
JSON-serializable frame/log carrier lists. Keep deterministic timing and
truth-safe output.

## Task 2: Enforce canonical relationships and deployable resources

Add a small shared predicate module (for example
`src/underwater_tracking/domain/availability.py`) and use it in
`agent/nodes/directives.py`, `agent/nodes/optimize.py`,
`agent/nodes/commit.py`, active-verification/allocator paths, and
`ui/src/components/assistant/AssignmentPanel.tsx`. Only a non-failed UUV with
`deployment_state == deployed` is a deployable planning/assignment resource;
all paths must produce a deterministic conflict or exclusion message for
onboard/returning resources. Add tests for preview, apply/validation and
optimization/commit behavior, including stale plan-version preservation.

Add model validators to `CarrierState`/`CarrierView` and their containing
snapshot/frame as appropriate: relationship lists are pairwise disjoint,
listed IDs exist, and each listed ID's UUV deployment state matches its list;
failed UUVs are omitted from carrier lists. Preserve `carrier=None` and old
missing deployment fields as compatibility defaults. Update inconsistent test
fixtures rather than weakening the invariant. Add UI tests proving onboard
and returning UUVs are not selectable.

## Task 3: Correct Canvas sprite semantics and hit testing

Modify only `ui/src/components/CanvasMap.tsx`,
`ui/src/components/map/geometry.ts`, and their focused tests. Keep the carrier
asset/vector fallback heading convention aligned at heading 0 and pi/2 by
using a documented carrier orientation offset. When the UUV image loads,
retain active/failed/reserved color halo/outline and selected styling. Make
recovery-link geometry use the current zoom/pan view (or remove the unused
helper) and derive the UUV hit area from the rendered sprite dimensions while
keeping reasonable click tolerance. Add focused tests for heading, visual
state branches, recovery links and edge hit selection; preserve fallback and
all pan/zoom behavior.

## Task 4: Exercise real browser and legacy frame paths

Modify `tests/e2e/command-center.spec.ts` and add/update backend replay tests
only where needed. Use map bounds containing the carrier and a returning UUV
with matching IDs. Assert the carrier card, carrier/link-visible map pixels or
stable screenshot region, and UUV selection/replay. Route each scene image to
404 in a separate test and assert the UI still renders vector/background
fallback without console errors. Add a JSONL replay fixture with both
`carrier` and every UUV `deployment_state` missing, asserting normalized
compatibility values reach the frontend. Keep the existing assignment
preview/apply and next-LangGraph-replan assertions.

## Task 5: Verify, review, and merge

Run the complete project-compatible Python 3.11 suite, Ruff, Mypy, frontend
Vitest/build/Playwright, and visual inspection at 1440x900. Review the full
branch against this remediation plan and the original plan. Resolve every
Critical/Important finding, then fast-forward merge `carrier-scene-assets`
into `master` without touching unrelated `.claude/` or user-provided files.

## Invariants

- `OperationalFrame` never contains target truth or evaluation-only state.
- A non-failed UUV appears in exactly one carrier relationship list and its
  `deployment_state` matches that list; failed UUVs are omitted from carrier
  lists and carry `failed`.
- Only deployed, non-failed, non-returning UUVs are assignment/planning
  resources.
- Live WebSocket and JSONL replay serialize/parse the same frame contract;
  old frames normalize missing carrier/deployment fields to compatibility
  values.
- Human assignment remains preview -> explicit apply -> next LangGraph
  strategic replanning; no UI scheduler or renderer may mutate plan state.
