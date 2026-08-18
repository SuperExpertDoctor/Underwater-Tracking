# Visual Command Center And Unified LLM Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the underwater tracking command center readable at a glance by focusing the map on the predicted corridor, rendering fine-grained regional tasks at stable screen sizes, reorganizing the right workspace, and routing expert feedback/evidence questions through one LLM Client conversation.

**Architecture:** Keep the simulation and planner in world units. Add a frontend-only view configuration and camera-fit/level-of-detail layer, then consolidate the sidebar around situation, prediction/handoff, and conversation. The new conversation endpoint classifies turns with the existing structured LLM client, delegates evidence answers to the read-only question path and plan changes to the existing directive preview/apply path, and never lets classification alone mutate a plan.

**Tech Stack:** Python 3.10, Pydantic, existing LLM/LangGraph runtime, FastAPI transport, React 18, TypeScript, Vite, Vitest, HTML Canvas, SVG, Playwright.

---

## Task 1: Add the display-scale and camera contract

**Files:**

- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/types/viewConfig.ts`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/map/geometry.ts`
- Test: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/CanvasMap.test.ts`

- [ ] **Step 1: Write the failing view-scale tests**

```ts
it("fits the prediction corridor without including the hidden detection circle", () => {
  const bounds = cameraBoundsForFrame(frame, {
    focusMode: "prediction_corridor",
    showDetectionRange: false,
    predictionPadding: 1.15,
  });
  expect(bounds.maxX - bounds.minX).toBeLessThan(0.8 * fullScenarioWidth);
});

it("clamps marker size in screen pixels", () => {
  expect(clampMarkerPixels(24, 0.5, 16, 36)).toBe(16);
  expect(clampMarkerPixels(24, 2, 16, 36)).toBe(36);
});
```

Run:

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/CanvasMap.test.ts
```

Expected: FAIL because the camera helper and pixel clamp do not exist.

- [ ] **Step 2: Implement the pure view helpers**

Add `ViewConfig`, `DEFAULT_VIEW_CONFIG`, `cameraBoundsForFrame`, and
`clampMarkerPixels`. `cameraBoundsForFrame` must use target/prediction/region
geometry, apply 15% padding, and include the detection bounds only when the
layer is enabled or `focusMode` is `full_area`. Keep `frame` values unchanged.

- [ ] **Step 3: Wire camera and controls into `CanvasMap` and `App`**

Store `ViewConfig` in `App`, pass it to `CanvasMap`, reset the camera only on
new target/prediction revisions, and keep pan/zoom user adjustments stable
between ordinary frames. Add a “聚焦目标/全域” control and make hidden
detection range the default.

- [ ] **Step 4: Run tests and commit**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/CanvasMap.test.ts
npm run build
git add src/underwater_tracking/ui/src/types/viewConfig.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/components/map/geometry.ts src/underwater_tracking/ui/src/components/CanvasMap.test.ts
git commit -m "feat: focus command center on prediction corridor"
```

## Task 2: Render fine-grained regions and persistent platform status

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.css`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/types/regionalTasks.test.ts`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx`

- [ ] **Step 1: Add red tests for LOD and marker visibility**

```ts
it("keeps all regional facts but aggregates labels at low zoom", () => {
  expect(regionRenderMode(0.7)).toBe("grouped");
  expect(regionRenderMode(1.4)).toBe("individual");
});

it("draws platform rings before selection", () => {
  const result = renderMap(frame, { selectedUuvId: null });
  expect(result.platformRings).toContain("uuv_01");
  expect(result.platformRings).toContain("usv_01");
});
```

Run the focused Vitest files and confirm the new assertions fail.

- [ ] **Step 2: Implement fixed-pixel platform markers and map LOD**

Clamp target/UUV/USV marker dimensions to 30/24/28 pixels with the existing
screen transform. Draw status rings and role badges for all visible platforms;
use selection only for the extra focus ring. Keep the detection circle on a
separate layer controlled by `showDetectionRange`.

- [ ] **Step 3: Implement fine-grid rendering**

Use 16 local divisions by default. Render region polygons from
`frame.regional_plans`, and at low zoom use stable group labels while retaining
all cells for hit testing and details. Use square aspect-preserving transforms
and never stretch cells with the viewport.

- [ ] **Step 4: Improve graph readability**

Change the graph viewport to a horizontally scrollable, larger node canvas;
use at least 96x44px region nodes, 24px entity circles, and readable labels.
Keep temporal arrows separate from responsibility edges and show relay edges
with a distinct dashed style. Add selected-region synchronization callbacks.

- [ ] **Step 5: Run UI tests/build and commit**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/types/regionalTasks.test.ts src/components/CanvasMap.test.ts src/components/assistant/RegionTaskGraph.test.tsx
npm run build
git add src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.css src/underwater_tracking/ui/src/types/regionalTasks.test.ts src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx
git commit -m "feat: make regional tracking visuals readable"
```

## Task 3: Reorganize the right workspace into three primary cards

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/SidebarPanels.css`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`

- [ ] **Step 1: Write the sidebar contract tests**

Assert that the rendered sidebar has exactly one top-level `LLM Client`
section, no visible standalone `专家反馈` or `态势问答` cards, independent
`aria-expanded` controls, and a prediction/handoff section with graph,
timeline, and list tabs.

- [ ] **Step 2: Implement the three-section shell**

Keep native `details/summary` behavior. Move status, brain, carrier, UUV/USV
roster, and adversary facts under `当前态势`; place AssignmentPanel,
AssignmentReview, and RegionTaskGraph under `预测与接力`; mount one new
`ConversationPanel` under `LLM Client`. Do not render scheme constraints as a
user-facing section.

- [ ] **Step 3: Add responsive layout and visual hierarchy**

Set a desktop sidebar width of 360-440px, give the prediction/handoff card the
largest scroll budget, and turn the sidebar into a mobile drawer. Keep the map
and bottom timeline occupying layout space rather than covering the map.

- [ ] **Step 4: Run tests/build and commit**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/RightSidebar.test.tsx src/components/assistant/AssignmentPanel.test.tsx
npm run build
git add src/underwater_tracking/ui/src/components/RightSidebar.tsx src/underwater_tracking/ui/src/components/SidebarPanels.css src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx src/underwater_tracking/ui/src/components/RightSidebar.test.tsx
git commit -m "feat: organize command center sidebar"
```

## Task 4: Add the unified LLM Client conversation contract

**Files:**

- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/ConversationPanel.tsx`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/ConversationPanel.test.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/services/assistantApi.ts`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/domain/conversation_models.py`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/conversation.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/directives.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/questions.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/runtime.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/api/app.py`
- Test: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/agent/test_conversation.py`

- [ ] **Step 1: Write failing domain/API tests**

```python
def test_evidence_query_is_read_only(conversation_service, active_plan):
    result = conversation_service.handle(message("为什么 region_4 质量下降？"))
    assert result.classification == "evidence_query"
    assert result.needs_confirmation is False
    assert result.plan_version == active_plan.revision
    assert conversation_service.pending_events() == ()

def test_plan_revision_requires_explicit_apply(conversation_service):
    result = conversation_service.handle(message("提高 region_4 的接力优先级"))
    assert result.classification == "plan_revision"
    assert result.proposal.status == "preview"
    assert result.needs_confirmation is True
```

Run:

```bash
conda run -n lang_py310 python -m pytest tests/agent/test_conversation.py -q
```

Expected: FAIL because the conversation models/service do not exist.

- [ ] **Step 2: Add structured conversation models**

Define `ConversationClassification` as `plan_revision | evidence_query | mixed | clarification`, plus message, turn result, evidence references, proposal, and `expected_plan_version`. Reject unknown evidence IDs and malformed version fields before invoking a mutating path.

- [ ] **Step 3: Implement the classifier node and safe routing**

Create one structured LLM operation that receives bounded conversation context,
current target/region selection, known evidence IDs, and active plan revision.
For `evidence_query`, call the existing read-only question answer path. For
`plan_revision`, call the existing directive preview path. For `mixed`, return
both sub-results without auto-applying. For `clarification`, return one follow-
up question. Preserve existing question and directive ledger records.

- [ ] **Step 4: Add a single frontend API and conversation panel**

Add `sendConversationMessage`, `applyConversationProposal`, and typed result
models in `assistantApi.ts`. Render role messages, classification badge,
evidence chips, proposal diff, and confirmation action in one panel. Keep the
old APIs available to avoid breaking existing tests while the new panel uses
the unified path.

- [ ] **Step 5: Run backend/frontend tests and commit**

```bash
conda run -n lang_py310 python -m pytest tests/agent/test_conversation.py tests/agent/test_directives.py tests/agent/test_questions.py -q
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/assistant/ConversationPanel.test.tsx src/services/assistantApi.test.ts
npm run build
git add src/underwater_tracking/domain/conversation_models.py src/underwater_tracking/agent/nodes/conversation.py src/underwater_tracking/agent/nodes/directives.py src/underwater_tracking/agent/nodes/questions.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/ui/src/components/assistant/ConversationPanel.tsx src/underwater_tracking/ui/src/components/assistant/ConversationPanel.test.tsx src/underwater_tracking/ui/src/services/assistantApi.ts src/underwater_tracking/ui/src/App.tsx tests/agent/test_conversation.py
git commit -m "feat: unify expert feedback and evidence chat"
```

## Task 5: Make the timeline demonstrate tracking performance

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/PlaybackBar.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/PlaybackBar.test.tsx`

- [ ] **Step 1: Add red tests for simulation-time progress**

Assert that the slider uses `sim_time_s` and `sim_duration_s`, labels show
seconds, event ticks are rendered, and no primary label contains a frame
fraction such as `3/3`.

- [ ] **Step 2: Implement time-axis and tracking events**

Compute the time domain from replay frames/events, render handoff and plan
revision ticks, and keep play/pause/step controls. Use `playbackRate` only to
control presentation speed; it must not alter simulation physics.

- [ ] **Step 3: Surface effect metrics**

Add compact coverage, quality, target deviation, and handoff latency rows to
the prediction/handoff card and event drawer. Label proxy metrics as proxy.

- [ ] **Step 4: Run UI tests/build and commit**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/PlaybackBar.test.tsx
npm run build
git add src/underwater_tracking/ui/src/components/PlaybackBar.tsx src/underwater_tracking/ui/src/components/BottomDrawer.tsx src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/components/PlaybackBar.test.tsx
git commit -m "feat: show tracking performance on simulation timeline"
```

## Task 6: Verify real `main.py` and browser behavior

**Files:**

- Modify only if needed: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/main.py`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/integration/test_visual_command_center_runtime.py`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts`

- [ ] **Step 1: Run backend/frontend regression tests**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python -m pytest tests/agent tests/api tests/simulation -q
cd src/underwater_tracking/ui
npm test -- --run
npm run build
```

- [ ] **Step 2: Start the real entry point**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python main.py --steps 0 --host 0.0.0.0 --port 8020
```

Record the printed Web UI/API URLs. Do not substitute a mock app or a copied
demo entry point.

- [ ] **Step 3: Run browser checks at desktop and narrow widths**

Verify: prediction-corridor focus, hidden detection range, visible marker rings,
fine regions, graph/timeline/list tabs, one LLM Client section, evidence-only
turn with unchanged plan version, plan-revision preview/apply, and no console
errors or overlapping sidebar content.

- [ ] **Step 4: Run final checks and commit**

```bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
git diff --check
git status --short
```

Report exact test results, commit IDs, running `main.py` process, UI URL, and
any remaining runtime limitation. Preserve unrelated dirty files.
