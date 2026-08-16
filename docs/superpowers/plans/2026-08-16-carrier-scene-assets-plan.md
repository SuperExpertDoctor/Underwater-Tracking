# Carrier Scene Assets Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将载体舰艇、UUV、目标潜艇和海域背景接入实时/回放态势地图，并让舰艇位置与 UUV 发送/回收状态由后端帧动态驱动。

**Architecture:** 在现有 `SituationSnapshot -> frame_builder -> OperationalFrame -> WebSocket/JSONL -> CanvasMap` 链路中增加 `CarrierState/CarrierView` 与 UUV `deployment_state`。Canvas 继续负责世界坐标渲染和缩放，React 负责载体状态卡片；图片失败时保留现有矢量渲染。

**Tech Stack:** Python 3.11、Pydantic v2、现有 SimulationEngine/LangGraph runtime、React 19、TypeScript、Canvas 2D、Vite、Vitest、Playwright。

## Global Constraints

- Python requirement remains `>=3.11,<3.13`; do not make the implementation depend on Python 3.10-only behavior.
- `OperationalFrame` remains truth-safe; carrier and deployment fields must not contain target truth or evaluation-only data.
- Live WebSocket and JSONL replay must serialize the same `OperationalFrame` contract.
- Existing LangGraph plan-version checks, human assignment flow, UUV selection, drawer tabs, and replay controls must remain intact.
- User assets stay in root `assets/` and are copied to stable English paths under `src/underwater_tracking/ui/public/assets/scene/`.
- Every behavior change follows red-green-refactor: write a focused failing test, run it, implement the minimum, rerun the test, then commit.

---

## File Map

| File | Responsibility after this plan |
| --- | --- |
| `src/underwater_tracking/domain/models.py` | Backend carrier/deployment enums and snapshot state contracts |
| `src/underwater_tracking/domain/ui_models.py` | Truth-safe browser-facing carrier/deployment view contracts |
| `src/underwater_tracking/simulation/carrier.py` | Deterministic carrier kinematics and UUV relationship aggregation |
| `src/underwater_tracking/simulation/engine.py` | Advance carrier state and include it in raw/situation frames |
| `src/underwater_tracking/api/frame_builder.py` | Map `CarrierState` and deployment states into `OperationalFrame` |
| `src/underwater_tracking/ui/src/types/frames.ts` | Authoritative TypeScript mirror of the extended frame contract |
| `src/underwater_tracking/ui/src/components/map/sceneAssets.ts` | Load scene images and provide fallback/geometry helpers |
| `src/underwater_tracking/ui/src/components/CanvasMap.tsx` | Draw background, sprites, carrier links, labels, and vector fallbacks |
| `src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx` | Render carrier status and send/recovery counts in React |
| `src/underwater_tracking/ui/src/App.tsx` / `App.css` | Mount the carrier panel and style responsive states |
| `assets/*` | User-provided source assets; never imported directly by browser code |
| `src/underwater_tracking/ui/public/assets/scene/*` | Stable browser-served copies of the four assets |
| `tests/simulation/test_carrier.py` | Carrier kinematics and relationship aggregation tests |
| `tests/api/test_frame_contracts.py` / `test_frame_pipeline.py` | Python frame contract and adapter tests |
| `src/underwater_tracking/ui/src/components/map/sceneAssets.test.ts` | Asset URL and geometry helper tests |
| `src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx` | Carrier panel rendering and legacy fallback tests |
| `tests/e2e/command-center.spec.ts` | End-to-end asset loading, carrier display, replay, and existing UI regression |

---

### Task 1: Add carrier and UUV deployment contracts

**Files:**
- Modify: `src/underwater_tracking/domain/models.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/domain/__init__.py`
- Test: `tests/domain/test_models.py`
- Test: `tests/api/test_frame_contracts.py`

**Interfaces:**
- Produces `CarrierStatus`, `DeploymentState`, `CarrierState`, `CarrierView`.
- Adds `deployment_state: DeploymentState` to `UUVState` and `deployment_state: DeploymentState` to `UUVView`.
- Adds `carrier: CarrierState | None = None` to `SituationSnapshot` and `carrier: CarrierView | None = None` to `OperationalFrame`.
- Defaults preserve old payloads: missing carrier becomes `None`; missing UUV deployment state becomes `deployed`.

- [ ] **Step 1: Write failing contract tests**

Add tests proving the new fields round-trip and old payloads still parse:

```python
def test_carrier_and_deployment_state_round_trip():
    carrier = CarrierState(
        carrier_id="carrier-01",
        position_xy=(-3000.0, -3000.0),
        heading_rad=0.25,
        speed_mps=1.5,
        status="recovering",
        onboard_uuv_ids=("uuv_03",),
        deployed_uuv_ids=("uuv_01",),
        returning_uuv_ids=("uuv_02",),
    )
    restored = CarrierState.model_validate_json(carrier.model_dump_json())
    assert restored == carrier


def test_old_uuv_and_snapshot_payloads_get_compatible_defaults():
    uuv = UUVState.model_validate({
        "uuv_id": "uuv_01",
        "position_xy": [0.0, 0.0],
        "heading_rad": 0.0,
        "speed_mps": 1.0,
        "energy_fraction": 0.9,
        "status": "available",
    })
    assert uuv.deployment_state == DeploymentState.DEPLOYED
    assert SituationSnapshot.model_validate({
        "scenario_id": "scenario-1",
        "snapshot_revision": 1,
        "sim_time_s": 30,
        "uuvs": [uuv.model_dump()],
        "group_reports": [],
        "pending_events": [],
    }).carrier is None
```

Extend the existing `_full_frame()` fixture in `tests/api/test_frame_contracts.py` with a carrier and assert `OperationalFrame.model_dump_json()` contains no truth fields.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
python -m pytest tests/domain/test_models.py tests/api/test_frame_contracts.py -q
```

Expected: FAIL because `CarrierState`, `CarrierView`, and `deployment_state` do not exist yet.

- [ ] **Step 3: Implement the strict contracts**

Add these enum values and strict models, following the existing `StrEnum`, `StrictModel`, `Field`, and `Literal` patterns:

```python
class CarrierStatus(StrEnum):
    STANDBY = "standby"
    TRANSIT = "transit"
    DEPLOYING = "deploying"
    RECOVERING = "recovering"


class DeploymentState(StrEnum):
    ONBOARD = "onboard"
    DEPLOYED = "deployed"
    RETURNING = "returning"
    FAILED = "failed"


class CarrierState(StrictModel):
    carrier_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    status: CarrierStatus = CarrierStatus.TRANSIT
    onboard_uuv_ids: tuple[str, ...] = ()
    deployed_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()
```

Use `DeploymentState.DEPLOYED` as the default on `UUVState` and `UUVView`. Add `CarrierView` with the same relationship fields and `Point2D position`. Export all new public models from `domain/__init__.py`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the command from Step 2. Expected: all contract tests pass.

- [ ] **Step 5: Commit the contract slice**

```bash
git add src/underwater_tracking/domain tests/domain/test_models.py tests/api/test_frame_contracts.py
git commit -m "feat: add carrier and UUV deployment contracts"
```

### Task 2: Add deterministic carrier kinematics to the simulation

**Files:**
- Create: `src/underwater_tracking/simulation/carrier.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/simulation/test_carrier.py`
- Test: `tests/integration/test_headless_loop.py`

**Interfaces:**
- `CarrierEntity.step(dt_s: float) -> None` advances a reproducible outer patrol route.
- `CarrierEntity.state_for(uuvs: Sequence[UUVState]) -> CarrierState` aggregates onboard/deployed/returning IDs and computes carrier status.
- `SimulationEngine.step()` raw frames and `_build_situation()` snapshots include the same carrier state.

- [ ] **Step 1: Write failing carrier tests**

Create `tests/simulation/test_carrier.py`:

```python
def _uuv(uuv_id: str, deployment_state: str) -> UUVState:
    return UUVState(
        uuv_id=uuv_id,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=1.0,
        energy_fraction=0.9,
        status="available",
        deployment_state=deployment_state,
    )


def test_carrier_patrol_is_deterministic_and_moves():
    left = CarrierEntity()
    right = CarrierEntity()
    left.step(30.0)
    right.step(30.0)
    assert left.state_for(()) == right.state_for(())
    assert left.state_for(()).position_xy != (-3000.0, -3000.0)


def test_carrier_status_and_uuv_lists_follow_deployment_state():
    uuvs = (
        _uuv("uuv_01", "onboard"),
        _uuv("uuv_02", "deployed"),
        _uuv("uuv_03", "returning"),
        _uuv("uuv_04", "failed"),
    )
    state = CarrierEntity().state_for(uuvs)
    assert state.status == CarrierStatus.RECOVERING
    assert state.onboard_uuv_ids == ("uuv_01",)
    assert state.deployed_uuv_ids == ("uuv_02",)
    assert state.returning_uuv_ids == ("uuv_03",)
```

Add a headless engine assertion that the first raw step includes a `carrier` mapping and all default UUVs have `deployment_state == "deployed"`.

- [ ] **Step 2: Run the carrier tests and verify they fail**

```bash
python -m pytest tests/simulation/test_carrier.py tests/integration/test_headless_loop.py -q
```

Expected: FAIL because `CarrierEntity` and the raw frame/snapshot carrier field are absent.

- [ ] **Step 3: Implement the minimum deterministic entity and engine wiring**

Implement `CarrierEntity` with a fixed rectangle route around the outside of the existing `DEFAULT_MAP_BOUNDS`, starting at `(-3000.0, -3000.0)`. Advance only by `dt_s`, use no random source, clamp `dt_s >= 0`, update `heading_rad` from the current leg, and reflect at each route corner. In `state_for`, sort every ID tuple and compute:

```python
returning = tuple(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.RETURNING)
onboard = tuple(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.ONBOARD)
deployed = tuple(u.uuv_id for u in uuvs if u.deployment_state is DeploymentState.DEPLOYED)
status = CarrierStatus.RECOVERING if returning else CarrierStatus.DEPLOYING if onboard and deployed else CarrierStatus.TRANSIT
```

Use `CarrierStatus.STANDBY` only when the carrier speed is zero. Add `_carrier_entity = CarrierEntity()` to `SimulationEngine`, call `step()` from `_advance_world`, include `self._carrier_entity.state_for(...)` in `_build_frame()`, and include the same state in `_build_situation()`. Set `deployment_state` in `_uuv_state()` to `FAILED` for failed UUVs, `RETURNING` for `UUVStatus.RETURNING`, and `DEPLOYED` otherwise.

- [ ] **Step 4: Run the focused tests and verify they pass**

```bash
python -m pytest tests/simulation/test_carrier.py tests/integration/test_headless_loop.py -q
```

Expected: all focused tests pass and the raw engine frame contains carrier coordinates.

- [ ] **Step 5: Commit the simulation slice**

```bash
git add src/underwater_tracking/simulation/carrier.py src/underwater_tracking/simulation/engine.py tests/simulation/test_carrier.py tests/integration/test_headless_loop.py
git commit -m "feat: publish deterministic carrier state"
```

### Task 3: Adapt carrier state into live and replay operational frames

**Files:**
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/replay.py` only if old-frame validation requires an explicit compatibility path
- Test: `tests/api/test_frame_pipeline.py`
- Test: `tests/api/test_live_publisher.py`

**Interfaces:**
- `build_operational_frame(...)` maps `snapshot.carrier` to `OperationalFrame.carrier` without reading runtime truth.
- Every UUV view copies `state.deployment_state.value`.
- JSONL replay accepts frames with and without `carrier`.

- [ ] **Step 1: Write failing adapter and publisher tests**

Add a snapshot fixture with a recovering carrier and a returning UUV, then assert:

```python
frame = build_operational_frame(snapshot, plan=None, ledger_tail=(), events=(), metrics=())
assert frame.carrier is not None
assert frame.carrier.status == "recovering"
assert frame.carrier.returning_uuv_ids == ("uuv_03",)
assert frame.uuvs[0].deployment_state == "returning"
```

Add a live publisher assertion that `publisher.publish(snapshot).carrier.position == snapshot.carrier.position_xy` and that its logged JSONL frame contains the same carrier object.

- [ ] **Step 2: Run the adapter tests and verify they fail**

```bash
python -m pytest tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py -q
```

Expected: FAIL because `frame_builder` currently omits carrier and deployment state.

- [ ] **Step 3: Implement the pure mapping**

Add `_build_carrier_view(carrier: CarrierState | None) -> CarrierView | None`, map `position_xy` to `Point2D`, copy sorted ID tuples and enum values, and pass `carrier=_build_carrier_view(snapshot.carrier)` into `OperationalFrame`. Update `_build_uuv_view` to set `deployment_state=state.deployment_state`.

Do not add a second carrier source to `OperationalFramePublisher`; it must continue forwarding the `SituationSnapshot` received from the engine. Keep the optional field default for old snapshots and let the existing replay parser use Pydantic defaults.

- [ ] **Step 4: Run the adapter tests and verify they pass**

```bash
python -m pytest tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py -q
```

Expected: all adapter/publisher tests pass, including the legacy no-carrier fixture.

- [ ] **Step 5: Commit the frame pipeline slice**

```bash
git add src/underwater_tracking/api/frame_builder.py tests/api/test_frame_pipeline.py tests/api/test_live_publisher.py
git commit -m "feat: carry carrier state through operational frames"
```

### Task 4: Add browser types, static scene assets, and image helpers

**Files:**
- Add: `assets/UUV.png`
- Add: `assets/潜艇.png`
- Add: `assets/背景图.png`
- Add: `assets/舰艇.png`
- Create: `src/underwater_tracking/ui/public/assets/scene/background.png`
- Create: `src/underwater_tracking/ui/public/assets/scene/carrier.png`
- Create: `src/underwater_tracking/ui/public/assets/scene/uuv.png`
- Create: `src/underwater_tracking/ui/public/assets/scene/submarine.png`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Create: `src/underwater_tracking/ui/src/components/map/sceneAssets.ts`
- Test: `src/underwater_tracking/ui/src/components/map/sceneAssets.test.ts`

**Interfaces:**
- TypeScript adds `CarrierStatus`, `DeploymentState`, `CarrierView`, `OperationalFrame.carrier`, and `UUVView.deployment_state`.
- `SCENE_ASSET_URLS` has keys `background`, `carrier`, `uuv`, and `submarine` with `/assets/scene/*.png` values.
- `loadSceneAssets(loader?: ImageLoader): Promise<SceneAssets>` resolves each image to `HTMLImageElement | null`; a failed image never rejects the aggregate promise.
- `coverImageRect(imageWidth, imageHeight, width, height)` returns a centered cover rectangle for the Canvas background.

- [ ] **Step 1: Write failing browser tests**

Add tests with a controlled `Image` fake:

```ts
it("keeps a failed scene image nullable without rejecting other assets", async () => {
  const loader: ImageLoader = async (url) => url.endsWith("carrier.png") ? null : validImage;
  const assets = await loadSceneAssets(loader);
  expect(assets.background).not.toBeNull();
  expect(assets.carrier).toBeNull();
});

it("computes a centered cover rectangle", () => {
  const rect = coverImageRect(1672, 941, 1200, 700);
  expect(rect.x).toBeCloseTo(-21.4, 1);
  expect(rect.y).toBe(0);
  expect(rect.width).toBeCloseTo(1242.7, 1);
  expect(rect.height).toBe(700);
});
```

- [ ] **Step 2: Run the focused browser test and verify it fails**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/map/sceneAssets.test.ts
```

Expected: FAIL because the helper module and extended frame types do not exist.

- [ ] **Step 3: Copy assets and implement the loader/helpers**

Run from the repository root:

```bash
mkdir -p src/underwater_tracking/ui/public/assets/scene
cp assets/背景图.png src/underwater_tracking/ui/public/assets/scene/background.png
cp assets/舰艇.png src/underwater_tracking/ui/public/assets/scene/carrier.png
cp assets/UUV.png src/underwater_tracking/ui/public/assets/scene/uuv.png
cp assets/潜艇.png src/underwater_tracking/ui/public/assets/scene/submarine.png
```

Extend `frames.ts` with:

```ts
export type CarrierStatus = "standby" | "transit" | "deploying" | "recovering";
export type DeploymentState = "onboard" | "deployed" | "returning" | "failed";
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
```

Export `type ImageLoader = (url: string) => Promise<HTMLImageElement | null>`. Use a per-image `loadImage(url)` promise with `image.onerror` resolving `null`, and make `loadSceneAssets(loader = loadImage)` call all four promises through `Promise.all`. Keep the helper free of React state so CanvasMap can cache the result in a ref.

- [ ] **Step 4: Run the focused browser test and verify it passes**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/map/sceneAssets.test.ts
```

Expected: all scene helper tests pass.

- [ ] **Step 5: Commit the browser contract/assets slice**

```bash
git add assets src/underwater_tracking/ui/public/assets/scene src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/components/map/sceneAssets.ts src/underwater_tracking/ui/src/components/map/sceneAssets.test.ts
git commit -m "feat: add browser scene assets and carrier types"
```

### Task 5: Render the background, sprites, and recovery relationship on the map

**Files:**
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/map/geometry.ts` only for shared screen/world scale helpers
- Test: `src/underwater_tracking/ui/src/components/map/geometry.test.ts`

**Interfaces:**
- `CanvasMap` accepts the extended `OperationalFrame` without new props.
- `drawMap` draws the scene background first, then existing geometry, then carrier/recovery links, then sprites and labels.
- Missing images use the existing `drawUuvs` vector triangle and existing target marker; missing carrier uses a small vector ship marker.

- [ ] **Step 1: Write a failing geometry regression test**

Add a helper test for a returning UUV connection:

```ts
it("returns a screen-space recovery segment for a returning UUV", () => {
  expect(recoverySegment(
    { x: -3000, y: -3000 },
    { x: -1200, y: -900 },
    bounds,
    800,
    600,
  )).toEqual({
    start: expect.objectContaining({ x: expect.any(Number), y: expect.any(Number) }),
    end: expect.objectContaining({ x: expect.any(Number), y: expect.any(Number) }),
  });
});
```

- [ ] **Step 2: Run the focused geometry test and verify it fails**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/map/geometry.test.ts
```

Expected: FAIL because `recoverySegment` is not defined.

- [ ] **Step 3: Implement asset-backed drawing with vector fallback**

Load the scene assets once in a `useEffect`, store the resolved map in a ref, and request a redraw when loading completes. Add these draw operations without changing the existing pan/zoom handlers:

```ts
drawSceneBackground(context, assets.background, width, height);
drawCarrier(context, frame.carrier, assets.carrier, transform, scale);
drawRecoveryLinks(context, frame, transform);
drawTargetSprites(context, frame, assets.submarine, transform, scale);
drawUuvSprites(context, frame, assets.uuv, transform, scale, selectedId);
```

Use `context.save()`/`restore()` around every rotated sprite. Clamp sprite dimensions so the carrier remains legible without covering the map, UUVs remain selectable near their current 18px hit radius, and submarine sprites remain behind target labels. Draw a translucent deep-blue overlay after the background and before tactical geometry. Keep the current vector functions as explicit fallback branches.

- [ ] **Step 4: Run all map tests and the existing UI suite**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/map
npm test
```

Expected: all map and existing component tests pass.

- [ ] **Step 5: Commit the map rendering slice**

```bash
git add src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/map/geometry.ts src/underwater_tracking/ui/src/components/map/geometry.test.ts
git commit -m "feat: render carrier scene sprites on tactical map"
```

### Task 6: Add the carrier status card and end-to-end regression coverage

**Files:**
- Create: `src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx`
- Create: `src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/App.tsx`
- Modify: `src/underwater_tracking/ui/src/App.css`
- Modify: `tests/e2e/command-center.spec.ts`

**Interfaces:**
- `CarrierStatusPanel({ frame }: { frame: OperationalFrame | null })` renders a labeled `载体舰 / 发送回收` card.
- With `frame.carrier === null`, it renders `等待载体态势` and does not throw.
- With a carrier, it renders carrier ID/status, onboard/deployed/returning/failed counts, and returning UUV IDs.

- [ ] **Step 1: Write failing component tests**

```tsx
const frameWithoutCarrier: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 1,
  sim_time_s: 30,
  plan_version: 4,
  map_bounds: { min_x: -4000, min_y: -4000, max_x: 4000, max_y: 4000 },
  uuvs: [{
    uuv_id: "uuv_01",
    status: "available",
    deployment_state: "deployed",
    position: { x: -1200, y: -900 },
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 0.8,
    group_id: null,
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: "passive",
    reserved: false,
  }],
  target_estimates: [],
  bearing_rays: [],
  groups: [],
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
};

const frameWithCarrier: OperationalFrame = {
  ...frameWithoutCarrier,
  carrier: {
    carrier_id: "carrier-01",
    position: { x: -3000, y: -3000 },
    heading_rad: 0,
    speed_mps: 1.5,
    status: "recovering",
    onboard_uuv_ids: ["uuv_04"],
    deployed_uuv_ids: ["uuv_01", "uuv_02"],
    returning_uuv_ids: ["uuv_03"],
  },
  uuvs: [{ ...frameWithoutCarrier.uuvs[0], uuv_id: "uuv_03", deployment_state: "returning" }],
};

it("shows carrier deployment and recovery counts", () => {
  render(<CarrierStatusPanel frame={frameWithCarrier} />);
  expect(screen.getByText("载体舰 / 发送回收")).toBeInTheDocument();
  expect(screen.getByText("回收 1")).toBeInTheDocument();
  expect(screen.getByText("uuv_03")).toBeInTheDocument();
});

it("keeps legacy frames usable without a carrier", () => {
  render(<CarrierStatusPanel frame={null} />);
  expect(screen.getByText("等待载体态势")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the focused component tests and verify they fail**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/CarrierStatusPanel.test.tsx
```

Expected: FAIL because the component is not present.

- [ ] **Step 3: Implement and mount the status card**

Use `frame.uuvs` to calculate failed count, use `frame.carrier` for carrier lists/status, and keep all labels in the existing Chinese command-center vocabulary. Mount the card in `RightSidebar` through `App.tsx` beside the overview and UUV list. Add responsive styles that collapse the four counts to two columns below 560px and preserve keyboard focus rings.

- [ ] **Step 4: Run component tests and verify they pass**

```bash
cd src/underwater_tracking/ui
npm test -- src/components/CarrierStatusPanel.test.tsx
```

Expected: all carrier panel tests pass.

- [ ] **Step 5: Extend the Playwright fixture and test the asset flow**

Update `tests/e2e/command-center.spec.ts` mock frames with:

```ts
carrier: {
  carrier_id: "carrier-01",
  position: { x: -3000, y: -3000 },
  heading_rad: 0,
  speed_mps: 1.5,
  status: "recovering",
  onboard_uuv_ids: ["uuv_04"],
  deployed_uuv_ids: ["uuv_01", "uuv_02"],
  returning_uuv_ids: ["uuv_03"],
},
```

Before the UI interaction assertions, request `/assets/scene/background.png`, `/assets/scene/carrier.png`, `/assets/scene/uuv.png`, and `/assets/scene/submarine.png` and assert HTTP 200. Assert the carrier card text and existing UUV selection/replay behavior in the same test.

- [ ] **Step 6: Run the full frontend suite and E2E test**

```bash
cd src/underwater_tracking/ui
npm test
npm run build
npm run test:e2e -- --reporter=line tests/e2e/command-center.spec.ts
```

Expected: all Vitest tests pass, TypeScript/Vite build exits 0, and the command-center E2E test passes at 1440×900.

- [ ] **Step 7: Commit the status/E2E slice**

```bash
git add src/underwater_tracking/ui/src/components/CarrierStatusPanel.tsx src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/App.css tests/e2e/command-center.spec.ts
git commit -m "feat: show carrier launch and recovery status"
```

### Task 7: Run cross-layer verification and document the operator behavior

**Files:**
- Modify: `README.md` with the scene asset locations and carrier card behavior
- Test: all existing Python and frontend tests

- [ ] **Step 1: Add concise operator documentation**

Document that the carrier is emitted in every new operational frame, that `onboard/deployed/returning/failed` are backend-owned states, that old replays may show the compatibility empty state, and that the four source assets live under root `assets/`.

- [ ] **Step 2: Run Python verification in a project-compatible environment**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m 'not real_llm' -q
python -m ruff check src tests
python -m mypy src/underwater_tracking
git diff --check
```

Expected: the non-real-LLM suite passes with zero failures, Ruff and mypy pass under the versions declared by `pyproject.toml`, and `git diff --check` is clean. Do not use a Python 3.10 environment because `pyproject.toml` explicitly requires Python 3.11–3.12.

- [ ] **Step 3: Run the complete frontend verification**

```bash
cd src/underwater_tracking/ui
npm test
npm run build
npm run test:e2e -- --reporter=line tests/e2e/command-center.spec.ts
```

Expected: Vitest, TypeScript/Vite build, and Playwright all exit 0.

- [ ] **Step 4: Inspect the final visual state**

Use Playwright at 1440×900 to capture the live and replay views. Confirm the sea background is visible but subdued, the carrier does not obscure the target, returning links are legible, labels remain readable, and the sidebar card remains usable at the responsive breakpoint.

- [ ] **Step 5: Commit documentation and verification changes**

```bash
git add README.md
git commit -m "docs: explain carrier scene assets and deployment states"
```

## Self-Review Checklist

- [x] Every design section has a corresponding task: contracts (1), deterministic carrier (2), live/replay flow (3), assets/loader (4), Canvas rendering (5), React status/E2E (6), documentation and verification (7).
- [x] All later task interfaces use the exact names introduced earlier: `CarrierState`, `CarrierView`, `CarrierEntity`, `DeploymentState`, `OperationalFrame.carrier`, `UUVView.deployment_state`, `loadSceneAssets`, and `coverImageRect`.
- [x] Legacy frames remain valid because `carrier` is optional and deployment defaults to `deployed`.
- [x] Fallback behavior is tested rather than assumed.
- [x] No task adds evaluation truth, changes LangGraph decision semantics, or introduces a second scheduler.
- [x] No placeholder step is left without an exact file, command, expected result, or interface.
