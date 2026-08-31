# UUV Initialization, Local Perception, and Moving Carrier Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default live run a deterministic UUV-only mission in which one carrier and three mother ships follow a configured moving formation, UUVs remain onboard until physically deployed, the target perceives only platforms within its local sensor range, region entry and handoff use explicit probabilistic evidence, and periodic truth-safe situation summaries enter the existing asynchronous memory pipeline.

**Architecture:** Keep `MissionController` as the single execution-state owner and `SimulationEngine` as the physics/observation owner. Replace loader-synthesized platforms with an explicit roster, add a deterministic moving-rendezvous boundary for mother ships, add a target-owned local-sensing projection that is the only source for adversary inputs, and pass typed probability/handoff observations into `MissionController`. Preserve the current Smart Assistant and MemoryWorker architecture; add only a low-frequency persisted source event. Live paths reject USV scenarios, while the replay adapter may parse and discard historical USV fields.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, NumPy/SciPy, deterministic simulation and A* routing, SQLite, LangGraph structured LLM ports, FastAPI/WebSocket, React 18, TypeScript, Vite, Vitest, Playwright, pytest, Ruff, and mypy.

## 中文审阅摘要

本计划不是重新开发全部任务系统，而是基于当前已经存在的 `MissionController`、UUV 区域任务、Smart Assistant 和 MemoryWorker，修正四个根因并补齐三个证据闭环：旧场景初始化、固定坐标返航、目标全局感知、舰载 UUV 无条件绘制，以及区域概率、有效交接、周期记忆输入。

| 阶段 | 主要修改 | 审阅时应看到的结果 |
| --- | --- | --- |
| 1. 显式场景 | 删除 loader 动态复制航母和清空 USV 的做法；用独立 YAML 配置 1 航母、3 母舰、12 UUV、固定归属和 1200 m 目标探测范围 | `python main.py` 默认进入 UUV-only 场景；实时入口拒绝带 USV 的配置 |
| 2. 初始化与投放 | 以 `home_carrier_id` 初始化母舰库存；增加权威 `physically_exposed` 字段；投放状态和 `uuv_deployed` 在同一发布边界生效 | 初始地图没有 UUV 圆圈；库存仍可看到 12 台舰载 UUV；到达区域外围后才显示 UUV |
| 3. 移动会合算法 | 新增航母槽位投影、巡航轨迹纯预测、带禁区/地图/服务窗的迭代会合求解 | 母舰返航目标随航母移动，不再驶向启动坐标；路线更新不瞬移、不跳过停靠点 |
| 4. 编队物理集成 | 航母先按既定循环航迹运动，待命母舰跟随旋转槽位，任务母舰离队并在回收后重新入列 | 连续两次航次均只产生一次各自的返队事件；会合不可行时明确降级 |
| 5. 目标局部感知 | 建立目标自有局部检测投影；进入门限 1200 m、丢失门限 1300 m；主动发射可听范围始终严格为 1200 m | 远处平台、我方被动观测和私有坐标不进入目标 LLM；局部证据无变化不调用目标 LLM |
| 6. 区域进入概率 | 使用公开目标估计的二维高斯分布计算方形区域概率质量；连续两次达到 0.70 才切换 | 不再以“均值是否在框内”的 0/1 值判断；缺测会清零连续计数但不改变生命周期 |
| 7. 有效交接 | 用带计划版本、观测周期、观测 ID、observer ID 和时间戳的 `HandoffEvidence` 代替两个布尔量 | 未部署、不健康、非被动、观测不足、旧周期或硬保护失败均不能交接；失败时进入降级而非伪造完成 |
| 8. 清除实时 USV 契约 | 同时清理主/从规划提示词、区域策略模型、当前帧、侧栏、任务图和 Canvas；仅 legacy adapter 可读取并丢弃旧字段 | 新方案输入、实时帧、UI 和目标侧输入都不存在 USV；旧回放仍能打开 |
| 9. 周期记忆来源 | 新增真值安全摘要和独立有界 SQLite writer；MemorySourceReader/MemoryWorker 继续用现有游标与工作队列 | 每个 `progress_report_s` 边界幂等落库；LLM 断线或记忆降级不阻塞物理执行，也不触发方案重规划 |
| 10. 端到端验收 | 从真实默认入口记录初始化、投放、扫描、跟踪、交接、回收、移动返队完整时序，并做桌面/移动端 Playwright 截图 | 展示效果与参考图的布局风格一致，但修正初始 UUV 圆圈、目标距离、探测圈和标签重叠问题 |

保留而不重写的能力：`MissionController` 的普通/专属任务模式和资源轮转主结构、Smart Assistant 的预览/确认/应用、证据回溯、四类记忆窗口、LLM Thinking、Memory Steam、长期记忆版本链和真实 SentenceTransformer/LongCat 边界。本次只为这些能力补充新的执行证据和周期来源。

执行门禁：本文当前仅供审阅。未收到批准前，不修改生产代码、不运行长时仿真、不提交本计划或现有未跟踪截图。

## Global Constraints

- Live simulation contains exactly `carrier_01` (carrier), `carrier_02..04` (mother ships), and `uuv_00..11`. There is no USV live entity, observation, assignment, planning input, operational-frame field, or UI rendering path.
- Historical JSONL/replay payloads may contain `usvs` or `usv_assignments`; only `legacy_frame_adapter` may accept them, and it must discard them before constructing a current `OperationalFrame`.
- The carrier owns no UUV. Each UUV has one configured mother ship for the complete run; plan application cannot transfer ownership.
- The carrier follows its configured cyclic patrol route. A mother ship that is not on a mission follows a configured relative formation slot. A returning mother ship rendezvous with the future moving slot and never returns to a frozen startup coordinate.
- UUVs begin `onboard`, follow their owning mother ship internally, and are absent from the map until a successful deployment stop. They do not initialize in a circle or any other waterborne formation.
- The target detection radius comes from `SubmarineInitialConfig.detection_range_m`, default `1200.0`. The same value drives sensor gating, adversary input, the operational frame, and the UI circle.
- Private world coordinates may be used only inside the target sensor boundary to decide whether a detection occurred. Adversary models receive noisy local range/bearing estimates, never the private target/platform coordinates or the private gate distance.
- Blue passive observations are never reflected into target-owned observations. The adversary LLM does not run when there is no target-local evidence or target-local change.
- Region entry uses the current public target estimate and covariance, never target truth. Invalid covariance holds the current lifecycle rather than falling back to a 0/1 mean-in-polygon test.
- A handoff completes only from typed, current-plan, current-cycle evidence. Boolean `successor_passive_ready` and immediate exit-trigger completion are removed.
- `MissionController` remains the sole owner of region lifecycle, UUV mission mode, recovery creation, resource episodes, and handoff completion. `SimulationEngine` only computes and submits observations.
- Periodic situation summaries contain only public estimates and execution state. They persist at `timing.progress_report_s`, are idempotent, do not trigger replanning, and are consumed asynchronously by the existing MemoryWorker.
- The diagnostic `underwater_tracking.cli simulate` path has no assistant database or MemoryWorker; it enforces the UUV-only roster but does not claim memory persistence. Periodic summary requirements apply to the agent-coupled `_agent_run` and `_serve` paths.
- Existing Smart Assistant preview/confirm/apply, evidence trace, Memory Window, LLM Thinking, and Memory Steam behavior is retained. No private chain-of-thought is stored or displayed.
- The reference screenshots are acceptance references, not files to modify: `src/underwater_tracking/ui/ui-review-current-open.png`, `ui-review-later.png`, and `ui-review-sidebar.png`.
- Implementation must preserve unrelated user changes and must not stage the untracked UI review screenshots.

---

### Task 1: Replace synthesized initialization with an explicit UUV-only live roster

**Files:**
- Modify: `src/underwater_tracking/config/platform_core.py`
- Modify: `src/underwater_tracking/config/loader.py`
- Create: `configs/environment_uuv_only.yaml`
- Modify: `configs/scenario/uuv_only_single_target.yaml`
- Modify: `main.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/config/test_uuv_only_config.py`
- Modify: `tests/config/test_platform_core_loader.py`
- Modify: `tests/main/test_main.py`
- Modify: `tests/integration/test_uuv_only_runtime_entrypoints.py`

**Interfaces:**

```python
class InitialPlatformConfig(StrictConfig):
    # Required for every UUV in a UUV-only live environment.
    home_carrier_id: str | None = None


class CarrierInitialConfig(StrictConfig):
    formation_slot_offset_xy: CoordinateXY = (0.0, 0.0)


class EnvironmentConfig(StrictConfig):
    rendezvous_tolerance_m: PositiveFloat = 250.0
```

`load_app_config()` must validate the explicit roster; it must not clone carriers, offset patrol routes, clear USVs, or derive UUV ownership. Add a production guard used by `_simulate`, `_agent_run`, and `_serve`:

```python
def _require_uuv_only_live_config(config: AppConfig) -> None:
    environment = config.environment
    if not _is_uuv_only_config(config) or environment is None or environment.usvs:
        raise SystemExit("live runtime requires an explicit UUV-only scenario")
```

The guard does not run in replay readers.

- [ ] Step 1: Add failing config tests for an explicit one-carrier/three-mother roster, zero USVs, 12 onboard UUVs, fixed ownership of four UUVs per mother, unique slot offsets, 250 m rendezvous tolerance, and one target with a 1200 m range.

```python
def test_uuv_only_roster_is_explicit_and_owned() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    environment = config.environment
    assert environment is not None
    carriers = (environment.carrier, *environment.carriers)
    assert [(item.platform_id, item.role) for item in carriers] == [
        ("carrier_01", "carrier"),
        ("carrier_02", "mother_ship"),
        ("carrier_03", "mother_ship"),
        ("carrier_04", "mother_ship"),
    ]
    assert environment.usvs == ()
    assert {uuv.deployment_state for uuv in environment.uuvs} == {"onboard"}
    assert [uuv.home_carrier_id for uuv in environment.uuvs[:4]] == ["carrier_02"] * 4
    assert environment.submarines[0].detection_range_m == 1200.0
```

- [ ] Step 2: Add rejection tests for missing owner, owner=`carrier_01`, unknown owner, duplicate carrier ID, nonzero USV roster, carrier inventory imbalance, and target-to-nearest-mother distance outside `[2500, 4000]` in the default explicit environment.

- [ ] Step 3: Run the focused tests and confirm that the current carrier-synthesis behavior fails them.

```bash
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/config/test_platform_core_loader.py tests/main/test_main.py tests/integration/test_uuv_only_runtime_entrypoints.py -q
```

- [ ] Step 4: Implement the schema validation. For UUV-only mode require exactly one carrier, three mothers, all four explicit IDs/roles, one unique slot per mother, 12 onboard UUVs, exactly four owned by each mother, no UUV owned by the carrier, no USV, and exactly one target. Do not infer missing values.

- [ ] Step 5: Create `configs/environment_uuv_only.yaml` with deterministic values. Use a cyclic carrier patrol route, leader slot `(0, 0)`, three non-overlapping mother slots, initial mother positions equal to the world-space slots at route start, UUV owner IDs, and a target position near the planned task corridor but outside 1200 m. Point `uuv_only_single_target.yaml` at this file.

- [ ] Step 6: Change root `_DEFAULT_CONFIG` to `configs/scenario/uuv_only_single_target.yaml`, add `_require_uuv_only_live_config()` to all three live CLI handlers, and leave `configs/environment.yaml` available only to explicitly named legacy test scenarios.

- [ ] Step 7: Verify config and live-entry behavior.

```bash
PYTHONPATH=src python -m pytest tests/config/test_uuv_only_config.py tests/config/test_platform_core_loader.py tests/main/test_main.py tests/integration/test_uuv_only_runtime_entrypoints.py -q
ruff check main.py src/underwater_tracking/config src/underwater_tracking/cli.py tests/config tests/main tests/integration/test_uuv_only_runtime_entrypoints.py
```

- [ ] Step 8: Commit.

```bash
git add main.py configs/environment_uuv_only.yaml configs/scenario/uuv_only_single_target.yaml src/underwater_tracking/config/platform_core.py src/underwater_tracking/config/loader.py src/underwater_tracking/cli.py tests/config/test_uuv_only_config.py tests/config/test_platform_core_loader.py tests/main/test_main.py tests/integration/test_uuv_only_runtime_entrypoints.py
git commit -m "fix: make live initialization explicitly uuv only"
```

---

### Task 2: Make configured ownership and onboard state authoritative in the engine

**Files:**
- Modify: `src/underwater_tracking/simulation/carrier.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `tests/simulation/test_uuv_only_carrier_group.py`
- Modify: `tests/simulation/test_carrier.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Modify: `tests/api/test_uuv_only_frame_contract.py`

**Interfaces:**

```python
def _configured_uuv_owner_ids(self) -> dict[str, str]: ...

def _uuv_is_physically_exposed(self, uuv_id: str) -> bool: ...
```

The engine stores `_waterborne_uuv_ids: set[str]`. Deployment adds the UUV; completed recovery removes it; failure does not remove it. `UUVView` gains an authoritative `physically_exposed: bool`, populated from this set. This distinguishes an onboard failure from a failed UUV still in the water without overloading `DeploymentState.FAILED`.

- [ ] Step 1: Add failing engine tests proving every UUV is initially colocated internally with its configured mother, all are `onboard`, `waterborne_uuv_ids` is empty, carrier inventory is empty, and mother inventories are exactly the configured four IDs.

- [ ] Step 2: Add failing transition tests proving: deploy adds the UUV to `_waterborne_uuv_ids`; waterborne failure remains exposed; recovery removes it; an onboard failure has `deployment_state="failed"` but `physically_exposed=False`; plan application that attempts cross-mother ownership is rejected atomically.

- [ ] Step 3: Run the focused tests before implementation.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_uuv_only_carrier_group.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py -q
```

- [ ] Step 4: Replace the round-robin owner assignment in `SimulationEngine.__init__` and `_spawn_explicit_world` with `home_carrier_id`. Initialize onboard UUV internal positions from the owner, not from the primary carrier and not from YAML water coordinates.

- [ ] Step 5: Pass the same immutable ownership map into `MissionController` plan validation. Preserve owner inventory across resource rotation; recovery returns the UUV to its configured owner only.

- [ ] Step 6: Keep all UUVs in the backend operational frame as inventory records, including onboard entries. Do not solve map visibility by deleting them from the API; Task 8 filters physical rendering while the sidebar retains inventory.

- [ ] Step 7: Make the deployment state change and `uuv_deployed` event observable in the same publication boundary. No operational frame may expose `physically_exposed=True` for a newly deployed UUV unless that frame's event set also contains its deployment event; do not defer the event to a later frame.

- [ ] Step 8: Verify deterministic initialization twice with the same seed and compare serialized initial frame bytes.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_uuv_only_carrier_group.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py -q
ruff check src/underwater_tracking/simulation/engine.py src/underwater_tracking/runtime/mission_controller.py tests/simulation/test_uuv_only_carrier_group.py tests/runtime/test_mission_controller.py
```

- [ ] Step 9: Commit.

```bash
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py tests/simulation/test_uuv_only_carrier_group.py tests/runtime/test_mission_controller.py tests/api/test_uuv_only_frame_contract.py
git commit -m "fix: preserve mother ship uuv ownership and onboard state"
```

---

### Task 3: Add deterministic moving formation and rendezvous primitives

**Files:**
- Create: `src/underwater_tracking/simulation/carrier_group.py`
- Modify: `src/underwater_tracking/simulation/carrier.py`
- Modify: `src/underwater_tracking/planning/carrier_tasks.py`
- Modify: `src/underwater_tracking/planning/astar.py`
- Create: `tests/simulation/test_carrier_group.py`
- Modify: `tests/simulation/test_carrier.py`
- Modify: `tests/planning/test_carrier_tasks.py`

Do not reuse `simulation/formation_control.py`; that module shapes UUV tracking slots around a target. The new module owns surface carrier-group geometry and rendezvous only.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class RendezvousSolution:
    endpoint_xy: tuple[float, float]
    eta_s: int
    route: RoutePlan
    iterations: int


@dataclass(frozen=True, slots=True)
class CommittedServiceStop:
    point_xy: tuple[float, float]
    earliest_s: int
    latest_s: int


def carrier_slot_position(
    leader_position_xy: tuple[float, float],
    leader_heading_rad: float,
    slot_offset_xy: tuple[float, float],
) -> tuple[float, float]: ...


def solve_moving_rendezvous(
    *,
    start_xy: tuple[float, float],
    current_time_s: int,
    committed_stops: tuple[CommittedServiceStop, ...],
    mother_speed_mps: float,
    project_slot_at: Callable[[int], tuple[float, float]],
    route_planner: AStarRoutePlanner,
    forbidden_regions: tuple[Bounds, ...],
    map_bounds: Bounds,
    tolerance_m: float,
    max_iterations: int = 8,
) -> RendezvousSolution | None: ...
```

Add pure projection and route-tail update methods to `CarrierEntity`:

```python
def project_patrol_state(self, delta_s: float) -> tuple[tuple[float, float], float]: ...
def remaining_committed_stops(self) -> tuple[tuple[float, float], ...]: ...
def replace_unfinished_return_segment(self, route_xy: tuple[tuple[float, float], ...]) -> None: ...
def clear_completed_mission(self) -> None: ...
```

- [ ] Step 1: Write failing geometry tests for slot rotation at headings `0`, `pi/2`, and `pi`, and pure carrier route projection across a corner and route wrap.

- [ ] Step 2: Write failing rendezvous tests for convergence on a moving cyclic route, stable results for identical input, preservation of committed stop order/windows, waiting until an earliest service time, rejection after a latest service time, map-bound/forbidden-region avoidance, `None` on bounded non-convergence, and no route endpoint farther than `rendezvous_tolerance_m` from the slot projected at ETA.

- [ ] Step 3: Write a failing `CarrierEntity` test that updates only the unfinished return segment and proves current position, completed stop indices, and uncompleted service stops are unchanged.

- [ ] Step 4: Run focused tests.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_carrier_group.py tests/simulation/test_carrier.py tests/planning/test_carrier_tasks.py -q
```

- [ ] Step 5: Implement `carrier_slot_position()` as `leader + rotate(slot_offset, heading)`. Implement patrol projection without mutating the entity. Implement rendezvous as a bounded fixed-point iteration: call A* with the real map bounds and forbidden regions, preserve the ordered committed stops, include earliest-window waiting in travel time, reject latest-window misses, re-project the slot at the resulting ETA, and stop when endpoint movement is within tolerance.

- [ ] Step 6: Change `CarrierTaskPlanner.build_routes(..., home_positions=...)` to `build_routes(..., rendezvous_positions=...)`; the planner still validates all service windows and returns a route ending at the supplied predicted rendezvous point. Remove wording and assertions that require the endpoint to equal startup home.

- [ ] Step 7: Implement the `CarrierEntity` route-tail API without changing position. Route replacement must reject a first point different from the current position and any omission/reordering of remaining committed stops.

- [ ] Step 8: Verify the primitives and type/style checks.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_carrier_group.py tests/simulation/test_carrier.py tests/planning/test_carrier_tasks.py -q
ruff check src/underwater_tracking/simulation/carrier_group.py src/underwater_tracking/simulation/carrier.py src/underwater_tracking/planning/carrier_tasks.py tests/simulation/test_carrier_group.py
mypy src/underwater_tracking/simulation/carrier_group.py src/underwater_tracking/simulation/carrier.py
```

- [ ] Step 9: Commit.

```bash
git add src/underwater_tracking/simulation/carrier_group.py src/underwater_tracking/simulation/carrier.py src/underwater_tracking/planning/carrier_tasks.py src/underwater_tracking/planning/astar.py tests/simulation/test_carrier_group.py tests/simulation/test_carrier.py tests/planning/test_carrier_tasks.py
git commit -m "feat: add moving carrier group rendezvous"
```

---

### Task 4: Integrate formation following, mission departure, and moving return

**Files:**
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/domain/mission_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `tests/simulation/test_uuv_only_carrier_group.py`
- Modify: `tests/domain/test_mission_models.py`
- Modify: `tests/api/test_uuv_only_frame_contract.py`

**State contract:**

```text
CarrierExecutionMode.FORMATION_FOLLOW
  -> mission accepted
CarrierExecutionMode.MISSION_ROUTE
  + existing CarrierRouteStatus TO_DEPLOY / DEPLOYING / EN_ROUTE_NEXT_DEPLOY / RECOVERING
  -> all committed stops complete and every recoverable batch member is onboard
CarrierExecutionMode.RENDEZVOUS_RETURN
  + CarrierRouteStatus.RETURNING_TO_FLEET
  -> distance to moving slot <= rendezvous_tolerance_m
CarrierExecutionMode.FORMATION_FOLLOW
  + CarrierRouteStatus.COMPLETE
  + one carrier_returned_to_fleet event per mission revision
```

Add `CarrierExecutionMode` as an engine/entity enum and `CarrierRouteStatus.RENDEZVOUS_BLOCKED` as the recoverable route-planning degradation. Publish the mission route status, but keep the entity execution mode internal. Arrival at a recovery stop begins/continues recovery service; it does not complete that stop until every assigned recoverable UUV reports `onboard` on the owning mother. A mother must hold at the recovery point while members approach. Do not infer recovery or return completion only from `mission_route_complete`.

Add an explicit entity/engine service-stop handshake:

```python
def set_mission_route(
    self,
    route_xy: tuple[tuple[float, float], ...],
    *,
    stop_windows: Mapping[int, tuple[int, int]] | None = None,
    externally_released_stop_indices: frozenset[int] = frozenset(),
    rendezvous_xy: tuple[float, float] | None = None,
) -> None: ...

@property
def awaiting_release_stop_index(self) -> int | None: ...

def release_mission_stop(self, route_index: int) -> None: ...
```

When `CarrierEntity` first reaches an externally released stop after its earliest service time, it reports that arrival once and holds position without incrementing `_mission_route_index`. Only `SimulationEngine` may call `release_mission_stop()` after validating the assigned batch's physical state.

- [ ] Step 1: Add failing integration tests proving the carrier moves along its configured patrol, standby mothers follow their rotating slots with bounded speed/turn motion, and no step teleports a mother more than `speed_mps * dt_s + epsilon`.

- [ ] Step 2: Add a failing full sortie test: a mother leaves its slot, reaches deployment stops before any owned UUV becomes waterborne, holds at its recovery stop until every assigned ordinary-task UUV is actually `onboard`, then follows an updated return tail, rendezvous with the moving slot, and emits exactly one `carrier_returned_to_fleet` event. Assert the mother has zero displacement toward rendezvous while any required recovery member is still `deployed`/`returning`, and its returned-to-fleet event frame lists no outstanding deployed/returning member from that batch.

- [ ] Step 3: Add a second-sortie test proving the return event can occur once in each mission revision rather than being globally deduplicated forever.

- [ ] Step 4: Add a failure test: if the rendezvous solver returns `None`, retain the current safe route/position, set `CarrierRouteStatus.RENDEZVOUS_BLOCKED`, and emit a plan-impact candidate event `carrier_rendezvous_infeasible`; do not redirect to `_carrier_home_positions`. A later successful solve returns the status to `RETURNING_TO_FLEET`.

- [ ] Step 5: Add a blocked-recovery test: if a required UUV is permanently unable to reach the mother, keep the mother at the recovery stop, set the mission degraded/failed according to the verified resource event, and emit a plan-impact recovery-blocked event. Do not let the mother return to formation and leave the UUV chasing a moving ship.

- [ ] Step 6: Add `CarrierEntity` unit tests proving an externally released recovery stop reports arrival once, remains at the same route index and position across repeated steps, rejects release of the wrong index, and advances only after `release_mission_stop(current_index)`.

- [ ] Step 7: Run focused tests and capture the current frozen-home failures.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_uuv_only_carrier_group.py tests/domain/test_mission_models.py tests/api/test_uuv_only_frame_contract.py -q
```

- [ ] Step 8: Replace `_carrier_home_positions` with configured `_carrier_slot_offsets`. Step the leader first; step standby mothers toward current world slots with bounded kinematics. Mission mothers continue their installed finite routes.

- [ ] Step 9: Mark recovery route indices as externally released when installing a mission. At each observation boundary, recompute only unfinished rendezvous tails. Never replace an active service leg during a physical tick. Call `release_mission_stop()` only after its assigned recoverable members are onboard. Declare fleet return only after all service/recovery stops are complete and the mother is within configured tolerance of the current slot; then clear the mission route and resume formation following.

- [ ] Step 10: Publish carrier role, execution phase, inventory, deployed IDs, returning IDs, and current status. Do not publish private route-planner internals.

- [ ] Step 11: Verify the engine integration.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_uuv_only_carrier_group.py tests/simulation/test_carrier.py tests/planning/test_carrier_tasks.py tests/api/test_uuv_only_frame_contract.py -q
ruff check src/underwater_tracking/simulation src/underwater_tracking/domain/mission_models.py src/underwater_tracking/api/frame_builder.py tests/simulation
```

- [ ] Step 12: Commit.

```bash
git add src/underwater_tracking/simulation/carrier.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/mission_models.py src/underwater_tracking/api/frame_builder.py tests/simulation/test_carrier.py tests/simulation/test_uuv_only_carrier_group.py tests/domain/test_mission_models.py tests/api/test_uuv_only_frame_contract.py
git commit -m "feat: execute sorties against a moving carrier group"
```

---

### Task 5: Build the target-owned local sensing boundary

**Files:**
- Create: `src/underwater_tracking/simulation/adversary_sensing.py`
- Modify: `src/underwater_tracking/domain/adversary_models.py`
- Modify: `src/underwater_tracking/simulation/target.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/agent/nodes/adversary.py`
- Create: `tests/simulation/test_adversary_local_sensing.py`
- Modify: `tests/agent/test_adversary_graph.py`
- Modify: `tests/agent/test_runtime_master_slave_adversary.py`

**Interfaces:**

```python
TargetPlatformKind = Literal["carrier", "mother_ship", "uuv"]


@dataclass(frozen=True, slots=True)
class ExposedPlatform:
    platform_id: str
    platform_kind: TargetPlatformKind
    position_xy: tuple[float, float]  # private; sensor-boundary input only
    sensor_mode: Literal["active", "passive"]
    relay_available: bool


class LocalPlatformDetection(StrictModel):
    platform_id: str
    platform_kind: TargetPlatformKind
    observed_at_s: int
    estimated_range_m: float
    relative_bearing_rad: float
    confidence: float
    sensor_mode: Literal["active", "passive"]
    relay_available: bool


@dataclass(frozen=True, slots=True)
class TargetLocalSensingResult:
    detections: tuple[LocalPlatformDetection, ...]
    acquired_platform_ids: frozenset[str]
    lost_platform_ids: frozenset[str]
    audible_active_emitter_ids: frozenset[str]


def update_local_platform_detections(
    *,
    target_id: str,
    target_position_xy: tuple[float, float],
    target_heading_rad: float,
    detection_range_m: float,
    release_margin_m: float,
    candidates: Sequence[ExposedPlatform],
    previous_ids: frozenset[str],
    sim_time_s: int,
    seed: int,
) -> TargetLocalSensingResult: ...
```

Private coordinates must not be fields on `LocalPlatformDetection` or any Pydantic adversary input model.

- [ ] Step 1: Add pure sensor tests for acquisition at `<=1200 m`, no first acquisition at 1201 m, retention of an already acquired platform from 1200 through 1300 m, loss at `>1300 m`, deterministic noisy range/bearing, and one acquisition/loss pair per `(target, platform, episode)`. Separately assert that an active emitter is audible only while its private sensor-boundary distance is `<=1200 m`, even when its platform detection is retained by hysteresis at 1250 m.

- [ ] Step 2: Add exposure tests: carrier and mother ship are candidates; deployed/returning/waterborne-failed UUVs are candidates; onboard and recovered UUVs are absent. Verify no USV kind can validate.

- [ ] Step 3: Add engine/adversary tests proving a never-acquired platform first observed at 1201 m is absent from `platform_threats`, active emitters, communication exposure, trigger events, and detected IDs; the same platform at 1199 m is represented only by noisy local estimates. For an already-acquired platform at 1250 m, retain its platform threat but exclude its active ping from `audible_active_emitter_ids`.

- [ ] Step 4: Add a regression test proving a blue `BearingObservation` and `platform_observation` about the target never becomes an `AdversaryObservation`. Add a gate test proving the adversary provider is not called when local detections, local triggers, and local active emitters are all empty.

- [ ] Step 5: Run the focused tests.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_graph.py tests/agent/test_runtime_master_slave_adversary.py -q
```

- [ ] Step 6: Change `SubmarineInitialConfig` and `TargetEntity` fallback defaults from 5000 to 1200. Keep explicit non-default scenario values valid.

- [ ] Step 7: Implement `_target_local_detections[target_id]`, `_target_audible_active_emitters[target_id]`, and per-platform episode counters in the engine. Build candidate exposure from carrier/mothers and `_uuv_is_physically_exposed()`. Detection hysteresis affects the retained platform set only; active-ping audibility is recomputed against the configured range every observation cycle. Emit deterministic IDs:

```python
f"target_detection_acquired:{target_id}:{platform_id}:e{episode}"
f"target_detection_lost:{target_id}:{platform_id}:e{episode}"
```

- [ ] Step 8: Rewrite `build_adversary_inputs()` to consume only `LocalPlatformDetection` plus `audible_active_emitter_ids`. Remove the loop that reverses blue observations, the loop over all platform states, and all references to global USV relay state. Remove USV language from `ADVERSARY_SYSTEM_PROMPT`.

- [ ] Step 9: Update `AdversaryDecisionGate.should_request()` so a first decision requires local evidence and cooldown expiry alone never invokes the graph. Invoke only for a new target-local trigger, acquired/lost platform, materially changed local risk/range bucket, or a two-cycle material target-kinematic revision after the minimum cooldown. An unchanged retained detection produces no new LLM call.

- [ ] Step 10: Verify that adversary payload serialization contains no `position`, `position_xy`, `true_distance`, or gate-only field.

```bash
PYTHONPATH=src python -m pytest tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_graph.py tests/agent/test_runtime_master_slave_adversary.py tests/api/test_uuv_only_frame_contract.py -q
ruff check src/underwater_tracking/simulation/adversary_sensing.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/agent/nodes/adversary.py tests/simulation/test_adversary_local_sensing.py
mypy src/underwater_tracking/simulation/adversary_sensing.py src/underwater_tracking/domain/adversary_models.py
```

- [ ] Step 11: Commit.

```bash
git add src/underwater_tracking/simulation/adversary_sensing.py src/underwater_tracking/domain/adversary_models.py src/underwater_tracking/simulation/target.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/agent/nodes/adversary.py tests/simulation/test_adversary_local_sensing.py tests/agent/test_adversary_graph.py tests/agent/test_runtime_master_slave_adversary.py
git commit -m "fix: restrict target awareness to local sensing"
```

---

### Task 6: Replace binary region entry with Gaussian probability mass

**Files:**
- Create: `src/underwater_tracking/tracking/region_probability.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Create: `tests/tracking/test_region_probability.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Modify: `tests/simulation/test_uuv_only_carrier_group.py`

**Interface:**

```python
def gaussian_probability_in_axis_aligned_region(
    *,
    mean_xy: tuple[float, float],
    covariance_xy: tuple[tuple[float, float], tuple[float, float]],
    polygon_xy: Sequence[tuple[float, float]],
) -> float | None: ...
```

The function accepts only the current axis-aligned rectangular task regions, finite means, a symmetric positive-definite 2x2 covariance, and a nonzero rectangle. Use `scipy.integrate.quad` with the conditional normal CDF (`scipy.special.ndtr`) so correlated covariance is handled deterministically. Return `None` on invalid input or integration failure and clamp valid results to `[0, 1]`.

- [ ] Step 1: Write failing numerical tests for a tight distribution well inside (`>0.99`), far outside (`<0.01`), a boundary-centered distribution, correlated covariance, repeatability, invalid polygon, nonfinite values, nonsymmetric covariance, and non-positive-definite covariance.

- [ ] Step 2: Add controller tests proving two consecutive probabilities `>=0.70` are required, and that either a subthreshold or missing/invalid probability resets `entry_confirmations` to zero while holding the current lifecycle. The sequence `0.8, missing, 0.8` must remain `ACTIVE_SCAN` with one confirmation.

- [ ] Step 3: Run focused tests.

```bash
PYTHONPATH=src python -m pytest tests/tracking/test_region_probability.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py -q
```

- [ ] Step 4: Implement the integrator without Monte Carlo sampling. In `_mission_entry_probabilities()`, take mean/covariance only from the public belief/report and omit a region key if the integrator returns `None`. Remove mean-in-polygon 0/1 behavior for executable polygon plans.

- [ ] Step 5: Change `_apply_entry_observations()` to distinguish missing from numeric zero:

```python
if region_id not in probabilities:
    self._regions[region_id] = region.model_copy(update={"entry_confirmations": 0})
    continue
probability = _float(probabilities[region_id], default=None)
if probability is None:
    self._regions[region_id] = region.model_copy(update={"entry_confirmations": 0})
    continue
```

- [ ] Step 6: Verify numerical and lifecycle behavior.

```bash
PYTHONPATH=src python -m pytest tests/tracking/test_region_probability.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py -q
ruff check src/underwater_tracking/tracking/region_probability.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/runtime/mission_controller.py tests/tracking/test_region_probability.py
mypy src/underwater_tracking/tracking/region_probability.py
```

- [ ] Step 7: Commit.

```bash
git add src/underwater_tracking/tracking/region_probability.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/runtime/mission_controller.py tests/tracking/test_region_probability.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py
git commit -m "fix: use belief probability for region entry"
```

---

### Task 7: Require typed current-cycle evidence for region handoff

**Files:**
- Modify: `src/underwater_tracking/domain/mission_models.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `tests/domain/test_mission_models.py`
- Modify: `tests/runtime/test_mission_controller.py`
- Modify: `tests/simulation/test_uuv_only_carrier_group.py`

**Interface:**

```python
class AcceptedHandoffObservation(StrictModel):
    observation_id: str
    observer_uuv_id: str
    observed_at_s: int = Field(ge=0)


class HandoffEvidence(StrictModel):
    predecessor_region_id: str
    successor_region_id: str
    plan_revision: int = Field(ge=1)
    observation_cycle_s: int = Field(ge=0)
    required_uuv_ids: tuple[str, ...]
    deployed_uuv_ids: tuple[str, ...]
    healthy_uuv_ids: tuple[str, ...]
    passive_mode_uuv_ids: tuple[str, ...]
    accepted_observations: tuple[AcceptedHandoffObservation, ...]
    hard_guard_reasons: tuple[str, ...] = ()
    blocked_reason: str | None = None
```

`MissionController(..., group_min_size: int)` validates evidence. `cli._mission_controller_for()` passes `config.tracking.group_min_size`.

- [ ] Step 1: Add model invariant tests: unique IDs, every accepted observation's observer is a required member in passive mode, every `observed_at_s` equals `observation_cycle_s`, and blocked evidence cannot validate as complete. Event source IDs are derived exactly from `accepted_observations[*].observation_id`.

- [ ] Step 2: Replace controller boolean tests with typed cases. Each of stale plan revision, `observation_cycle_s != MissionController.sim_time_s`, wrong successor, missing required deployment, unhealthy member, non-passive member, fewer than `group_min_size` distinct current-cycle observers, and hard guard must prevent completion.

- [ ] Step 3: Add ordering tests proving complete evidence activates the successor group first, emits `handoff_completed`, then marks the predecessor `TRACKING_COMPLETED` and creates ordinary recovery work. Dedicated UUVs remain dedicated.

- [ ] Step 4: Add blocked tests proving permanent successor unavailability or predecessor resource exhaustion produces `DEGRADED` plus one `handoff_blocked` event containing plan revision and source IDs, never a fabricated `handoff_completed`.

- [ ] Step 5: Add an engine test that joins the current `GroupReport.belief.source_observation_ids` back to current-cycle `BearingObservation` records and creates `AcceptedHandoffObservation(observation_id, observer_uuv_id, observed_at_s)`. Rejected rays, old-cycle IDs/timestamps, active observations, and nonmembers do not count.

- [ ] Step 6: Run focused tests.

```bash
PYTHONPATH=src python -m pytest tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py -q
```

- [ ] Step 7: Implement `_mission_handoff_evidence(snapshot, reports, sim_time_s)` in the engine. Build required members from the current successor region, calculate health from controller resources plus physical deployment/mode, copy hard guards from the current successor report, and materialize the validated observation-ID-to-observer mapping while the engine still owns the current observation records.

- [ ] Step 8: Replace `handoff_ready` and `successor_passive_ready` observation keys with `handoff_evidence`. In `_apply_external_events`, `target_exit_predicted` may move a tracked predecessor into `HANDOFF_PENDING`, but it must not complete tracking or create recovery by itself.

- [ ] Step 9: Implement controller completion/degradation exactly once per plan/region episode. Insufficient evidence keeps `HANDOFF_PENDING` and the predecessor in passive mode.

- [ ] Step 10: Verify state transitions and event payloads.

```bash
PYTHONPATH=src python -m pytest tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py tests/agent/test_event_policy.py -q
ruff check src/underwater_tracking/domain/mission_models.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/simulation/engine.py tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py
```

- [ ] Step 11: Commit.

```bash
git add src/underwater_tracking/domain/mission_models.py src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/simulation/engine.py src/underwater_tracking/cli.py tests/domain/test_mission_models.py tests/runtime/test_mission_controller.py tests/simulation/test_uuv_only_carrier_group.py
git commit -m "fix: require effective observations for mission handoff"
```

---

### Task 8: Remove live USV contracts and render only physically waterborne UUVs

**Files:**
- Modify: `src/underwater_tracking/domain/__init__.py`
- Modify: `src/underwater_tracking/domain/agent_models.py`
- Modify: `src/underwater_tracking/domain/regional_models.py`
- Modify: `src/underwater_tracking/domain/slave_models.py`
- Modify: `src/underwater_tracking/domain/mission_adapters.py`
- Modify: `src/underwater_tracking/agent/prompts.py`
- Modify: `src/underwater_tracking/agent/nodes/slave.py`
- Modify: `src/underwater_tracking/agent/nodes/strategy.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Modify: `src/underwater_tracking/agent/nodes/verify.py`
- Modify: `src/underwater_tracking/agent/nodes/commit.py`
- Modify: `src/underwater_tracking/agent/nodes/regional_strategy.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/planning/regions.py`
- Modify: `src/underwater_tracking/planning/region_cap.py`
- Modify: `src/underwater_tracking/planning/regional_allocation.py`
- Modify: `src/underwater_tracking/planning/regional_validation.py`
- Modify: `src/underwater_tracking/planning/mission_validation.py`
- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RegionTimelineRow.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx`
- Modify: `src/underwater_tracking/ui/src/components/CanvasMap.test.ts`
- Modify: `src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx`
- Modify: `src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.test.tsx`
- Modify: `src/underwater_tracking/ui/src/types/regionalTasks.test.ts`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/legacy_frame_adapter.py`
- Modify: `tests/agent/test_central_graph.py`
- Modify: `tests/agent/test_regional_graph.py`
- Modify: `tests/agent/test_regional_plan_pipeline.py`
- Modify: `tests/agent/test_regional_strategy.py`
- Modify: `tests/agent/test_slave_graph.py`
- Modify: `tests/agent/test_strategy.py`
- Modify: `tests/agent/test_agent_models.py`
- Modify: `tests/agent/test_commit.py`
- Modify: `tests/domain/test_models.py`
- Modify: `tests/domain/test_regional_models.py`
- Modify: `tests/planning/test_regions.py`
- Modify: `tests/planning/test_region_cap.py`
- Modify: `tests/planning/test_regional_allocation.py`
- Modify: `tests/planning/test_regional_plan_validator.py`
- Modify: `tests/planning/test_regional_validation.py`
- Modify: `tests/api/test_uuv_only_frame_contract.py`
- Modify: `tests/api/test_replay_compatibility.py`

**Interfaces:**

```typescript
export function isWaterborneUuv(uuv: UUVView): boolean {
  return uuv.physically_exposed;
}

export function waterborneUuvs(frame: OperationalFrame): UUVView[] {
  return frame.uuvs.filter(isWaterborneUuv);
}
```

Current planning models/prompts remove assigned-USV and USV-relay branches. Current `OperationalFrame` removes `USVView`, `usvs`, live `usv_assignments`, and the `"usv"` branch of `RegionAssignmentView.platform_kind`. The Python current-frame model likewise has no serialized live USV collection. The legacy adapter reads old fields into a private compatibility model and drops them; legacy platform classes may remain inside legacy simulation/config modules but are not re-exported as current contracts.

- [ ] Step 1: Add failing negative contract tests proving current `TrackingPlan`, `PlanCommand`, master/slave/regional prompts, serialized planning payloads, and TypeScript regional-task mirrors contain no `USV`, `usv`, assigned-USV IDs, USV actions, relay policy, or mixed-platform branch. A current planning request that carries a legacy USV assignment must fail validation rather than silently influencing the plan.

- [ ] Step 2: Remove USV fields from current regional/slave strategy models, region generation/capping, allocation, validation, verification, commit, prompt schemas, optimizer inputs, central graph payloads, and current domain exports. Keep historical parsing in the legacy adapter only. Replace mixed-platform prompt language with UUV sensor mode, passive cooperation, carrier support, endurance, and communication constraints.

- [ ] Step 3: Add failing map-helper tests proving `physically_exposed=False` UUVs, including failed onboard units, do not enter camera bounds, hit testing, trails, labels, sprites, communication links, recovery links, target-detection links, or detected-platform highlighting. Deployed, returning, and failed-waterborne UUVs with `physically_exposed=True` remain visible.

- [ ] Step 4: Add component tests proving the carrier/sidebar inventory still lists onboard UUVs with owner, energy, mileage, and health. Map filtering must not mutate `frame.uuvs`.

- [ ] Step 5: Add frame tests proving a new live frame has no `usvs`/`usv_assignments`, while a historical replay containing them parses successfully and yields neither field in the current frame. Cover `frame_builder`, top-level domain exports, sidebar, timeline rows, region overlays, and assistant region graphs so no stale current-frame consumer remains.

- [ ] Step 6: Add a non-default detection-range test proving the circle reads `frame.adversary.detection_range_m` or target `detection_range_m`; change only the compatibility fallback from 1800 to 1200.

- [ ] Step 7: Run frontend/backend focused tests.

```bash
npm --prefix src/underwater_tracking/ui test -- src/types/regionalTasks.test.ts src/components/CanvasMap.test.ts src/components/CarrierStatusPanel.test.tsx src/components/RightSidebar.test.tsx src/components/RegionTimelinePanel.test.tsx src/components/map/RegionOverlay.test.tsx src/components/assistant/RegionTaskGraph.test.tsx src/components/assistant/AssignmentPanel.test.tsx
PYTHONPATH=src python -m pytest tests/agent/test_agent_models.py tests/agent/test_commit.py tests/agent/test_central_graph.py tests/agent/test_regional_graph.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_regional_strategy.py tests/agent/test_slave_graph.py tests/agent/test_strategy.py tests/domain/test_models.py tests/domain/test_regional_models.py tests/planning/test_regions.py tests/planning/test_region_cap.py tests/planning/test_regional_allocation.py tests/planning/test_regional_plan_validator.py tests/planning/test_regional_validation.py tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py -q
```

- [ ] Step 8: Compute `const visibleUuvs = waterborneUuvs(frame)` once per render and use it consistently in every spatial/render loop. When a selected non-exposed UUV disappears from the map, clear only the map selection; do not remove it from the inventory panel.

- [ ] Step 9: Remove live UI USV types, sidebar blocks, draw helpers, sensor/communication range helpers, timeline/region-graph assignment branches, and tests fixtures. Do not add a hidden empty USV array to satisfy types.

- [ ] Step 10: Keep all text within current compact panels, preserve existing visual language, and make no unrelated style redesign. The expected initial viewport contains the moving carrier group, public target estimate, and task overlays; no UUV ring appears until deployment.

- [ ] Step 11: Verify build and tests.

```bash
npm --prefix src/underwater_tracking/ui test
npm --prefix src/underwater_tracking/ui run build
PYTHONPATH=src python -m pytest tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py -q
ruff check src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/legacy_frame_adapter.py tests/api
```

- [ ] Step 12: Commit.

```bash
git add src/underwater_tracking/domain/__init__.py src/underwater_tracking/domain/agent_models.py src/underwater_tracking/domain/regional_models.py src/underwater_tracking/domain/slave_models.py src/underwater_tracking/domain/mission_adapters.py src/underwater_tracking/agent/prompts.py src/underwater_tracking/agent/nodes/slave.py src/underwater_tracking/agent/nodes/strategy.py src/underwater_tracking/agent/nodes/optimize.py src/underwater_tracking/agent/nodes/verify.py src/underwater_tracking/agent/nodes/commit.py src/underwater_tracking/agent/nodes/regional_strategy.py src/underwater_tracking/agent/graphs/central.py src/underwater_tracking/planning/regions.py src/underwater_tracking/planning/region_cap.py src/underwater_tracking/planning/regional_allocation.py src/underwater_tracking/planning/regional_validation.py src/underwater_tracking/planning/mission_validation.py src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/legacy_frame_adapter.py src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/types/regionalTasks.test.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/RightSidebar.tsx src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx src/underwater_tracking/ui/src/components/RegionTimelineRow.tsx src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/components/CarrierStatusPanel.test.tsx src/underwater_tracking/ui/src/components/RightSidebar.test.tsx src/underwater_tracking/ui/src/components/RegionTimelinePanel.test.tsx src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.test.tsx tests/agent/test_agent_models.py tests/agent/test_commit.py tests/agent/test_central_graph.py tests/agent/test_regional_graph.py tests/agent/test_regional_plan_pipeline.py tests/agent/test_regional_strategy.py tests/agent/test_slave_graph.py tests/agent/test_strategy.py tests/domain/test_models.py tests/domain/test_regional_models.py tests/planning/test_regions.py tests/planning/test_region_cap.py tests/planning/test_regional_allocation.py tests/planning/test_regional_plan_validator.py tests/planning/test_regional_validation.py tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py
git commit -m "fix: show uuvs only after physical deployment"
```

---

### Task 9: Add idempotent periodic situation summaries to the memory source stream

**Files:**
- Create: `src/underwater_tracking/memory/situation_summary.py`
- Modify: `src/underwater_tracking/persistence/events.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/agent/event_policy.py`
- Create: `tests/memory/test_situation_summary.py`
- Create: `tests/persistence/test_events.py`
- Modify: `tests/memory/test_source_reader.py`
- Modify: `tests/memory/test_worker.py`
- Modify: `tests/agent/test_background_cycle.py`
- Modify: `tests/agent/test_event_policy.py`

**Interfaces:**

```python
class PeriodicSituationSummary(StrictModel):
    scenario_id: str
    sim_time_s: int
    plan_version: int
    region_states: tuple[RegionSummary, ...]
    carrier_states: tuple[CarrierSummary, ...]
    uuv_counts: UUVCountSummary
    target_estimates: tuple[PublicTargetSummary, ...]
    changes_since_previous: tuple[SituationChange, ...]
    source_event_ids: tuple[str, ...]


def build_periodic_situation_summary(
    situation: SituationSnapshot,
    mission: MissionSnapshot,
    source_events: Sequence[RuntimeEvent],
    previous: PeriodicSituationSummary | None,
) -> tuple[PeriodicSituationSummary, RuntimeEvent]: ...


class EventRepository:
    def append_if_absent(
        self,
        *,
        event_id: str,
        event_type: str,
        scenario_id: str,
        sim_time_s: int,
        payload: dict[str, Any],
        target_id: str | None = None,
        severity: str = "info",
    ) -> int | None: ...


class PeriodicSituationSummaryWriter:
    def __init__(self, database_path: Path, *, queue_limit: int = 64) -> None: ...
    def start(self) -> None: ...
    def submit(self, event: RuntimeEvent) -> bool: ...  # non-blocking
    def stop(self, *, timeout: float = 5.0) -> bool: ...
```

The event ID is exactly `periodic_situation_summary:{scenario_id}:{sim_time_s}`. Its payload includes both a bounded human-readable `summary` for the current source reader and typed public fields for audit. It excludes target truth, truth positions, private local-sensing gate distances/coordinates, raw model reasoning, secrets, and full historical payloads.

- [ ] Step 1: Add serialization tests for regions, carrier mission/return state, UUV mode/deployment/energy/mileage/health counts, public target estimate quality/intent/prediction state, and bounded sorted source event IDs. Pass the previous typed summary explicitly and assert deterministic `changes_since_previous` for lifecycle, mode, health, quality, intent, and prediction-revision changes; the first summary has an empty delta. Assert forbidden key fragments (`truth`, `private`, `chain_of_thought`, `gate_distance`) are absent recursively.

- [ ] Step 2: Add repository tests proving two `append_if_absent()` calls for the same event ID create one row and return `row_id, None`; a different simulation time creates a second row.

- [ ] Step 3: Add scheduling tests proving summaries are generated once when simulation time crosses each `progress_report_s` boundary and are not generated every physical tick. The scheduling check runs for every observation delivered to `on_situation`, even while a plan-LLM background cycle is busy and newer situations are being coalesced in its mailbox. Accumulate bounded unique source event IDs between summary boundaries so mailbox coalescing cannot erase the period's evidence references.

- [ ] Step 4: Add writer-isolation tests using a blocking/failing event repository: `submit()` returns without waiting for SQLite, physical stepping continues, writer metrics expose degradation, and summaries can be retried with the same IDs. Block the writer across at least three `progress_report_s` boundaries and prove the immutable boundary snapshots are later persisted in order with their original simulation times and deltas. The writer opens and closes its own `EventRepository` on its daemon thread; it never shares the physics-thread SQLite connection.

- [ ] Step 5: Add source-reader/worker tests proving the persisted event becomes one durable work item, the source cursor advances only after that work item is committed, duplicate polling does not duplicate work, and restart resumes after the last committed cursor.

- [ ] Step 6: Add event-policy tests proving `periodic_situation_summary` is audit/memory-only with `plan_impact=False`; `handoff_blocked` and `carrier_rendezvous_infeasible` remain eligible for plan-impact evaluation.

- [ ] Step 7: Run focused tests.

```bash
PYTHONPATH=src python -m pytest tests/memory/test_situation_summary.py tests/persistence/test_events.py tests/memory/test_source_reader.py tests/memory/test_worker.py tests/agent/test_background_cycle.py tests/agent/test_event_policy.py -q
```

- [ ] Step 8: Implement the pure summary builder, idempotent repository insert using SQLite `INSERT ... ON CONFLICT(event_id) DO NOTHING`, and the bounded `PeriodicSituationSummaryWriter`. Keep payload sizes bounded and all tuples deterministically sorted. Queue saturation must retain every already accepted immutable boundary and return `False`; it must not replace an older event with a newer summary.

- [ ] Step 9: Own the writer in `_AgentLoop`: start it during `attach()`, stop/drain it during `close()`, and call a non-blocking `_submit_due_periodic_summary(situation)` at the start of `on_situation()` before choosing synchronous versus background carrier execution. Maintain `_periodic_summary_source_ids`, `_last_built_periodic_summary`, and an ordered `_pending_periodic_summaries: deque[tuple[PeriodicSituationSummary, RuntimeEvent]]`; build and retain an immutable pair for every crossed boundary, then flush oldest-first whenever the writer accepts work. Bound this producer backlog at 64 boundaries (more than 10 simulated hours at the default interval); on overflow retain the oldest entries, reject the newest, increment an explicit `periodic_summary_backlog_overflow` degradation counter/event, and never reconstruct an old timestamp from a newer snapshot. Read the current immutable `MissionSnapshot` only to build the public event; never invoke the memory LLM or plan LLM from this method. Summary scheduling is independent of `_BackgroundCarrierCycle`, chat credentials, and LLM retry state.

- [ ] Step 10: Let `MemorySourceReader` consume the existing payload `summary` and source IDs. Apart from the bounded persistence writer mailbox, do not add a second memory work queue, direct MemoryWorker call, or special LLM prompt in `SimulationEngine`.

- [ ] Step 11: Verify focused memory behavior and existing assistant views.

```bash
PYTHONPATH=src python -m pytest tests/memory tests/agent/test_background_cycle.py tests/agent/test_event_policy.py tests/api/test_memory_routes.py -q
ruff check src/underwater_tracking/memory/situation_summary.py src/underwater_tracking/persistence/events.py src/underwater_tracking/memory/source_reader.py src/underwater_tracking/cli.py src/underwater_tracking/agent/event_policy.py tests/memory tests/persistence/test_events.py
```

- [ ] Step 12: Commit.

```bash
git add src/underwater_tracking/memory/situation_summary.py src/underwater_tracking/persistence/events.py src/underwater_tracking/memory/source_reader.py src/underwater_tracking/cli.py src/underwater_tracking/agent/event_policy.py tests/memory/test_situation_summary.py tests/persistence/test_events.py tests/memory/test_source_reader.py tests/memory/test_worker.py tests/agent/test_background_cycle.py tests/agent/test_event_policy.py
git commit -m "feat: persist truth safe periodic memory summaries"
```

---

### Task 10: Prove the complete execution timeline and visual result

**Files:**
- Create: `tests/integration/test_uuv_initialization_local_perception.py`
- Modify: `tests/integration/test_uuv_only_runtime_entrypoints.py`
- Modify: `tests/integration/test_uuv_only_8h_replay_acceptance.py`
- Create: `tests/e2e/uuv-live-timeline.spec.ts`
- Create: `docs/superpowers/reports/2026-08-21-uuv-initialization-local-perception-acceptance.md`

The acceptance run must use the real root/default scenario, not a test-only engine constructor that bypasses configuration or live guards.

- [ ] Step 1: Add an integration timeline test that records frames/events and asserts this order:

```text
initial: 1 carrier + 3 mothers + 12 onboard UUVs + 0 USVs
carrier group moves on configured route
mother mission accepted and mother detaches
mother reaches perimeter deployment stop
uuv_deployed occurs and only then UUV is waterborne/map-visible
multiple UUVs follow distinct serpentine routes
two-cycle probabilistic entry changes ACTIVE_SCAN to PASSIVE_TRACK
typed successor observations complete handoff
old group receives recovery work
mother recovers UUVs and rendezvous with moving slot
carrier_returned_to_fleet occurs once for the mission revision
```

- [ ] Step 2: Add episode-aware local-perception assertions throughout that run: a never-acquired platform outside 1200 m does not appear in adversary evidence; an acquired platform may remain as a retained threat through 1300 m; a retained platform outside 1200 m contributes no active-ping evidence; loss occurs once beyond 1300 m; blue observations never leak; and the target LLM is not invoked when the target-local evidence fingerprint is unchanged.

- [ ] Step 3: Add persistence assertions: new operational frames contain no USV fields, periodic summaries exist and are truth-safe/idempotent, Memory Steam can incrementally read the resulting processing events, and periodic summaries do not create a new plan version.

- [ ] Step 4: Run backend unit and integration suites.

```bash
PYTHONPATH=src python -m pytest \
  tests/config \
  tests/domain/test_mission_models.py \
  tests/simulation \
  tests/tracking/test_region_probability.py \
  tests/runtime/test_mission_controller.py \
  tests/agent/test_adversary_graph.py \
  tests/agent/test_runtime_master_slave_adversary.py \
  tests/agent/test_background_cycle.py \
  tests/memory \
  tests/api \
  tests/integration/test_uuv_initialization_local_perception.py \
  tests/integration/test_uuv_only_runtime_entrypoints.py -q
```

- [ ] Step 5: Run static checks and the complete UI suite.

```bash
ruff check main.py src tests
mypy src/underwater_tracking
npm --prefix src/underwater_tracking/ui test
npm --prefix src/underwater_tracking/ui run build
```

- [ ] Step 6: Start the real local application with `python main.py` using the default 60x demo pacing, seed 42, and the explicit default UUV-only config. Use the printed API/UI ports and run the new non-mocked `uuv-live-timeline.spec.ts` against that `PLAYWRIGHT_BASE_URL` at `1440x900` and `390x844`. Unlike `command-center.spec.ts`, the new test must not intercept snapshot, replay, WebSocket, or memory APIs. Give this real timeline test an explicit bounded timeout of 10 minutes and advance assertions by polling frame/event state, never fixed sleeps. Capture initial, post-deployment, handoff, and returned-to-fleet screenshots. Verify with canvas pixel sampling that sprites/routes are nonblank and with DOM assertions that no UI text or payload contains `USV`.

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:<printed-ui-port> npm --prefix src/underwater_tracking/ui run test:e2e -- uuv-live-timeline.spec.ts
```

- [ ] Step 7: Compare the captures with `ui-review-current-open.png`, `ui-review-later.png`, and `ui-review-sidebar.png`. Record objective differences in the acceptance report: carrier/target framing, no initial UUV circle, no label overlap, inventory still listing onboard UUVs, target detection circle scale, and post-deployment serpentine distribution.

- [ ] Step 8: Run the opt-in 8-hour accelerated replay only after the short acceptance is green. Do not silently enable the environment flag in ordinary CI.

```bash
UNDERWATER_TRACKING_RUN_8H=1 PYTHONPATH=src python -m pytest tests/integration/test_uuv_only_8h_replay_acceptance.py -m long_running -q
```

- [ ] Step 9: Write the acceptance report with commands, test counts, screenshot paths, seed/config, known residual risks, and explicit evidence for every requirement in the design specification.

- [ ] Step 10: Commit the acceptance artifacts, excluding temporary run databases, JSONL output, node modules, and the user's existing review screenshots.

```bash
git add tests/integration/test_uuv_initialization_local_perception.py tests/integration/test_uuv_only_runtime_entrypoints.py tests/integration/test_uuv_only_8h_replay_acceptance.py tests/e2e/uuv-live-timeline.spec.ts docs/superpowers/reports/2026-08-21-uuv-initialization-local-perception-acceptance.md
git commit -m "test: verify uuv deployment and local perception timeline"
```

---

## Requirement Traceability

| Requirement | Primary implementation task | Acceptance evidence |
| --- | --- | --- |
| No USV in the algorithm | Tasks 1, 5, 8 | Live config/frame/input/UI assertions; legacy replay discard test |
| One carrier, three mothers, fixed ownership | Tasks 1, 2 | Explicit roster and inventory tests |
| Carrier group follows configured route | Tasks 3, 4 | Patrol/slot/no-teleport tests |
| UUVs deploy at task area, not at startup | Tasks 2, 4, 8 | Timeline and waterborne visibility tests |
| No circular UUV initialization | Tasks 1, 2, 8 | Initial screenshot and distinct serpentine routes |
| Target starts near the task corridor but outside local sensing | Task 1 | Default geometry validation |
| Target senses only a nearby area | Task 5 | 1200/1300 m boundary and payload tests |
| No reversal of blue observations | Task 5 | Adversary payload leakage regression test |
| Probabilistic region entry | Task 6 | Gaussian integration and two-cycle state tests |
| Effective handoff before recovery | Task 7 | Typed evidence, blocked, and ordering tests |
| Task/resource rotations remain distinct | Tasks 2, 7 | Dedicated-mode and recovery regression tests |
| Memory stays asynchronous | Task 9 | Blocking/failure and cursor-restart tests |
| Periodic truth-safe memory input | Task 9 | Idempotence, forbidden-field, and event-policy tests |
| Smart Assistant/Memory UI retained | Tasks 8, 10 | Existing suites plus Playwright acceptance |

## Execution Gate

This document authorizes no production-code changes by itself. Implementation begins only after the user reviews and approves this plan. At approval time, execute tasks in order because Tasks 2-10 depend on contracts established by earlier tasks. Each task starts with its failing tests, ends with the listed verification, and is committed independently so review or rollback can stop at a coherent boundary.
