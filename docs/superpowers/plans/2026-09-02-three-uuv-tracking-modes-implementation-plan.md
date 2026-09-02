# Three-UUV Tracking Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 UUV-only live 场景改造成四个动态正方形区域、每区三艇主动覆盖、全组被动跟踪、区域无缝交接和唯一组持续跟踪的一体化前后端可视系统。

**Architecture:** 保留现有 IMM 预测和 ExecutionCoordinator 提交边界，将四区域计划与 MissionController 持有的部署实例/跟踪所有权组合成唯一的 OperationalExecutionSnapshot。SimulationEngine 只执行该快照中的组级模式与边界运动，OperationalFrame 原样投影权威尺寸和过渡状态，React UI 只渲染真实 frame，不维护平行状态机或显示专用几何。

**Tech Stack:** Python 3.11、Pydantic 2、Shapely、NumPy、pytest、FastAPI/WebSocket/JSONL Replay、React 18、TypeScript 5.9、Vitest、Playwright。

## Review Gate

本文档完成后停止，不执行任何业务代码修改。只有用户明确审核通过本实施方案后，才能从
Task 1 开始实施；审核意见必须先回写本文档并重新完成计划自审。

## Global Constraints

- UUV-only 场景始终包含 4 个边长 2000m 的动态正方形任务区域。
- 每个 task group instance 严格包含 3 艘 UUV。
- 目标探测半径为 1000m；UUV 主动和被动探测半径均为 600m。
- 区域扫描阶段三艇全部主动；区域和专属跟踪阶段三艇全部被动。
- 入区概率阈值为 0.70，连续两个观测周期确认，不使用额外边界 buffer。
- 最大里程为 50000m；专属组三艇任一成员剩余里程 <=7000m 时触发整组退出。
- 模式 2 只能锁定当前区域被动跟踪所有者，不能手动提前退出。
- 新组必须先形成有效被动观测并获得所有权，旧组随后从区域边界退出。
- 不实施母舰回收、储备、维护冷却、补给或健康故障调度；退出实例立即可用且里程重置。
- 稳态 regional 为 12 艘，稳态 dedicated 为 3 艘；并行替换允许最多 24 艘。
- HTTP、WebSocket、JSONL、Replay 和 UI 必须使用同一个不可变 OperationalFrame。
- 非 UUV-only 场景和 legacy replay 只能通过兼容 adapter 保持原行为，不能驱动新 live 状态机。

---

## File Structure

**Create**

- src/underwater_tracking/runtime/task_group_instances.py：创建三艇部署实例、生成稳定 deployment-aware ID、合并每槽位待执行区域 revision。
- src/underwater_tracking/ui/src/state/executionSelectors.ts：从 frame 选择真实可见 UUV、当前 owner、区域/组过渡和统计，避免组件各自推断。
- tests/runtime/test_task_group_instances.py：实例工厂和每槽位过渡队列测试。
- src/underwater_tracking/ui/src/state/executionSelectors.test.ts：前端权威选择器测试。
- src/underwater_tracking/ui/e2e/three-uuv-tracking-modes.spec.ts：真实/可控 frame 的视觉状态序列测试。
- tests/acceptance/test_three_uuv_tracking_modes.py：后端完整模式 1/模式 2 验收。

**Modify**

- src/underwater_tracking/config/models.py 和 configs/scenario/uuv_only_single_target.yaml：唯一 tracking policy。
- configs/environment_uuv_only.yaml 和 configs/sensors.yaml：1000m/600m 物理配置并与 policy 交叉校验。
- src/underwater_tracking/domain/execution_models.py：正方形区域、三艇实例、tracking control 和过渡 cardinality。
- src/underwater_tracking/domain/mission_models.py：组级生命周期和部署实例字段。
- src/underwater_tracking/planning/region_baseline.py：固定 2000m 正方形。
- src/underwater_tracking/planning/task_groups.py：三艇、无固定声呐角色、无 reserve 驱动。
- src/underwater_tracking/planning/coverage.py：复用蛇形算法并暴露覆盖完整性计算。
- src/underwater_tracking/runtime/execution_snapshot_factory.py：生成四个初始三艇组和新的 policy/control 字段。
- src/underwater_tracking/runtime/execution_coordinator.py：CAS 提交后同步 MissionController 的 runtime projection。
- src/underwater_tracking/runtime/mission_controller.py：模式 1、区域替换、模式 2 和所有权状态机。
- src/underwater_tracking/simulation/engine.py：动态部署实例、三艇路径、边界动画、里程阈值和有效被动证据。
- src/underwater_tracking/agent/nodes/directives.py、src/underwater_tracking/agent/runtime.py、src/underwater_tracking/agent/prompts.py：仅锁定当前 owner，禁止 regional 手动释放。
- src/underwater_tracking/domain/event_registry.py：新结构化事件。
- src/underwater_tracking/cli.py：live 状态同步、事件持久化和 frame 发布。
- src/underwater_tracking/domain/ui_models.py 和 src/underwater_tracking/api/frame_builder.py：新 ExecutionView/UUVView 投影。
- src/underwater_tracking/api/legacy_frame_adapter.py、src/underwater_tracking/api/replay.py：旧两艇 frame 只读兼容。
- src/underwater_tracking/ui/src/types/frames.ts：与 Pydantic schema 对齐。
- src/underwater_tracking/ui/src/components/CanvasMap.tsx：真实区域/探测范围/全部部署实例。
- src/underwater_tracking/ui/src/components/map/geometry.ts、camera.ts、RegionOverlay.tsx：删除显示扩张。
- src/underwater_tracking/ui/src/components/regionTimeline.ts、RegionTimelinePanel.tsx、RightSidebar.tsx：owner、生命周期和过渡计数。
- src/underwater_tracking/verification/uuv_tracking_coverage_runner.py、uuv_tracking_coverage_audit.py、uuv_tracking_coverage_render.py：真实 live 证据和视频。

---

### Task 1: 建立单一 Tracking Policy 配置

**Files:**
- Modify: src/underwater_tracking/config/models.py
- Modify: configs/scenario/uuv_only_single_target.yaml
- Modify: configs/environment_uuv_only.yaml
- Modify: configs/sensors.yaml
- Test: tests/config/test_uuv_only_config.py

**Interfaces:**
- Produces: TrackingPolicyConfig，字段与 Global Constraints 完全一致。
- Consumes: 现有 ScenarioConfig、EnvironmentConfig、SensorCatalogConfig。
- Later consumers: region_baseline、snapshot factory、MissionController、SimulationEngine、frame_builder。

- [ ] **Step 1: 写严格配置失败测试**

在 tests/config/test_uuv_only_config.py 增加：

~~~python
def test_uuv_only_tracking_policy_is_single_validated_contract() -> None:
    config = load_app_config(Path("configs/scenario/uuv_only_single_target.yaml"))
    policy = config.scenario.tracking_policy
    assert policy.region_count == 4
    assert policy.task_group_size == 3
    assert policy.task_region_side_m == 2_000.0
    assert policy.target_detection_radius_m == 1_000.0
    assert policy.uuv_active_detection_radius_m == 600.0
    assert policy.uuv_passive_detection_radius_m == 600.0
    assert policy.region_entry_probability_threshold == 0.70
    assert policy.region_transition_confirm_cycles == 2
    assert policy.max_uuv_mileage_m == 50_000.0
    assert policy.dedicated_release_remaining_mileage_m == 7_000.0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"region_count": 3}, "region_count must equal 4"),
        ({"task_group_size": 2}, "task_group_size must equal 3"),
        ({"task_region_side_m": 1_000.0}, "region side must exceed target detection"),
        ({"target_detection_radius_m": 600.0}, "target detection must exceed UUV"),
        ({"dedicated_release_remaining_mileage_m": 50_000.0}, "release threshold"),
    ],
)
def test_tracking_policy_rejects_invalid_invariants(update, message) -> None:
    payload = TrackingPolicyConfig().model_dump()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        TrackingPolicyConfig.model_validate(payload)
~~~

- [ ] **Step 2: 运行配置测试并确认旧模型失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/config/test_uuv_only_config.py -q

Expected: FAIL，ScenarioConfig 尚无 tracking_policy，旧配置仍返回 5000m 和 3500m/4500m。

- [ ] **Step 3: 实现 TrackingPolicyConfig 和跨配置校验**

在 config/models.py 添加冻结严格模型：

~~~python
class TrackingPolicyConfig(StrictModel):
    region_count: Literal[4] = 4
    task_group_size: Literal[3] = 3
    task_region_side_m: PositiveFloat = 2_000.0
    target_detection_radius_m: PositiveFloat = 1_000.0
    uuv_active_detection_radius_m: PositiveFloat = 600.0
    uuv_passive_detection_radius_m: PositiveFloat = 600.0
    region_entry_probability_threshold: float = Field(0.70, gt=0, le=1)
    region_transition_confirm_cycles: int = Field(2, ge=1)
    max_uuv_mileage_m: PositiveFloat = 50_000.0
    dedicated_release_remaining_mileage_m: PositiveFloat = 7_000.0

    @model_validator(mode="after")
    def validate_geometry_and_range(self) -> "TrackingPolicyConfig":
        if self.task_region_side_m <= self.target_detection_radius_m:
            raise ValueError("region side must exceed target detection radius")
        if self.target_detection_radius_m <= max(
            self.uuv_active_detection_radius_m,
            self.uuv_passive_detection_radius_m,
        ):
            raise ValueError("target detection must exceed UUV detection radius")
        if self.dedicated_release_remaining_mileage_m >= self.max_uuv_mileage_m:
            raise ValueError("release threshold must be below maximum mileage")
        return self
~~~

将 ScenarioConfig.tracking_policy 设置为 default_factory，并在 AppConfig validator 中校验 UUV-only
environment 的 target detection range 和 uuv_dual_sonar 的 active/passive range 与 policy 相等。
保留旧 ScenarioConfig 字段供非 UUV-only/legacy 读取，但 UUV-only controller 不再读取
region_entry_buffer_m 或 resource_warning_mileage_fraction。

- [ ] **Step 4: 更新 YAML 为确认值**

scenario YAML 写入嵌套 tracking_policy，设置 region_entry_buffer_m 为 0 作为兼容值；environment
目标 detection_range_m 改为 1000；sensor profile 的 active_source_range_m、
active_receive_range_m 和 passive_range_m 均改为 600。

- [ ] **Step 5: 运行测试和静态检查**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/config/test_uuv_only_config.py -q

Run: conda run --no-capture-output -n underwater-tracking python -m ruff check src/underwater_tracking/config tests/config/test_uuv_only_config.py

Expected: PASS。

- [ ] **Step 6: 提交**

~~~bash
git add src/underwater_tracking/config/models.py configs/scenario/uuv_only_single_target.yaml configs/environment_uuv_only.yaml configs/sensors.yaml tests/config/test_uuv_only_config.py
git commit -m "feat: define three-uuv tracking policy"
~~~

---

### Task 2: 迁移权威执行领域模型

**Files:**
- Modify: src/underwater_tracking/domain/execution_models.py
- Modify: src/underwater_tracking/domain/mission_models.py
- Test: tests/domain/test_execution_models.py
- Test: tests/runtime/test_mission_controller.py

**Interfaces:**
- Produces: TaskGroupInstance、TaskGroupLifecycle、GroupSensorMode、TrackingControlState。
- Produces: OperationalExecutionSnapshot.task_groups 为 1..8 个 runtime instances。
- Produces: ExecutionRegion.center、side_length_m 和严格正方形校验。

- [ ] **Step 1: 写正方形和三艇实例失败测试**

~~~python
def test_execution_region_requires_exact_configured_square() -> None:
    region = execution_region(
        center=(1_000.0, 2_000.0),
        side_length_m=2_000.0,
        geometry=((0.0, 1_000.0), (2_000.0, 1_000.0),
                  (2_000.0, 3_000.0), (0.0, 3_000.0)),
    )
    assert region.center == (1_000.0, 2_000.0)
    assert region.side_length_m == 2_000.0


def test_task_group_instance_requires_exactly_three_members() -> None:
    group = TaskGroupInstance(
        group_instance_id="T1:task:01:deploy:000004",
        target_id="T1",
        region_id="T1:task:01",
        deployment_revision=4,
        member_uuv_ids=("T1:01:4:1", "T1:01:4:2", "T1:01:4:3"),
        lifecycle=TaskGroupLifecycle.ENTERING,
        sensor_mode=GroupSensorMode.ACTIVE,
        ownership_status="candidate",
        reason="initial_deployment",
        evidence_ids=("plan:4",),
    )
    assert len(group.member_uuv_ids) == 3
~~~

再添加 2/4 成员拒绝、active/passive 固定角色字段拒绝、重复成员拒绝、非正方形 geometry 拒绝测试。

- [ ] **Step 2: 写 snapshot cardinality 和 owner 失败测试**

覆盖 regional 稳态 4 组、dedicated 稳态 1 组、四槽位并行 replacement 8 组、错误的 9 组、
同一实例跨组重复成员、owner 不存在、两个 owner 等情况。

~~~python
def test_snapshot_accepts_parallel_four_slot_replacement() -> None:
    snapshot = execution_snapshot(
        groups=tuple(
            group(slot=slot, phase=phase)
            for slot in range(1, 5)
            for phase in ("entering", "exiting")
        ),
        tracking_control=TrackingControlState(
            mode="regional",
            tracking_owner_group_id="T1:task:01:deploy:2",
        ),
    )
    assert len(snapshot.task_groups) == 8
~~~

- [ ] **Step 3: 运行领域测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/domain/test_execution_models.py -q

Expected: FAIL，现有模型固定 4 个两艇组且要求 active/passive role IDs。

- [ ] **Step 4: 实现新枚举和模型**

~~~python
class TaskGroupLifecycle(str, Enum):
    ENTERING = "entering"
    ACTIVE_SCAN = "active_scan"
    PASSIVE_TRACK = "passive_track"
    DEDICATED_TRACK = "dedicated_track"
    DEDICATED_RELEASE_PENDING = "dedicated_release_pending"
    EXITING = "exiting"
    DISAPPEARED = "disappeared"


class GroupSensorMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    OFF = "off"


class TrackingControlState(ExecutionModel):
    mode: Literal["regional", "dedicated"] = "regional"
    tracking_owner_group_id: str | None = None
    pending_successor_group_id: str | None = None
    dedicated_release_triggered_at_m: float | None = Field(default=None, ge=0)
    source_event_ids: tuple[str, ...] = ()
~~~

TaskGroupInstance validator 必须执行 exact-three、唯一成员、lifecycle/sensor 合法组合：
ACTIVE_SCAN 只能 ACTIVE；PASSIVE_TRACK/DEDICATED_TRACK/RELEASE_PENDING 只能 PASSIVE；
EXITING 可保持之前传感器状态或 OFF；DISAPPEARED 必须 OFF。

- [ ] **Step 5: 更新 MissionSnapshot 领域投影**

MissionSnapshot 增加 task_groups、tracking_control、pending_region_revisions；旧
dedicated_target_by_uuv、carrier_missions 和 reserve 字段仅保留 legacy adapter 输入，不再作为
UUV-only 新状态机的 owner。

- [ ] **Step 6: 运行领域和序列化测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/domain/test_execution_models.py tests/api/test_execution_frame_contract.py -q

Expected: PASS，新 JSON schema 不包含 active_verifier_uuv_id/passive_tracker_uuv_id。

- [ ] **Step 7: 提交**

~~~bash
git add src/underwater_tracking/domain/execution_models.py src/underwater_tracking/domain/mission_models.py tests/domain/test_execution_models.py tests/runtime/test_mission_controller.py
git commit -m "refactor: model three-uuv deployment instances"
~~~

---

### Task 3: 固定四个 2000m 正方形并验证三艇覆盖

**Files:**
- Modify: src/underwater_tracking/planning/region_baseline.py
- Modify: src/underwater_tracking/planning/coverage.py
- Test: tests/planning/test_region_baseline.py
- Test: tests/simulation/test_task_group_waypoints.py

**Interfaces:**
- Consumes: TrackingPolicyConfig.task_region_side_m 和现有 IMM centerline。
- Produces: square_geometry(center, side_length_m)。
- Produces: coverage_gap_area_m2(region, routes, detection_radius_m)。

- [ ] **Step 1: 写四个固定正方形失败测试**

~~~python
def test_baseline_emits_four_exact_two_kilometre_squares() -> None:
    baseline = build_four_region_baseline(
        _accepted(status="valid", regime="imm"),
        target_id="T1",
        execution_revision=7,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=MAP_BOUNDS,
        task_region_side_m=2_000.0,
    )
    assert len(baseline.regions) == 4
    for region in baseline.regions:
        polygon = Polygon(region.geometry)
        min_x, min_y, max_x, max_y = polygon.bounds
        assert max_x - min_x == pytest.approx(2_000.0)
        assert max_y - min_y == pytest.approx(2_000.0)
        assert polygon.area == pytest.approx(4_000_000.0)
~~~

同时验证中心仍按预测中心线排序、ID/相邻拓扑稳定、geometry revision 只在中心变化时递增。

- [ ] **Step 2: 写三艇全覆盖失败测试**

~~~python
def test_three_serpentine_routes_cover_every_point_of_square() -> None:
    square = ((0.0, 0.0), (2_000.0, 0.0),
              (2_000.0, 2_000.0), (0.0, 2_000.0))
    routes = serpentine_coverage_waypoints_by_uuv(
        square,
        ("U1", "U2", "U3"),
        detection_radius_m=600.0,
    )
    assert set(routes) == {"U1", "U2", "U3"}
    assert coverage_gap_area_m2(square, routes, 600.0) <= 1e-6
~~~

- [ ] **Step 3: 运行规划测试并确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/planning/test_region_baseline.py tests/simulation/test_task_group_waypoints.py -q

Expected: FAIL，region baseline 仍输出可变 polygon，coverage helper 尚不存在。

- [ ] **Step 4: 实现正方形规范化**

~~~python
def square_geometry(center: Point2, side_length_m: float) -> tuple[Point2, ...]:
    half = side_length_m / 2.0
    cx, cy = center
    return (
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    )
~~~

沿用现有 centerline 分段选择区域中心；只替换 geometry 构造和 map-bound clamp。若地图无法容纳
四个完整 2000m 正方形，返回明确 planning failure，不缩小区域。

- [ ] **Step 5: 复用现有蛇形算法并加入覆盖审计**

让 serpentine_coverage_waypoints_by_uuv 接受 detection_radius_m。以最大 1200m 的 lane spacing
生成三艘分区路径。coverage_gap_area_m2 使用 Shapely：

~~~python
covered = unary_union(
    LineString(route).buffer(detection_radius_m, cap_style="square")
    for route in routes.values()
)
return Polygon(region).difference(covered).area
~~~

如果 gap 大于容差，规划失败并记录 coverage_path_incomplete，不允许把 coverage 标为 1.0。

- [ ] **Step 6: 运行测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/planning/test_region_baseline.py tests/simulation/test_task_group_waypoints.py -q

Expected: PASS。

- [ ] **Step 7: 提交**

~~~bash
git add src/underwater_tracking/planning/region_baseline.py src/underwater_tracking/planning/coverage.py tests/planning/test_region_baseline.py tests/simulation/test_task_group_waypoints.py
git commit -m "feat: generate fixed square coverage regions"
~~~

---

### Task 4: 创建三艇部署实例和初始执行快照

**Files:**
- Create: src/underwater_tracking/runtime/task_group_instances.py
- Modify: src/underwater_tracking/planning/task_groups.py
- Modify: src/underwater_tracking/runtime/execution_snapshot_factory.py
- Test: tests/runtime/test_task_group_instances.py
- Test: tests/planning/test_task_groups.py
- Test: tests/runtime/test_execution_snapshot_factory.py

**Interfaces:**
- Produces: AlwaysAvailableTaskGroupFactory.create(region, deployment_revision, reason, sensor_mode)。
- Produces: RegionTransitionQueue.offer(region) / pop_latest(slot)。
- Produces: 初始 snapshot 四个 ENTERING 三艇实例，无 reserve。

- [ ] **Step 1: 写 deterministic instance factory 失败测试**

~~~python
def test_factory_creates_unique_three_member_deployment_instances() -> None:
    factory = AlwaysAvailableTaskGroupFactory(scenario_id="S1")
    first = factory.create(region_id="T1:task:01", deployment_revision=1,
                           reason="initial_deployment", sensor_mode="active")
    second = factory.create(region_id="T1:task:01", deployment_revision=2,
                            reason="region_replacement", sensor_mode="active")
    assert len(first.member_uuv_ids) == len(second.member_uuv_ids) == 3
    assert set(first.member_uuv_ids).isdisjoint(second.member_uuv_ids)
    assert first.group_instance_id != second.group_instance_id
~~~

ID 格式固定为 S1:T1:task:01:deploy:000001:member:01..03；同 seed 和相同 revision 必须可重现。

- [ ] **Step 2: 写每槽位 latest-wins 队列测试**

连续 offer revisions 2、3、4，pop_latest 必须只返回 revision 4；entering/exiting 未完成时
active transition 不被覆盖。

- [ ] **Step 3: 运行测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_task_group_instances.py tests/planning/test_task_groups.py tests/runtime/test_execution_snapshot_factory.py -q

Expected: FAIL，旧 allocator 固定 2 艘并生成 4 reserve。

- [ ] **Step 4: 实现实例工厂与三艇 allocator**

~~~python
class AlwaysAvailableTaskGroupFactory:
    def create(
        self,
        *,
        target_id: str,
        region_id: str,
        deployment_revision: int,
        reason: DeploymentReason,
        sensor_mode: GroupSensorMode,
    ) -> TaskGroupInstance:
        group_id = f"{region_id}:deploy:{deployment_revision:06d}"
        members = (
            f"{group_id}:member:01",
            f"{group_id}:member:02",
            f"{group_id}:member:03",
        )
        return TaskGroupInstance(
            group_instance_id=group_id,
            target_id=target_id,
            region_id=region_id,
            deployment_revision=deployment_revision,
            member_uuv_ids=members,
            lifecycle=TaskGroupLifecycle.ENTERING,
            sensor_mode=sensor_mode,
            ownership_status="candidate",
            reason=reason,
            evidence_ids=(f"{group_id}:created",),
        )
~~~

删除 TaskGroupPolicy.active_role/passive_role/reserve_count 的 live 使用。TaskGroupPolicy 固定
group_count=4、group_size=3。ReplacementQueue 只保留 legacy import compatibility；新 runtime
使用 RegionTransitionQueue。

- [ ] **Step 5: 更新 snapshot factory**

build_execution_snapshot 接收 tracking_policy 和 instance_factory。首次计划创建四个 ENTERING
active groups；滚动计划把区域 proposal 交给 controller reconcile，不直接保留旧成员。快照填充：

~~~python
tracking_control=TrackingControlState(mode="regional"),
tracking_policy=tracking_policy,
task_groups=initial_groups,
reserve_uuvs=(),
~~~

- [ ] **Step 6: 运行聚焦测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_task_group_instances.py tests/planning/test_task_groups.py tests/runtime/test_execution_snapshot_factory.py -q

Expected: PASS，断言 4×3=12 成员且无 reserve。

- [ ] **Step 7: 提交**

~~~bash
git add src/underwater_tracking/runtime/task_group_instances.py src/underwater_tracking/planning/task_groups.py src/underwater_tracking/runtime/execution_snapshot_factory.py tests/runtime/test_task_group_instances.py tests/planning/test_task_groups.py tests/runtime/test_execution_snapshot_factory.py
git commit -m "feat: create always-available three-uuv groups"
~~~

---

### Task 5: 实现模式 1 组级扫描、入区和所有权交接

**Files:**
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/domain/mission_models.py
- Test: tests/runtime/test_mission_controller.py
- Test: tests/runtime/test_observation_boundary.py

**Interfaces:**
- Consumes: MissionObservation.region_entry_probabilities、passive_observer_ids、deployed_uuv_ids。
- Produces: MissionSnapshot.task_groups、tracking_control、RuntimeEvent。
- Produces: reconcile_execution_snapshot(candidate) -> MissionSnapshot。

- [ ] **Step 1: 写全组三艇模式切换失败测试**

~~~python
def test_region_entry_switches_all_three_members_from_active_to_passive() -> None:
    controller = controller_with_four_three_member_groups()
    controller.observe({"region_entry_probabilities": {"R1": 0.70}})
    assert group(controller, "R1").lifecycle == "active_scan"
    controller.observe({"region_entry_probabilities": {"R1": 0.81}})
    tracked = group(controller, "R1")
    assert tracked.lifecycle == "passive_track"
    assert all(controller.snapshot().uuv_modes[u] == UUVMissionMode.PASSIVE_TRACK
               for u in tracked.member_uuv_ids)
~~~

加入 0.69、missing、0.80 序列重置 confirmation，以及禁止任何 active member 留在 passive group 的测试。

- [ ] **Step 2: 写证据门控所有权失败测试**

新组进入概率满足但 passive_observer_ids 少于三个时 owner 保持旧组；三个当前周期 observer
全部存在后，先发布 tracking_ownership_transferred，再把旧组置 EXITING。

- [ ] **Step 3: 运行 controller 测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py tests/runtime/test_observation_boundary.py -q

Expected: FAIL，旧 _apply_plan 强制一主动一被动。

- [ ] **Step 4: 用组级 helper 替换角色级赋值**

~~~python
def _set_group_phase(
    self,
    group_instance_id: str,
    lifecycle: TaskGroupLifecycle,
    sensor_mode: GroupSensorMode,
) -> None:
    group = self._groups[group_instance_id]
    self._groups[group_instance_id] = group.model_copy(
        update={"lifecycle": lifecycle, "sensor_mode": sensor_mode}
    )
    member_mode = (
        UUVMissionMode.ACTIVE_SCAN
        if sensor_mode is GroupSensorMode.ACTIVE
        else UUVMissionMode.PASSIVE_TRACK
    )
    for member_id in group.member_uuv_ids:
        self._uuv_modes[member_id] = member_mode
~~~

删除 active_execution_uuv_ids 保护集合。execution snapshot refresh 必须保持 controller 当前组级
lifecycle，不能重新开启 active。

- [ ] **Step 5: 实现精确正方形入区和 owner transfer**

controller 只消费 engine 计算的概率，不读取 truth。连续周期确认后先把候选组全被动；只有
accepted passive observers 等于三名成员时，调用 _transfer_tracking_owner(old, new, evidence_ids)。

- [ ] **Step 6: 实现末端/无 successor 保持**

第四区域无接班组、目标暂时在四区域外或 prediction stale 时保持旧 owner 和 PASSIVE_TRACK，
发出有界 handoff_waiting 事件，不转 TRACKING_COMPLETED。

- [ ] **Step 7: 运行聚焦测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py tests/runtime/test_observation_boundary.py -q

Expected: PASS。

- [ ] **Step 8: 提交**

~~~bash
git add src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/domain/mission_models.py tests/runtime/test_mission_controller.py tests/runtime/test_observation_boundary.py
git commit -m "feat: enforce group-level regional tracking"
~~~

---

### Task 6: 实现动态区域并行替换和 latest-wins 合并

**Files:**
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/runtime/task_group_instances.py
- Modify: src/underwater_tracking/runtime/execution_coordinator.py
- Test: tests/runtime/test_mission_controller.py
- Test: tests/runtime/test_execution_coordinator.py
- Test: tests/integration/test_uuv_only_replan_loop.py

**Interfaces:**
- Consumes: 新 committed snapshot 的四个 stable regions。
- Produces: RegionReplacementState(slot, outgoing_group_id, incoming_group_id, target_revision)。
- Produces: controller.runtime_execution_snapshot(base_snapshot)。

- [ ] **Step 1: 写未变化槽位不替换测试**

相同 center/side、不同 prediction metadata 不创建新 group instance；只有中心变化的槽位创建
incoming/outgoing pair。

- [ ] **Step 2: 写四槽位并行替换测试**

~~~python
def test_four_changed_regions_allow_eight_transition_groups() -> None:
    controller = regional_controller()
    controller.reconcile_execution_snapshot(snapshot_with_shifted_centres())
    waterborne = [
        g for g in controller.snapshot().task_groups
        if g.lifecycle != TaskGroupLifecycle.DISAPPEARED
    ]
    assert len(waterborne) == 8
    assert sum(g.lifecycle == TaskGroupLifecycle.ENTERING for g in waterborne) == 4
    assert sum(g.lifecycle == TaskGroupLifecycle.EXITING for g in waterborne) == 4
~~~

当前 owner 所在槽位必须进入 passive successor pending，其他槽位 incoming 为 active。

- [ ] **Step 3: 写高频 revision 合并测试**

replacement revision 2 尚未完成时提交 3、4；不创建第三个可见组；完成 revision 2 后下一次
reconcile 直接选择 revision 4。

- [ ] **Step 4: 运行测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_execution_coordinator.py tests/integration/test_uuv_only_replan_loop.py -q

Expected: FAIL，现有滚动 refresh 保留旧组成员并移动 geometry。

- [ ] **Step 5: 实现 per-slot replacement state**

~~~python
class RegionReplacementState(StrictModel):
    region_id: str
    source_geometry_revision: int
    target_geometry_revision: int
    outgoing_group_id: str
    incoming_group_id: str
    latest_pending_region: ExecutionRegion | None = None
~~~

非 owner 槽位 incoming ACTIVE_SCAN 与 outgoing EXITING 并行。owner 槽位 incoming
PASSIVE_TRACK，直到有效观测后 transfer，再让 outgoing EXITING。

- [ ] **Step 6: 让 coordinator 保存 runtime projection**

ExecutionCoordinator 提交规划 candidate 后调用 controller.reconcile_execution_snapshot；
controller 返回包含实际 groups/control 的不可变 snapshot。CAS 成功后 coordinator 存储该
projection。每个 observation boundary 的 lifecycle 更新通过 update_runtime_projection：

~~~python
def update_runtime_projection(
    self,
    snapshot: OperationalExecutionSnapshot,
    *,
    expected_execution_revision: int,
) -> bool:
    with self._lock:
        current = self._load_current_locked()
        if current is None:
            return False
        if current.execution_revision != expected_execution_revision:
            return False
        if snapshot.execution_revision != current.execution_revision:
            return False
        current_plan = tuple(
            (region.region_id, region.geometry_revision, region.geometry)
            for region in current.regions
        )
        candidate_plan = tuple(
            (region.region_id, region.geometry_revision, region.geometry)
            for region in snapshot.regions
        )
        if candidate_plan != current_plan:
            return False
        self._current = snapshot.model_copy(deep=True)
        return True
~~~

该方法只允许同 execution revision 的 runtime 字段变化，禁止覆盖较新规划 revision。

- [ ] **Step 7: 运行测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_execution_coordinator.py tests/runtime/test_mission_controller.py tests/integration/test_uuv_only_replan_loop.py -q

Expected: PASS。

- [ ] **Step 8: 提交**

~~~bash
git add src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/runtime/task_group_instances.py src/underwater_tracking/runtime/execution_coordinator.py tests/runtime/test_mission_controller.py tests/runtime/test_execution_coordinator.py tests/integration/test_uuv_only_replan_loop.py
git commit -m "feat: replace changed regions with visible groups"
~~~

---

### Task 7: 实现严格模式 2 与 7000m 恢复流程

**Files:**
- Modify: src/underwater_tracking/runtime/mission_controller.py
- Modify: src/underwater_tracking/agent/nodes/directives.py
- Modify: src/underwater_tracking/agent/runtime.py
- Modify: src/underwater_tracking/agent/prompts.py
- Test: tests/runtime/test_mission_controller.py
- Test: tests/agent/test_assignment_directives.py
- Test: tests/agent/test_semantic_nodes.py

**Interfaces:**
- Produces: set_dedicated_owner(target_id, owner_group_id) -> bool。
- Consumes: 当前 TrackingControlState 和每成员 mileage。
- Removes live use: clear_dedicated_group / tracking_mode="regional" directive。

- [ ] **Step 1: 写 dedicated entry 失败测试**

验证只有 current owner 且 lifecycle=PASSIVE_TRACK 可进入；active group、非 owner、任意 UUV
列表和无 owner 均拒绝。成功后 owner 三艇全被动且其他三个组 EXITING。

- [ ] **Step 2: 写唯一退出条件失败测试**

~~~python
def test_dedicated_release_starts_when_any_member_reaches_7000_remaining() -> None:
    controller = dedicated_controller(
        mileage={"U1": 42_999.0, "U2": 40_000.0, "U3": 39_000.0}
    )
    controller.observe(resource_observation(mileage={"U1": 43_000.0}))
    snapshot = controller.snapshot()
    assert snapshot.tracking_control.mode == "dedicated"
    assert owner(snapshot).lifecycle == "dedicated_release_pending"
    assert count_events(snapshot, "dedicated_release_threshold_reached") == 1
~~~

重复 observation 不得重复事件。手动 regional directive、区域 revision、member health=false 均不
释放 dedicated。

- [ ] **Step 3: 写无缝恢复失败测试**

阈值触发后四个 latest-region groups ENTERING；当前目标区域组全被动，其他三组全主动。
新组三名 observer 不全时 dedicated owner 保持；齐全后 owner transfer、mode regional、旧组 EXITING。
水中组/艇计数依次验证 1/3、5/15、4/12。

- [ ] **Step 4: 运行测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py tests/agent/test_assignment_directives.py tests/agent/test_semantic_nodes.py -q

Expected: FAIL，旧实现按 UUV 映射 dedicated，允许 clear，并返区重置而不消失。

- [ ] **Step 5: 实现 group-owner dedicated 状态机**

set_dedicated_group 改为只接收当前 group_instance_id，内部重新验证 owner/lifecycle/三成员。
其他 active region groups 调 begin_group_exit。_update_resource_thresholds 计算：

~~~python
remaining = self.max_uuv_mileage_m - resource.mileage_m
if remaining <= self.dedicated_release_remaining_mileage_m:
    self._begin_dedicated_release(owner_group_id, remaining)
~~~

删除 RETURN_TO_REGION 和 returned_to_region_uuv_ids 的新路径。

- [ ] **Step 6: 收紧 directive**

freeze_dedicated_tracking_members 从 execution.tracking_control.tracking_owner_group_id 读取三名成员；
LLM 输出成员始终被覆盖。移除“resume regional handoffs”提示和 apply 分支，regional 提前释放
返回 needs_clarification，原因 dedicated_mode_releases_only_at_mileage_threshold。

- [ ] **Step 7: 运行测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_mission_controller.py tests/agent/test_assignment_directives.py tests/agent/test_semantic_nodes.py -q

Expected: PASS。

- [ ] **Step 8: 提交**

~~~bash
git add src/underwater_tracking/runtime/mission_controller.py src/underwater_tracking/agent/nodes/directives.py src/underwater_tracking/agent/runtime.py src/underwater_tracking/agent/prompts.py tests/runtime/test_mission_controller.py tests/agent/test_assignment_directives.py tests/agent/test_semantic_nodes.py
git commit -m "feat: lock dedicated tracking until range threshold"
~~~

---

### Task 8: 让物理引擎真实执行三艇模式和边界可见性

**Files:**
- Modify: src/underwater_tracking/simulation/engine.py
- Modify: src/underwater_tracking/domain/platforms.py
- Test: tests/simulation/test_uuv_only_carrier_group.py
- Test: tests/simulation/test_uuv_boundary_rotation.py
- Test: tests/simulation/test_execution_group_activation.py

**Interfaces:**
- Consumes: MissionSnapshot.task_groups 和 tracking_control。
- Produces: 每个 deployment-aware UUV 的 position、heading、mileage、physically_exposed。
- Produces: current-cycle passive_observer_ids 和 boundary completion observations。

- [ ] **Step 1: 写动态实例 materialization 失败测试**

engine 收到 8 个 transition groups 时物化 24 个不同 UUV entity；不存在于新 snapshot 且已
DISAPPEARED 的 entity 不再出现在 waterborne map。

- [ ] **Step 2: 写真实边界进入/退出失败测试**

~~~python
def test_three_member_group_enters_and_exits_at_exact_square_boundary() -> None:
    engine = engine_with_square_region(side_m=2_000.0)
    group = deploy_group(engine, member_count=3)
    for member in group.member_uuv_ids:
        assert point_on_square_boundary(engine.position(member), side_m=2_000.0)
        assert engine.view(member).physically_exposed is True
    request_group_exit(engine, group.group_instance_id)
    advance_until_disappeared(engine, group.group_instance_id)
    assert all(not engine.view(member).physically_exposed
               for member in group.member_uuv_ids)
~~~

- [ ] **Step 3: 写声呐事件约束测试**

ACTIVE_SCAN 三成员都可产生 active_ping；PASSIVE_TRACK、DEDICATED_TRACK 和
DEDICATED_RELEASE_PENDING 在多次 snapshot refresh 后 active_ping 数量仍为 0。

- [ ] **Step 4: 写 passive handoff evidence 测试**

handoff_evidence.required_uuv_ids 和 accepted_observer_ids 必须等于接班组三名成员；任何一个
缺失均 blocked_reason=missing_effective_successor_observations。

- [ ] **Step 5: 运行物理测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/simulation/test_execution_group_activation.py tests/simulation/test_uuv_boundary_rotation.py tests/simulation/test_uuv_only_carrier_group.py -q

Expected: FAIL，engine 只认识配置中的旧两艇成员并保留 RETURN_TO_REGION。

- [ ] **Step 6: 实现 deployment-aware entity lifecycle**

新增 _ensure_group_entities(group)，为新 instance 创建三名 UUV runtime entity，复用 uuv motion
profile 和 sensor profile。entity key 使用 member_uuv_id；退出完成后保留最小审计状态但从
active physics 集合移除。每槽位同一 group instance 只 materialize 一次。

- [ ] **Step 7: 改写任务航点选择**

- ENTERING/ACTIVE_SCAN：调用三艇 scan_waypoints_by_uuv，并在路线末尾循环；
- PASSIVE_TRACK：使用现有 group tracking/FIM 航点，约束接班准备但允许旧 owner 短暂越界；
- DEDICATED_TRACK/RELEASE_PENDING：跟随最新公开目标估计，不读取 truth；
- EXITING：调用 _begin_uuv_boundary_exit，使用关联正方形最近边界点；
- DISAPPEARED：停止物理推进和传感器输出。

- [ ] **Step 8: 改写 observation payload**

移除新路径的 returned_to_region_uuv_ids，增加：

~~~python
{
    "entered_group_instance_ids": ("T1:task:02:deploy:000004",),
    "disappeared_group_instance_ids": ("T1:task:01:deploy:000003",),
    "passive_observer_ids_by_group": {"group-id": ("U1", "U2", "U3")},
    "mileage_m_by_uuv": {"U1": 43_000.0, "U2": 40_000.0, "U3": 39_000.0},
}
~~~

_mission_entry_probabilities 使用原始 region.geometry，删除 _mission_entry_polygon 的 buffer。

- [ ] **Step 9: 运行物理测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/simulation/test_execution_group_activation.py tests/simulation/test_uuv_boundary_rotation.py tests/simulation/test_uuv_only_carrier_group.py -q

Expected: PASS。

- [ ] **Step 10: 提交**

~~~bash
git add src/underwater_tracking/simulation/engine.py src/underwater_tracking/domain/platforms.py tests/simulation/test_uuv_only_carrier_group.py tests/simulation/test_uuv_boundary_rotation.py tests/simulation/test_execution_group_activation.py
git commit -m "feat: execute visible three-uuv group transitions"
~~~

---

### Task 9: 统一 coordinator、CLI 和结构化事件

**Files:**
- Modify: src/underwater_tracking/domain/event_registry.py
- Modify: src/underwater_tracking/runtime/execution_coordinator.py
- Modify: src/underwater_tracking/cli.py
- Test: tests/runtime/test_execution_coordinator.py
- Test: tests/api/test_execution_evidence.py
- Test: tests/integration/test_uuv_only_runtime_entrypoints.py

**Interfaces:**
- Consumes: controller.runtime_execution_snapshot()。
- Produces: 同 revision 的 runtime projection 发布和持久化。
- Produces: 设计文档列出的 12 类事件。

- [ ] **Step 1: 写事件 registry 失败测试**

每个新事件调用 event_definition 不得抛错；validate_event_payload 要求 target_id、region_id、
geometry_revision、group_instance_id、member_uuv_ids、sim_time_s、reason、source_event_ids。

- [ ] **Step 2: 写 runtime projection CAS 失败测试**

同 execution revision 更新 lifecycle 可以提交；旧 revision、不同 region plan hash、未知 group
instance 和非 controller 来源都拒绝。持久化失败时 coordinator 和 controller checkpoint 一起回滚。

- [ ] **Step 3: 写 CLI 单次发布测试**

一个 observation boundary 只能发布一个包含最新 runtime groups 的 frame；不得先发布规划组、
随后再发布物理组造成 UI 闪回。dedicated threshold 事件只触发 runtime transition，不提交战略
LLM replan；恢复使用当前 deterministic four-region snapshot。

- [ ] **Step 4: 运行测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_execution_coordinator.py tests/api/test_execution_evidence.py tests/integration/test_uuv_only_runtime_entrypoints.py -q

Expected: FAIL，新事件未知且 CLI 仍监听 dedicated_mode_released 后发起 regional replan。

- [ ] **Step 5: 注册事件和严格 payload**

在 EVENT_REGISTRY 中设置这些执行事件为 BLUE_PLANNING/OPERATOR 可见，但默认不触发战略 LLM。
handoff_waiting 使用 episode/rate limit 防止每帧刷屏。

- [ ] **Step 6: 重接 CLI 同步**

删除 _sync_dedicated_tracking_groups 对 dedicated_mode_released 的自动 replan 分支。每个 engine
observation boundary 完成后：

~~~python
runtime_snapshot = mission_controller.runtime_execution_snapshot(
    execution_coordinator.current
)
execution_coordinator.update_runtime_projection(
    runtime_snapshot,
    expected_execution_revision=runtime_snapshot.execution_revision,
)
publish_frame(runtime_snapshot)
~~~

必须在 ScenarioTransitionCoordinator 锁内完成 controller、repository 和 publisher bundle 冻结。

- [ ] **Step 7: 运行测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/runtime/test_execution_coordinator.py tests/api/test_execution_evidence.py tests/integration/test_uuv_only_runtime_entrypoints.py -q

Expected: PASS。

- [ ] **Step 8: 提交**

~~~bash
git add src/underwater_tracking/domain/event_registry.py src/underwater_tracking/runtime/execution_coordinator.py src/underwater_tracking/cli.py tests/runtime/test_execution_coordinator.py tests/api/test_execution_evidence.py tests/integration/test_uuv_only_runtime_entrypoints.py
git commit -m "feat: publish authoritative group transition events"
~~~

---

### Task 10: 迁移 OperationalFrame、HTTP/WebSocket 和 Replay

**Files:**
- Modify: src/underwater_tracking/domain/ui_models.py
- Modify: src/underwater_tracking/api/frame_builder.py
- Modify: src/underwater_tracking/api/legacy_frame_adapter.py
- Modify: src/underwater_tracking/api/replay.py
- Test: tests/api/test_execution_frame_contract.py
- Test: tests/api/test_uuv_only_frame_contract.py
- Test: tests/api/test_replay_compatibility.py
- Test: tests/api/test_uuv_only_replay_acceptance.py

**Interfaces:**
- Produces: TrackingPolicyView、TaskGroupInstanceView、TrackingControlView。
- Produces: UUVView.group_instance_id、deployment_revision、group_lifecycle、sensor_mode。
- Contract consumed by: ui/src/types/frames.ts。

- [ ] **Step 1: 写新 frame JSON 失败测试**

~~~python
def test_frame_projects_real_tracking_policy_and_all_visible_groups() -> None:
    frame = build_operational_frame(
        execution_snapshot=parallel_replacement_snapshot(),
        **frame_inputs(),
    )
    assert frame.execution.tracking_policy.task_region_side_m == 2_000.0
    assert frame.execution.tracking_policy.target_detection_radius_m == 1_000.0
    assert frame.execution.tracking_policy.uuv_active_detection_radius_m == 600.0
    assert len(frame.execution.task_groups) == 8
    assert len([u for u in frame.uuvs if u.physically_exposed]) == 24
~~~

断言 payload 不含 active_verifier_uuv_id、passive_tracker_uuv_id 和 live reserve_uuv_ids。

- [ ] **Step 2: 写传输一致性测试**

同一 frame 分别经过 model_dump_json、WebSocket serializer、JSONL writer 和 replay reader，比较
canonical SHA-256；必须相等。3/12/15/24 艘 fixture 都覆盖。

- [ ] **Step 3: 写 legacy replay adapter 测试**

旧两艇 frame 输入时 adapter 标记 schema_version=legacy、保留只读显示；不得转换成
MissionController 可执行 snapshot。新 frame round-trip 不经过 legacy 分支。

- [ ] **Step 4: 运行 API 测试确认失败**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/api/test_execution_frame_contract.py tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py tests/api/test_uuv_only_replay_acceptance.py -q

Expected: FAIL，TaskGroupView 固定两艇且 frame_builder 只投影四个 group assignments。

- [ ] **Step 5: 实现 UI models 和 frame projection**

TaskGroupInstanceView 与 Python domain 使用相同 enum literal，member_uuv_ids 类型固定为
[string, string, string] tuple。ExecutionView 增加 tracking_policy、tracking_control、replacements。
frame_builder 从 runtime snapshot 投影全部非 DISAPPEARED groups，
并按 group_instance_id 关联每艘 UUV；不再用 groups_by_region 字典覆盖同槽位 entering/exiting。

- [ ] **Step 6: 更新 replay schema adapter**

legacy adapter 为旧字段提供 LegacyTaskGroupView，UI 只显示历史数据标签；live API schema 使用
新字段。明确拒绝把 legacy object 传入 apply_verified_execution_snapshot。

- [ ] **Step 7: 运行 API 测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/api/test_execution_frame_contract.py tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py tests/api/test_uuv_only_replay_acceptance.py -q

Expected: PASS。

- [ ] **Step 8: 提交**

~~~bash
git add src/underwater_tracking/domain/ui_models.py src/underwater_tracking/api/frame_builder.py src/underwater_tracking/api/legacy_frame_adapter.py src/underwater_tracking/api/replay.py tests/api/test_execution_frame_contract.py tests/api/test_uuv_only_frame_contract.py tests/api/test_replay_compatibility.py tests/api/test_uuv_only_replay_acceptance.py
git commit -m "feat: expose live tracking transitions in frames"
~~~

---

### Task 11: 对齐 TypeScript 类型和权威前端选择器

**Files:**
- Modify: src/underwater_tracking/ui/src/types/frames.ts
- Create: src/underwater_tracking/ui/src/state/executionSelectors.ts
- Create: src/underwater_tracking/ui/src/state/executionSelectors.test.ts
- Modify: src/underwater_tracking/ui/src/App.tsx
- Modify: src/underwater_tracking/ui/src/components/regionTimeline.ts
- Test: src/underwater_tracking/ui/src/types/regionalTasks.test.ts

**Interfaces:**
- Consumes: Task 10 的新 OperationalFrame JSON。
- Produces: visibleExecutionUuvs、ownerGroup、groupsByRegionSlot、executionCounts。

- [ ] **Step 1: 更新类型测试 fixture 并确认编译失败**

fixture 包含一个 region slot 的 entering+exiting groups、tracking owner 和真实 policy。成员数组
必须长度 3，虽 TypeScript 运行时仍需 selector 防御 malformed payload。

- [ ] **Step 2: 写选择器失败测试**

~~~typescript
it("keeps incoming and outgoing instances visible for the same region", () => {
  const visible = visibleExecutionUuvs(frameWithParallelReplacement);
  expect(visible).toHaveLength(24);
  expect(new Set(visible.map((uuv) => uuv.group_instance_id)).size).toBe(8);
});

it("derives counts from lifecycle without assuming twelve", () => {
  expect(executionCounts(dedicatedRestoreFrame)).toEqual({
    visibleUuvs: 15,
    enteringGroups: 4,
    exitingGroups: 1,
    activeScanGroups: 3,
    passiveTrackGroups: 1,
  });
});
~~~

- [ ] **Step 3: 运行前端测试确认失败**

Run: npm test -- --run src/types/regionalTasks.test.ts src/state/executionSelectors.test.ts

Workdir: src/underwater_tracking/ui

Expected: FAIL，旧 TaskGroupView 要求 active_verifier/passive_tracker，selector 尚不存在。

- [ ] **Step 4: 实现与 Pydantic 对齐的 TypeScript 类型**

添加 TaskGroupLifecycle、GroupSensorMode、TrackingMode、TrackingPolicyView、
TrackingControlView、RegionReplacementView。删除 live TaskGroupView 的固定角色字段。UUVView 增加：

~~~typescript
group_instance_id: string | null;
deployment_revision: number | null;
group_lifecycle: TaskGroupLifecycle | null;
sensor_mode: GroupSensorMode | null;
~~~

- [ ] **Step 5: 实现纯选择器**

~~~typescript
export function visibleExecutionUuvs(frame: OperationalFrame): UUVView[] {
  const visibleGroupIds = new Set(
    frame.execution?.task_groups
      .filter((group) => group.lifecycle !== "disappeared")
      .map((group) => group.group_instance_id) ?? [],
  );
  return frame.uuvs.filter(
    (uuv) => uuv.physically_exposed
      && !!uuv.group_instance_id
      && visibleGroupIds.has(uuv.group_instance_id),
  );
}
~~~

App、timeline 和 sidebar 都调用 selector；禁止各自构造 groupsByRegion 后覆盖 transition group。

- [ ] **Step 6: 运行类型、测试和 build**

Run: npm test -- --run src/types/regionalTasks.test.ts src/state/executionSelectors.test.ts src/components/RegionTimelinePanel.test.tsx

Run: npm run build

Workdir: src/underwater_tracking/ui

Expected: PASS。

- [ ] **Step 7: 提交**

~~~bash
git add src/underwater_tracking/ui/src/types/frames.ts src/underwater_tracking/ui/src/state/executionSelectors.ts src/underwater_tracking/ui/src/state/executionSelectors.test.ts src/underwater_tracking/ui/src/App.tsx src/underwater_tracking/ui/src/components/regionTimeline.ts src/underwater_tracking/ui/src/types/regionalTasks.test.ts
git commit -m "refactor(ui): consume authoritative group instances"
~~~

---

### Task 12: 实现真实区域、探测范围和进出动画

**Files:**
- Modify: src/underwater_tracking/ui/configs/map_display.ts
- Modify: src/underwater_tracking/ui/src/components/CanvasMap.tsx
- Modify: src/underwater_tracking/ui/src/components/map/geometry.ts
- Modify: src/underwater_tracking/ui/src/components/map/camera.ts
- Modify: src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx
- Modify: src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx
- Modify: src/underwater_tracking/ui/src/components/RightSidebar.tsx
- Test: src/underwater_tracking/ui/src/components/CanvasMap.test.ts
- Test: src/underwater_tracking/ui/src/components/map/geometry.test.ts
- Test: src/underwater_tracking/ui/src/components/map/camera.test.ts
- Test: src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx

**Interfaces:**
- Consumes: frame.execution.tracking_policy、真实 region.geometry 和 executionSelectors。
- Produces: canvas 中严格对应 live frame 的区域、目标圆、UUV fan、路径和 transition UI。

- [ ] **Step 1: 写真实几何失败测试**

删除 sharedRegionDisplaySide 相关断言，改为：

~~~typescript
it("renders each authoritative square without display expansion", () => {
  expect(displayRegionPoints(region2000)).toEqual(region2000.geometry);
  expect(regionBounds(region2000).width).toBe(2000);
  expect(regionBounds(region2000).height).toBe(2000);
});
~~~

camera bounds 必须包含真实 1000m target circle、600m UUV footprint 和 entering/exiting positions。

- [ ] **Step 2: 写模式相关探测图层失败测试**

~~~typescript
expect(targetDetectionRange(frame)).toBe(1000);
expect(uuvDetectionFootprint(activeUuv, frame)?.radiusM).toBe(600);
expect(uuvDetectionFootprint(passiveUuv, frame)?.radiusM).toBe(600);
expect(uuvDetectionFootprint(passiveUuv, frame)?.mode).toBe("passive");
~~~

active 三艇均有主动扇形；passive/dedicated 三艇均使用被动样式；不得读取 MAP_DISPLAY_CONFIG
业务半径。

- [ ] **Step 3: 写 24 艘同屏和 owner 高亮失败测试**

CanvasMap data attributes 至少暴露 visible-uuv-count、entering-group-count、
exiting-group-count、tracking-owner-group-id，测试 24 个不同 deployment-aware keys 均被绘制。

- [ ] **Step 4: 运行组件测试确认失败**

Run: npm test -- --run src/components/CanvasMap.test.ts src/components/map/geometry.test.ts src/components/map/camera.test.ts src/components/map/RegionOverlay.test.tsx

Workdir: src/underwater_tracking/ui

Expected: FAIL，当前 UI 使用固定 3000/2000 半径、统一扩大区域并只选八个 execution members。

- [ ] **Step 5: 删除 display-only 业务几何**

MAP_DISPLAY_CONFIG 只保留像素间距、ellipse readability 等纯展示参数。删除
uuvSensorRadiusM、targetDetectionRadiusM 和 sharedRegionDisplaySide。RegionOverlay、hit test、
camera、timeline 统一使用 region.geometry。

- [ ] **Step 6: 改造 CanvasMap 数据流**

CanvasMap 接收整个 frame 或显式 trackingPolicy，调用 visibleExecutionUuvs。所有 UUV 图层以
member_uuv_id + deployment_revision 为 key。区域路径、探测 footprint 和状态 badge 都从同一
group instance 查找 lifecycle/sensor_mode。

- [ ] **Step 7: 实现稳定动画约束**

- ENTERING：后端位置从边界向内移动，opacity 按 transition progress 从 0 到 1；
- EXITING：后端位置向边界移动，opacity 从 1 到 0；
- DISAPPEARED：不渲染；
- owner：使用现有高亮色/线宽，不添加遮挡目标的大卡片；
- active/passive：使用不同 stroke/fill pattern，并保持 600m 几何相同；
- label 布局保持固定尺寸，24 艘时按 owner、目标、entering、exiting 的优先级避让。

- [ ] **Step 8: 更新 timeline/sidebar**

区域行显示稳定 slot、geometry revision、当前 active/incoming/outgoing group；任务统计分别显示
扫描组、跟踪组、进入组、退出组和可见 UUV 数。dedicated 模式突出唯一 owner，不显示虚假三个
空区域组。

- [ ] **Step 9: 运行组件测试和 build**

Run: npm test -- --run src/components/CanvasMap.test.ts src/components/map/geometry.test.ts src/components/map/camera.test.ts src/components/map/RegionOverlay.test.tsx src/components/RegionTimelinePanel.test.tsx src/components/RightSidebar.test.tsx

Run: npm run build

Workdir: src/underwater_tracking/ui

Expected: PASS。

- [ ] **Step 10: 提交**

~~~bash
git add src/underwater_tracking/ui/configs/map_display.ts src/underwater_tracking/ui/src/components/CanvasMap.tsx src/underwater_tracking/ui/src/components/map/geometry.ts src/underwater_tracking/ui/src/components/map/camera.ts src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx src/underwater_tracking/ui/src/components/RegionTimelinePanel.tsx src/underwater_tracking/ui/src/components/RightSidebar.tsx src/underwater_tracking/ui/src/components/CanvasMap.test.ts src/underwater_tracking/ui/src/components/map/geometry.test.ts src/underwater_tracking/ui/src/components/map/camera.test.ts src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx
git commit -m "feat(ui): visualize live three-uuv transitions"
~~~

---

### Task 13: 建立端到端语义与视觉验收

**Files:**
- Create: tests/acceptance/test_three_uuv_tracking_modes.py
- Modify: tests/integration/test_live_tracking_health_pipeline.py
- Modify: tests/integration/test_uuv_only_mission_acceptance.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_runner.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_audit.py
- Modify: src/underwater_tracking/verification/uuv_tracking_coverage_render.py
- Create: src/underwater_tracking/ui/e2e/three-uuv-tracking-modes.spec.ts
- Modify: src/underwater_tracking/ui/playwright.live.config.ts
- Modify: docs/verification/2026-08-30-world-model-uuv-control-audit.md

**Interfaces:**
- Consumes: 真实 main.py live server、WebSocket frames、event JSONL。
- Produces: 可机器验证的 trajectory/events/metrics、桌面/移动截图和 MP4/WebM。

- [ ] **Step 1: 写后端完整序列 acceptance test**

测试用固定 seed 和可控时间推进，不注入手写业务 frame。断言事件顺序：

~~~python
expected = [
    "task_group_entering",
    "active_scan_started",
    "passive_track_started",
    "tracking_ownership_transferred",
    "dedicated_tracking_started",
    "task_group_exiting",
    "task_group_disappeared",
    "dedicated_release_threshold_reached",
    "regional_mode_restored",
]
assert is_ordered_subsequence(event_types, expected)
assert final.execution.tracking_control.mode == "regional"
assert len(visible_uuvs(final)) == 12
~~~

同时断言扫描阶段每区域三 active、跟踪阶段零 active ping、模式 2 区域 revision 变化而 owner 不变、
恢复过渡出现 15 艘、并行 replacement fixture 出现 24 艘。

- [ ] **Step 2: 让覆盖 runner 输出新指标**

metrics.json 增加：

~~~json
{
  "region_side_m": 2000.0,
  "target_detection_radius_m": 1000.0,
  "uuv_detection_radius_m": 600.0,
  "task_group_size": 3,
  "max_coverage_gap_area_m2": 0.0,
  "active_ping_count_during_passive": 0,
  "tracking_owner_gap_frames": 0,
  "max_visible_uuv_count": 24
}
~~~

audit 对每帧执行 schema、正方形、成员数、owner、sensor/lifecycle、boundary 和 transport hash 检查。

- [ ] **Step 3: 写 Playwright live 失败测试**

从真实 WebSocket 等待 data attributes/文本状态，不 mock route：

~~~typescript
await expect(map).toHaveAttribute("data-region-count", "4");
await expect(map).toHaveAttribute("data-task-group-size", "3");
await expect(map).toHaveAttribute("data-region-side-m", "2000");
await expect(map).toHaveAttribute("data-target-radius-m", "1000");
await expect(map).toHaveAttribute("data-uuv-radius-m", "600");
~~~

等待并截图 active scan、passive owner、dedicated 3 艘、regional restore 15 艘过渡和最终 12 艘。
并行 24 艘可使用后端 acceptance fixture endpoint 启动真实 engine 状态，但不得在浏览器端伪造 frame。

- [ ] **Step 4: 增加 canvas pixel 和布局检查**

对每张截图检查地图 canvas 非空、目标圆和至少一个 UUV footprint 的预期颜色像素存在；
用 getBoundingClientRect 验证 timeline/sidebar 不覆盖地图控制条，所有 label 在容器内。视口至少：

- 1440×900 desktop；
- 1280×720 compact desktop；
- 390×844 mobile。

- [ ] **Step 5: 运行后端 acceptance**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest tests/acceptance/test_three_uuv_tracking_modes.py tests/integration/test_live_tracking_health_pipeline.py tests/integration/test_uuv_only_mission_acceptance.py -q

Expected: PASS。

- [ ] **Step 6: 运行前端 E2E**

Run: npm run test:e2e -- three-uuv-tracking-modes.spec.ts

Run: npm run test:e2e:live -- three-uuv-tracking-modes.spec.ts

Workdir: src/underwater_tracking/ui

Expected: PASS，生成 desktop/mobile 截图和 video。

- [ ] **Step 7: 手工查看关键截图**

使用 view_image 检查每个视口的 active、passive、dedicated、15 艘和 24 艘画面。必须确认：

- 四个区域是相同真实比例的 2000m 正方形；
- 1000m 目标圆和 600m UUV 范围比例正确；
- entering/exiting UUV 位于或朝向相应边界；
- owner 清晰但不遮挡目标；
- 24 艘情况下无不可读文本堆叠；
- canvas 非空且地图取景未裁掉关键状态。

- [ ] **Step 8: 更新旧审计文档**

将“4 个双 UUV 组 + 4 reserve”和固定 active/passive 角色标记为已废弃契约，链接新设计和新
acceptance 证据。不要改写历史结果本身。

- [ ] **Step 9: 提交**

~~~bash
git add tests/acceptance/test_three_uuv_tracking_modes.py tests/integration/test_live_tracking_health_pipeline.py tests/integration/test_uuv_only_mission_acceptance.py src/underwater_tracking/verification/uuv_tracking_coverage_runner.py src/underwater_tracking/verification/uuv_tracking_coverage_audit.py src/underwater_tracking/verification/uuv_tracking_coverage_render.py src/underwater_tracking/ui/e2e/three-uuv-tracking-modes.spec.ts src/underwater_tracking/ui/playwright.live.config.ts docs/verification/2026-08-30-world-model-uuv-control-audit.md
git commit -m "test: verify three-uuv tracking visualization"
~~~

---

## Final Verification

- [ ] **Step 1: 后端全量测试**

Run: conda run --no-capture-output -n underwater-tracking python -m pytest -q

Expected: PASS。若 real_llm、long_running 或 live_acceptance markers 默认跳过，记录跳过数量和原因。

- [ ] **Step 2: Python lint 和类型检查**

Run: conda run --no-capture-output -n underwater-tracking python -m ruff check src tests

Run: conda run --no-capture-output -n underwater-tracking python -m mypy src/underwater_tracking

Expected: PASS。

- [ ] **Step 3: 前端单测和生产构建**

Run: npm test

Run: npm run build

Workdir: src/underwater_tracking/ui

Expected: PASS。

- [ ] **Step 4: 前端 synthetic 和 live Playwright**

Run: npm run test:e2e

Run: npm run test:e2e:live

Workdir: src/underwater_tracking/ui

Expected: PASS。

- [ ] **Step 5: 真实入口长时语义探针**

使用 main.py 和默认 uuv-only 配置启动受管 live run，至少覆盖一个完整 regional handoff 和一个
dedicated restore episode。保存 run ID、frame JSONL、event JSONL、metrics、截图和视频。确认：

- 没有 active ping 出现在 passive/dedicated frame；
- tracking_owner_group_id 不出现空窗；
- frame、WebSocket 和 replay hash 一致；
- 最终不存在 ENTERING/EXITING 卡死实例；
- 浏览器 console、后端日志和 WebSocket 均无未处理异常。

- [ ] **Step 6: 检查工作树和提交序列**

Run: git status --short

Run: git log --oneline --decorate -15

Expected: 只存在用户原有未提交文件；实现提交按 Task 1 至 Task 13 排列，没有测试产物或敏感运行
数据被意外纳入版本控制。

---

## Implementation Notes

1. Task 1 至 Task 4 是 schema foundation；在它们完成前不要修改 React。
2. Task 5 至 Task 9 形成真正后端闭环；Task 10 只能消费其已冻结的 frame schema。
3. Task 11 和 Task 12 不得重新解释业务状态。发现缺字段时回到 Python frame contract 补字段。
4. Task 13 必须最后执行真实 live 验收；仅 synthetic component test 不能证明视觉闭环。
5. 每个任务完成后运行列出的聚焦测试并提交，发现无关工作树变化时保留不动。
6. 若旧测试断言 2 艘、8 艘、reserve、active verifier 或 RETURN_TO_REGION，应按新设计重写断言；
   不得通过兼容分支让新 live 路径继续产生旧行为。
