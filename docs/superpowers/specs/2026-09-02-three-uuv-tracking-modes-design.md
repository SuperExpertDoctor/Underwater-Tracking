# Three-UUV Tracking Modes Design

## 1. Goal

将 UUV-only 场景的执行逻辑统一为一套可审计、可回放、可真实可视化的跟踪状态机：

- 始终根据现有轨迹预测算法维护四个动态任务区域；
- 每个任务区域是边长 `2000m` 的正方形；
- 每个执行 task group 固定包含三艘 UUV；
- 区域覆盖扫描阶段三艘全部使用主动探测；
- 区域跟踪和唯一组跟踪阶段三艘全部使用被动探测；
- 模式 1 按相邻区域完成跟踪权交接；
- 模式 2 锁定当前跟踪组，直至里程阈值触发无缝恢复模式 1；
- UUV 从任务区域边界出现，从任务区域边界退出并消失；
- 不再让储备、母舰回收、维护和冷却影响跟踪方案。

本设计修改现有权威执行链路，不建立第二套平行控制器。`OperationalExecutionSnapshot`
仍是 UUV-only 物理执行、API、回放和 UI 的唯一权威执行状态。

## 2. Confirmed Decisions

以下决策已经与用户确认：

1. 四个任务区域均为边长 `2000m` 的正方形，中心和顺序随预测轨迹更新。
2. 目标探测圆半径为 `1000m`。
3. UUV 主动和被动探测半径均为 `600m`。
4. 每个 task group 固定三艘 UUV。
5. `ACTIVE_SCAN` 阶段三艘全部开启主动探测并联合覆盖整个区域。
6. `PASSIVE_TRACK` 和 `DEDICATED_TRACK` 阶段三艘全部使用被动探测，禁止主动验证例外。
7. 目标进入区域使用公开目标估计的区域概率判定；阈值为 `0.70`，连续两个观测周期确认。
8. 删除当前 `1500m` 入区外扩缓冲；概率针对真实 `2000m` 正方形计算。
9. 模式 2 只能锁定当前正在被动跟踪且持有所有权的 task group。
10. 模式 2 中其他三个 task group 驶离各自区域并从边界消失。
11. 模式 2 期间四区域预测继续滚动，但区域更新不能改变唯一跟踪所有权。
12. UUV 最大里程为 `50000m`；任意专属成员剩余里程 `<=7000m` 时触发整组退出流程。
13. 模式 2 不能手动提前取消，也不因成员故障提前结束。
14. 模式 2 恢复时按最新四区域部署四个新三艇组；新跟踪组先接管，旧专属组后退出。
15. 普通区域交接同样要求接班组先形成有效被动观测，旧组随后退出。
16. 动态区域改变时，不让旧 UUV 跟着区域平移；旧区域组退出，新区域组进入。
17. 一次预测更新只替换实际变化的区域，多个区域可以并行替换。
18. 交接和替换期间允许新旧组同时可见，UUV 数量可以暂时超过 12，最多出现 24 艘。
19. 不关注有限资源调度；退出并消失的 UUV 立即恢复可用，里程重置。

## 3. Scope

### 3.1 In Scope

- UUV-only 权威领域模型与配置契约；
- 四个固定尺寸正方形区域的生成和滚动修订；
- 三艇任务组实例和目标级跟踪所有权；
- 模式 1、模式 2、区域替换、边界进入和边界退出状态机；
- 现有三艇覆盖路径调用和覆盖完整性验证；
- 主动/被动探测模式约束；
- API、WebSocket、JSONL、Replay 和 React UI 的一致投影；
- 单元、集成、端到端和视觉验收。

### 3.2 Out of Scope

- 主动声呐传播、互扰、散射或复杂探测概率物理模型；
- UUV 储备数量优化、母舰容量、回收航线、维护冷却和补给调度；
- UUV 健康故障导致的任务组重组；
- 非 UUV-only 场景的任务逻辑重写；
- 轨迹预测和 IMM 核心算法重写；
- 让 LLM 直接生成坐标、物理航点或 UUV ID。

## 4. Selected Architecture

采用“迁移现有权威执行契约”方案：

```text
GlobalTargetTrack / IMM Prediction
                |
                v
     Fixed-Square Four-Region Planner
                |
                v
 OperationalExecutionSnapshot
   - four planned regions
   - target tracking mode
   - tracking owner
   - deployed group instances
   - entering/exiting transitions
                |
                v
       MissionController
   - lifecycle and ownership authority
                |
                v
       SimulationEngine
   - coverage routes
   - passive tracking routes
   - boundary motion and mileage
                |
                v
      Immutable OperationalFrame
        /        |         \
      HTTP    WebSocket   JSONL/Replay
        \        |         /
                v
           React UI
```

规划区域和水中任务组必须分开建模。四区域是预测规划结果，始终恰好四个；水中任务组是
执行实例，过渡期可以同时包含进入组和退出组。不能再使用“快照中永远只有四个两艇组”
作为领域不变量。

## 5. Configuration Contract

新增单一 `tracking_policy` 配置，由后端校验并随操作帧发布：

```yaml
tracking_policy:
  region_count: 4
  task_group_size: 3
  task_region_side_m: 2000.0
  target_detection_radius_m: 1000.0
  uuv_active_detection_radius_m: 600.0
  uuv_passive_detection_radius_m: 600.0
  region_entry_probability_threshold: 0.70
  region_transition_confirm_cycles: 2
  max_uuv_mileage_m: 50000.0
  dedicated_release_remaining_mileage_m: 7000.0
```

启动时必须校验：

- `region_count == 4`；
- `task_group_size == 3`；
- `task_region_side_m > target_detection_radius_m`；
- `target_detection_radius_m > uuv_active_detection_radius_m`；
- `target_detection_radius_m > uuv_passive_detection_radius_m`；
- `0 < dedicated_release_remaining_mileage_m < max_uuv_mileage_m`；
- 入区概率位于 `[0, 1]`，连续确认周期至少为 1。

前端不得复制这些业务常量。它只能读取操作帧中的已验证配置。

## 6. Authoritative Data Model

### 6.1 Planned Region

每个 `ExecutionRegion` 表达一个正方形规划区域：

- `region_id`：稳定槽位 ID，格式保持 `<target_id>:task:01..04`；
- `slot_index`：`0..3`；
- `center`：来自现有预测轨迹区域算法；
- `side_length_m`：固定为 `2000.0`；
- `geometry`：由中心和边长确定的四个真实顶点；
- `geometry_revision`：几何变化时递增；
- `predecessor_region_id` / `successor_region_id`：相邻交接拓扑；
- `prediction_id` / `execution_revision` / `evidence_ids`：保持可追溯性。

区域业务对象是正方形。Shapely polygon 只作为概率、覆盖、最近边界点等几何计算的派生值，
不能成为另一套显示几何。

### 6.2 Task Group Instance

用 `TaskGroupInstance` 替换固定角色的两艇 `TaskGroupAssignment`：

- `group_instance_id`：一次部署实例的唯一 ID；
- `target_id`；
- `region_id`：区域组关联槽位；专属跟踪时保留来源区域用于审计；
- `deployment_revision`；
- `member_uuv_ids`：严格三个不同 ID；
- `lifecycle`；
- `sensor_mode`；
- `ownership_status`；
- `entry_boundary_point` / `exit_boundary_point`；
- `source_group_instance_id`：替换或恢复关系；
- `reason`：初始部署、区域替换、区域交接或专属恢复。

删除 `active_verifier_uuv_id`、`passive_tracker_uuv_id` 及固定主动/被动角色校验。声呐模式
完全由任务组阶段决定。

任务组生命周期为：

```text
ENTERING
  -> ACTIVE_SCAN
  -> PASSIVE_TRACK
  -> DEDICATED_TRACK
  -> DEDICATED_RELEASE_PENDING
  -> EXITING
  -> DISAPPEARED
```

不是所有状态都必须顺序经过。例如非跟踪区域替换可以从 `ACTIVE_SCAN` 直接进入 `EXITING`，
新接班组可以从 `ENTERING` 直接进入 `PASSIVE_TRACK`。

### 6.3 Tracking Ownership

每个目标维护一个 `TrackingControlState`：

- `mode`: `regional | dedicated`；
- `tracking_owner_group_id`：最多一个；
- `pending_successor_group_id`：等待有效观测的新组；
- `dedicated_release_triggered_at_m`；
- `dedicated_release_reason`：固定为里程阈值；
- `source_event_ids`。

只有 `tracking_owner_group_id` 指向的组拥有持续目标跟踪权限。区域状态、UUV 路径和 UI
高亮都从这一字段派生，禁止在各层分别推断当前所有者。

### 6.4 Snapshot Cardinality

权威快照约束为：

- planned regions 始终恰好 4 个；
- 每个 task group instance 始终恰好 3 艘；
- regional 稳态有 4 个水中任务组，共 12 艘；
- dedicated 稳态只有 1 个水中任务组，共 3 艘；
- 四区域并行替换时每个槽位最多一个进入组和一个退出组，最多 8 组、24 艘；
- dedicated 恢复时允许四个新区域组和一个退出专属组同时存在，共 5 组、15 艘；
- `DISAPPEARED` 组不属于水中集合，但保留在事件历史中。

不设置有限 UUV 总量。执行实例工厂按需提供三艇组；退出完成后立即释放实例资源并重置里程。

## 7. Sensor and Coverage Semantics

### 7.1 Active Coverage

`ACTIVE_SCAN` 是组级阶段：

- 三艘成员都设置为主动探测；
- 直接复用现有区域蛇形覆盖搜索；
- 正方形按三艘成员划分扫描分区；
- 每条路径使用 `600m` 探测半径形成覆盖带；
- 三条覆盖带并集必须覆盖完整 `2000m × 2000m` 正方形；
- 一轮结束后循环执行，直到入区确认或区域被替换。

本设计不新增声呐传播和互扰模型。主动探测的业务本质是 UUV 能通过现有底层扫描路径主动获取
区域信息。

### 7.2 Passive Tracking

`PASSIVE_TRACK` 和 `DEDICATED_TRACK` 是严格被动阶段：

- 三艘成员都设置为被动探测；
- 禁止执行快照刷新把任意成员恢复为主动模式；
- 禁止主动验证指令绕过组级状态；
- FIM、测向几何、IMM/UIF 和现有跟踪航迹可以继续工作；
- 有效交接观测必须来自当前周期、健康、已部署且处于被动模式的接班组三个成员。

## 8. Mode 1: Regional Tracking

### 8.1 Initial Deployment and Entry

1. 最新预测产生四个正方形区域。
2. 每个区域创建一个三艇组实例。
3. 三艇组从各自区域最近边界点出现并驶入。
4. 三艘进入 `ACTIVE_SCAN`，执行分区蛇形覆盖。
5. 使用公开目标估计及协方差计算落入真实正方形的概率。
6. 概率 `>=0.70` 且连续两个观测周期成立后，该组三艘统一切换为 `PASSIVE_TRACK`。
7. 第一个形成有效被动跟踪的组成为 `tracking_owner_group_id`。

不得读取目标真值，不得使用 `1500m` 外扩多边形，不得因预测中心靠近区域就提前切换。

### 8.2 Adjacent Handoff

1. 当前所有者持续被动跟踪。
2. 相邻下一地区组在入区条件成立时从主动扫描切换为全组被动。
3. 接班组三艘均已部署、均为被动且当前周期存在有效观测后，进入候选接管状态。
4. `tracking_owner_group_id` 原子地从旧组切换到新组。
5. 先发布所有权转移事件，再将旧组置为 `EXITING`。
6. 旧组可短暂越过旧区域边界继续维持观测，直到接班证据成立；不得制造跟踪空窗。
7. 旧组三艘驶向旧正方形最近边界点，到达后消失。

### 8.3 Terminal or Missing Successor

目标离开第四区域、下一地区尚未形成或预测暂时无效时，当前所有者继续被动跟踪。系统等待
新的有效四区域方案和接班组，不因缺少 successor 自动释放所有权。

## 9. Dynamic Region Replacement

预测和区域算法保持持续滚动。区域变化采用“旧组退出、新组进入”，不让旧 UUV 跟随正方形平移。

- 对每个稳定槽位比较新旧正方形中心和 geometry revision；
- 未变化槽位不执行任何部署动作；
- 非跟踪槽位变化时，新组从新边界进入并主动扫描，旧组同时驶向旧边界；
- 当前跟踪槽位变化时，新组以被动模式进入，形成有效观测并接管后，旧组退出；
- 多个槽位可以并行替换；
- 每个槽位最多有一个 entering 和一个 exiting 实例；
- 替换未完成时到达的更新只保留最新已提交区域版本；当前替换完成后，再与最新版本比较；
- 不为每个中间预测版本生成新的可视 UUV 实例。

这使并行替换可见且有界，避免更新频率高于动画速度时无限累积组实例。

## 10. Mode 2: Dedicated Tracking

### 10.1 Entry

模式 2 只能通过既有受控指令流程进入：

- 当前必须存在唯一 `PASSIVE_TRACK` 所有者；
- 指令语义可以是“持续跟踪”或“设为唯一跟踪组”；
- 系统自动解析当前所有者，用户和 LLM 不提供任意 UUV ID；
- 当前没有有效跟踪所有者时拒绝指令且状态不变；
- 应用后冻结当前三名成员和 `group_instance_id`。

进入模式 2 后：

1. 当前所有者切换为 `DEDICATED_TRACK` 并保持全被动。
2. 该组跟随最新公开目标估计，不受区域边界或区域 revision 影响。
3. 其他三个 task group 转为 `EXITING`，从各自关联正方形最近边界点驶离并消失。
4. 四个预测区域继续滚动，但不创建新的区域执行组。
5. 普通区域交接、区域替换和主动验证都不能改变专属所有权。

### 10.2 Exit Trigger and Regional Restoration

最大里程为 `50000m`。当专属组三名成员中任意一名的剩余里程从大于 `7000m` 变为
小于或等于 `7000m` 时：

1. 发布一次幂等 `dedicated_release_threshold_reached` 事件。
2. 专属组进入 `DEDICATED_RELEASE_PENDING`，继续被动跟踪以避免空窗。
3. 使用最新有效四区域方案创建四个新三艇组。
4. 目标当前区域的新组以被动模式从边界进入；其他三组以主动模式进入。
5. 新跟踪组三艘形成有效被动观测后，所有权原子转移。
6. 模式切换为 `regional`。
7. 原专属组进入 `EXITING`，驶向当前目标区域最近边界并消失。
8. 原实例消失后立即释放，里程重置；后续部署使用新的 deployment revision。

模式 2 不支持手动提前退出，也不因成员故障提前结束。成员健康信息可以继续显示，但不参与本
设计的跟踪控制状态转换。

## 11. Boundary Motion and Physical Visibility

- 新 UUV 的初始可见位置必须位于关联正方形边界；
- `ENTERING` 路径从边界点进入扫描分区或被动跟踪阵位；
- `EXITING` 路径终点是关联正方形上的最近边界点；
- 到达边界前保持 `physically_exposed=true`；
- 到达边界后原子设置 `DISAPPEARED` 和 `physically_exposed=false`；
- UI 可以根据后端发布的 transition progress 做透明度动画，但不能提前隐藏；
- 同一个 group instance 及 UUV ID 不能同时出现在两个位置；
- 新旧实例必须用不同 `group_instance_id` 和 deployment revision 区分。

UUV-only 新路径不创建母舰回收任务，不等待 onboard、维护或 refuel 状态。旧场景兼容逻辑
可以保留，但不得驱动本设计的 live UUV-only 执行。

## 12. Events and Error Handling

至少发布以下结构化事件：

- `task_group_entering`；
- `active_scan_started`；
- `passive_track_started`；
- `tracking_ownership_transferred`；
- `task_group_exiting`；
- `task_group_disappeared`；
- `region_replacement_started`；
- `region_replacement_completed`；
- `dedicated_tracking_started`；
- `dedicated_release_threshold_reached`；
- `regional_mode_restored`；
- `handoff_waiting_for_passive_observation`。

每个事件包含 scenario、target、region slot、region revision、group instance、三个成员、模式、
里程、仿真时间、原因和 source evidence IDs。事件 ID 必须包含部署 revision，保证重试幂等。

安全降级规则：

- 无效或过期预测：冻结最后一份有效四区域方案，不触发 UUV 进出；
- 新组无有效被动观测：旧所有者继续跟踪，发布有界等待事件，不强制交接；
- 目标暂时不属于任何区域：当前所有者继续跟踪，等待下一份有效区域方案；
- 模式 2 非法指令：拒绝并保留当前状态；
- passive/dedicated 阶段主动验证请求：拒绝并记录原因；
- 高频区域 revision：按槽位合并为最新待执行 revision；
- 不完整三艇组：不得进入水中执行，也不得成为 tracking owner。

## 13. API and Replay Contract

`OperationalFrame` 必须直接投影：

- tracking policy 的真实尺寸和阈值；
- 四个真实正方形区域；
- 所有当前水中 group instances，包括 entering、active、passive、dedicated 和 exiting；
- target-level tracking mode 和唯一 owner group ID；
- 每艘 UUV 的 group instance、sensor mode、deployment lifecycle、位置、里程和可见状态；
- 当前区域替换和所有权交接关系。

HTTP、WebSocket、JSONL 和 Replay 发布同一不可变帧。旧两艇回放通过兼容 adapter 读取为
legacy read-only view，不得重新进入 live controller。新回放必须逐帧还原过渡期 3、12、15、
24 艘状态和边界动画。

## 14. Frontend Visualization

前端不再持有业务尺寸或组规模常量。需要完成以下投影：

1. 删除 `MAP_DISPLAY_CONFIG` 中 UUV `2000m` 和目标 `3000m` 的硬编码半径。
2. 删除 `sharedRegionDisplaySide` 和扩大区域的显示几何。
3. 区域 overlay、点击命中、camera bounds、时间线和 sidebar 使用相同真实正方形顶点。
4. 地图显示 `1000m` 目标探测圆。
5. `ACTIVE_SCAN` 三艇显示 `600m` 主动探测扇形和各自覆盖路径。
6. `PASSIVE_TRACK` / `DEDICATED_TRACK` 显示 `600m` 被动探测范围，视觉样式与主动阶段明确区分。
7. UUV 渲染集合来自 frame 中所有 `physically_exposed` group instances，不能只筛四个当前组。
8. React key 使用 deployment-aware UUV instance ID，确保旧组退出与新组进入可同时动画。
9. owner group 使用稳定高亮；entering 和 exiting 使用方向/透明度反馈，但不遮挡目标与区域标签。
10. 稳态统计正确显示 12 或 3 艘；过渡统计允许 15 至 24 艘并标注 entering/exiting 数量。
11. UI 不推断交接完成、模式切换或消失时机，只消费后端 lifecycle 和 owner 字段。

真正的视觉可视化必须由真实 live frame 驱动。不得为截图注入手写 frame、在前端复制状态机、
或用 display-only geometry 掩盖后端区域。

## 15. Verification Strategy

### 15.1 Domain and Configuration

- 四个固定边长正方形；
- 每组三个不同成员；
- 移除固定 active/passive 成员角色；
- tracking owner 唯一性；
- 过渡 cardinality 上限；
- `2000 > 1000 > 600` 和里程阈值校验；
- 旧 replay compatibility adapter。

### 15.2 Planning and Coverage

- 区域中心来自现有预测分段，边长固定 `2000m`；
- stable slot 和相邻拓扑保持；
- 三条扫描路线按 `600m` buffer 后的并集覆盖整个正方形；
- 扫描路线完成后循环；
- 未变化区域不替换，变化区域独立替换；
- 高频 revision 每槽位只保留最新待执行版本。

### 15.3 Controller and Physics

- 三艇全主动到三艇全被动的原子切换；
- 无 active verifier 回退；
- 精确正方形概率 `0.70 × 2` 入区；
- 有效被动观测前所有权不转移；
- 普通交接、区域替换、模式 2 恢复均为新组先接管、旧组后退出；
- 边界出现、移动、淡入、退出和消失；
- 专属模式其他三组退出；
- `7001m -> 7000m` 触发一次整组 release pending；
- 手动取消和区域 revision 不释放模式 2；
- UUV 消失后里程重置并立即可重新部署。

### 15.4 API and UI

- Python schema、JSON payload 和 TypeScript 类型逐字段一致；
- HTTP、WebSocket、JSONL、Replay frame hash 一致；
- 地图使用真实区域和探测半径；
- entering/exiting groups 不被 task-group filter 丢弃；
- timeline、sidebar、map 的 owner/mode/count 一致；
- desktop 和 mobile 无文本、控件、标签或地图元素非预期重叠。

### 15.5 End-to-End Acceptance

固定种子 live 运行必须真实经历：

```text
four 3-UUV groups enter
-> all three active scan per region
-> exact-square entry confirmation
-> all three passive track
-> evidence-gated adjacent handoff
-> dedicated mode locks current owner
-> other three groups exit and disappear
-> prediction regions continue changing without owner change
-> any member reaches 7000m remaining
-> four latest-region groups enter
-> new passive owner acquires observations
-> old dedicated group exits and disappears
-> regional mode stable with 12 UUVs
```

验收同时记录结构化事件、操作帧、覆盖几何指标、截图和视频。Playwright 至少检查桌面和移动
视口，并用 canvas pixel sampling 验证区域、目标圆、UUV 探测范围和 UUV 标记真实可见、非空、
未越出地图容器。最终运行 Python 全量测试、前端单元测试、生产构建和 live Playwright 测试。

## 16. Migration Boundaries

- 先迁移领域模型和配置，再迁移 controller/engine，最后迁移 API/UI；
- 每个阶段使用 adapter 保持代码可运行，不允许长期保留双重权威状态；
- 旧 reserve、carrier recovery 和 fixed-role 字段只在 legacy adapter 中存在；
- 所有新 live UUV-only 路径必须在最终阶段停止读取这些旧字段；
- 文档、fixture、验收基线和界面文案统一更新为四个三艇组。

## 17. Completion Criteria

只有在以下条件全部满足时，本改造才算完成：

1. live 权威快照、物理仿真、API、Replay 和 UI 使用同一组新契约；
2. 四个区域、三艇组、尺寸、模式、所有权和过渡 cardinality 均有严格模型校验；
3. 扫描全主动、跟踪全被动，不存在任何刷新路径重新开启主动探测；
4. 模式 1 和模式 2 的所有权交接均无跟踪空窗；
5. 边界出现、边界退出和消失由物理状态真实驱动；
6. 动态区域替换可并行、可合并且不会无限堆积 UUV 实例；
7. UI 可从真实 live frame 显示 3、12、15、24 艘状态和正确探测几何；
8. 自动测试、真实 live 验收、桌面/移动截图和回放检查全部通过。
