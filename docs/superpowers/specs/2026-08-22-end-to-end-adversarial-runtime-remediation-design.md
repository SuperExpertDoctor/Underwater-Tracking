# 敌我对抗运行闭环修复总设计

日期：2026-08-22

状态：设计与四份详细修改计划已完成，等待统一审核

基线：`master` @ `17f9099`

范围：默认 `main.py` 实时运行、UUV-only 任务执行、目标潜艇 LLM、二维定深运动学、操作帧、智能助理、记忆后台和发布验收

## 1. 文档地位

本文是以下既有设计的纠偏与收敛文档：

- `2026-08-19-uuv-only-carrier-region-mission-design.md`
- `2026-08-20-long-running-runtime-dataflow-hardening-design.md`
- `2026-08-20-smart-assistant-memory-design.md`
- `2026-08-21-uuv-initialization-local-perception-design.md`

既有文档中与本文冲突的运行时 freshness、初始化目标先验、执行 group 创建、脑状态派生、目标任务语义、边界运动和发布验收规则，以本文为准。本文不删除既有任务状态机和记忆模型，而是修复这些能力在真实默认入口中没有形成闭环的问题。

## 2. 已验证的基线问题

默认真实 LongCat 配置运行约 5 分 22 秒、推进到 7435 仿真秒后，系统仍满足以下失败状态：

- `plan_version == 0`；
- 12 艘 UUV 全部为 `onboard`；
- 没有投放、主动扫描、被动跟踪、交接、回收和返队事件；
- 没有目标侧 `adversary_escape` 决策；
- 目标估计由无来源的 `(0, 0)` 粗略先验生成；
- 舰载 UUV 已被放入执行 group；
- 从脑和对手脑被 UI 错误显示为在线；
- 周期摘要已产生，但 Memory Steam 和长期记忆没有形成可验收活动；
- 长时运行后 API 失去及时响应，进程不能在信号后正常退出。

根因不是缺少单个组件，而是规划、物理、任务执行、UI 和记忆之间的版本与真值契约不一致。

## 3. 目标

1. 真实 LLM 延迟大于多个物理快照周期时，首轮有效方案仍能经过重校验并提交，物理仿真和 API 不被阻塞。
2. 默认世界仅包含 1 艘航母、3 艘母舰、12 艘 UUV 和目标潜艇，不存在 USV 实时实体或算法分支。
3. UUV 初始全部舰载并保持固定母舰归属；母舰到达投放点后才建立水中实体和执行 group。
4. 目标初始空间关系、任务先验和传感器估计相互分离，不再生成无来源中心点估计。
5. 目标潜艇拥有真实参与任务的 LLM，只使用自身任务命令和 1200 m 局部感知信息。
6. 目标运动采用二维定深受约束运动学，所有命令统一限制速度、加速度、减速度、转向率和最小转弯半径。
7. `MissionController` 继续作为唯一执行状态源，UI 状态只能从权威执行快照和真实调用账本派生。
8. MemoryWorker 持续异步消费持久化来源，但不阻塞物理 tick，也不把普通观测升级为方案重规划。
9. 默认 `main.py` 端到端验收能够观察完整的敌我博弈时序，并在运行期间保持 API 可用和可关闭。

## 4. 非目标

- 不在本轮实现三维深度、俯仰、升沉速度、六自由度或水动力模型。
- 不允许 LLM 直接写物理坐标、速度、部署状态、资源状态或当前执行方案。
- 不把每个物理帧发送给主脑、目标脑或记忆 LLM。
- 不重写区域概率算法、IMM/UIF、A*、Hungarian 或长期记忆的三类分类定义。
- 不以确定性伪方案冒充真实 LLM 方案；LLM 不可用时保持最后一个已验证方案或安全待命状态。
- 不把摘要、记忆或 UI 文本作为事实证据；证据仍来自事件、决策、方案、观测和知识记录。

## 5. 总体架构

系统保持“两条并行链路 + 一个统一执行状态源”，并增加独立规划纪元：

```text
SimulationEngine
  物理运动、局部观测、目标传感器边界
        │
        ├── ObservationBatch / CriticalEventBatch
        │          │
        │          ▼
        │    PlanningEpoch
        │    主脑 + 从脑 + 确定性优化器
        │          │
        │          ▼
        │    提交前语义重校验
        │          │
        ▼          ▼
MissionController（唯一执行状态源）
  方案版本、UUV 模式、区域生命周期、资源轮转
        │
        ├── OperationalFrame -> API / UI
        └── 持久化事件 -> MemoryWorker
```

### 5.1 三类版本

| 版本 | 含义 | 推进条件 |
| --- | --- | --- |
| `physics_revision` | 当前物理和观测版本 | 每个 observation boundary |
| `planning_epoch_id` | 一次冻结规划输入及其关键事件集合 | 首次有效观测、关键事件或专家确认 |
| `plan_version` | 已验证并进入 MissionController 的方案版本 | 原子提交成功 |

物理 revision 前进只表示时间和观测更新，不能单独使规划结果失效。规划是否失效由显式语义不变量决定。

### 5.2 四阶段依赖

1. 规划活性与运行时生命周期；
2. 初始化、库存和 UI 真值；
3. 目标 LLM 与二维定深运动学；
4. 记忆闭环、前端功能和发布验收。

后续阶段的端到端验收必须建立在前一阶段门禁通过的基础上。阶段二不能用模拟方案绕过阶段一；阶段四不能用静态 fixture 代替阶段一至三的真实运行证据。

## 6. 阶段一：规划活性与运行时生命周期

### 6.1 PlanningEpoch

新增不可变规划输入模型：

```python
class PlanningEpoch(BaseModel):
    epoch_id: str
    scenario_id: str
    base_physics_revision: int
    base_sim_time_s: int
    observation_batch_id: str
    critical_event_ids: tuple[str, ...]
    public_target_prior_ids: tuple[str, ...]
    public_target_estimate_ids: tuple[str, ...]
    resource_manifest_hash: str
    active_plan_version: int
    expert_request_version: int | None
```

每个 epoch 必须随一份 `PlanningEpochCapture` 持久化完整的公开 `SituationSnapshot`、`MissionSnapshot`、observation batch 引用、公开先验/估计引用和资源清单。LLM 节点在整个周期中只读取 captured epoch，不得通过 live provider 混入后续 revision。首轮无目标估计时以有效公开先验规划；先验到期或被替换属于可重校验的语义变化。

### 6.2 单 worker 与 mailbox

- 同一 scenario 最多运行一个中央规划 epoch。
- 普通快照只覆盖 `latest_situation`，不自动排队多个规划任务。
- 当前 epoch 运行期间产生的关键事件进入按事件 ID 幂等的 mailbox。
- 完成后先执行重校验；若 mailbox 中存在仍有效且未被当前方案覆盖的事件，再创建下一 epoch。
- 恢复事件可以使对应异常事件失效，但不能删除持久化审计记录。
- 同一事件失败采用 5、15、45 秒退避，最多自动尝试 3 次；永久 schema/配置错误直接进入 dead-letter。达到上限后等待新事件、配置变化或专家重试，不围绕同一事件无限调用 provider。

### 6.3 两层验证

第一层是结构验证：

- 模型 schema 完整；
- 最多 4 个最终执行区域；
- UUV 永久归属未改变；
- 任务状态转换合法；
- 能量、里程、健康和服务窗满足硬约束；
- 专家确认版本与预览版本一致。

第二层是实时语义重校验：

- 目标、区域、母舰和 UUV 仍存在，UUV 永久 owner 与 captured epoch 一致；
- 首轮所用公开先验仍有效且未变更，或目标公开估计仍位于候选区域外扩 500 m 的方案适用包络；
- 计划使用的 UUV 未被回收、故障或更高版本方案占用；
- 当前资源清单仍可支持航路、投放、区域运动和回收；
- 没有更高优先级专家方案或执行方案已经提交；
- 触发 epoch 的关键事件没有被恢复事件消除。

只有 ETA、当前路线起点、未来会合点和预计资源余量可以由确定性适配器重算。区域优先级、任务意图和目标选择不得在重校验阶段被静默改变。

### 6.4 提交结果

```python
class EpochCommitResult(BaseModel):
    epoch_id: str
    status: Literal["committed", "invalidated", "rejected", "failed"]
    plan_id: str | None
    plan_version: int | None
    validation_report_id: str | None
    executable_plan: ExecutableMissionPlan | None
    invalidated_reason: str | None
    failure_category: str | None
    failure_message: str | None
    consumed_event_ids: tuple[str, ...]
```

`committed` 必须携带持久化重校验报告和完整 executable plan；`invalidated` 表示方案在生成时有效、在提交时因语义变化失效；`rejected` 表示方案自身未通过验证；`failed` 表示 provider 在候选或报告产生前失败，此时允许没有 `validation_report_id`。四者必须分别进入决策账本和 UI 状态。提交端口使用同一 SQLite 事务写入报告、审计方案和 epoch result，并在提交前对 `MissionController` 做可回滚的 copy-on-write apply；SQL 或 apply 任一失败都回滚两侧状态。

规划提交和物理观测提交必须共享同一个 scenario-scoped `ScenarioTransitionCoordinator` 及其唯一 `RLock`，锁顺序固定为 transition lock -> SQLite transaction；锁内禁止 provider/embedding 调用。publisher 只接受在该锁内完成并随后冻结的 `CommittedStateBundle`，因此 plan rollback 不能覆盖已提交物理更新，物理 rollback 也不能覆盖已提交 plan。

### 6.5 服务隔离与关闭

- LLM、embedding 和记忆推理只能在线程 worker 或独立异步任务中运行。
- FastAPI/WebSocket 发布路径不得等待规划 future。
- 首次规划超时保持 UUV 舰载，并发布真实超时/降级状态。
- 已有方案时，规划失败继续执行最后一个仍有效的方案。
- 关闭时按“停止接收 -> 取消或限时等待规划 -> 停止记忆 -> 停止发布 -> 提交数据库 -> 退出”执行。
- `SIGINT` 和 `SIGTERM` 使用相同幂等关闭入口；Vite 子进程、线程、HTTP client 和 SQLite repository 均有明确 owner。

## 7. 阶段二：初始化、库存与 UI 真值

### 7.1 三类目标数据

| 数据 | 可见范围 | 允许用途 |
| --- | --- | --- |
| 私有世界真值 | SimulationEngine 内部 | 物理推进、传感器门控、测试裁判 |
| 公开任务先验 | 主脑和操作员 | 初始搜索区域与任务规划 |
| 传感器目标估计 | 我方规划、执行和 UI | 跟踪、区域概率与交接 |

公开先验必须包含 `prior_id`、来源、时间、中心、协方差/边界和置信度。到达 `valid_until_s` 后先验从活动帧移除并产生一次 `target_prior_expired` 关键事件；没有更新先验或传感器估计时，相关新方案必须失效并保持/返回安全搜索状态。没有公开先验时，不得创建目标估计或地图目标标记。禁止用 `(0, 0)`、私有目标坐标或空 group report 填充目标估计。

### 7.2 默认世界

- 实时平台严格为 `carrier_01`、`carrier_02..04`、`uuv_00..11` 和一个目标潜艇。
- 航母无 UUV 库存；每艘母舰固定拥有 4 艘 UUV。
- 全部 UUV 从 `onboard` 开始，内部位置随所属母舰移动。
- 航母战斗群沿配置循环航迹运动，待命母舰跟随相对槽位。
- 目标位于任务走廊附近，距最近母舰 2.5-4.0 km，位于目标 1200 m 感知圈外。
- 同配置和 seed 的初始权威快照逐字段一致。

### 7.3 分配、运输和执行 group

```text
库存归属
  -> planned_assignment
  -> 母舰运输
  -> 到达投放点
  -> 物理投放
  -> execution_group
  -> ACTIVE_SCAN / PASSIVE_TRACK
```

`planned_assignment` 只存在于 `MissionSnapshot`，可以引用舰载 UUV，但不产生 tracking `GroupReport`、目标估计、从脑观测或地图实体。成功物理投放后建立不带目标 belief 的 `ExecutionGroupState`，用于主动区域扫描；只有真实观测融合产生带 `source_observation_ids` 的 belief 后才发布 tracking `GroupReport` 并允许被动跟踪。回收完成后，UUV 从 execution group 移除并恢复舰载库存状态。

每个观测边界由一个 `ObservationBoundaryCommitter` 在 scenario transition lock 下提交：物理 delta -> `MissionController` copy-on-write 更新 -> execution group 对账 -> 同 revision 状态束 -> frame 发布。任一阶段失败都恢复 controller/engine checkpoint，且不发布该 revision，禁止半完成投放帧。批次部分投放超时后，已入水成员进入 `RETURN_REQUIRED` 并由原母舰回收，不能形成降级扫描组。

`MissionController` 启动时登记每艘 UUV 的永久 owner、部署状态、能量、里程和健康。方案只能改变任务分配，不能改变 owner。

### 7.4 操作帧和脑状态

`BrainView.status` 扩展为：

```text
unconfigured | ready | running | succeeded | degraded | failed
```

- `master` 来自中央规划 epoch 和真实调用账本；
- `slave` 来自已经投放的 execution group 及其真实从脑调用；
- `adversary` 来自目标 LLM 的真实调用和目标自身局部证据；
- `connected_platform_ids` 只引用生成最近一次决策的实际证据平台。

`ready` 仅表示已配置但从未调用；`running` 仅表示当前调用；`succeeded/degraded/failed` 保持为最近一次终态并携带时间，直到下一次该角色调用覆盖。`brains` 按 `master, slave, adversary` 固定排序，但测试和 UI 必须按 `role` 查找，不依赖数组下标。

启动顺序固定为：构造权威世界和 controller -> 发布唯一 bootstrap frame -> 置位 `bootstrap_published` barrier -> 在同一 `sim_time_s=0` 排队初始化 planning/target 事件 -> 启动后台调用。因而 bootstrap frame 中已配置但未调用的 adversary 固定为 `ready`，后续帧才允许 `running/succeeded/degraded/failed`。

UI 不得用“存在 UUV”“存在 group report”或固定全量 roster 推导脑在线状态。

### 7.5 地图和库存展示

- `onboard` UUV 只出现在库存与计划预览中。
- `deployed`、`returning` 或仍在水中的故障 UUV 才绘制地图实体。
- 初始视口综合战斗群、任务区域和有来源的公开先验。
- 没有公开目标估计时，不自动聚焦私有目标。
- 投放帧必须同时出现部署事件、物理暴露状态和执行 group，禁止跨帧半完成状态。
- 实时操作帧、算法和 UI 均不出现 USV；旧回放字段只在 legacy adapter 中读取并丢弃。

## 8. 阶段三：目标 LLM 与二维定深运动学

### 8.1 目标任务状态

```python
class AdversaryMissionState(BaseModel):
    target_id: str
    task_region_id: str
    mission_route_xy: tuple[tuple[float, float], ...]
    escape_region_ids: tuple[str, ...]
    current_intent: str
    current_route_index: int
    local_contact_ids: tuple[str, ...]
    last_decision_id: str | None
```

`task_region_id`、`escape_region_ids` 和任务航线必须从配置进入目标实体、目标图和决策账本，不能只做 loader 校验。

### 8.2 目标本地输入

目标 LLM 允许读取：

- 自身导航状态和任务时间；
- 自身任务区域、任务航线和候选逃逸区域；
- 1200 m 范围内带噪声的距离、方位和平台类别；
- 自身此前决策和带 TTL 的局部接触记忆；
- 边界、陆地、禁航区和运动能力。

目标 LLM 禁止读取：

- 我方全局平台位置或库存；
- 我方针对目标的估计和传感器内部观测；
- 我方主脑方案、区域概率和资源计划；
- 用于传感器门控的私有平台真值距离。

### 8.3 触发和输出

目标 LLM 在以下事件触发：初始任务命令、局部接触进入/离开、威胁等级显著变化、主动声呐暴露、当前路线失效和任务阶段转换。普通物理 tick 不调用 LLM。

输出只包含决策 ID、目标 ID、以下高层意图、可选的已配置逃逸区 ID、置信度、有限理由和触发事件 ID：

```text
continue_mission | avoid_contact | break_contact |
escape_to_region | hold_position
```

`escape_to_region` 必须引用配置中的候选逃逸区。确定性 guidance 将意图转换为航点和速度曲线。目标 LLM 失败时继续最近有效命令；命令不可行时使用确定性任务航线或安全避碰，不停止物理仿真。

### 8.4 二维定深运动学

物理状态固定为 `x`、`y`、`heading`、`speed`。统一运动执行器限制：

- 速度上下界（本轮允许静止，最低速度为 0）；
- 最大加速度和减速度；
- 最大转向角速度；
- 当前速度对应的最小转弯半径；
- 地图边界、陆地和禁航区安全距离。

所有 LLM、规则、任务航线和 fallback 命令经过同一执行器。边界处理采用前视安全缓冲和可行转弯航点，禁止瞬时反射速度。模型和 prompt 删除 `depth_change`；本轮没有深度和俯仰状态。

## 9. 阶段四：记忆、前端和发布验收

### 9.1 记忆调度

MemoryWorker 使用两个独立周期：

- `source_poll_interval_s`：发现事件、决策、方案、对话和周期摘要；
- `maintenance_interval_s`：衰减、归档和清理。

worker 启动后立即发现 scenario 并从持久化 cursor 消费。普通观测允许进入记忆来源，但不能直接进入中央规划 mailbox。

### 9.2 Memory Steam 事件

最少支持以下真实事件类型：

```text
context_loaded
retrieval_started / retrieval_completed
memory_filtered / memory_extracted
short_term_compression_started / short_term_compressed
memory_version_created / memory_version_superseded
memory_accessed / memory_archived / memory_deleted
evidence_trace_started / evidence_trace_completed
source_read_degraded / work_degraded / work_retry_scheduled / worker_recovered
```

长期筛选允许忽略输入；因此周期摘要不要求强制创建长期记忆。验收使用一条明确长期价值的专家消息验证创建、检索、更新、版本失效和来源回溯。

### 9.3 前端功能

- 地图实体和部署时序与 API 一致；
- 智能助理方案调整执行“预览 -> 差异 -> 确认 -> 版本校验 -> 应用”；
- 证据回溯展示问题、记忆版本、来源事件/决策/知识和方案版本；
- 记忆窗口支持短期、情景、语义和程序四类视图，以及版本链和删除；
- `LLM 思考过程` 只显示面向操作员的方案理由；
- `Memory Steam` 只显示后台记忆处理事件；
- 桌面和移动视口无重叠、截断、横向溢出和虚假脑状态。

Playwright 只扫描当前 UI 工程。共享仿真的 live tests 串行运行，或为每个 worker 使用独立端口、数据库和 scenario。测试等待业务状态，不使用固定 sleep 证明任务完成，也不以非透明 canvas 像素代替语义断言。

## 10. 统一不变量

1. `MissionController` 是 region lifecycle、UUV mode、资源、恢复任务和已提交方案的唯一 owner。
2. `SimulationEngine` 只能提交观测和物理状态，不能直接宣告任务交接或资源轮换完成。
3. 物理 revision 前进不能单独使 planning epoch 失效。
4. 任一 plan version 最多提交一次，并且只能基于一份持久化重校验报告提交。
5. UUV 永久归属在一次运行中不变；`onboard` UUV 不属于 execution group。
6. 私有目标真值不进入我方 LLM、操作帧、记忆摘要或 UI。
7. 目标脑只看到自身任务和局部传感器证据。
8. 所有目标运动命令经过同一个受约束运动学执行器。
9. 普通观测和常规资源轮换不逐帧触发主脑；关键事件按 episode 触发。
10. 记忆处理异步运行，完整事实来源保存在 SQLite、事件台账、决策账本和方案仓库。
11. 实时链路中不存在 USV。
12. API 和 WebSocket 可用性不依赖 LLM 是否完成。

## 11. 降级与错误语义

| 故障 | 物理行为 | 执行行为 | UI/审计 |
| --- | --- | --- | --- |
| 首次主脑超时 | 继续编队航行 | UUV 保持舰载 | `master=degraded`，记录 epoch failure |
| 重规划失败 | 继续物理推进 | 保留仍有效旧方案 | 显示失败原因和旧版本 |
| 从脑失败 | UUV 执行最近安全模式 | group 标记 degraded | 不伪造新观测 |
| 目标脑失败 | 执行最近有效或确定性任务命令 | 不影响我方控制器 | `adversary=degraded` |
| MemoryWorker 失败 | 不影响物理和方案执行 | 持久化来源保留待重试 | Steam 发布 degraded/recovered |
| WebSocket 断开 | 仿真继续 | 无状态变化 | 清理连接任务，可重新获取快照 |
| 正常 SIGINT | 停止接受新工作 | 完成有界清理 | 10 秒内以 130 退出 |
| 关闭超时 | 停止接受新工作 | 限时取消后台调用 | 非 130 故障状态和明确日志，不静默挂起 |

## 12. 发布验收矩阵

### 12.1 单元层

- planning epoch 捕获、mailbox 合并、语义重校验和幂等提交；
- 初始化先验、永久库存、planned assignment 与 execution group 分离；
- BrainView 从账本派生；
- 目标任务配置进入目标状态和 LLM 输入；
- 局部感知信息隔离；
- 加减速、转向率、最小转弯半径和边界前视；
- waypoint 纯平移保持对应关系；
- MemoryWorker 立即发现、cursor 恢复和 stream 事件。

### 12.2 集成层

- 受控慢 LLM 跨越多个物理 revision 后仍能提交；
- 关键资源变化使 epoch 明确 invalidated；
- 初始、投放前、投放、投放后四帧真值一致；
- 完整 ACTIVE_SCAN、PASSIVE_TRACK、HANDOFF_PENDING、回收和返队状态机；
- 目标本地接触触发真实目标脑决策且无全局泄漏；
- API 在慢/失败 LLM 和记忆处理期间持续响应；
- 信号关闭无残留线程、子进程、连接和锁。

### 12.3 真实 provider smoke

使用显式 `real_llm` 标记和独立命令运行，必须观察主脑、从脑、目标脑和记忆 LLM 的真实账本记录。普通 pytest 不因存在 `.env` 自动调用外部 provider。

### 12.4 默认 main.py 长时验收

固定 seed 的验收运行必须按顺序观察：

```text
战斗群航行
 -> 首个 plan_version 提交
 -> 母舰离队
 -> 到达任务区外围
 -> UUV 投放
 -> 主动蛇形扫描
 -> 目标局部感知与 LLM 规避
 -> 我方被动协同跟踪
 -> 区域交接
 -> 资源退出与母舰回收
 -> 母舰重新加入移动战斗群
```

运行期间健康 API 必须持续在测试时限内响应，桌面和移动 Playwright 必须通过，最后一次 `SIGINT` 必须在关闭时限内退出。

## 13. 发布门禁

以下命令全部通过后才能宣称实现设计目标：

```bash
PYTHONPATH=src python -m pytest -q
ruff check main.py src tests
mypy src/underwater_tracking
npm --prefix src/underwater_tracking/ui test -- --run
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

真实 provider smoke 和长时验收使用单独、显式命令，结果写入新的验收报告。任何失败不得以“历史基线问题”从本轮发布门禁中豁免；超出某阶段范围的既有错误必须在进入最终发布验收前清零。

## 14. 实施与审核顺序

本文对应四份独立修改计划：

1. `2026-08-22-planning-epoch-runtime-liveness-plan.md`
2. `2026-08-22-truthful-bootstrap-ui-state-plan.md`
3. `2026-08-22-adversary-llm-kinematics-plan.md`
4. `2026-08-22-memory-e2e-release-gates-plan.md`

每份计划单独形成可测试交付物并设置进入下一阶段的门禁。四份计划全部完成后才能执行真实默认入口的最终验收。

## 15. 已确认决策

- 文档组织采用 1 份总设计和 4 份按依赖顺序执行的修改计划。
- 运行时核心采用“规划纪元 + 提交前语义重校验”。
- 目标运动维持二维定深模型，不扩展三维潜艇动力学。
- 设计文档完成后立即制定详细修改计划，全部文档完成后统一交付审核。
