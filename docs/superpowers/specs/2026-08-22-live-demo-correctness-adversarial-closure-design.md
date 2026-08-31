# 实时演示正确性与敌我闭环修复设计

日期：2026-08-22

状态：草案，等待审核

基线：`master` @ `2d58376`

范围：默认 `main.py`、规划 epoch、区域策略、事件隔离、仿真时钟、运行生命周期、操作帧、前端状态、目标潜艇 LLM 与受限水下运动

## 1. 文档目的

本文针对默认 `main.py` 实机审查暴露的问题，定义一套可运行、可解释、可停止、可端到端验收的修复设计。本文不推翻既有 UUV 区域任务、记忆和智能助理设计，而是修正这些能力在真实 LongCat、60 倍仿真时钟和默认前端入口下没有形成闭环的问题。

与以下文档冲突时，以本文的规划终态、事件可见性、启动门禁、运行结束、日志采样、脑状态和潜艇深度运动约束为准：

- `2026-08-22-end-to-end-adversarial-runtime-remediation-design.md`
- `2026-08-22-planning-epoch-runtime-liveness-plan.md`
- `2026-08-22-adversary-llm-kinematics-plan.md`
- `2026-08-22-truthful-bootstrap-ui-state-plan.md`
- `2026-08-22-memory-e2e-release-gates-plan.md`

## 2. 已验证的基线事实

在默认真实 LongCat 配置下运行约十分钟，最终操作帧为：

```text
sim_time_s       = 30725
frame_id         = 6145
plan_version     = 0
target_priors    = 0
target_estimates = 0
execution_groups = 0
onboard_uuvs     = 12
waterborne_uuvs  = 0
```

持久化台账显示 5 个 planning epoch 失败、1 个仍在运行，方案和决策记录均为 0。主要失败依次为：

1. `handoff predecessor references unknown candidate target_00:cell:-30:-46`；
2. `resource_optimizer requires an approved strategy or active plan`；
3. `unknown event type: 'target_mission_decision'`；
4. graph 错误路径结束后没有 `epoch_commit_result`，被外层统一记为 internal failure。

其中 `target_00:cell:-30:-46` 实际存在于完整候选图。它只是不在当前四区域 LLM batch 中，因此这是批次局部校验错误，不是 LLM 引用了真实不存在的区域。

初始化方面已经满足以下事实，不应回退：

- 仅有 1 艘航母、3 艘母舰、12 艘 UUV 和 1 个目标潜艇；
- `usvs == []`，实时模型和 UI 不存在 USV；
- 12 艘 UUV 初始全部 `onboard`，不会在地图围成圆圈；
- 目标初始距战斗群约 3.8 km，目标自身局部感知半径为 1200 m；
- 公开目标先验与私有目标真值分离；
- 前端主要面板可以展开，组件测试和构建通过。

## 3. 目标

1. 每个已启动 planning epoch 必须产生且只产生一个持久化终态。
2. 真实 LLM 返回合法跨批次交接关系时，不得被局部 batch 校验误杀。
3. 结构或语义无效的 LLM 输出只允许一次有界纠正，不得转化为无限 internal retry。
4. 首方案未提交前，不允许 60 倍物理时钟使公开先验过期。
5. 每次新观测都推进 coordinator 的 latest physics revision，即使已有 LLM 调用正在运行。
6. 目标潜艇私有 LLM 决策不进入我方规划输入；我方只能响应公开情报和传感器可观测结果。
7. 默认 8 小时场景到达 `duration_s` 后停止物理推进，并保持最终帧可查询和回放。
8. 单次默认场景输出不超过 250 MiB；API 和 WebSocket 不等待 LLM、embedding 或磁盘回放扫描。
9. UI 明确区分 LLM 子调用、planning epoch 和已提交方案，不得把子调用成功显示成总体规划成功。
10. 目标潜艇运动扩展为受约束三自由度模型：水平位置、航向、深度和垂向速度受统一限制。
11. 默认真实入口在发布验收中形成首方案、运输、投放、扫描、跟踪、交接、资源轮转、回收和返队闭环。

## 4. 非目标

- 不实现六自由度刚体动力学、横摇、纵摇力矩、流体阻力场或螺旋桨模型。
- 不让 LLM 直接写坐标、深度、速度、部署状态、资源状态或方案版本。
- 不把敌方私有位置、私有任务航线、私有决策理由或引导点泄漏给我方规划图。
- 不用确定性伪方案冒充真实 LLM 方案。
- 不把每个物理 tick 发送给主脑、从脑、对手脑或 MemoryWorker。
- 不自动删除既有历史输出目录；历史清理由显式维护命令完成。
- 不改变“航母无 UUV 库存、每艘母舰固定拥有 4 艘 UUV”的归属规则。

## 5. 方案比较

### 5.1 方案 A：只修当前三个异常

修改 batch validator、注册 `target_mission_decision`、在 graph error 时补一个 result。改动最小，但 60 倍时钟、先验过期、运行不结束、日志膨胀和 UI 假成功仍然存在。下一次真实 LLM 输出变化仍可能让演示停在 v0。

### 5.2 方案 B：契约优先的闭环修复，推荐

把区域策略权限、epoch 终态、事件可见性、bootstrap 门禁、运行结束和 UI 状态作为明确契约分别修复，再用真实入口发布门禁统一验收。该方案保留现有组件和数据库结构，只增加必要的状态与边界，风险和收益平衡最好。

### 5.3 方案 C：重写规划和仿真编排

用新的 actor/event-sourcing runtime 替换 `_AgentLoop`、LangGraph 和当前 RunController。理论边界最干净，但会扩大到 MissionController、记忆、回放和 API 的全面迁移，无法在本轮可靠完成。

本文采用方案 B。

## 6. 总体架构

```text
RunController
  BOOTSTRAP_PLANNING
        │  captured public situation @ sim 0
        ▼
PlanningEpochWorker ── LongCat regional policy
        │              deterministic candidate graph / optimizer
        ▼
EpochTerminalResult ── commit / rejected / invalidated / failed
        │
        ├── committed ───────────────┐
        │                            ▼
        └── other ── AWAITING_RETRY  RUNNING @ configured time scale
                                     │
                     SimulationEngine + MissionController
                                     │
                   OperationalFrame / EventLedger / MemoryWorker
                                     │
                              COMPLETED @ duration_s
```

系统仍保持“两条并行链路 + 一个统一执行状态源”：MissionController 是任务执行状态源，MemoryWorker 异步沉淀记忆；PlanningEpoch 是独立的方案思考链，不占用物理 tick。

## 7. 区域策略和交接拓扑

### 7.1 权限边界

LLM 决定：

- 候选区域优先级；
- `active_scan`、`passive_track` 或 `handoff_reserve` 策略；
- 所需质量和 UUV 数量；
- 可选的 UUV 硬锁；
- 面向操作员的简短理由和证据引用。

确定性候选图和优化器决定：

- predecessor/successor 关系；
- 最终最多 4 个执行区域；
- 交接顺序和窗口；
- 具体运输、投放、回收与返队路径；
- 未硬锁时的最终 UUV 分配。

LLM 输出不再拥有修改交接拓扑的权限。已有 `predecessor_candidate_id` 和 `successor_candidate_id` 字段仅作为兼容输入读取；实时 UUV-only 路径要求它们为空，并从候选图生成 resolved policy。

### 7.2 批次语义

候选区域仍可按最多 4 个一批发送给 LongCat，但 batch validator 只校验：

- 本批 policy 的 candidate ID 完整且不重复；
- mode、质量、UUV 数量和证据合法；
- UUV 硬锁属于当前资源池且不跨 policy 重复。

合并后 validator 使用完整候选图执行全局覆盖、资源和交接验证。任何跨批次候选引用都只能来自确定性候选图，不能由 batch 局部集合判定为未知。

### 7.3 有界语义纠正

Pydantic 结构解析成功但业务语义失败时，regional node 生成一次 correction payload，内容包括错误码、错误字段和允许值，不包含隐藏真值。第二次仍失败时返回：

```text
status           = rejected
failure_category = semantic
failure_message  = bounded operator-safe summary
```

语义失败不进入 provider/internal 自动重试；只有 timeout、connection、429 和 5xx 可以按 epoch retry policy 重试。

## 8. PlanningEpoch 终态与 freshness

### 8.1 终态不变量

每个 reserved/running epoch 必须满足：

```text
exactly one of committed | invalidated | rejected | failed
```

Graph 的所有出口统一经过 `finalize_epoch`：

- `commit_plan` 产生 committed/invalidated/rejected；
- `handle_error` 把已分类错误转为 rejected 或 failed；
- provider cancellation 转为 failed/cancelled；
- graph 无终态属于 invariant violation，写 failed/internal 并触发测试失败指标。

`_AgentLoop._finish_epoch()` 只负责断言并持久化 graph 已产生的终态，不再为正常错误路径猜测结果。

### 8.2 错误分类

| 分类 | 终态 | 自动重试 |
| --- | --- | --- |
| schema/content/semantic | rejected | 否 |
| stale prior/resource/plan version | invalidated | 新有效事件到来后 |
| timeout/connection/429/5xx | failed | 5、15、45 秒，最多 3 次 |
| invariant/database/internal | failed | 否，进入 degraded/dead-letter |
| shutdown cancellation | failed | 否 |

### 8.3 最新观测

`on_situation()` 在任何 background-cycle 分支之前调用 `coordinator.observe(situation)`。该操作只覆盖 latest snapshot，不创建 epoch、不调用 LLM。active epoch 始终使用 captured base revision；提交端通过 latest revision 做语义重校验。

健康状态同时暴露：

- `base_physics_revision`；
- `latest_physics_revision`；
- `base_sim_time_s`；
- `latest_sim_time_s`；
- `data_age_s`；
- 当前 node、attempt、started_at 和 deadline；
- 最近终态及 operator-safe failure。

## 9. 敌我事件隔离

### 9.1 事件受众

新增事件受众：

```text
blue_planning     我方规划可读取
adversary_private 目标潜艇 LLM 可读取
operator_audit    UI/审计可读取，但不能作为任一方规划事实
memory_source     MemoryWorker 可读取来源摘要
```

一个事件可以拥有多个受众，但受众在创建时固定，不能由 UI 或 MemoryWorker 提升权限。

### 9.2 目标决策

`target_mission_decision` 属于 `adversary_private + operator_audit + memory_source`，不得进入 `SituationSnapshot.pending_events` 的 blue planning 投影。操作员可以看到目标脑发生过决策及其审计 ID，但我方 LLM 不能看到真实 intent、escape region、私有 waypoint 或理由。

我方能够响应的是传感器派生事件，例如：

- `target_maneuver_observed`；
- `target_speed_regime_changed`；
- `target_depth_regime_changed`；
- `target_lost` / `target_reacquired`。

这些事件必须带公开 observation/estimate ID，不得带私有目标状态。

### 9.3 事件注册

所有进入 EventMonitor 的公开事件必须在一个单一 registry 中声明 level、coalescing family、plan-impact policy 和允许 payload。SimulationEngine、MissionController 和 agent runtime 的公开事件测试共同读取该 registry，防止生产者和消费者再次漂移。

## 10. Bootstrap 与仿真时钟

### 10.1 运行阶段

```text
CREATED
  -> BOOTSTRAP_PLANNING
  -> RUNNING
  -> COMPLETED

BOOTSTRAP_PLANNING
  -> AWAITING_RETRY    首方案 rejected/failed
  -> STOPPING          用户终止
```

API 和前端在所有阶段可用。

### 10.2 首方案门禁

默认入口先发布 `sim_time_s=0` 的权威 bootstrap frame，然后启动 initialization epoch。物理 worker 在首个 executable plan 原子提交前不推进仿真时间，因此：

- 公开先验不会因 provider 墙钟延迟过期；
- UUV、母舰和潜艇不会在方案产生前跨越任务区域；
- UI 明确显示“初始方案制定中”，而不是显示一个快速增长但没有任务的空场景。

首方案墙钟 deadline 默认 180 秒。regional batch 最多并发 3 个调用，每个 batch 最多一次语义纠正。超时或拒绝后进入 `AWAITING_RETRY`，保持 UUV 舰载和 sim 0；专家可通过显式 retry 操作重新开始，系统不得自动无限调用。

### 10.3 运行阶段

首方案提交后，物理时钟按 `demo_time_scale=60` 运行。后续重规划不暂停物理，继续执行最后一个有效方案；若没有仍有效方案，则 MissionController 进入安全扫描、返航或待命状态，而不是使用未验证候选。

## 11. 场景结束、日志和关闭

### 11.1 场景结束

默认物理 worker 在 `sim_time_s >= scenario.duration_s` 时停止推进并设置 `COMPLETED`。FastAPI 保持运行以展示最终帧和回放。只有显式 `--continuous` 才允许越过 duration。

### 11.2 操作帧持久化

WebSocket 仍可按每个 physics step 发布实时帧，但 JSONL 只在以下条件写入：

- 到达 30 秒操作帧采样边界；
- plan version、run phase 或 planning epoch 终态变化；
- 出现任务、部署、交接、资源、目标公开估计或关键故障事件；
- 最终 COMPLETED 帧。

每个写入帧仍是完整可独立解码的 OperationalFrame。8 小时默认场景的 `operational_frames.jsonl` 必须小于 250 MiB。历史 run 清理由独立显式命令执行，不在启动或关闭时自动删除。

### 11.3 关闭

关闭顺序固定为：

```text
stop accepting mutations
-> stop physics worker
-> cancel active role HTTP clients
-> stop planning/background cycles
-> stop MemoryWorker and summary writer
-> flush/close frame loggers and repositories
-> stop Vite child
-> stop API server
```

`SIGINT`、`SIGTERM` 和 API shutdown 共用幂等入口。默认关闭 deadline 为 10 秒；超时必须报告仍存活资源的具体 owner 和线程/进程 ID，不能只输出笼统的 `owned resources remain active`。

## 12. UI 诚实状态

### 12.1 三层状态

UI 分开显示：

1. **LLM 调用**：单次 `regional_strategy`、adversary decision、memory filter 等调用；
2. **规划纪元**：running、committed、invalidated、rejected、failed；
3. **执行方案**：当前 `plan_version` 和 MissionController 执行状态。

一个 LLM 子调用成功不能把主脑总体状态标记为 succeeded。主脑终态以 planning epoch 为准；对手脑可以显示自身最近真实调用，但不得将其私有理由注入我方状态。

### 12.2 空场景和失败状态

当 `plan_version == 0` 时，右栏必须明确显示当前 run phase 和原因：

- 初始方案制定中；
- 初始方案被拒绝；
- LLM 不可用，等待重试；
- 已停止；
- 已完成。

地图继续遵守真值边界：没有有效公开先验或传感器估计时，不画目标位置。该空白是正确状态，但必须通过右栏解释，而不是让操作员误以为地图损坏。

### 12.3 智能助理和记忆

- 方案 v0 时允许证据回溯和查看记忆；
- 方案调整 preview 在没有基线方案时转为“初始化方案建议”，仍需确认和 epoch 提交；
- apply 冲突、epoch rejected/failed 和 retry 状态必须可见；
- LLM 思考过程只展示 operator-safe summary；
- Memory Steam 保持独立，继续展示真实 cursor 增量事件。

## 13. 目标潜艇三自由度运动

### 13.1 状态与约定

使用 `depth_m` 正值向下：

```python
class SubmarineMotionState(BaseModel):
    position_xy: tuple[float, float]
    depth_m: float
    heading_rad: float
    speed_mps: float
    vertical_speed_mps: float
```

配置增加：

- 初始 `depth_m`；
- `min_depth_m`、`max_depth_m`；
- `max_vertical_speed_mps`；
- `max_vertical_acceleration_mps2`；
- `max_pitch_rad`，仅作为轨迹坡度约束，不引入完整姿态动力学。

### 13.2 LLM 权限

目标 LLM 只输出高层深度意图：

```text
maintain_depth | go_deeper | go_shallower
```

确定性 guidance 根据水深边界、当前深度、局部联系和任务阶段生成 `desired_depth_m`。积分器每个子步限制水平速度、加减速、转弯率、垂向速度、垂向加速度和最大坡度。越界命令转为 safe hold，并记录导航退化事件。

### 13.3 感知和公开投影

目标本地探测使用三维距离；水面舰深度固定为 0，水中 UUV 使用其配置/状态深度。目标真值深度只存在私有世界和对手脑输入。只有传感器确实产生深度估计时，我方 OperationalFrame 才发布估计深度及不确定度。

二维地图仍投影 x/y；目标公开估计和 UUV tooltip 可显示有来源的估计深度。不得把私有目标深度作为 UI fallback。

## 14. 端到端验收

### 14.1 无外部 provider 的确定性门禁

- 跨 batch 的全局候选关系不被局部 validator 拒绝；
- 每条 graph 出口产生一个 epoch result；
- target private event 不进入 blue planning snapshot；
- 新观测在 active epoch 期间持续更新 latest revision；
- duration 到达后不再产生新 physics frame；
- 8 小时回放输出小于 250 MiB；
- 三自由度轨迹逐步满足全部运动约束；
- 前端 build、Vitest 和本地 Playwright 全通过。

### 14.2 真实 LongCat 发布门禁

使用默认 `main.py` 和真实配置，自动采集 API、SQLite、WebSocket 和输出目录：

1. 180 秒墙钟内产生首个 committed plan，或明确进入可操作的 AWAITING_RETRY；发布门禁要求 committed。
2. bootstrap frame 中 12 艘 UUV 全部 onboard、0 execution groups、1 个有效公开先验、0 私有目标真值泄漏。
3. 物理时钟只在首方案提交后启动。
4. 母舰运输到投放区后才出现 waterborne UUV 和 execution group。
5. 场景内至少出现一次 ACTIVE_SCAN、PASSIVE_TRACK、HANDOFF_PENDING、handoff completed、resource rotation、recovery 和 carrier returned。
6. 潜艇至少产生一次真实 LLM 决策，且轨迹满足三自由度限制。
7. Memory Steam 至少出现 source discovery、filter/extract 或 short-term compression 的真实事件。
8. API health p95 小于 200 ms，WebSocket 无断流，关闭小于 10 秒。
9. `sim_time_s` 不超过 `duration_s + physics_step_s`。
10. 单次输出小于 250 MiB，方案和事件证据可回放。

真实 provider 不可用时，该门禁必须明确标记 SKIPPED/BLOCKED，不能用 fake LLM 结果替代通过。

### 14.3 全场景敌我博弈监控验收

发布前必须再执行一次独立于单元测试的最终验收：直接运行默认 `main.py`，从 bootstrap 开始监控到 `sim_time_s=28800` 的场景完成状态。该任务不注入固定方案、假目标观测、人工部署事件或 fake LLM 输出。

验收必须同时证明两条因果链真实发生：

```text
蓝方跟踪：
公开先验 -> LLM 区域方案 -> 母舰运输/投放 -> 主动扫描
-> 有来源目标估计 -> 被动跟踪 -> 出口预测 -> 区域交接
-> 资源轮转 -> 回收 -> 母舰返队

潜艇反跟踪：
任务航线 -> 1200 m 局部平台感知/可听主动声纳
-> 目标 LLM 决策 -> 受约束航向/航速/深度引导
-> 蓝方传感器观测到机动效果 -> 蓝方重新估计或重规划
```

“双方脑都调用过”不能替代博弈验收。目标 LLM 决策必须能追溯到目标自身可用的局部证据，蓝方后续响应必须能追溯到公开观测/估计，二者之间不得使用目标私有决策事件作为捷径。

全程逐物理步检查全部 17 个实体（1 艘航母、3 艘母舰、12 艘 UUV、1 艘目标潜艇）：

- 航母沿配置巡逻路线运动，速度、加速度、转向率和边界合法；
- 待命母舰保持编队槽位，执行任务的母舰只沿已提交运输/会合/回收路线偏离，返队后重新收敛到槽位；
- UUV 舰载时随母舰且不作为水中实体运动，投放位置与母舰投放点连续，水中速度、加减速、转向率、里程、能量和边界合法，回收位置与会合点连续；
- 潜艇水平速度、加减速、转向率、深度、垂向速度、垂向加速度、坡度和边界合法；
- 任何部署、回收或状态切换造成的位置变化都必须有显式物理事件解释，不能表现为无事件瞬移。

验收监控器只输出每实体的最大观测值、限制值、违反次数和对应帧 ID，不把私有坐标或目标真实深度发布给 UI/API。UI 在桌面和移动视口定期截图并录制阶段证据，检查目标公开信息来源、UUV 可见性、任务时间线、双方脑状态、LLM 思考和 Memory Steam 与台账一致。

最终通过条件：全部必需战术阶段出现；至少形成一次“目标局部感知 -> 目标 LLM 机动 -> 蓝方公开观测响应”的完整反跟踪证据链；全部实体物理违反次数为 0；浏览器错误和失败请求为 0；运行按 duration 完成且干净关闭。

## 15. 迁移和兼容性

- 旧回放缺少 run phase、event audience 或 depth 字段时，由 replay adapter 填充 `legacy/unknown`，不得写回旧文件。
- 旧 UUV policy 中的 predecessor/successor 仅供旧回放展示；新实时方案由 deterministic graph 重新解析。
- 数据库新增字段使用可空列或新表，旧 planning epoch 仍可读取。
- `position_xy` 保留为地图投影接口；潜艇内部新增 depth，不把全系统一次性改成任意三维向量。
- UI 对旧 frame 保持可读，但实时模式不得使用旧 fallback 伪造状态。

## 16. 实施顺序

1. 区域策略与 epoch 终态；
2. 事件隔离与公开事件 registry；
3. coordinator freshness 和 bootstrap 门禁；
4. duration、日志和关闭；
5. UI 诚实状态；
6. 潜艇三自由度扩展；
7. 真实入口自动发布门禁；
8. 完整 `main.py` 敌我博弈与全实体物理监控验收。

任何阶段不得通过静态前端 fixture 或确定性伪方案绕过前置门禁。
