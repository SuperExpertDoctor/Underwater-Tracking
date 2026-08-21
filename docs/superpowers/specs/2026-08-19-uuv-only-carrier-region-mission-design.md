# UUV-Only 区域任务与母舰连续投放设计

日期：2026-08-19  
状态：设计已确认，核心链路已实施，持续验收中
范围：Underwater-Tracking

## 1. 目标与第一阶段范围

系统收敛为“UUV 负责全部探测与跟踪，母舰负责后勤投放与回收”的单目标全闭环任务系统。

第一阶段支持一个敌方潜艇目标、多个 UUV、多个母舰和一个航母战斗群基地，完整验证：

~~~text
IMM/UIF 跟踪
  -> 概率栅格与预测轨迹
  -> LLM 选择方形任务区域和 UUV 编组建议
  -> 全局效能优化与分批投放
  -> 母舰多站点匈牙利匹配
  -> 母舰 A* 航行
  -> UUV 主动区域扫描
  -> 目标进入后切换被动协同跟踪
  -> 区域交接与 UUV 回收
  -> 里程、意图和置信度事件触发全局重规划
~~~

数据模型保留 target_id，但第一阶段不要求多目标之间的资源竞争优化。

## 2. 已确认的设计决策

### 2.1 平台边界

- UUV 是唯一执行主动声呐区域扫描、被动测向和协同跟踪的平台。
- 母舰是纯后勤平台，不参与测向、声呐探测、跟踪质量计算、UUV group 观测或通信中继。
- USV 从新运行链路中移除：不创建、不观测、不编组、不分配、不绘制、不写入新操作帧。
- 历史回放可读取含有 USV 字段的旧帧，但读取时忽略这些字段；新运行不得生成 USV 数据。
- 母舰每次任务的起点和最终终点都是同一个航母战斗群。任务区域和其他母舰位置不能作为最终驻留点。

### 2.2 兼容迁移的含义

复用稳定的 IMM/UIF、轨迹预测、LangGraph、LLM 结构化端口、FastAPI/WebSocket、React 命令中心和 A* 基础能力，但不把新逻辑做成旧 USV 逻辑的补丁集合。

内部建立新的任务编排和后勤调度边界：

- MissionController 是区域生命周期、传感器切换、交接、回收事件和计划执行的唯一任务状态源。
- SimulationEngine 只负责物理推进和观测生成。
- 旧 TrackingPlan、旧区域字段和旧回放字段只通过边界适配器提供兼容视图，不再控制新执行路径。
- 新操作帧不发布 USV；旧帧读取时忽略旧 USV 数据。

### 2.3 效能原则

投放采用带未来可行性约束的滚动边际收益最大化，而不是一次性派出全部 UUV。

硬约束：

- 当前跟踪区域满足最低被动跟踪规模。
- 高概率、短时间内即将进入的预测区域保留最低可行投放资源。
- UUV 轮换、故障、回收和航行时间计入未来资源需求。
- 追加一批 UUV 后若后续高概率区域不可行，则缩小或拒绝当前批次。
- 目标意图、IMM 置信度、区域状态、UUV 里程和母舰容量变化时重新滚动计算。

效能收益包括预测概率覆盖、主动扫描覆盖、被动跟踪质量/FIM、区域交接连续性和跟踪可用时长。代价包括母舰 A* 距离、UUV 航行/扫描能耗、投放延迟、区域空窗、计划变更和过早消耗备用 UUV。

## 3. 总体架构

~~~text
Operational observation
      |
      v
IMM/UIF belief and intent evidence
      |
      v
PredictionGridBuilder
      |
      v
CandidateRegionGenerator
      |
      v
RegionalStrategyGenerationNode (LLM)
      |
      v
RegionalPlanValidator
      |
      v
MissionOptimizer (UUV batch and reserve)
      |
      v
CarrierTaskPlanner
      |
      v
Multi-slot Hungarian matcher + A* route validator
      |
      v
MissionController
      |
      +--> UUV active scan / passive track / return
      +--> carrier multi-stop deployment / recovery / return
      +--> event ledger and next plan revision
~~~

LLM 不直接修改实体状态。LLM 输出先成为候选计划，再由确定性校验器和优化器生成可执行计划；计划只在观测边界原子应用。

模块职责：

| 模块 | 责任 |
| --- | --- |
| PredictionGridBuilder | IMM 概率投影、栅格颜色证据、时间窗 |
| CandidateRegionGenerator | 连续栅格和方形候选区域 |
| RegionalStrategy | 从候选区域中选择区域链、优先级和编组建议 |
| RegionalPlanValidator | 几何、时间、证据和区域连通性校验 |
| MissionOptimizer | UUV 分批、未来储备和效能最大化 |
| CarrierTaskPlanner | 投放/回收事项和多站点任务链 |
| HungarianMatcher | 任务与母舰虚拟服务槽匹配 |
| AStarRoutePlanner | 区域外安全路线和返航路线 |
| MissionController | 区域生命周期、模式切换、事件和计划执行 |
| SimulationEngine | 物理运动、传感器观测、时间推进 |
| OperationalFrameBuilder | 内部状态到实时帧和回放帧的投影 |

## 4. 预测概率栅格与方形区域

每个目标的每次计划修订建立一个局部栅格：

- 原点是规划时刻的 IMM 目标估计均值。
- grid_x/grid_y 是整数坐标。
- 栅格边长由预测包络面积、协方差、横向机动风险和配置边界计算。
- 同一计划修订内栅格 ID 和颜色证据稳定；新的计划修订可以建立新的原点和栅格。

建议尺寸规则：

~~~text
raw_cell_size = sqrt(predicted_envelope_area / target_grid_cell_count)
cell_size = clamp_and_round(raw_cell_size, min_cell_size, max_cell_size)
~~~

高置信度、低协方差使用更细栅格；低置信度、高协方差或高机动风险扩大预测走廊，并允许更大的方形区域。

每个候选单元保存目标进入概率、首次进入时间、最后离开时间、IMM 模型概率、协方差摘要、目标意图、意图置信度、中心线/走廊标记和当前区域 ID。

最终任务区域必须由连续栅格组成，外形为轴对齐方形，边长是整数个栅格边长的倍数，并且：

- 覆盖预测轨迹的一段；
- 有进入/离开时间窗；
- 能找到外围投放点和回收点；
- 能与相邻区域形成有序接力链；
- 不超出地图边界或生成不可达几何。

LLM 只能从候选区域中选择栅格、优先级、时间顺序、主动扫描数量建议、被动跟踪数量建议、备用数量建议和交接关系。LLM 不能创建候选空间外坐标、修改 IMM 证据、引用不存在的平台或绕过资源与路径约束。

概率只负责栅格颜色强度和排序证据，不改变任务区域黄色底色的语义。

## 5. UUV 传感器与区域状态机

UUV 执行模式：

~~~text
ONBOARD
TRANSIT_TO_REGION
ACTIVE_SCAN
PASSIVE_TRACK
RETURN_REQUIRED
RECOVERING
FAILED
~~~

区域生命周期：

~~~text
PLANNED
  -> CARRIER_DEPLOYING
  -> ACTIVE_SCAN
  -> PASSIVE_TRACK
  -> HANDOFF_PENDING
  -> TRACKING_COMPLETED
  -> CARRIER_RECOVERY
  -> RECOVERED
~~~

资源不足时区域进入 DEGRADED 或 UNCOVERED，不能伪造已部署成员。

### 5.1 目标进入前主动扫描

母舰按当前批次将 UUV 投放到区域外围。UUV 使用确定性的覆盖路径进入专属区域，按网格分片执行主动声呐周期扫描。扫描覆盖率、刷新间隔和主动声呐能耗进入区域效果指标。该阶段的目标是提高目标进入时的区域发现概率，不把主动扫描结果直接当作持续被动跟踪权。

主动扫描规模由区域面积、主动声呐作用范围、扫描刷新周期、UUV 速度、预计到达时间和区域可行路径决定。

### 5.2 目标进入后被动协同跟踪

当目标进入概率达到配置的 region_entry_probability_threshold，并连续满足 region_transition_confirm_cycles 个观测周期后：

- 区域切换为 PASSIVE_TRACK；
- UUV 停止常规主动声呐扫描；
- UUV 切换为被动测向；
- 被动观测进入 IMM/UIF 和 group tracking；
- UUV 按 FIM、测向几何和目标机动状态调整编队；
- 正常情况下不持续主动发射，除非人工指定或主动验证事件允许。

默认建议为概率 0.70、连续确认 2 个观测周期，均由配置提供。

### 5.3 交接与轮换

目标接近当前区域出口时，当前区域进入 HANDOFF_PENDING，下一地区提前准备被动跟踪组。交接需要下一地区存在健康 UUV、获得有效观测且计划版本和交接引用一致。当前区域不能越过自身边界继续跟踪目标；交接完成后旧区域 UUV 生成回收任务，除非人工锁定某些 UUV 执行特殊任务。

UUV 累计里程包括投放到区域、区域内部运动和前往回收点的距离。达到轮换阈值时发预警，达到最大里程或能量耗尽时：

1. 停止扫描或跟踪；
2. 切换为 RETURN_REQUIRED；
3. 生成母舰回收任务；
4. 触发全局区域、编组和后勤重规划；
5. 回收和健康检查完成后重新进入资源池。

## 6. UUV 编组与投放优化

LLM 给出主动扫描、被动跟踪和备用 UUV 数量建议，确定性优化器计算实际投放批次。

主动扫描最低需求由区域面积、单 UUV 刷新周期内有效扫描面积、目标到达时间和路径修正计算。被动跟踪最低需求由目标机动等级、协方差、FIM 几何、区域大小、被动测向能力和交接时间计算。

每个规划周期：

1. 计算当前与未来区域的最低需求；
2. 计算候选批次的边际覆盖收益；
3. 检查投放后未来高概率区域是否仍可行；
4. 过滤会破坏储备约束的批次；
5. 选择整体效能最高的批次；
6. 将批次追加到母舰连续投放任务链。

## 7. 母舰多站点连续投放与回收

每艘母舰维护 total_uuv_capacity、available_uuv_count、reserved_uuv_count、ready_uuv_count 和 recoverable_uuv_count。只有 ready_uuv_count 可以用于新的投放匹配。

母舰一次任务可以连续访问多个区域：

~~~text
航母战斗群
  -> 区域 A 外围投放点
  -> 区域 B 外围投放点
  -> 区域 C 外围投放点
  -> 航母战斗群
~~~

投放点和回收点在任务区域外围，母舰不得穿过任何任务区域。回收任务也必须最终返回航母战斗群。投放和回收可以组合到同一条多站点航线，但每个新增停靠点都要重新通过容量、时间窗、禁行和返航校验。

母舰是否可继续匹配由以下谓词决定：

~~~text
ready_uuv_count > 0
and carrier is healthy
and a new task can be inserted into the current route
and the route still returns to the home battle group
and future reserve constraints remain feasible
~~~

因此母舰在 TO_DEPLOY、DEPLOYING、EN_ROUTE_NEXT_DEPLOY 和 RETURNING_TO_FLEET 状态下仍可获得新的投放任务。没有可用 UUV 时不能接收新的投放任务，但继续完成既有航线和回收任务。

### 7.1 A* 航行

母舰航行栅格将任务区域内部设为禁行，安全边界设为高代价或禁行，投放/回收点设为可到达点，地图边界和动态障碍设为禁行，当前母舰位置为起点，航母战斗群为强制终点。

新增停靠点时必须重新计算从当前母舰位置经所有承诺停靠点回到航母战斗群的完整路线，不能只校验当前一段。

### 7.2 多槽位匈牙利匹配

物理母舰展开为虚拟服务槽，例如 carrier_01.slot_1、carrier_01.slot_2 和 carrier_02.slot_1。每个槽位代表当前航线中可插入的一个投放或回收停靠点。

匹配代价包括新增停靠点的增量 A* 距离、预计到达时间、时间窗违约、任务所需 UUV 数量、母舰剩余 ready UUV、插入后的返航距离和未来区域资源损失。

匈牙利匹配完成后由路线构建器重新生成停靠顺序并校验；不满足约束的匹配被拒绝，不能通过高分强行执行。

## 8. 事件与全局重规划

战略重规划事件：

~~~text
target_intent_changed
imm_confidence_shifted
target_entered_region
target_exit_predicted
handoff_completed
uuv_range_exhausted
uuv_energy_depleted
uuv_failed
region_coverage_degraded
carrier_dispatch_completed
carrier_recovery_completed
~~~

每次重规划使用不可变快照，包含 IMM/UIF 证据、概率栅格、区域执行状态、UUV 里程/能量/模式、母舰位置/库存/承诺航线、待处理事件、人工锁定关系和当前计划版本。

提交规则：

- 新计划校验完成前沿用上一版计划；
- 新计划必须拥有更高版本号；
- 旧版本不能覆盖新版本；
- 事件和原因进入事件台账与回放；
- 人工 UUV—目标关系是硬约束；
- 人工锁定 UUV 失效时标记降级并要求可行替代方案。

LLM 不可用时不伪造成功，沿用上一版已验证计划并记录降级事件。

## 9. 操作帧与地图

操作帧新增或重构以下视图：

~~~text
PredictionGridView
  target_id, origin, cell_size_m, cells

PredictionGridCellView
  grid coordinates, bounds, probability, entry/exit time,
  IMM model probabilities, region_id

RegionalMissionView
  region, square geometry, state, target window,
  active_scan_uuv_ids, passive_track_uuv_ids, reserve_uuv_ids,
  coverage, tracking quality, handoff, carrier task

CarrierMissionView
  carrier_id, home_battle_group_id, mission type, route,
  route status, stop list, onboard/ready/reserved counts
~~~

地图层顺序：

1. IMM 概率栅格，概率越高颜色越明显；
2. 预测轨迹中心线和方向；
3. 黄色、透明度 0.66 的方形任务区域；
4. UUV 主动扫描路径和被动测向关系；
5. 母舰从航母战斗群出发、访问多个停靠点并返回的 A* 路径；
6. 区域接力、回收关系和事件标记。

区域状态建议使用 ACTIVE_SCAN、PASSIVE_TRACK、HANDOFF_PENDING、TRACKING_COMPLETED 和 DEGRADED/UNCOVERED 的不同颜色。地图必须明确区分目标到达前主动扫描和目标进入后被动跟踪，并显示未派出的 UUV 是未来储备。

## 10. 失败处理

### 10.1 LLM 不可用

保留上一版已验证计划，记录 llm_degraded，界面显示沿用计划/决策暂停；确定性安全校验继续运行。

### 10.2 A* 无路

拒绝当前匹配，尝试其他母舰服务槽或其他投放点。所有路径均不可行时，区域标记降级并触发重新区域划分；不得生成穿越任务区域的直线兜底路径。

### 10.3 资源不足

当前区域和高概率下一地区优先，低优先级区域显示未覆盖或降级，不创建虚假的 UUV 成员，回收完成后重新进入滚动优化。

## 11. 智能助理、记忆闭环与演示时钟

### 11.1 智能助理边界

界面中的 `LLM Client` 统一命名为“智能助理”，只通过真实 LLM 接口完成以下两类工作：

1. 方案调整：专家提交反馈后，LLM 对反馈分类并生成候选调整方案；方案必须经过确定性校验，专家显式确认后才能实施。
2. 证据回溯：专家追问方案依据时，LLM 只能基于当前短期上下文、经验证的检索材料和来源链回答，不能把长期记忆摘要直接当作事实。

LLM 的方案思考与记忆处理是两个独立过程。方案思考在每次初始制定或动态调整时产生可供操作员查看的说明；记忆处理由可停止的后台 worker 持续执行，不占用仿真推进锁，也不阻塞主任务执行。

### 11.2 短期和长期记忆

短期记忆按 `(user_id, scenario_id, conversation_id)` 隔离，保存滚动摘要和有限条近期原始消息。消息数量、估算 token 或时间达到阈值时，后台 worker 必须调用真实 LLM 生成摘要，保留关键事实、偏好、目标、来源 ID 和未完成事项；摘要版本递增，原始消息只保留有界窗口。

长期记忆分为三类：

- 情景记忆：过往事件、历史对话片段和当时的上下文；
- 语义记忆：静态事实、用户偏好、规则和知识；
- 程序记忆：技能、行动范式和经过验证的操作流程。

长期记忆处理必须经过真实 LLM 的记忆筛选器和记忆提炼器。筛选器过滤一次性闲聊，识别关键事实、偏好、长期目标和“记住这件事”等显式指令；提炼器将一轮或多轮原文凝练为可向量化摘要。检索时先在同一 `user_id` 和 `scenario_id` 范围内做向量 Top-K 召回，再按记忆类型、时间范围、最低重要性、时间衰减和访问频次重排，并限制最终数量和 token 预算。

长期记忆只作为素材供给，永远不能直接替代短期记忆；最终推理输入必须在短期上下文中组装。长期记忆版本不可变，新信息形成新版本并标记旧版本失效，保留 `memory_family_id`、版本号和 supersedes 关系。访问频次会提升权重，长期未访问的记忆会衰减；低于归档阈值的记忆由维护任务归档，用户可以显式删除。

### 11.3 Memory Steam 与证据因果链

主界面底部任务详情栏中，`LLM 思考过程` 旁边固定提供 `Memory Steam` 标签。它只展示后台记忆处理事件，不能用 LLM 思考字段或前端 mock 事件填充。事件通过真实的 `/api/assistant/memory/stream` 增量接口读取，按 `after_cursor`/`next_cursor` 推进并在前端保留最近 300 条。

每条 Memory Stream 事件只允许结构化审计字段：`work_id`、`memory_ids`、`memory_family_id`、`version`、`plan_version`、`source_message_ids`、`source_event_ids`、`source_decision_ids`、`source_knowledge_ids` 和 `source_plan_ids`。短期压缩、筛选、提炼、版本替代、归档、删除、访问和降级都要有明确事件状态。证据回溯产生 `evidence_trace_started` 和 `evidence_trace_completed` 事件，使用 trace ID 幂等写入；操作员展开事件即可从问题/工作 ID 经记忆版本追溯到分类型来源 ID，并点击来源回到主界面证据定位。

证据回溯的开始与完成事件使用确定性事件 ID，并在同一 SQLite 事务中原子写入；并发请求或重试只复用已有事件，不产生重复链路。未显式声明类型的来源 ID 按事件来源保守保存，调用方提供的决策/知识/方案来源保持独立字段；单条事件的五类来源合计最多 64 个 ID。

### 11.4 仿真时钟与回放

常规演示仿真时钟固定以配置 `demo_time_scale=60` 为默认值，即仿真 1 秒对应真实 1 分钟，8 小时仿真约在 8 分钟内完成。运行控制器按仿真时间映射到单调时钟的绝对 deadline 节流，避免每步累计误差；显式 speed=0 只用于测试或不节流运行。

视频回放速度仅允许 `1x`、`4x`、`10x`，只改变回放帧间隔，不能反向修改仿真调度器的演示时钟。长时间验收必须使用真实后端、真实 LLM/Embedding 配置和真实 SQLite 数据流；未配置 provider 时必须显式显示 degraded，不能用规则、哈希向量或 mock 数据冒充成功。

当前生产语义检索使用本地 `sentence-transformers` 模型生成向量，默认模型为多语言 `paraphrase-multilingual-MiniLM-L12-v2`。模型权重必须预先存在于本机缓存或本地路径，provider 始终以 `local_files_only=true` 加载；LongCat 只负责聊天 LLM，不被当作 embedding 服务。缺少 `sentence-transformers` 包或本地权重时，检索和后台记忆任务进入显式 `degraded`，不联网下载、不切换 HTTP embedding、不生成哈希/零向量。

结构化区域策略请求必须受输出预算约束：在 `max_tokens=4096` 的 LongCat 配置下，每次请求最多携带 4 个区域/候选区域，避免嵌套策略对象与模型 reasoning 内容把 JSON 截断。批次结果仍须逐批通过严格 schema、区域覆盖和资源校验后合并；超时、截断或非法响应只能记录真实 LLM 错误并进入降级/重试路径，禁止用规则或静态策略冒充 LLM 结果。

## 12. 测试与完成定义

### 12.1 算法与领域测试

覆盖概率栅格坐标和稳定 ID、置信度尺寸关系、连续方形区域、时间窗和接力、主动到被动切换门限、区域外围投放/回收点、UUV 里程/能量、扫描和跟踪最低规模、未来储备拒绝、连续多站点路线、多槽位匈牙利、A* 区域禁行和强制返航、非法 LLM 输出、LLM 降级。

### 12.2 仿真集成测试

固定种子场景必须验证：至少两艘母舰；同一母舰连续访问至少三个投放点；从航母战斗群出发并最终返回；路线不穿过区域；投放前 ACTIVE_SCAN；目标进入后 PASSIVE_TRACK；目标离开后交接与回收；最大里程触发轮换和重规划；意图/置信度变化重新生成区域；资源不足不全派；新帧无 USV；同 seed 与确定性 LLM provider 产生一致计划和路线。

### 12.3 API、回放和前端测试

覆盖新操作帧合约、旧帧兼容读取、新帧不发布 USV、概率栅格、黄色 0.66 区域、UUV 模式、母舰多站点路线、区域详情、事件顺序、计划版本和桌面/窄屏 Playwright 验证。

第一阶段只有在以下条件全部满足时完成：UUV 完成主动扫描与被动跟踪切换；区域权严格交接；里程事件触发回收和重规划；LLM 只能选择确定性候选；投放按整体效能分批；母舰可连续投放且有可用 UUV 时保持可匹配；每条母舰路线从航母战斗群出发并返回；A* 不穿越区域；匈牙利考虑距离、时间窗、容量和未来资源；地图显示概率栅格和黄色透明区域；旧回放兼容且新帧无 USV；后端、API、前端和确定性回放测试通过。
