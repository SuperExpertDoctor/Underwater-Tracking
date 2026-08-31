# Long-Running Runtime and Data-Flow Hardening Design

日期：2026-08-20  
状态：已确认，进入实施  
基线：`master` @ `39f3497`

## 1. 背景

当前算法链路在短时运行中能够完成仿真、组跟踪、载体任务规划、UUV 任务轮转、资源轮转、事件触发重规划和操作帧发布。但长时间运行审计发现，运行时状态生命周期与实时数据发布边界没有统一约束，导致性能和数据一致性会随运行时间恶化。

已复现的基线现象：UUV-only 显式平台核心运行 300 个 physics tick 时，LangGraph 组检查点的 `writes` 从 53 增长到 458、`blobs` 从 183 增长到 1443；累计耗时从前 30 步约 0.316 秒增加到最后 30 步约 6.164 秒，RSS 从约 126 MB 增长到约 137 MB。原因是每个 tick 的显式回滚会深拷贝持续增长的 `InMemorySaver` 容器。

此外还确认了以下数据流风险：

- 后台 LLM 周期运行时，新 situation 只会被丢弃，完成的旧周期没有 snapshot freshness guard。
- 组历史、事件去重集合、mission event 列表、carrier payload store、指令任务字典均没有统一的生命周期上限。
- 操作帧只发布语义结果，没有明确告诉消费者语义数据相对物理时钟的 revision/age/status。
- Replay 和运行目录摘要会一次性读取完整帧历史。
- CLI 与 `RunController` 对 UUV-only 的边界判断不一致。
- 实时事件、方案重规划事件和记忆输入没有统一的凝练边界；正常生命周期事件可能被误送入战略 LLM 分支。
- WebSocket 客户端断开后，发送和心跳任务可能继续调用已关闭的 transport。

## 2. 目标

1. 在默认 8 小时场景内保持 physics tick 的耗时和内存占用有界，不改变确定性仿真结果。
2. 保证后台 LLM 不阻塞物理线程、不静默丢失最新事件、不把过期结果应用到更新后的物理状态。
3. 保持事件、任务、检查点、计划和操作帧之间的 revision、时间和去重语义一致。
4. 让进程重启后仍能解析检查点引用的 planning snapshot/candidate payload。
5. 让实时 API、WebSocket 和 Replay 在历史增长后仍然具有有界的单次内存和响应规模。
6. 保留已有的 UUV-only 执行边界：`ExecutableMissionPlan` 是唯一物理执行契约，legacy `TrackingPlan` 只用于兼容审计和非 UUV-only 场景。

## 3. 非目标

- 不重写 IMM/UIF、区域规划、Hungarian、A* 或 MissionController 的业务规则。
- 不删除 SQLite runtime events、plans、decisions 或 JSONL 原始帧；内存边界不等于审计数据删除。
- 不把后台 LLM 改成同步调用，也不在 LLM 不可用时生成未经验证的替代计划。
- 不改变已有 API 字段的含义；新增字段提供兼容默认值。

## 4. 设计

### 4.1 紧凑组检查点

新增 `BoundedInMemorySaver`，保持 LangGraph `BaseCheckpointSaver` 接口，默认只保留每个 group thread 的最近少量 checkpoint。每次成功写入后：

- 保留最新 checkpoint 及有限 parent checkpoint；
- 删除不再被保留 checkpoint 引用的 `writes`；
- 删除不再被当前保留 checkpoint channel versions 引用的 `blobs`；
- 保持 `get_tuple()`、正常 invoke、`delete_thread()` 和显式 tick rollback 语义不变。

`SimulationEngine` 继续在 tick 前对必要的 group runtime 做显式回滚快照，但快照对象大小变为有界大小。组状态中的 `last_observations`、质量窗口和 emitted event tail 同样按窗口/上限保留，避免“检查点数量有界但单个状态无限增长”。

长时测试验证的是 saver 容器和单 tick 计时曲线的上界，而不是依赖机器绝对耗时的脆弱阈值。

### 4.2 运行时内存边界

所有实时容器均采用显式 retention policy，持久化存储继续作为完整审计源：

- `SimulationEngine._belief_histories` 保留预测器需要的时间窗口加最小初始化余量；
- engine public event ledger、MissionController event tail、CarrierRuntime processed-event dedupe 和 mission-controller forwarding dedupe 保留有界 tail，并在淘汰时同步维护 ID 集合；
- `CarrierRuntime._payload_store` 使用有界缓存，payload 通过 SQLite durable payload table 保存，重启时按引用懒加载；
- carrier conversation turn、LLM degradation marker、graph output/error tail 和 directive jobs 采用有界 retention；
- directive executor 接受有限数量的 queued/running jobs，完成任务在保留窗口外淘汰；超限返回可识别的队列满错误，而不是无限排队。

事件完整性由 SQLite `runtime_events`、决策 ledger、计划表和 JSONL 帧日志保证；内存 tail 仅用于当前帧、实时推送和快速状态查询。

### 4.3 Durable planning payload store

新增 `runtime_payloads` 表和 `PayloadStore` 映射适配器。`SnapshotNode`、`OptimizeNode`、`VerifyPlanNode` 和 `RecordDecisionNode` 仍使用原有 string reference，但读写经过该适配器：

- payload 以带类型的 JsonPlus 序列化形式存储，避免依赖 Python `repr` 或非结构化字符串；
- `put` 使用 reference 主键幂等 upsert；
- 内存 cache 只保留最近 payload，SQLite 保存最近运行所需的引用和受控保留窗口；
- 运行重启后首次 graph cycle 可以解析 checkpoint 中的 snapshot/candidate reference；
- 旧 payload 淘汰前不影响当前 checkpoint 的 `snapshot_ref`、`selected_plan_ref` 和最近 decision audit。

该 store 不保存高频 raw observation history；raw events 仍写入 `runtime_events`，避免把每个 physics tick 复制进 graph checkpoint。

### 4.4 后台 LLM mailbox 和过期结果保护

后台 carrier loop 改成单 worker + latest-value mailbox：

1. 每个 observation boundary 都更新 `latest_situation`；worker 运行期间只保留最新尚未处理的 situation，不静默丢弃。
2. worker 对一个不可变 captured situation 执行 local brains 和 central graph；graph 内所有节点通过 captured provider 读取同一 revision。
3. `CarrierDependencies` 增加 current live snapshot revision provider。commit 前再次比较 captured revision 与当前 revision；不一致则返回 `stale`，不写入新的 active plan。
4. physics boundary 只应用与当前 observation revision 相同的完成结果。旧结果只保留审计状态，不应用 sensor control、mission plan、slave decision 或 adversary decision。
5. 丢弃旧结果后立即调度 mailbox 中最新 situation；其 pending events 重新进入下一 cycle，保证事件不会因旧 cycle 完成而丢失。
6. LLM 错误保留 pending event/control，并继续发布物理帧；恢复后从最新 situation 重试。

同步模式复用相同 freshness guard，但由于没有并发 mailbox，行为保持原有确定性。

### 4.5 操作帧时效与数据质量

`OperationalFrame` 新增兼容字段：

- `planning_snapshot_revision`；
- `planning_sim_time_s`；
- `planning_data_age_s`；
- `planning_data_status`：`current`、`stale` 或 `unavailable`。

publisher 使用物理 observation revision 与 checkpointed planning revision 计算 status。物理 UUV/carrier 状态始终来自当前 `SituationSnapshot`；语义计划、意图、预测来自同一 checkpoint state，并明确标记 age，禁止消费者误认为两者属于同一时刻。

mission events 和普通 event views 使用有界 tail；完整事件通过持久化事件表/帧日志查询。mission controller snapshot 也只暴露有界事件 tail，资源状态、region lifecycle、UUV mode 和 episode 映射始终保留完整当前状态。

实时发布和记忆处理使用不同的过滤边界：事件、周期性观测批次和人机交互先完整进入持久化与 Memory Stream 输入；只有经过 episode 凝练和方案影响评估的关键事件进入实时事件窗口或重规划控制路径。实时物理帧可以持续发布，区域方案正文只在首帧或 `plan_version` 变化时发布；完整方案仍写入帧日志、快照和回放，不能把实时增量流当作审计源。

关键事件必须证明当前计划的质量、效能或可行性受到影响。质量、通信、协方差和资源事件使用统一的异常进入/升级/恢复滞回；正常投放、回收、健康检查和 LLM 降级只记录审计/记忆状态，不能触发失败的 LLM 重规划循环。

### 4.6 Replay 和运行目录读取边界

`ReplayService.range()` 增加 `offset/limit`，默认和最大值均有界；API `/api/replay` 暴露分页参数并拒绝非法范围。Replay index 只保留 offsets 和时间，不把所有 frame payload 放入内存。

`RunCatalog._summary()` 只读取 index count 和最后一帧，不调用无界 `range()`，因此 `/api/runs` 的响应和内存不随帧总数线性复制。

### 4.7 入口和关闭一致性

- CLI 和 `RunController` 共用一个 UUV-only 判定函数，确保 MissionController、frame serializer 和 engine execution boundary 采用同一规则。
- 后台 worker 关闭时先进入 closing 状态、停止接收新 mailbox item、等待当前 cycle 进入安全完成点，再关闭 publisher/runtime/repositories；超时通过状态报告而不是静默关闭仍在使用的资源。
- WebSocket 发送、接收和心跳共享幂等关闭状态；客户端断开或 transport 关闭后取消并回收所有协程，不向 ASGI 传播关闭竞态异常。

## 5. 一致性不变量

- 任意时刻 active executable mission plan 的 revision 不大于其验证时的 planning snapshot revision，且不会由过期后台 cycle 覆盖。
- 所有进入 central graph 的 event ID 至少一次送达；SQLite event repository 以 unique event ID 幂等，内存淘汰不产生重复持久化记录。
- 每个操作帧的 physical `sim_time_s` 单调不减；`planning_data_age_s` 不为负；`current` 只在 planning revision 对应当前 observation revision 时成立。
- 单个 group thread 的 checkpoint、writes、blobs、belief history、mission event tail 和 directive job 数量均有明确上界。
- UUV-only 帧不出现 USV；legacy frame 仍可读取但不会重新进入 UUV-only execution。
- 实时方案版本不变时不重复发送区域正文；版本跳跃或缺少正文时由快照补偿，不使用旧方案冒充当前方案。
- 单个异常 episode 在恢复前最多产生一个方案控制触发；同一周期内多个来源事件合并为一个有证据链的重规划批次。

## 6. 验证策略

- 先为每个问题增加失败回归测试，再修改生产代码。
- 运行 group saver/engine 长时测试，检查容器增长和分段耗时。
- 运行后台 LLM 阻塞、过期 snapshot、事件合并、故障恢复和关闭测试。
- 运行事件 episode/方案影响评估、实时方案增量与快照补偿、WebSocket 客户端断连回收测试。
- 运行 payload restart、frame freshness、mission event tail、Replay pagination、RunCatalog summary 测试。
- 最后运行后端全量 pytest、ruff、前端测试/build，以及固定种子长时 smoke run。
