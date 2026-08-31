# Bounded Events, Four-Region Plans, and Live Protocol Implementation Plan

> 本计划在 `master` 上制定。设计文档已确认并提交；实现必须在从 `master` 创建的新分支中进行，完成验证后合并回 `master`。

## 目标

将当前“事件类型直接决定战略路径”的实现改为基于事件 episode 和方案影响的低频关键事件闭环；将每个目标的最终执行区域限制为最多 4 个；保留完整候选、观测、事件和记忆来源；实时 WebSocket 只增量发送方案；修复客户端断连异常；让事件、周期性观测和人机交互全部进入真实后台记忆处理。

## 全局约束

- 所有原始事件、周期性观测批次和人机交互消息仍写入持久化审计/记忆输入，不因实时降频而丢失。
- 关键事件必须证明当前计划质量、效能或可行性受到影响；事件类型本身不能直接触发 LLM 战略重规划。
- 同一异常 episode 在恢复前最多产生一次控制触发；同一规划边界内的多个来源事件合并为一个有证据链的批次。
- 最终执行计划按目标最多 4 个区域；完整候选保留并标记 `region_cap_not_selected`，不能伪装成执行区域。
- 持久化帧、HTTP 快照和回放帧始终完整；实时帧省略未变化的区域正文时，前端必须按 `plan_version` 合并或请求快照补偿。
- LongCat 只负责真实聊天/结构化 LLM；语义检索必须使用本地 `sentence-transformers`，禁止 HTTP embedding、哈希向量、零向量和 mock 成功结果。
- 不改变 UUV-only 的物理执行契约，不恢复 USV 新链路，不删除已有 SQLite/JSONL 审计数据。

## Task 1：建立事件影响策略和 episode 状态机

**文件：**

- Create: `src/underwater_tracking/agent/event_policy.py`
- Modify: `src/underwater_tracking/agent/nodes/event_monitor.py`
- Modify: `src/underwater_tracking/domain/models.py`
- Test: `tests/agent/test_event_policy.py`
- Test: `tests/agent/test_event_monitor.py`

**步骤：**

- [ ] 先写失败测试，覆盖事件的 `audit_only`、`tactical`、`candidate`、`key` 四种处置；验证 `llm_degraded`、正常投放/回收、普通检测变化和 `active_ping` 不会直接得到战略重规划资格。
- [ ] 写失败测试，验证质量、通信、协方差、能量预警和低电量事件只有在异常进入/升级时发射；持续异常不重复，恢复后同一 episode 才能重新触发。
- [ ] 写失败测试，验证意图/置信度必须连续确认，并且事件没有 `plan_impact` 时不会进入战略分支。
- [ ] 增加不可变的事件影响结果和有界 episode 状态；记录 `episode_id`、状态边沿、触发原因、影响指标、受影响目标/区域和 `source_event_ids`。
- [ ] 将 `EventMonitor` 的质量、目标丢失、意图确认和冷却逻辑统一到 episode 状态机；冷却只作为保护，不替代恢复边沿。
- [ ] 保留 `RuntimeEvent` 原始类型/载荷，不覆盖原始来源，只在控制路径附加结构化影响判定。
- [ ] 运行 `tests/agent/test_event_policy.py tests/agent/test_event_monitor.py`，确认先红后绿。
- [ ] Commit: `test/fix: add event episode impact policy`

## Task 2：统一事件来源和重规划触发

**文件：**

- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Test: `tests/agent/test_central_graph.py`
- Test: `tests/agent/test_regional_graph.py`
- Test: `tests/runtime/test_mission_controller.py`
- Test: `tests/integration/test_agent_loop.py`

**步骤：**

- [ ] 写失败集成测试：连续 `active_ping`、组报告、普通目标机动和正常载体投放/回收只写入审计/记忆，不增加 `plan_version`，不调用战略 LLM。
- [ ] 写失败测试：质量退化、通信断链、协方差越界和资源阈值经过持续确认后只产生一个合并的重规划批次；恢复后再次退化可以产生新的 episode。
- [ ] 写失败测试：UUV 失效、航程耗尽、能量耗尽和任务窗口违约立即触发一次关键事件；多个 UUV 同周期故障合并，所有来源 ID 保留。
- [ ] 写失败测试：周期性 `strategic_review` 无质量/效能变化时只执行健康检查；`llm_degraded` 不重新进入失败的 LLM 路径。
- [ ] 在 `central.py` 增加 `PlanImpactEvaluator` 接入点，比较当前执行方案与区域覆盖、时间窗、通信、资源、返航和预测走廊；`REGIONAL_REPLAN_EVENT_TYPES` 不再直接提升事件等级。
- [ ] 让 `EventMonitorNode` 先完成事件归一化，再根据影响评估产出一个 `KeyEventBatch`；路由只依据批次的最高影响等级，不依据原始事件名字。
- [ ] 删除/移除 `SimulationEngine._update_fast_regional_replan_events` 与中央质量/通信判断之间的重复控制触发；引擎只发布观测事实，统一 monitor 负责 episode 判定。
- [ ] 修正 `MissionController` 的生命周期去重：正常 dispatch/recovery/health-check 事件只记录一次；持续 coverage degraded 不因计划 revision 变化重新发射，恢复时清理 episode。
- [ ] 将 `_feedback_events` 中的电池逻辑改为进入轮换状态边沿；低电量只触发确定性轮换，只有覆盖/资源可行性受影响才升级到 LLM 方案调整。
- [ ] 运行中央图、区域图、mission controller 和 agent loop 定向测试。
- [ ] Commit: `fix: route only plan-impacting event batches`

## Task 3：实现每目标最多 4 个最终执行区域

**文件：**

- Create: `src/underwater_tracking/planning/executable_region_limits.py`
- Modify: `src/underwater_tracking/planning/mission_optimizer.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/domain/ui_models.py`
- Test: `tests/planning/test_mission_optimizer.py`
- Test: `tests/agent/test_optimize.py`
- Test: `tests/api/test_uuv_only_frame_contract.py`

**步骤：**

- [ ] 写失败测试：候选区域超过 4 个时，最终 executable mission 和 `regional_plans[target].regions` 都不超过 4 个；完整候选仍可通过候选证据读取。
- [ ] 写失败测试：当前 active 区域及连续后继优先；剩余槽位按概率、优先级、时间窗和代价进行稳定排序；相同输入产生相同选择。
- [ ] 写失败测试：超出上限的候选带有 `region_cap_not_selected` 原因，不进入 UUV 分配、carrier stop、地图执行区域或任务状态集合。
- [ ] 增加共享常量 `MAX_EXECUTABLE_REGIONS_PER_TARGET = 4` 和纯选择函数，避免把 LLM 请求批量常量当成执行上限。
- [ ] 在 UUV-only `MissionOptimizer` 生成 executable plan 前应用上限；在 legacy regional materialization 增加同一防线，避免默认场景绕过限制。
- [ ] 在 UI view 中保留候选/执行状态和未选原因；frame builder 只把执行区域放入执行区域字段，候选证据单独输出。
- [ ] 运行 planning、optimizer、frame contract 定向测试，确认资源分配和路线只消费最多 4 个执行区域。
- [ ] Commit: `fix: cap executable regions per target`

## Task 4：统一持久化帧和实时方案增量协议

**文件：**

- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/api/hub.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/api/frame_logger.py`
- Test: `tests/api/test_live_publisher.py`
- Test: `tests/api/test_hub.py`
- Test: `tests/api/test_frame_pipeline.py`

**步骤：**

- [ ] 写失败测试：logger、HTTP snapshot 和 replay 每帧都含完整 `regional_plans`；WebSocket 在 plan version 不变时不重复发送方案正文。
- [ ] 写失败测试：首帧、计划版本变化和当前没有可用缓存时发送 `plan_payload_status="full"`；版本不变发送 `"unchanged"`；无法保证连续性发送 `"sync_required"`。
- [ ] 写失败测试：实时关键事件窗口不包含普通 `active_ping`/进度/组报告，但持久化事件和 Memory Stream 来源仍完整。
- [ ] 为 `OperationalFrame` 增加兼容的 `plan_payload_status`；构建完整 frame 后再生成 live payload，不能用 live payload 覆盖 logger 或 hub snapshot。
- [ ] 扩展 `OperationalHub` 保存完整 latest frame，同时向订阅者投递可省略区域正文的 live message；有界队列丢弃普通物理帧时保留版本检查语义。
- [ ] 在 publisher 中按上次已发布 `plan_version` 生成 live payload，计划版本变化必须携带完整区域方案。
- [ ] 更新 API payload 转换和 schema 测试，保持旧 replay frame 可读取。
- [ ] 运行 API/live/hub 定向测试。
- [ ] Commit: `fix: publish plan-versioned live frames`

## Task 5：修复 WebSocket 客户端断连和任务回收

**文件：**

- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/api/hub.py`
- Test: `tests/api/test_app.py`
- Test: `tests/api/test_hub.py`
- Test: `tests/api/test_websocket_lifecycle.py`

**步骤：**

- [ ] 写失败测试：客户端在发送首帧、心跳或 `ping` 回复前关闭连接时，不抛出未处理 `RuntimeError`/`OSError`，subscriber 被移除。
- [ ] 写失败测试：send、receive、heartbeat 任一任务先结束时，其余任务都取消并通过 `gather(..., return_exceptions=True)` 回收；不出现 pending task 或未检索异常。
- [ ] 增加连接级幂等关闭标志和受控的 closed-transport 异常转换；只捕获 WebSocketDisconnect、关闭 transport 的 RuntimeError/OSError，不吞掉业务编码错误。
- [ ] 保持发送锁，禁止 heartbeat 与 command response 并发写同一 socket；连接关闭后不再从 hub 取帧。
- [ ] 用 FastAPI TestClient/WebSocket 或受控 fake transport 重现附件日志中的两类异常，运行生命周期测试并检查日志不含 ASGI traceback。
- [ ] Commit: `fix: close operational websockets idempotently`

## Task 6：接入统一记忆输入和 Memory Stream

**文件：**

- Modify: `src/underwater_tracking/memory/service.py`
- Modify: `src/underwater_tracking/memory/worker.py`
- Modify: `src/underwater_tracking/memory/source_reader.py`
- Modify: `src/underwater_tracking/domain/memory_models.py`
- Modify: `src/underwater_tracking/persistence/memory.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/api/app.py`
- Test: `tests/memory/test_memory_service.py`
- Test: `tests/memory/test_memory_worker.py`
- Test: `tests/api/test_memory_api.py`

**步骤：**

- [ ] 写失败测试：关键事件、普通事件、周期性观测批次和专家对话都进入短期上下文，并保留 `source_event_ids`/`source_message_ids`/`plan_version`。
- [ ] 写失败测试：短期阈值触发真实 LLM 摘要；长期筛选/提炼、版本替代、访问和证据 trace 都生成结构化 Memory Stream 事件。
- [ ] 写失败测试：长期检索只能在同一 `user_id`/`scenario_id` 范围内调用本地 sentence-transformers；缺包/缺模型显式 `degraded`，不生成伪向量。
- [ ] 在观测边界把周期性观测压缩为有界结构化 batch 入队，原始帧/事件仍由持久化层保存，避免把高频 raw frame 复制进长期记忆。
- [ ] 复用现有真实 LLM memory filter/summarizer 和本地 embedding provider，补齐事件来源映射和后台 worker 的有界队列；worker 不持有仿真推进锁。
- [ ] 确认长期记忆只作为检索素材，最终方案/证据推理上下文由短期记忆组装；Memory Stream 不与 LLM thinking 混用。
- [ ] 运行 memory service/worker/API 定向测试；涉及模型调用的验收测试必须使用真实 LongCat 和本地模型配置。
- [ ] Commit: `fix: ingest events observations and dialogue into memory flow`

## Task 7：更新前端实时合并和同步状态

**文件：**

- Modify: `src/underwater_tracking/ui/src/types/frames.ts`
- Modify: `src/underwater_tracking/ui/src/state/frameStore.ts`
- Modify: `src/underwater_tracking/ui/src/hooks/useWebSocket.ts`
- Modify: `src/underwater_tracking/ui/src/hooks/useReplay.ts`
- Modify: `src/underwater_tracking/ui/src/components/BottomDrawer.tsx`
- Test: `src/underwater_tracking/ui/src/state/frameStore.test.ts`
- Test: `src/underwater_tracking/ui/src/hooks/useWebSocket.test.ts`
- Test: `src/underwater_tracking/ui/src/hooks/replayApi.test.ts`
- Test: `src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx`

**步骤：**

- [ ] 写失败 TypeScript/Vitest 测试：`full` 替换区域方案，`unchanged` 保留缓存，`sync_required` 清除当前方案状态并触发 snapshot refresh。
- [ ] 写失败测试：收到新的 `plan_version` 但没有 `regional_plans` 时不能继续把旧方案标记为当前；收到版本跳跃时请求快照。
- [ ] 更新 `OperationalFrame` 类型和 stream message 判定，保持快照/回放完整帧兼容。
- [ ] 在 `frameStore` 中实现按 `frame_id`/`sim_time_s` 排序和按 plan version 合并；WebSocket hook 管理同步中状态和一次性补偿请求，避免循环请求。
- [ ] 在 UI 中只展示实时关键事件，普通事件通过回放/Memory Stream 进入详情，不恢复前端 mock 事件。
- [ ] 运行 UI unit tests、`npm run build` 和真实后端联调 smoke。
- [ ] Commit: `fix: merge plan-versioned operational frames in ui`

## Task 8：真实链路验收、长时测试和合并

**文件/环境：**

- Modify: `docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md` only if implementation reveals a contract discrepancy.
- Modify: `src/underwater_tracking/ui/FRONTEND_INTEGRATION.md` only if the actual wire schema needs a documented correction.
- Test: backend full pytest, frontend Vitest/build, Playwright acceptance, real provider smoke.

**步骤：**

- [ ] 运行后端定向测试、`ruff check`、前端测试/build；确认计划、文档和 wire schema 一致。
- [ ] 预检真实 provider：LongCat chat/structured endpoint 可用；`sentence-transformers` 已安装；本地模型权重可用且 `local_files_only=true`；SQLite/JSONL 路径可写。
- [ ] 使用真实 LLM 和本地 Embedding 启动 `main.py`，验证自动端口、API、UI、WS、Memory Stream 和智能助理接口均使用同一真实后端。
- [ ] 以 100x 运行完整 8 小时仿真，记录 plan revisions、关键事件数、普通事件持久化数、WS 关键事件数、每目标执行区域数、记忆版本/摘要数、队列峰值、RSS 和 checkpoint 容器大小。
- [ ] 验收断连：浏览器关闭/刷新和进程 Ctrl+C 后不残留 API/UI 端口、不出现两类 ASGI transport 异常、不留下后台 task。
- [ ] 使用 Playwright 验证地图最多显示 4 个执行区域、方案版本变化、同步中/快照补偿、Memory Steam 三类长期记忆和短期摘要、智能助理方案调整/证据回溯。
- [ ] 运行 `git diff --check`、后端全量 pytest、前端全量测试/build 和固定种子回放校验；不得用 mock provider 作为验收通过依据。
- [ ] 请求代码审查，修复高风险问题后重新验证。
- [ ] 在功能分支提交最终变更，切换主工作树到 `master`，合并功能分支；保留并确认 9 个未跟踪 UI 截图未被修改。

## 执行顺序

按 Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 顺序执行。事件判定和最终区域上限先于实时协议，实时协议先于 UI 合并；记忆输入与事件归一化共享来源 ID，但后台 worker 不阻塞方案思考或物理推进。

## 分支

- 计划提交在：`master`
- 实现分支：`fix/bounded-events-region-cap-live-protocol`
- 实现完成后：在主工作树切回 `master`，合并实现分支并运行最终验收。
