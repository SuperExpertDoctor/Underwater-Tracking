# Long-Running Runtime and Data-Flow Hardening Implementation Plan

> 本计划在 `master` 上制定，设计已获用户确认。实现必须在新隔离分支中进行；每个任务先增加失败测试，再修改生产代码。

## 目标

修复长时间算法运行中的检查点膨胀、后台 LLM 过期结果、无界内存、实时帧时效不明确、payload 重启失效、Replay 全量加载和入口边界不一致问题，形成可持续运行的后端数据闭环。

## 全局约束

- 不改变 UUV-only 任务规划和物理执行契约；`ExecutableMissionPlan` 仍是唯一执行计划。
- 不删除 SQLite/JSONL 完整审计数据；只限制实时内存和单次 API 读取规模。
- 不用 wall-clock 的绝对耗时阈值替代功能断言；性能测试优先断言增长上界和趋势。
- 所有生产修改均由对应回归测试覆盖，并保持 Python 3.11/3.12 兼容。

## Task 1：建立长时基线与 retention 配置

**文件：**

- Modify: `src/underwater_tracking/config/models.py`
- Modify: `configs/agent.yaml`
- Test: `tests/config/test_*`
- Create: `tests/integration/test_long_running_runtime.py`

**步骤：**

- [ ] 写失败测试，验证默认配置可表达 checkpoint、history、event、payload、frame 和 directive retention；验证非法值被拒绝。
- [ ] 写失败长时测试，运行显式 platform-core 240 个 tick，断言 group saver 的 checkpoint/writes/blobs 不随 tick 数线性增长，并记录每 30 tick 的耗时样本。
- [ ] 增加带明确默认值的 runtime retention 配置，默认值满足 8 小时场景且不影响预测器最小历史需求。
- [ ] 运行定向配置和长时测试，确认先失败测试变绿。
- [ ] Commit: `test: define long-running runtime bounds`

## Task 2：实现有界 group checkpointer

**文件：**

- Create/Modify: `src/underwater_tracking/groups/checkpoint.py`
- Modify: `src/underwater_tracking/groups/manager.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/groups/state.py`
- Modify: `src/underwater_tracking/groups/nodes.py`
- Test: `tests/groups/test_group_graph.py`
- Test: `tests/integration/test_platform_core_scenario.py`

**步骤：**

- [ ] 写失败测试，反复 invoke 一个 group 后断言 checkpoint、writes、blobs 被限制在配置上限内，且最新 report/belief 与无裁剪 saver 相同。
- [ ] 写失败回归，注入 carrier failure 后 engine rollback，下一次成功 tick 与 reference engine 帧完全一致。
- [ ] 实现 `BoundedInMemorySaver`，按 thread 保留最近 checkpoint，清理无引用 writes/blobs，支持 delete_thread。
- [ ] 将 `GroupManager` 默认 saver 切换为有界实现；保持测试传入的自定义 saver 不受影响。
- [ ] 对 group emitted events 做 tail 保留，质量窗口继续按时间窗口计算。
- [ ] 运行 group、rollback 和长时定向测试。
- [ ] Commit: `fix: bound group checkpoint retention`

## Task 3：限制 engine 和 MissionController 的内存历史

**文件：**

- Modify: `src/underwater_tracking/simulation/engine.py`
- Modify: `src/underwater_tracking/runtime/mission_controller.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Test: `tests/runtime/test_mission_controller.py`
- Test: `tests/integration/test_uuv_only_physical_execution.py`
- Test: `tests/integration/test_long_running_runtime.py`

**步骤：**

- [ ] 写失败测试，验证 belief history 覆盖预测窗口并有上限，旧样本不影响预测结果。
- [ ] 写失败测试，验证 engine event ledger、mission event tail、processed event IDs 和 controller forwarding IDs 有界，且 SQLite event IDs 仍唯一。
- [ ] 将 history/event/dedupe 容器改为带淘汰的 deque/ordered set，并提供统一 append/evict helper，避免只裁剪 list 而遗漏 ID 集合。
- [ ] 让 `MissionController.snapshot()` 返回有界 event tail，但保留完整当前 region/resource/mode/episode state。
- [ ] 运行资源轮转、handoff、故障和长时测试，确认事件类型和事件顺序不变。
- [ ] Commit: `fix: bound runtime event and belief history`

## Task 4：增加 durable payload store 和重启恢复

**文件：**

- Modify: `src/underwater_tracking/persistence/sqlite.py`
- Create/Modify: `src/underwater_tracking/persistence/payloads.py`
- Modify: `src/underwater_tracking/persistence/checkpoints.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/nodes/snapshot.py`
- Modify: `src/underwater_tracking/agent/nodes/optimize.py`
- Test: `tests/persistence/test_payloads.py`
- Test: `tests/integration/test_agent_loop.py`

**步骤：**

- [ ] 写失败测试：写入 snapshot/candidate，关闭 runtime，使用同一 SQLite 重建 runtime，按旧 reference 读取并完成下一 cycle。
- [ ] 写失败测试：重复 upsert 相同 reference 不产生重复 payload；cache 和数据库保留窗口有界。
- [ ] 增加 `runtime_payloads` schema 和带类型 JsonPlus 序列化的 `PayloadStore` MutableMapping 适配器。
- [ ] CarrierRuntime 使用 payload store 替换进程私有 dict；保留最近 references 的 bounded cache。
- [ ] 处理旧数据库无 payload 表的幂等迁移；缺失 payload 时产生可审计 node error，不让异常污染物理线程。
- [ ] 运行 payload、restart、central graph 和已有 integration tests。
- [ ] Commit: `fix: persist and bound carrier planning payloads`

## Task 5：实现 latest-value background mailbox 和 freshness guard

**文件：**

- Modify: `src/underwater_tracking/agent/graphs/central.py`
- Modify: `src/underwater_tracking/agent/nodes/commit.py`
- Modify: `src/underwater_tracking/agent/runtime.py`
- Modify: `src/underwater_tracking/cli.py`
- Modify: `src/underwater_tracking/agent/state.py`
- Test: `tests/agent/test_runtime_master_slave_adversary.py`
- Test: `tests/agent/test_central_graph.py`
- Test: `tests/integration/test_agent_loop.py`

**步骤：**

- [ ] 写失败测试：后台 provider 阻塞期间送入多个 observation，释放后最新 situation 被处理，事件不丢失。
- [ ] 写失败测试：cycle 运行期间 revision 前进时，旧结果返回 `stale`，不产生新的 active plan 或物理控制写入。
- [ ] 写失败测试：旧 cycle 被丢弃后 mailbox 自动运行最新 cycle；LLM 错误时 latest event/control 仍可重试。
- [ ] 为 central graph 增加独立 current-revision provider，在 CommitNode 和 UUV-only commit 分支写入数据库前执行 freshness guard。
- [ ] 将 `_AgentLoop._background_cycle` 改为 captured situation + latest pending mailbox，并让 graph cycle 始终读取 captured situation。
- [ ] 只在 revision 一致的 physics boundary 应用 mission plan、sensor、slave 和 adversary decisions；过期结果进入状态/计数审计但不执行。
- [ ] 运行后台阻塞、故障恢复、计划提交和同步模式回归测试。
- [ ] Commit: `fix: protect background planning from stale cycles`

## Task 6：补齐操作帧时效和 bounded mission event 数据流

**文件：**

- Modify: `src/underwater_tracking/domain/ui_models.py`
- Modify: `src/underwater_tracking/api/frame_builder.py`
- Modify: `src/underwater_tracking/api/live.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/api/test_frame_contracts.py`
- Test: `tests/api/test_live_publisher.py`
- Test: `tests/api/test_uuv_only_frame_contract.py`

**步骤：**

- [ ] 写失败测试：物理时间前进而 planning revision 未更新时，帧标记 `stale` 且 age 非负；没有 graph state 时标记 `unavailable`。
- [ ] 写失败测试：mission event 数量超过 retention 后帧只带 tail，资源/region/mode/episode 状态完整。
- [ ] 增加 planning revision/time/age/status 字段及 legacy defaults。
- [ ] Publisher 使用 observation cadence 计算物理 revision，读取 checkpoint state 的 semantic metadata，统一排序并限制事件 tail。
- [ ] 确认 UUV-only serializer 仍不输出 USV，旧 frame replay 仍可读。
- [ ] Commit: `fix: expose bounded operational data freshness`

## Task 7：限制 Replay、RunCatalog 和 directive queue

**文件：**

- Modify: `src/underwater_tracking/api/replay.py`
- Modify: `src/underwater_tracking/api/app.py`
- Modify: `src/underwater_tracking/runtime/run_catalog.py`
- Modify: `src/underwater_tracking/api/dependencies.py`
- Modify: `src/underwater_tracking/api/hub.py`
- Test: `tests/api/test_frame_pipeline.py`
- Test: `tests/api/test_app.py`
- Test: `tests/runtime/test_run_catalog.py`
- Test: `tests/api/test_hub.py`

**步骤：**

- [ ] 写失败测试：Replay 分页只读取 bounded frames，非法 limit/offset 被拒绝；旧范围查询的小数据结果不变。
- [ ] 写失败测试：RunCatalog summary 使用 index count/last frame，不复制全量 payload。
- [ ] 写失败测试：directive queue 达到上限时不会无限增长，终态 job 按 retention 淘汰，running job 不被淘汰。
- [ ] 为 ReplayService 增加 offset/limit/count/last；API 设置默认和最大 page size。
- [ ] 改造 RunCatalog summary，移除无界 `range()`。
- [ ] 为 directive jobs 增加 bounded admission、终态淘汰和可识别 queue-full 错误；关闭 executor 时等待已运行任务进入安全状态。
- [ ] Commit: `fix: bound replay and operator work queues`

## Task 8：统一 UUV-only 入口并完善关闭流程

**文件：**

- Modify: `src/underwater_tracking/runtime/run_controller.py`
- Modify: `src/underwater_tracking/cli.py`
- Test: `tests/runtime/test_run_controller.py`
- Test: `tests/agent/test_runtime_master_slave_adversary.py`
- Test: `tests/integration/test_uuv_only_runtime_entrypoints.py`

**步骤：**

- [ ] 写失败测试：仅 `environment.uuv_only` 为 true 时 RunController 也创建 MissionController，并与 CLI/engine/frame 使用同一边界。
- [ ] 写失败测试：后台线程关闭时不再访问已经关闭的 runtime/repository；closing 后不接受新的 mailbox situation。
- [ ] 抽取并复用 `_is_uuv_only_config` / `_mission_controller_for`，清理重复判定。
- [ ] 增加 closing 状态和安全 join；对无法在关闭窗口结束的 provider 记录错误并保证文件/数据库关闭顺序明确。
- [ ] 运行 runtime entrypoint、background close 和 UUV-only acceptance tests。
- [ ] Commit: `fix: align runtime entrypoints and shutdown`

## Task 9：全量验证与审查

**步骤：**

- [ ] 运行定向 pytest、长时 smoke、故障/重启 smoke、ruff。
- [ ] 运行后端全量 `PYTHONPATH=src pytest -q`。
- [ ] 运行前端 `npm test -- --run` 和 `npm run build`。
- [ ] 使用固定种子跑 8 小时等价 simulation/agent smoke，记录 tick 分段耗时、RSS、checkpoint 容器、event/payload/frame 数量。
- [ ] 检查 `git diff --check`、工作区变更、文档与实现一致性。
- [ ] 使用 `requesting-code-review` 做独立代码审查，修复阻断问题并重新验证。
- [ ] Commit: `test: verify long-running dataflow hardening`（仅在确有独立验证变更时提交）

## 建议执行顺序

按 Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 顺序执行。Task 2/3 先降低物理线程和内存风险，Task 4/5 再稳定 carrier 数据流，Task 6/7 最后收敛外部 API 读取边界，避免在数据源未稳定前调整展示层。
