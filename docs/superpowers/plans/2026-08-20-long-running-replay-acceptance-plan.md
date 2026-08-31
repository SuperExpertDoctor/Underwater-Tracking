# 长时闭环运行与 100x 回放实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复长时间运行中的时间、角度、资源计划和 UI 状态一致性问题，接入真实回放全量分页，支持 1x/4x/10x/.../100x 演示倍率，并在合并后的 `master` 上完整验收 8 小时仿真与 100x 回放。

**Architecture:** `SimulationEngine` 继续负责物理时钟和观测，`GroupManager` 接收显式周期时刻并在空观测周期预测，`MissionController` 只接受与运行时母舰集合完全一致的可执行计划，`LivePublisher` 以当前周期事件投影 UI 状态。前端 `useReplay` 按后端 `total_count` 分页加载真实帧，`PlaybackBar` 只修改演示计时，使用相邻帧仿真时间差驱动播放。

**Tech Stack:** Python 3、pytest、Pydantic、LangGraph、FastAPI、TypeScript、React、Vitest、Testing Library、Playwright、Vite。

## 全局约束

- 所有代码修改使用 `apply_patch`；不恢复或删除工作区中已有的 9 个未跟踪 UI 截图。
- 设计文档和本计划先在 `master` 提交；实现必须在新的隔离分支/ worktree 完成，完成后合并回 `master`，最终当前工作区停留在 `master`。
- 不引入业务 mock 数据，不用修改断言或放宽 schema 来掩盖真实数据缺陷；测试替身只能用于确定性 LLM、HTTP 边界或纯 UI 组件行为。
- 后端仿真时间固定覆盖 `0..28800s`；100x 只改变回放 wall-clock 延时，不改变 `sim_time_s`、事件顺序或实时运行路径。
- 每项任务先写失败测试，运行最小测试确认失败，再实现，运行最小测试确认通过，最后提交该项变更。

## 任务 1：修复航向角半开区间边界

**Files:**

- Modify: `src/underwater_tracking/simulation/kinematics.py`
- Test: `tests/simulation/test_kinematics.py`

- [ ] 新增边界回归测试：对 `-pi` 附近的浮点输入和多圈输入调用 `wrap_angle`，断言结果满足 `-pi <= result < pi`，并覆盖当前曾产生正 `pi` 的 `-pi - epsilon` 场景。
- [ ] 运行 `pytest -q tests/simulation/test_kinematics.py -k wrap`，确认新测试在修复前失败。
- [ ] 调整 `wrap_angle`，对非有限输入显式拒绝，并使用稳定的余数计算和正 `pi` 边界收敛，确保 `advance_motion` 写出的航向永远符合半开区间。
- [ ] 运行完整 `pytest -q tests/simulation/test_kinematics.py`，确认通过；提交 `fix: keep motion headings in half-open range`。

## 任务 2：让空观测周期推进编组时钟

**Files:**

- Modify: `src/underwater_tracking/groups/state.py`
- Modify: `src/underwater_tracking/groups/manager.py`
- Modify: `src/underwater_tracking/groups/nodes.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/groups/test_group_graph.py`
- Test: `tests/integration/test_agent_loop.py`

- [ ] 在编组图测试中增加“有效观测后空观测但周期时刻前进”的测试，传入显式 `sim_time_s`，断言 belief 时间前进、`last_accepted_sim_time_s` 保留上次有效更新时间、质量 age 增长。
- [ ] 运行该测试确认当前空批次仍停留在旧 belief 时间。
- [ ] 在 `GroupState` 增加可序列化的当前周期时刻字段；在 `GroupManager.invoke` 增加可选 `sim_time_s` 参数并将其写入图输入；不传该参数的旧调用保持原有兼容行为。
- [ ] 修改 `predict_and_update`，将当前周期时刻、最新观测时刻和 belief 时刻取最大值，空观测也执行预测；只在实际接收有效观测时更新 `last_accepted_sim_time_s`。
- [ ] 修改 legacy 和 platform-core 两条 engine 观测路径，始终传入当前 `sim_time_s`。
- [ ] 运行 `pytest -q tests/groups/test_group_graph.py tests/integration/test_agent_loop.py -k 'predict_only or checkpoint_failure'`，确认空观测后的报告时间不重复、不倒退；提交 `fix: advance group time on empty observation cycles`。

## 任务 3：拒绝不完整的母舰执行计划

**Files:**

- Modify: `src/underwater_tracking/planning/mission_validation.py`
- Modify: `src/underwater_tracking/simulation/engine.py`
- Test: `tests/simulation/test_physical_execution.py` 或现有对应 UUV-only 物理执行测试文件
- Test: `tests/planning/test_mission_validation.py` 或现有对应验证测试文件

- [ ] 增加计划验证测试，分别覆盖缺失 live carrier、额外未知 carrier、carrier mission key 不一致；断言返回稳定、可审计的 issue code。
- [ ] 增加 engine 应用测试，构造只含部分 `carrier_missions` 的 otherwise-valid 计划，断言不会安装部分任务。
- [ ] 在验证器中比较 `set(plan.carrier_missions)` 与平台快照的 carrier ID 集合，缺失和额外集合都产生明确问题；保持旧快照单 carrier 兼容路径可用。
- [ ] 在 `SimulationEngine.apply_verified_mission_plan` 中把子集检查改为严格集合相等，并在任何后续 route/task 校验失败时保持原计划不变。
- [ ] 运行现有 UUV-only 物理执行、任务验证和 mission acceptance 测试；提交 `fix: require complete carrier mission coverage`。

## 任务 4：隔离历史事件与当前周期 UI 状态

**Files:**

- Modify: `src/underwater_tracking/api/live.py`
- Test: `tests/api/test_live_publisher.py`

- [ ] 增加回归测试：事件账本含有旧的 directive/replan 事件，而当前快照没有当前周期事件时，断言 `llm_thinking_trigger` 使用当前周期评估或当前计划状态，不复用历史最后一条事件。
- [ ] 增加测试覆盖当前周期事件仍能正确触发 `event_trigger`/`dynamic_adjustment`，人工反馈事件在当前窗口仍能显示。
- [ ] 修改 `_operator_thinking`，`latest_event` 只从 `_current_cycle_events` 取得；没有当前事件时使用周期性评估或等待首轮输入。
- [ ] 修改 `_operational_stage_flags`，历史 sticky state 不能单独伪造当前 `human_feedback` 或 `dynamic_adjustment` 阶段；当前周期事件和当前可执行计划仍应正常投影。
- [ ] 运行 `pytest -q tests/api/test_live_publisher.py`，并检查 JSONL replay 与 hub 帧字段一致；提交 `fix: scope live thinking to current cycle`。

## 任务 5：实现真实回放全量分页加载

**Files:**

- Modify: `src/underwater_tracking/ui/src/state/frameStore.ts`
- Modify: `src/underwater_tracking/ui/src/hooks/useReplay.ts`
- Test: `src/underwater_tracking/ui/src/state/frameStore.test.ts`
- Test: `src/underwater_tracking/ui/src/hooks/useReplay.test.ts`（如不存在则新增纯 hook/API helper 测试）

- [ ] 先增加纯函数测试，验证多页数据按 `total_count` 合并、按 `(sim_time_s, frame_id)` 去重排序，并验证 5760 个 5 秒帧不会被旧的 600 帧上限截断。
- [ ] 将客户端回放上限改为明确覆盖 8 小时的值（例如 `10000`），并保留显式内存边界；达到上限但后端仍有未加载数据时返回可识别错误，而不是静默丢帧。
- [ ] 扩展回放响应类型，读取 `total_count`、`offset`、`limit`；`loadRange` 按稳定页大小循环请求 `/api/replay`，直到已读数量达到 `total_count`，每页使用真实 API 返回的帧。
- [ ] 为分页过程处理空页、重复页、HTTP 错误和响应总数不一致；加载失败时清空回放并保留用户可见错误。
- [ ] 保持 replay 与 live store 分离，保留时间范围输入、marker 生成和 seek 行为。
- [ ] 运行 UI 的 frame store、hook 相关 Vitest 测试；提交 `fix: load complete replay ranges from real api`。

## 任务 6：支持 1x 到 100x 的回放控制

**Files:**

- Modify: `src/underwater_tracking/ui/src/components/PlaybackBar.tsx`
- Modify: `src/underwater_tracking/ui/src/hooks/useReplay.ts`
- Test: `src/underwater_tracking/ui/src/components/PlaybackBar.test.tsx`
- Test: `src/underwater_tracking/ui/src/hooks/useReplay.test.ts`（如需要）

- [ ] 增加组件测试，断言选择器包含 `1x`、`4x`、`10x`、`20x`、`50x` 和 `100x`，并断言选择后调用数值倍率回调。
- [ ] 将 `SPEEDS` 设为稳定的产品选项集合；默认保持现有 `1x` 兼容行为，演示验收显式选择 `100x`。
- [ ] 将固定 `setInterval` 改为按当前帧与下一帧的 `sim_time_s` 差值计算的可取消定时器；倍率只作用于 wall-clock 延时，最小延时受保护，重复时间戳不会卡住。
- [ ] 到达末帧时停止播放，保持末帧和 `28800s` 时间读数，不循环、不跳回首帧。
- [ ] 运行 UI 单元测试和生产构建；提交 `feat: add variable-speed replay playback`。

## 任务 7：清理已知非实时回归并完成全套自动化测试

**Files:**

- Inspect and modify only the owning implementation/tests for the five baseline failures:
  `src/underwater_tracking/agent/counterfactual.py`,
  `src/underwater_tracking/planning/allocation.py`,
  `src/underwater_tracking/agent/nodes/strategy.py`,
  `src/underwater_tracking/agent/nodes/verify.py` and their focused tests.

- [ ] 在实现任务 1-6 后重跑 baseline 中的五个失败测试，区分由本次时钟改动引起的回归和已有契约不一致。
- [ ] 对仍失败的测试先定位数据/语义根因，补充最小回归测试后修复实现；禁止通过删除 semantic validation、放宽 schema 或把生产分支改回 mock 来消除失败。
- [ ] 运行非 real-LLM 全套 Python 测试，记录剩余仅因外部凭据/网络跳过的测试；运行 UI `npm test` 和 `npm run build`。
- [ ] 提交 `fix: restore deterministic agent regression contracts`（仅在确有实现变更时提交）。

## 任务 8：构建确定性 8h 运行与真实回放验收

**Files:**

- Test/utility: `tests/integration/test_uuv_only_8h_replay_acceptance.py`（或按现有 acceptance fixture 组织方式放置）
- Modify only if needed: `tests/integration/test_uuv_only_production_acceptance.py`
- Test: `tests/api/test_frame_pipeline.py`（补充大于默认页大小的 replay 分页合同测试）
- Test/utility: `src/underwater_tracking/ui/tests/e2e/` 或现有真实后端 Playwright 验收入口

- [ ] 复用现有确定性 `FixedSeedUUVLLM`/runtime fixture，生成完整 `0..28800s` 操作帧 JSONL；不调用外部真实 LLM，不把其返回值写成业务 mock。
- [ ] 对全量帧执行 schema、时间单调性、航向半开区间、报告时间、事件时间和 carrier mission 集合检查；断言至少包含计划修订、目标/区域事件、UUV 资源变化以及投放/回收或轮换数据。
- [ ] 用 `ReplayService` 通过多个 offset/limit 页面读取完整范围，断言累计数量等于 `total_count`，首尾时间覆盖验收范围。
- [ ] 为真实浏览器验收准备临时 FastAPI + 真实 `ReplayService` 后端和当前 Vite UI，禁止静态业务 mock；加载完整回放后选择 `100x`，等待最后一帧，断言 UI 显示 `28800s`、播放按钮已停止且请求过所有分页。
- [ ] 运行长测时先执行 `python ... --help` 检查 webapp-testing server helper，再按 helper 要求启动后端和 UI；记录实际 wall-clock 时长和帧数。
- [ ] 长测单独运行命令：`pytest -q tests/integration/test_uuv_only_8h_replay_acceptance.py`；浏览器验收使用 `npm run test:e2e` 配合真实后端环境变量。
- [ ] 提交 `test: add full duration replay acceptance`。

## 任务 9：合并前验证与回到 master

- [ ] 在 feature worktree 运行针对性 Python 测试、全套非 real-LLM Python 测试、UI Vitest、UI build、8h 后端验收和真实后端 Playwright 验收。
- [ ] 使用 `git diff --check`、`git status --short` 和提交历史检查实现范围；保留用户已有未跟踪截图，不把它们作为本次功能提交内容。
- [ ] 使用 `requesting-code-review` skill 做一次以缺陷和数据质量为先的审查，修复审查发现并重新验证。
- [ ] 使用 `finishing-a-development-branch` skill，确认所有验证通过后将 feature 分支合并到本地 `master`。
- [ ] 在合并后的 `master` 再运行最小关键验收和 `git status --short --branch`，确认当前分支为 `master`，向用户报告实际测试结果、8h 帧数、100x 回放最终时间和任何外部依赖限制。
