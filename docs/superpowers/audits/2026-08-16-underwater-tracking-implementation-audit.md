# Underwater Tracking 实施审计

审计日期：2026-08-16  
审计工作树：`.claude/worktrees/underwater-tracking-sdd`  
审计范围：`docs/superpowers/plans/` 中的 foundation、agent、UI、tactical-realism、evaluation 与 master roadmap。

## 结论摘要

面向实时指挥台的主链路已经贯通：

```text
SimulationEngine -> CarrierRuntime/LangGraph -> OperationalFramePublisher
                 -> JSONL + OperationalHub -> FastAPI/WebSocket -> React command center
专家预览 -> 明确确认 -> RuntimeDirectiveQueue -> CarrierRuntime event queue
         -> 下一轮 strategic route -> Verify/Optimize/Commit -> 新方案帧
```

本次补齐了 UI 计划中原先缺失的 API、实时帧桥接、地图交互、回放、人工指派确认、证据质询和 evaluation gate，并修复了两个会影响真实运行的缺陷：FIM 舍入负特征值，以及 React StrictMode 下 Canvas 首次 RAF 被取消后不再重绘。

## 已完成或已验证

### Foundation / Agent / Tactical realism

- bearing-only FIM 对数值舍入产生的极小负特征值做有限 ULP 修复，并保留真正奇异几何；聚焦 FIM 测试通过。
- `CarrierRuntime` 仍是 LangGraph checkpoint、pending event 和 SQLite ledger 的唯一拥有者；`apply_directive` 只持久化已确认指令并排入下一轮 `directive_applied` strategic event，当前方案不会被 HTTP 请求直接替换。
- `CarrierRuntime` 现在用可重入锁串行化 `tick`、`resume`、`get_state`、预览、确认和质询，避免 API worker 与仿真线程并发写同一 graph/checkpointer。
- Typed assignment 会写入 `ReservationRegistry`；下一轮图周期通过 snapshot/directive 读到 reservation，allocator 和 verification pinger pool 会避开已锁定资源。
- Strategy prompt 和 payload 明确包含 estimator quality/FIM、资源/能量、reservation、专家约束、active plan version、hard guards，并要求平衡质量、连续性、安全、能量储备、换组代价和 relay coverage。

### UI / API

- `OperationalFrame` 与 `EvaluationFrame` 独立；默认 operational REST/WebSocket/replay 不携带 truth 字段。
- `OperationalFramePublisher` 把真实 runtime state、intent/prediction、stored events、ledger、applied directives、breadcrumbs 和 metrics 持续写入 `operational_frames.jsonl` 并发布到 `OperationalHub`。
- FastAPI 已提供 health、snapshot、动态 replay、directive queue/status/apply、typed assignment、evidence question、bounded WebSocket 和显式 evaluation gate。
- `ReplayService` 会在 live JSONL 追加后刷新 index；不会要求重启服务才能看到新帧。
- React 指挥台包含深海视觉系统、Canvas 网格/预测走廊/航路/轨迹/方位线/协方差/目标/UUV/选中态、状态侧栏、方案/事件/台账/指标抽屉和回放控制。
- 专家指令和 typed assignment 都是“预览 -> 明确确认 -> applying/applied -> 下一轮重规划”的非阻塞流程；方案版本过期会拒绝并要求重新审阅。
- Vite 开发代理已转发 `/api` 与 `/ws` 到 FastAPI；新增 `underwater-tracking cli serve` 将仿真线程、FastAPI 和 WebSocket 连接到同一个 runtime/hub。

### 尚未完成的范围

`2026-08-14-underwater-tracking-evaluation-plan.md` 的正式实验系统仍未实现：目前只有 truth-only API/UI gate 和 evaluation frame contract，没有 B0-2/B0-3/B1/B2/Full policy runner、paired Monte Carlo、metric registry、formal freeze、acceptance CLI 和报告生成。因此不能宣称正式性能验收已经完成；该范围应作为下一阶段独立任务执行，不能用当前 operational UI 测试替代。

UI Task 10 的 truth gate 已完成基础版，但当前浏览器验收使用 deterministic API/WebSocket fixture；完整 headless engine + FastAPI + Vite 三进程 truth-isolation 运行仍属于 evaluation 阶段。

## 验证证据

在目标 worktree 执行：

- `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -m 'not real_llm' -q`：**292 passed, 21 deselected**。
- `PYTHONPATH=src .venv/bin/python -m ruff check src tests`：通过。
- `PYTHONPATH=src .venv/bin/python -m mypy src/underwater_tracking`：**77 source files, no issues**。
- `npm run build`：通过，Vite production bundle 生成成功。
- `npm test`：**10 test files, 22 tests passed**。
- `npm run test:e2e -- --reporter=line tests/e2e/command-center.spec.ts`：**1 passed**，1440×900；覆盖实时帧、Canvas 实际像素绘制、UUV 选择、typed assignment preview/confirm、详情抽屉和 replay。
- `serve` composition smoke：用隔离 fake uvicorn、单步仿真验证 runtime/hub/replay/app wiring，返回 0。

real LLM 标记测试没有在本审计中重复触发，避免将外部网络/供应商延迟混入离线回归；正式验收需要按 evaluation runbook 单独记录 provider、model 和请求结果。
