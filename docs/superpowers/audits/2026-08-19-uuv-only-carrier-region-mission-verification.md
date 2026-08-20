# UUV-only carrier-region production verification

日期：2026-08-20
设计：`docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md`
执行计划：`docs/superpowers/plans/2026-08-20-uuv-only-production-closed-loop-plan.md`
实现分支：`fix/uuv-only-production-loop`

## 结论

复核后的 UUV-only 链路已闭环：`ExecutableMissionPlan` 是唯一执行契约；LLM 只生成候选策略，确定性校验、实时资源预检、Hungarian 槽位约束、A* 完整返航和时间窗校验通过后才进入 `MissionController` 与真实载体物理执行。旧 `TrackingPlan` 仅保留为非 UUV 场景及历史审计/回放兼容视图；UUV-only 运行时的两个旧执行入口会显式拒绝调用。

## 需求—证据矩阵

| 设计要求 | 实现与证据 | 结果 |
|---|---|---|
| UUV 承担探测/跟踪，母舰只做后勤 | `mission_optimizer.py`、`mission_controller.py`、UUV-only frame；新帧无 `usvs` | 通过 |
| UUV 能力与资源持续监视 | `UUVResourceState`、`MissionController` 资源轮次、健康/主动能力事件、执行边界实时能量/里程/能力/健康/轮次/航程预检、`UUVResourceView` | 通过 |
| ACTIVE_SCAN→PASSIVE_TRACK | 载体投放后由真实引擎按候选时间窗回传 entry estimate，控制器连续确认 | 通过 |
| 任务轮转与资源轮转 | 优化器将 predecessor/successor 物化为连续多批 handoff；`handoff_to` 就绪判定、前继 UUV `RETURN_REQUIRED`、回收健康检查、资源轮次重置 | 通过 |
| 母舰多站点与最终回家 | `CarrierTaskPlanner`、`HungarianMatcher`、`AStarRoutePlanner`、`CarrierEntity` | 通过 |
| 时间窗、容量、未来 reserve 与禁行区域 | Hungarian 增量路线/ETA/ready reserve；任务路线完整校验；迟到拒绝 | 通过 |
| 里程/能量耗尽与故障轮换 | `uuv_range_exhausted`、`uuv_energy_depleted`、`uuv_failed`、回收及重规划 | 通过 |
| 事件触发快速重规划 | 公共 IMM/UIF belief 变化产生 `target_intent_changed`/`imm_confidence_shifted`，事件进入 EventMonitor、CarrierRuntime、LangGraph 并生成新 executable revision | 通过 |
| LLM 不可用时安全降级 | 保留上一版 executable plan，记录 `llm_degraded`，不伪造新计划 | 通过 |
| 新帧无 USV、旧回放兼容 | UUV-only frame serializer、ReplayService legacy field stripping | 通过 |
| 旧执行链不可绕过 | UUV-only `apply_tracking_plan()` 和 `apply_plan_command()` 显式拒绝 legacy execution | 通过 |
| 禁止区域边穿越 | A* 对搜索边和最终连续线段均做矩形内部相交检查 | 通过 |

## 关键固定种子证据

```text
PYTHONPATH=src pytest -q
884 passed, 65 skipped, 2 warnings
```

重点测试包括：

- `tests/integration/test_uuv_only_production_acceptance.py`：真实 `_AgentLoop`、`CarrierRuntime`、`SimulationEngine`、`MissionController` 和确定性 LLM provider；双载体、计划 revision 重规划、事件持久化、无 USV 输出。
- `tests/integration/test_uuv_only_physical_execution.py`：真实载体多站点路由、时间窗等待、ACTIVE_SCAN、PASSIVE_TRACK、交接、目标退出预测、UUV 回收和 ONBOARD 复位。
- `tests/planning/test_hungarian.py`、`test_carrier_tasks.py`：载体归属、容量、未来 reserve、不可达返航和时间窗拒绝。
- `tests/api/test_uuv_only_frame_contract.py`：资源里程/能量/能力轮次出现在 UUV-only 操作帧，且不生成 USV 字段。
- `tests/planning/test_mission_optimizer.py`：拓扑区域链生成多批物理 batch，并填充 predecessor/successor handoff。
- `tests/simulation/test_engine.py`：旧执行入口拒绝、注入 USV 不创建/不观测、实时低能量拒绝，以及公共 IMM 事件生成。
- `tests/planning/test_astar.py`：禁行矩形被连续边穿越时拒绝路线。

## 审查整改

独立代码审查发现的阻断项已全部整改并由定向回归覆盖：旧入口绕过、拓扑链仅生成首批、动态位置误作固定返航点、执行边界缺少实时资源复核、缺少公共意图/置信度事件、A* 边穿越、事件幂等粒度和 UUV-only USV 隔离。整改后重新运行全量后端测试并通过。

## 前端验证

```text
npm test -- --run       # 22 files, 81 tests passed
npm run build           # passed
```

定向 Playwright 命令运行了 2 个测试：1 个跳过，1 个历史截图基线失败。失败为既有 `command-center-carrier-returning` Canvas 截图 88 像素差异，未涉及新 UUV-only 资源字段或任务状态逻辑，作为独立视觉基线债务保留。

后端仅有两条既有 Pydantic `validation_alias` warning；未新增失败或未处理异常。
