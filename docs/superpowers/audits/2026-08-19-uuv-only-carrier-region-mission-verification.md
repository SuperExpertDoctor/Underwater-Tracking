# UUV-only carrier-region production verification

日期：2026-08-20
设计：`docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md`
执行计划：`docs/superpowers/plans/2026-08-20-uuv-only-production-closed-loop-plan.md`
实现分支：`fix/uuv-only-production-loop`

## 结论

UUV-only 链路已闭环：`ExecutableMissionPlan` 是唯一执行契约；LLM 只生成候选策略，确定性校验、资源/能力检查、Hungarian 槽位约束、A* 完整返航和时间窗校验通过后才进入 `MissionController` 与真实载体物理执行。旧 `TrackingPlan` 仅保留为非 UUV 场景及历史审计/回放兼容视图，不再控制 UUV-only 仿真。

## 需求—证据矩阵

| 设计要求 | 实现与证据 | 结果 |
|---|---|---|
| UUV 承担探测/跟踪，母舰只做后勤 | `mission_optimizer.py`、`mission_controller.py`、UUV-only frame；新帧无 `usvs` | 通过 |
| UUV 能力与资源持续监视 | `UUVResourceState`、`MissionController` 资源轮次、健康/主动能力事件、`UUVResourceView` | 通过 |
| ACTIVE_SCAN→PASSIVE_TRACK | 载体投放后由真实引擎按候选时间窗回传 entry estimate，控制器连续确认 | 通过 |
| 任务轮转与资源轮转 | `handoff_to` 就绪判定、前继 UUV `RETURN_REQUIRED`、回收健康检查、资源轮次重置 | 通过 |
| 母舰多站点与最终回家 | `CarrierTaskPlanner`、`HungarianMatcher`、`AStarRoutePlanner`、`CarrierEntity` | 通过 |
| 时间窗、容量、未来 reserve 与禁行区域 | Hungarian 增量路线/ETA/ready reserve；任务路线完整校验；迟到拒绝 | 通过 |
| 里程/能量耗尽与故障轮换 | `uuv_range_exhausted`、`uuv_energy_depleted`、`uuv_failed`、回收及重规划 | 通过 |
| 事件触发快速重规划 | EventMonitor strategic event 集合、CarrierRuntime → LangGraph → 新 executable revision | 通过 |
| LLM 不可用时安全降级 | 保留上一版 executable plan，记录 `llm_degraded`，不伪造新计划 | 通过 |
| 新帧无 USV、旧回放兼容 | UUV-only frame serializer、ReplayService legacy field stripping | 通过 |

## 关键固定种子证据

```text
PYTHONPATH=src pytest -q
876 passed, 65 skipped, 2 warnings
```

重点测试包括：

- `tests/integration/test_uuv_only_production_acceptance.py`：真实 `_AgentLoop`、`CarrierRuntime`、`SimulationEngine`、`MissionController` 和确定性 LLM provider；双载体、计划 revision 重规划、事件持久化、无 USV 输出。
- `tests/integration/test_uuv_only_physical_execution.py`：真实载体多站点路由、时间窗等待、ACTIVE_SCAN、PASSIVE_TRACK、交接、目标退出预测、UUV 回收和 ONBOARD 复位。
- `tests/planning/test_hungarian.py`、`test_carrier_tasks.py`：载体归属、容量、未来 reserve、不可达返航和时间窗拒绝。
- `tests/api/test_uuv_only_frame_contract.py`：资源里程/能量/能力轮次出现在 UUV-only 操作帧，且不生成 USV 字段。

## 前端验证

```text
npm test -- --run       # 22 files, 81 tests passed
npm run build           # passed
```

定向 Playwright 命令运行了 2 个测试：1 个跳过，1 个历史截图基线失败。失败为既有 `command-center-carrier-returning` Canvas 截图 88 像素差异，未涉及新 UUV-only 资源字段或任务状态逻辑，作为独立视觉基线债务保留。

后端仅有两条既有 Pydantic `validation_alias` warning；未新增失败或未处理异常。
