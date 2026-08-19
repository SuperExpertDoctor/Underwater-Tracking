# UUV-only carrier region mission verification

日期：2026-08-20
设计：`docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md`
计划：`docs/superpowers/plans/2026-08-19-uuv-only-carrier-region-mission-plan.md`
执行分支：`feature/uuv-only-carrier-region-mission`

## 验收范围

本记录验证第一阶段单目标、多 UUV、多母舰任务链：预测概率栅格和方形候选区域、LLM
候选策略校验、滚动 UUV 编组与未来储备、母舰多站点投放/回收、Hungarian/A* 路由、
MissionController 生命周期、UUV-only 操作帧和旧回放兼容。

固定验收输入为 seed `20260820`、provider
`deterministic-test-provider-v1`。目标真值不进入操作帧、计划或回放哈希。

## 需求—证据矩阵

| 设计要求 | 实现边界 | 验证证据 | 结果 |
|---|---|---|---|
| UUV 承担探测/跟踪，母舰只做后勤 | `config`, `mission_models`, `MissionController`, frame builder | UUV-only config/domain tests；固定种子 frame payload 无 `usv` | 通过 |
| 概率栅格和方形候选区域确定性生成 | `planning/prediction_grid.py`, `planning/candidate_regions.py` | prediction-grid、candidate-region focused tests；revision/cell hash | 通过 |
| LLM 只能选已生成候选 | `planning/regional_plan_validator.py`, regional strategy nodes | unknown region/UUV、越界几何、缺证据和 malformed candidate tests | 通过 |
| 当前任务与未来区域联合优化 | `planning/mission_optimizer.py` | batch marginal benefit、future reserve、degraded/uncovered focused tests | 通过 |
| 母舰连续多站点任务且最终回家 | `planning/carrier_tasks.py`, `planning/astar.py`, `planning/hungarian.py` | carrier-task、A*、Hungarian tests；闭环 route 首尾同 home 且避开 forbidden | 通过 |
| 区域状态与 UUV 模式由单一控制器维护 | `runtime/mission_controller.py` | entry threshold 连续确认、handoff、recovery、failure/event tests | 通过 |
| 里程/能量耗尽触发回收和重规划 | `MissionController`, mission optimizer/controller events | 固定 trace 含 `uuv_range_exhausted`、`RETURN_REQUIRED`、计划 revision 递增 | 通过 |
| 无效或不可用 LLM 保留上一版计划 | plan validator/controller revision gate | malformed candidate rejected；stale revision rejected；previous verified revision retained | 通过 |
| 新帧不发布 USV，旧帧只读兼容 | `api/frame_builder.py`, `api/legacy_frame_adapter.py`, replay service | `test_uuv_only_replay_acceptance.py`：JSONL hash 稳定、旧 `usvs` 被忽略 | 通过 |
| UI 展示概率证据、区域、UUV 模式和母舰路线 | `ui/src/components/CanvasMap.tsx`, `RegionOverlay.tsx`, `CarrierStatusPanel.tsx` | Vitest 22 files/81 tests；production build | 通过 |
| 任务区域保持黄色语义，概率只影响证据层 | Canvas map/region overlay | `MISSION_REGION_FILL = rgba(245, 194, 64, 0.66)`；probability evidence grid separate | 通过 |
| 旧运行时兼容和公开输出安全 | `runtime/__init__.py`, `cli.py`, legacy views | import-order regression、manifest seed privacy、legacy degraded view tests | 通过 |

## 固定种子闭环命令

```bash
PYTHONPATH=src pytest -q \
  tests/integration/test_uuv_only_mission_acceptance.py \
  tests/api/test_uuv_only_replay_acceptance.py
```

结果：`2 passed`。验收 trace 内包含三个 plan revisions、两个 grid revisions、三个
deployment stops、至少六个 carrier tasks、handoff、range recovery、intent/confidence
events、degraded resource lifecycle 和 stable frame/route/plan hashes。

## 回归门禁

```bash
PYTHONPATH=src pytest -q
```

结果：`848 passed, 65 skipped, 2 warnings`（Python 3.13.11；项目声明的正式支持范围仍为
Python 3.11/3.12）。两条 warning 来自既有 Pydantic `validation_alias` 用法，不影响本次
UUV-only 验收。

前端：

```bash
npm test -- --run
npm run build
```

结果：22 个测试文件、81 个测试通过，构建通过。

Playwright 定向命令：

```bash
npm run test:e2e -- --grep "uuv-only|mission|replay"
```

结果为 2 个测试，其中 1 个跳过、1 个历史截图失败。失败只表现为比例尺文字从旧基线
`1 km` 变为当前实际 `2 m`；该 fixture 不包含本方案新增字段，因此没有更新无关截图，
并在 `docs/audit-hyperparameters.md` 保留说明。

## 完成判定

方案代码、固定种子闭环、JSONL 回放、后端全量回归、前端单测和生产构建均通过。剩余
Playwright 截图差异是需要单独确认产品比例尺基线的视觉债务，不是 UUV-only 任务链的
功能失败。
