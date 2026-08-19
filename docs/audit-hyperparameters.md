# 参数审计与修复验收

## 基线

- 基线提交：`2d2c1d4` (`master`)
- 实施分支：`fix/hyperparameter-audit-v2`
- 后端环境：`lang_py310`，Python 3.10.20
- 后端基线：814 passed，9 failed，21 deselected (`real_llm`)
- 前端基线：22 个测试文件、74 个测试通过，`npm run build` 通过
- 项目声明支持 Python `>=3.11,<3.13`。因此 Python 3.10 结果单独记录，不把版本兼容失败当作算法回归。

## 已知基线失败

- `tests/agent/test_questions.py`：2 个 counterfactual 质量目标断言失败
- `tests/agent/test_semantic_nodes.py`：strategy payload evidence 断言失败
- `tests/agent/test_verify_graph.py`：semantic rule 断言失败
- `tests/config/test_regional_config.py`：grid 默认值断言失败
- `tests/domain/test_regional_compatibility.py`：legacy degraded view 断言失败
- `tests/integration/test_agent_loop.py`：checkpoint failure 的报告周期断言失败
- `tests/integration/test_platform_core_scenario.py`：2 个 Python 3.10 `ExceptionGroup` 名称兼容失败

后续每次回归必须记录新增失败测试节点；不能只比较失败总数。

## Top 10 验收矩阵

| 项目 | 修改边界 | 必须证明的行为 | 当前状态 |
|---|---|---|---|
| 1. 步长 | scenario timing、时钟和相关测试 | 5s 配置在 physics、能量、运动和回放语义中一致；观测周期仍准确；固定 seed 长跑无异常 | 待实施 |
| 2. 地图边界 | frame builder fallback 与环境边界 | 默认/显式边界不裁剪有效实体；live frame 使用环境边界；避免新增第二套硬编码边界 | 待实施 |
| 3. 质量阈值 | TrackingConfig、allocation、optimize、EventMonitor | 传入哨兵阈值能贯穿生产调用链；`quality_critical`、`active_quality_floor` 等不同语义不被错误合并 | 待实施 |
| 4. 平台运动学 | USV/UUV 配置和 carrier/uuv motion | 每个 physics tick 的速度、加速度、航向变化受限；单位和配置范围有来源；多步轨迹无瞬时 90 度跳变 | 待实施 |
| 5. IMM | process noise、commanded turns、tracking tests | 转弯轨迹的 RMSE/模型识别率改善或不退化；使用多 seed 容差，不使用单一概率阈值作为唯一验收 | 待实施 |
| 6. LLM 配置 | role config、retry/backoff、speed semantics | role 配置不被 CLI 覆盖；重试有上限；暂停后可重连；`--speed` 明确定义为 physics-time 倍率 | 待实施 |
| 7. 传感器概率化 | Pd、clutter、false alarm、multistatic migration | Pd 单调且可复现；虚警不改变真实目标状态；检测率/虚警率统计可解释；多基地代码删除前完成调用图迁移 | 待实施 |
| 8. 事件锁存 | CarrierRuntime replan event state | 同一事件连续触发只产生一次动作；恢复后可重新触发；异常时锁存状态回滚 | 待实施 |
| 9. 区域 standoff | waypoint candidate、regional allocation | 可行航点距离目标预测满足最小距离；无可行点时显式报告违反原因，不静默伪装为通过 | 待实施 |
| 10. UI 比例尺/回放 | frame contract、CanvasMap、useReplay | 比例尺随地图缩放；回放按 physics step 解释；真实 `main.py` 的桌面/移动端操作和旧帧 fallback 通过 | 待实施 |

## 实施门禁

1. 删除 multistatic API 或测试前，必须通过 `rg` 调用图检查，并先迁移 `usv_multistatic` 配置。
2. 改变 physics step 前，必须跑固定 seed 的 300 步和多 seed 的短跑，记录输出帧数、报告时间、能量和异常。
3. Top 1 传感器改动必须增加统计验收：Pd 曲线、clutter 虚警比例、真实目标 detection rate、滤波状态污染检查。
4. 每个主题提交后运行对应定向测试；全量测试只允许把本文件列出的 9 个基线失败保留，新增失败必须定位并处理。
5. 最终必须在 `lang_py310` 运行 `main.py`，检查 `outputs/` 日志、暂停/重连状态、目标侧动态博弈和真实浏览器交互；另在 Python 3.11 环境验证项目声明的正式支持路径。

## UUV-only carrier region mission addendum（2026-08-20）

本节记录合并 `fix/hyperparameter-audit-v2` 后，依据
`docs/superpowers/specs/2026-08-19-uuv-only-carrier-region-mission-design.md`
执行的 UUV-only 任务闭环。详细需求—证据矩阵见
`docs/superpowers/audits/2026-08-19-uuv-only-carrier-region-mission-verification.md`。

### 分支与变更边界

- 合并提交：`00c544c merge: integrate hyperparameter audit v2`
- master 上的详细计划：`docs/superpowers/plans/2026-08-19-uuv-only-carrier-region-mission-plan.md`
- 执行分支：`feature/uuv-only-carrier-region-mission`
- 执行 worktree：`.worktrees/uuv-only-carrier-region-mission`
- 计划阶段提交：`d928791`
- 方案执行提交：`ca58373`、`dd539bb`、`7301333`、`cdcb2cd`、`b0127f1`、`e84dfad`、`eadca72`、`fa4ee4e`、`bee7072`

原 `fix/hyperparameter-audit-v2` 的未提交改动已先提交为
`31f6c08 feat: add multi-run replay catalog and playback metadata`，再通过
`00c544c` 合并；未提交文件没有被静默丢弃。

### 固定种子验收

验收场景使用 seed `20260820` 和确定性 provider
`deterministic-test-provider-v1`。验收同时覆盖：

- 单目标、多母舰、多区域的预测栅格、候选区域与版本修订；
- UUV 主动扫描、被动跟踪、区域交接、里程耗尽回收；
- 意图变化、IMM 置信度变化和降级事件触发重规划；
- 未来资源不足时的 `DEGRADED` 生命周期；
- 母舰多站点任务、A* 禁行区域和回到 home battle group；
- malformed/stale LLM candidate 被拒绝并保留上一版已验证计划；
- UUV-only operational frame、JSONL replay 及旧帧 `usvs` 字段只读兼容。

执行命令：

```bash
PYTHONPATH=src pytest -q tests/integration/test_uuv_only_mission_acceptance.py tests/api/test_uuv_only_replay_acceptance.py
```

结果：`2 passed`。同一个 seed 重复运行的 trace、plan、route 和 frame hash 完全一致。

### 当前回归结果

在本次执行环境 Python `3.13.11` 下：

- 后端：`PYTHONPATH=src pytest -q` → `848 passed, 65 skipped, 2 warnings`；
- 前端：`npm test -- --run` → 22 个测试文件、81 个测试通过；
- 前端构建：`npm run build` 通过；
- 代码质量：`ruff check`、`python -m compileall -q src tests`、`git diff --check` 通过。

本轮同时修复了合并后暴露的运行时包循环导入、公开 manifest 泄露 seed、legacy
regional view 缺少顶层 `degraded_regions`，并把旧测试迁移到当前 5 秒 physics / 30 秒
observation 时序契约。

Playwright 定向命令仍有一个历史截图基线差异：

```bash
npm run test:e2e -- --grep "uuv-only|mission|replay"
```

该命令运行 2 个测试，其中 1 个跳过、1 个旧截图失败；失败截图仅差异于比例尺文本从基线的
`1 km` 变为当前实际 `2 m`，fixture 没有 UUV-only 新字段，因此未把无关基线截图改写。
