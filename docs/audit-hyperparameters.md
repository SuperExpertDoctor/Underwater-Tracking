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
